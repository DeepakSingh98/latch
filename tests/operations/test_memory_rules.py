"""LATCH-TRAIN-MEMORY: ops.memory_rules lifetime sink (pure planning; no device)."""
from megakernel_lab.operation_compiler.lower import TensorLayout
from megakernel_lab.operation_compiler.memory_rules import lifetime_sink


def _layouts(sizes):
    return {n: TensorLayout((k,), (1,), n) for n, k in sizes.items()}


def _phase(reads, writes):
    return {"kind": "elementwise", "reads": list(reads), "writes": list(writes), "code": ""}


def test_backward_body_sinks_to_its_reader():
    # x (forward) -> y = f(x) (only read at the end) ; z reads x late too, so x stays live anyway.
    phases = [_phase([], ["x"]), _phase(["x"], ["y"]), _phase(["x"], ["a"]), _phase(["a"], ["b"]),
              _phase(["b"], ["c"]), _phase(["c", "y", "x"], ["z"])]
    intervals = [[0], [1, 2], [3], [4], [5]]
    layouts = _layouts(dict(x=1024, y=4096, a=16, b=16, c=16, z=16))
    dtypes = {n: "float32" for n in layouts}
    out, record = lifetime_sink(phases, intervals, layouts, dtypes, list(layouts))
    level = {i: k for k, m in enumerate(out) for i in m}
    assert level[1] == level[5] - 1 and record["moved"] >= 1
    # every body still follows its producers
    assert level[0] < level[1] and level[2] < level[3] < level[4] < level[5]


def test_no_move_when_inputs_would_extend_more():
    # y is small, its input x is large and otherwise dead after interval 1: moving y would extend x.
    phases = [_phase([], ["x"]), _phase(["x"], ["y"]), _phase([], ["a"]), _phase(["a"], ["b"]), _phase(["b", "y"], ["z"])]
    intervals = [[0, 2], [1, 3], [4]]
    layouts = _layouts(dict(x=4096, y=16, a=16, b=16, z=16))
    dtypes = {n: "float32" for n in layouts}
    intervals2 = [[0, 2], [1], [3], [4]]
    out, record = lifetime_sink(phases, intervals2, layouts, dtypes, list(layouts))
    level = {i: k for k, m in enumerate(out) for i in m}
    assert level[1] == 1


def test_pinned_and_war_respected():
    # w is overwritten (WAW/WAR through the same root) by body 3: body 1 reading w must stay before it.
    phases = [_phase([], ["w"]), _phase(["w"], ["y"]), _phase([], ["q"]), _phase(["q"], ["w"]), _phase(["y"], ["z"])]
    intervals = [[0, 2], [1], [3], [4]]
    layouts = _layouts(dict(w=16, y=4096, q=16, z=16))
    dtypes = {n: "float32" for n in layouts}
    out, _ = lifetime_sink(phases, intervals, layouts, dtypes, list(layouts))
    level = {i: k for k, m in enumerate(out) for i in m}
    assert level[1] < level[3]
    out, record = lifetime_sink(phases, intervals, layouts, dtypes, list(layouts), pinned={1})
    level = {i: k for k, m in enumerate(out) for i in m}
    assert level[1] == 1 and all(move["phase"] != 1 for move in record.get("moves", ()))


def test_equal_size_chain_sinks_together():
    # q (live late) -> s -> t -> u (same sizes; u also reads a scalar c); u is read only by z after a longer chain.
    names = ["q", "c", "a", "s", "t", "u", "b", "d", "e", "f", "z"]   # program order: writers before readers
    reads = {"s": ["q"], "t": ["s"], "u": ["t", "c"], "b": ["a"], "d": ["b"], "e": ["d"], "f": ["e"], "z": ["f", "u", "q"]}
    phases = [_phase(reads.get(n, []), [n]) for n in names]
    index = {n: k for k, n in enumerate(names)}
    intervals = [[index[n] for n in group] for group in (["q", "c", "a"], ["s", "b"], ["t", "d"], ["u", "e"], ["f"], ["z"])]
    layouts = _layouts(dict(q=1024, s=4096, t=4096, u=4096, c=1, a=16, b=16, d=16, e=16, f=16, z=16))
    dtypes = {n: "float32" for n in layouts}
    out, record = lifetime_sink(phases, intervals, layouts, dtypes, list(layouts))
    level = {i: k for k, m in enumerate(out) for i in m}
    assert level[index["u"]] == level[index["z"]] - 1      # u computed just before its reader despite the scalar c
    assert level[index["t"]] == level[index["u"]] - 1      # the equal-size links sank together
    assert level[index["s"]] == level[index["t"]] - 1
    assert record["moved"] >= 3


def test_tiny_inputs_do_not_block_but_large_ones_do():
    from megakernel_lab.operation_compiler.memory_rules import TINY_BYTES
    assert TINY_BYTES <= 4096
    phases = [_phase([], ["x"]), _phase(["x"], ["y"]), _phase([], ["a"]), _phase(["a"], ["b"]), _phase(["b", "y"], ["z"])]
    intervals = [[0, 2], [1], [3], [4]]
    layouts = _layouts(dict(x=8192, y=4096, a=16, b=16, z=16))   # x larger than y and only read by y
    dtypes = {n: "float32" for n in layouts}
    out, _ = lifetime_sink(phases, intervals, layouts, dtypes, list(layouts))
    level = {i: k for k, m in enumerate(out) for i in m}
    assert level[1] == 1
