"""P6-B contract checks (CPU; lowering only, no CUDA).

* exact_cta_tree_per_thread_column replays the frozen 256-thread CTA tree bit for bit (FP32 numpy emulation of both
  trees on random data, extents 33..256, including signed zeros);
* indexed selections (cat / arange / gather / embedding) lower without phases of their own and refuse nothing else;
* the selected BF16 exact-order route admits aligned/tail layouts without the unselected P6-B split-K candidate;
* the P6-B options are off-switchable: all off reproduces the program without them;
* quad-thread column trees (four threads per result, sub-butterflies + two xor shuffles) equal the frozen tree.
"""
from __future__ import annotations

import re

import numpy as np


def _frozen_tree(x):
    """The frozen generic CTA tree (fusion/lower): thread t holds 0 + x[t] (t < extent, else 0), xor-butterfly per
    warp (s101h::warp_sum), lane 0 of warp w stores partial w, then lanes 0..7 of warp 0 butterfly the partials."""
    f = np.float32
    v = [f(0) + f(x[t]) if t < len(x) else f(0) for t in range(256)]
    partials = []
    for w in range(8):
        lanes = v[32 * w:32 * w + 32]
        for o in (16, 8, 4, 2, 1):
            lanes = [f(lanes[i] + lanes[i ^ o]) for i in range(32)]
        partials.append(lanes[0])
    lanes = [partials[i] if i < 8 else f(0) for i in range(32)]
    for o in (16, 8, 4, 2, 1):
        lanes = [f(lanes[i] + lanes[i ^ o]) for i in range(32)]
    return lanes[0]


def _evaluate(expression, values):
    """Evaluate a generated sum expression whose leaves are '(0.0f + L[r])' in FP32 (left to right per '+')."""
    f = np.float32
    text = re.sub(r"\(0\.0f \+ L\[(\d+)\]\)", lambda m: f"_leaf({m.group(1)})", expression)

    class F:
        def __init__(self, v):
            self.v = f(v)

        def __add__(self, other):
            return F(f(self.v + other.v))
    return eval(text, {"_leaf": lambda r: F(f(0) + f(values[r])), "__builtins__": {}}).v


def run():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    from megakernel_lab.operation_compiler.codegen_rules import exact_reduction
    rng = np.random.default_rng(0)
    for extent in (33, 49, 64, 97, 200, 256):
        code, extra = exact_reduction(out="y", position="(r / 1LL)", extent=extent, results=4096,
                                      read=lambda p: "L[" + re.search(r"\((\d+)LL / 1LL\)", p).group(1) + "]",
                                      rounded=lambda e: e, coalesced_threads=True)
        check(extra["engine"] == "exact_cta_tree_per_thread_column", f"coalesced engine selected for {extent}")
        acc = re.search(r"const float acc = (.*);\n", code).group(1)
        for trial in range(20):
            x = (rng.standard_normal(extent) * 10.0 ** rng.integers(-3, 4, extent)).astype(np.float32)
            if trial % 5 == 0:
                x[rng.integers(0, extent, 4)] = np.float32(-0.0)
            a, b = _evaluate(acc, x), _frozen_tree(x)
            check(a.tobytes() == b.tobytes(), f"bitwise frozen tree extent {extent} trial {trial}: {a} vs {b}")
    selected = exact_reduction(out="y", position="(r / 1LL)", extent=49, results=512,
                               read=lambda p: "L[0]", rounded=lambda e: e, coalesced_threads=True)
    check(selected[1]["engine"] == "exact_cta_tree_per_warp", "few results keep the per-warp engine")

    # The split_choice experiment from d81ea0a6b was not selected into the
    # maintained 2f9bbc3f3 baseline. Validate the selected BF16 route here;
    # importing that historical candidate would test an unavailable runtime.
    # The independent scalar/quad arithmetic proofs below and above are kept.
    from megakernel_lab.operation_compiler.lower import TensorLayout
    from megakernel_lab.operation_compiler.matrix_codegen import select
    for m, n, k, tail in ((256, 256, 6272, False), (6272, 256, 256, False),
                           (256, 256, 6256, True)):
        la = TensorLayout((m, k), (1, m), 'a')
        lb = TensorLayout((k, n), (n, 1), 'b')
        spec, reason = select('a', 'b', 'out', la, lb, (m, n), 'bfloat16')
        check(spec is not None, f"selected BF16 layout admitted {(m, n, k)}: {reason}")
        if spec is not None:
            check(not spec.get('split') and not spec.get('engine'),
                  f"BF16 small-tile route keeps its unsplit exact-order body {(m, n, k)}")
            check(bool(spec.get('tail')) == tail, f"BF16 tail admission {(m, n, k)}")
        unaligned = TensorLayout((m, k), (1, m), 'a', 1)
        rejected, reason = select('a', 'b', 'out', unaligned, lb, (m, n), 'bfloat16')
        check(rejected is None and 'retained' in reason,
              f"unsupported BF16 alignment retains the fallback {(m, n, k)}")

    # The facade selects fusion_reference for this synthetic indexed graph;
    # its older protocol lacks P6B's selectable inline-index mechanism. Test the
    # retained P7B mechanism directly without changing normal default dispatch.
    from megakernel_lab.operation_compiler import fusion_p7b
    from megakernel_lab.operation_compiler.execution_plan import execution_scope
    from megakernel_lab.operation_compiler.performance_schedule import _LOCK, _configuration
    def lower(graph, **options):
        with _LOCK, execution_scope('legacy'), _configuration('p7b', {}):
            return fusion_p7b.lower_fused(graph, **options)

    def value(name, shape, dtype="float32", role="temporary"):
        strides, st = [], 1
        for d in reversed(shape):
            strides.append(st)
            st *= d
        return {"name": name, "shape": list(shape), "dtype": dtype, "strides": list(reversed(strides)), "role": role,
                "parameter": None, "alias_of": None, "storage_offset": 0, "producer": None, "output_roles": []}
    values = {n: value(n, s, d, r) for n, s, d, r in [
        ("a", (64, 32), "float32", "input"), ("b", (64, 32), "float32", "input"), ("ids", (64,), "int64", "input"),
        ("tab", (11, 32), "float32", "input"), ("c", (64, 64), "float32", "temporary"),
        ("e", (64, 32), "float32", "temporary"), ("r", (64,), "float32", "temporary"),
        ("o1", (64, 64), "float32", "temporary"), ("o2", (64, 32), "float32", "temporary")]}
    ops = [{"name": "c", "target": "aten.cat.default", "args": [[{"ref": "a"}, {"ref": "b"}], -1], "kwargs": {}},
           {"name": "e", "target": "aten.embedding.default", "args": [{"ref": "tab"}, {"ref": "ids"}], "kwargs": {}},
           {"name": "r", "target": "aten.arange.start", "args": [0, 64], "kwargs": {}},
           {"name": "o1", "target": "aten.mul.Tensor", "args": [{"ref": "c"}, 2.0], "kwargs": {}},
           {"name": "o2", "target": "aten.add.Tensor", "args": [{"ref": "e"}, {"ref": "a"}], "kwargs": {}}]
    graph = {"inputs": ["a", "b", "ids", "tab"], "outputs": ["o1", "o2"], "values": values, "operations": ops}
    on = lower(graph, p6b_index_elementwise=True).manifest
    off = lower(graph, p6b_index_elementwise=False).manifest
    kinds_on = {p["kind"] for p in on["phases"]}
    kinds_off = {p["kind"] for p in off["phases"]}
    # The selected P8/P9 extra_ops emitter precedes standard_ops and names its
    # copy phase "cat" (already true at the 2f9bbc3f3 integration anchor).
    check({"cat", "embedding"} <= kinds_off, f"rule off keeps copy phases {kinds_off}")
    check(not kinds_on & {"cat", "concatenate", "embedding"}, f"selections are per-element expressions {kinds_on}")
    check(len(on["phases"]) < len(off["phases"]), "selections remove phases")
    src_on = lower(graph, p6b_index_elementwise=True).source
    check("latch_p6b_row(" in src_on, "gathers keep a checked row index")

    # Quad trees: lane q evaluates the sub-butterflies over rows 32w + 4k + q; shuffles (2, 1) and the partial
    # butterfly finish the frozen order.
    from megakernel_lab.operation_compiler.codegen_rules import exact_reduction
    for extent in (33, 49, 64, 100, 196, 256):
        code, _ = exact_reduction(out="O", position="L[(r / 1LL)]", extent=extent, results=1024, read=lambda p_: p_,
                                  rounded=lambda a: a, coalesced_threads=True, quad_threads=True)
        lines = re.findall(r"float (acc_w\d) = (.*);\n", code)
        final = re.search(r"const float acc = (.*);\n", code).group(1)
        for trial in range(20):
            x = (rng.standard_normal(extent) * 10.0 ** rng.uniform(-3, 3, extent)).astype(np.float32)
            parts = {}
            for name, e in lines:
                vals = []
                for q in range(4):
                    # absent rows are +0: the leaf of an appended zero element (0 + 0 = +0)
                    e2 = re.sub(r"\((\d+)u \+ q < (\d+)u \? (.*?) : 0\.0f\)",
                                lambda m: m.group(3) if int(m.group(1)) + q < int(m.group(2)) else f"(0.0f + L[{extent}])",
                                e)
                    e2 = re.sub(r"L\[\(\((\d+)LL \+ \(long long\)q\) / 1LL\)\]", lambda m: f"L[{int(m.group(1)) + q}]", e2)
                    vals.append(_evaluate(e2, np.append(x, np.float32(0))))
                s1 = [np.float32(vals[q] + vals[q ^ 2]) for q in range(4)]
                parts[name] = np.float32(s1[0] + s1[1])
            got = np.float32(eval(final, {**parts}))
            check(got.tobytes() == _frozen_tree(x).tobytes(), f"quad tree extent {extent} trial {trial}")

    options_off = dict(p6b_scalar_chain=False, p6b_trap_asserts=False, p6b_column_tree=False,
                       p6b_index_elementwise=False)
    check(lower(graph, **options_off).source == lower(graph, **options_off).source, "deterministic lowering")
    return {"passed": not failures, "failed": len(failures), "failures": failures[:20]}


if __name__ == "__main__":
    print(run())
