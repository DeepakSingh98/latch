"""LATCH-SELECTIVE-10: structural unit tests of ops.accumulation_rules and the rules it selects (CPU, no CUDA)."""
from types import SimpleNamespace as NS

import pytest

from megakernel_lab.operation_compiler import accumulation_rules as ar
from megakernel_lab.operation_compiler import group_rules as gr
from megakernel_lab.operation_compiler import row_chain
from megakernel_lab.operation_compiler import square_sum_association as sq
from megakernel_lab.operation_compiler import update_schedule as us


def test_rule_validation_and_features():
    assert ar.rules_of(None) == ()
    assert ar.rules_of("castview+fswarp") == ("castview", "fswarp")
    with pytest.raises(ValueError):
        ar.rules_of("castview+unknown")
    with ar.scope("fsgather+fswarp+cmulti+rowchain+fsmix"):
        assert ar.features() == "+fsgather+fswarp+cmulti+rowchain+fsmix"
        assert gr.features_of("base" + ar.features()) == {"fsgather", "fswarp", "cmulti", "rowchain", "fsmix"}
    assert ar.features() == ""


def _layouts(names, n=8):
    return {name: NS(root=name, shape=(n,), strides=(1,), offset=0, contiguous=True, numel=n) for name in names}


def _ew(name, reads, line):
    return dict(kind="elementwise", names=[name], targets=["aten.x"], reads=list(reads), writes=[name], shape=(8,),
                dtype="float32", body=[f"const float r_{name} = {line}; v_{name}[i] = r_{name};"])


def test_cmulti_merges_independent_readers_and_keeps_order():
    layouts = _layouts(["g", "m0", "m", "p", "pn", "mc", "f", "s"])
    a = _ew("m", ["g", "m0"], "v_m0[i] + v_g[i]")
    flag = dict(_ew("f", ["s"], "v_s[i]"), shape=(1,))
    b1 = _ew("pn", ["p", "m", "f"], "v_p[i] - v_m[i]")
    b2 = _ew("mc", ["m", "m0", "f"], "v_m[i] * v_m0[i]")
    out, rec = gr.sink_multi_reader_chains([a, flag, b1, b2], layouts, ident=lambda n: "v_" + n,
                                           index=lambda lay, shape, i: i, inplace={"m0": "mc"})
    assert [p["names"] for p in out] == [["f"], ["m", "pn", "mc"]]
    merged = out[-1]["body"]
    assert merged[1].startswith("const float r_pn = v_p[i] - r_m;") and merged[2].startswith("const float r_mc = r_m * v_m0[i];")


def test_cmulti_refuses_an_overwrite_of_a_moved_operand():
    layouts = _layouts(["g", "m0", "m", "p", "pn", "mc", "s"])
    a = _ew("m", ["g", "m0"], "v_m0[i] + v_g[i]")
    clobber = dict(_ew("x", ["s"], "v_s[i]"), writes=["m0"])
    b1 = _ew("pn", ["p", "m"], "v_p[i] - v_m[i]")
    b2 = _ew("mc", ["m"], "v_m[i] * 2.0f")
    out, rec = gr.sink_multi_reader_chains([a, clobber, b1, b2], layouts, ident=lambda n: "v_" + n,
                                           index=lambda lay, shape, i: i, inplace={})
    assert len(out) == 4 and rec[-1]["refused_multi_reader_candidates"]["intervening access"] == 1


COMBINE = """    __threadfence();
    acc = 0.0f;
    for (unsigned b = threadIdx.x; b < gridDim.x; b += blockDim.x) acc += __ldcg((v_p + 64LL) + b);
    acc = s101h::warp_sum(acc);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
      acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
      acc = s101h::warp_sum(acc);
      if (threadIdx.x == 0) { v_s[0] = acc; *reinterpret_cast<unsigned*>(v_p) = 0u; }
    }"""


def test_warp_combine_replicates_the_block_combine():
    code = us._warp_combine(COMBINE)
    assert code is not None
    assert "latch_w_ * 32u + (threadIdx.x & 31u); b < gridDim.x; b += 256u" in code
    assert "v_s[0] = latch_fw_" in code and "smem" not in code and "__syncthreads" not in code
    assert us._warp_combine(COMBINE.replace("b += blockDim.x", "b += 2")) is None


def _calculation(n, name):
    with ar.scope("fsdeep"):
        body = sq.lane_loop(f"{name}[r]")
    return (f"  const long long chunk_ = ({n}LL + gridDim.x - 1) / gridDim.x, lo_ = (long long)blockIdx.x * chunk_;\n"
            f"  const long long hi_ = lo_ + chunk_ < {n}LL ? lo_ + chunk_ : {n}LL;\n  float acc = 0.0f;\n"
            + sq.marker("s" + name) + "\n" + body + "\n")


def test_fsdeep_keeps_each_lane_order():
    with ar.scope("fsdeep"):
        text = sq.lane_loop("x[r]")
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("lane") and "+= t" in line]
    assert lines[:3] == ["lane0 += t0;", "lane1 += t1;", "lane2 += t2;"] and lines[8] == "lane0 += t8;"


def test_fsmix_interleaves_rounds_and_keeps_markers():
    calcs = [(0, _calculation(1000, "v_a")), (3, _calculation(2000, "v_b"))]
    code = us._mixed_calculations(calcs)
    assert code is not None
    assert sq.marker("sv_a") in code and sq.marker("sv_b") in code
    assert "latch_fsb_acc_[0] = ((l0_0 + l0_1)" in code and "latch_fsb_acc_[3] = ((l1_0 + l1_1)" in code
    assert us._mixed_calculations([(0, "  float acc = 0.0f;\n")]) is None


SOFTMAX = """for (long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < 2048LL; row += ((long long)gridDim.x * blockDim.x) >> 5) {
  const int lane = threadIdx.x & 31;
  x;
}"""


def test_rowchain_forms_and_restriction():
    phase = dict(kind="softmax", engine="exact_cta_tree_per_warp_narrow", writes=["s"], reads=["x"])
    assert row_chain._form(phase, SOFTMAX) == ("warp_row", 2048)
    restricted = row_chain.restrict(phase, ("warp_row", 2048), SOFTMAX, 16)
    assert "row = latch_rc_r0 + (threadIdx.x >> 5); row < latch_rc_r1" in restricted and "blockIdx" not in restricted
    flat = "for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < 4096LL; i += (long long)gridDim.x * blockDim.x) { y; }"
    gather = dict(kind="gather", writes=["g"], reads=["s"])
    assert row_chain._form(gather, flat) == ("flat", 4096)
    assert "i = latch_rc_r0 * 2LL + threadIdx.x" in row_chain.restrict(gather, ("flat", 4096), flat, 2)
    assert row_chain._form(dict(kind="softmax", engine="exact_cta_tree_per_warp"), SOFTMAX) is None


def _dcast_manifest(update_pieces):
    import torch
    from megakernel_lab.operation_compiler import derived_casts, update_schedule
    from megakernel_lab.operation_compiler.capture import capture_training
    from megakernel_lab.operation_compiler.compose import lower_composed_flat
    from megakernel_lab.operation_compiler.execution_plan import execution_scope
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update

    def update(parameters, gradients, state, _hp):
        out, new = {}, {}
        for name, p in parameters.items():
            changed = name == "fast" or update_pieces
            out[name] = p - 0.01 * gradients[name] if changed else p
            new["acc:" + name] = state["acc:" + name] + gradients[name]
        return out, new

    rule = tensor_update(update, init_state=lambda ps, _hp: {"acc:" + n: torch.zeros_like(p) for n, p in ps.items()},
                         name="dcast_probe")
    torch.manual_seed(0)
    params = {"fast": torch.randn(16, 8), "e0": torch.randn(8, 16), "e1": torch.randn(8, 16)}
    x = torch.randn(32, 16)

    def loss(p, x):
        w = torch.cat([p["e0"], p["e1"]], 0).to(torch.bfloat16)
        h = (x.to(torch.bfloat16) @ w.t()).float()
        return (h @ p["fast"]).square().mean()

    captured = capture_training(loss, params, None, (x,), rule)
    assert any(str(op.target) == derived_casts.TARGET for op in captured.ir.operations)
    ir, args, records = derived_casts.apply(captured.ir, captured.example_args)
    assert len(records) == 1 and len(args) == len(captured.example_args) + 1
    assert torch.equal(args[-1].view(torch.int16),
                       torch.cat([params["e0"], params["e1"]], 0).to(torch.bfloat16).view(torch.int16))
    from megakernel_lab.operation_compiler import accumulation_rules
    with execution_scope("compose"), update_schedule.scope("streamed_ustate"), accumulation_rules.scope("dcast"):
        program, _ = lower_composed_flat(ir)
    return program.manifest, records[0], ir


def test_dcast_reads_the_derived_copy_and_commits_only_when_pieces_change():
    for update_pieces in (False, True):
        m, record, ir = _dcast_manifest(update_pieces)
        roles = m["input_roles"]
        binding = [n for n in m["inputs"] if roles[n]["role"] == "state" and roles[n]["parameter"] == record["key"]]
        assert len(binding) == 1 and record["key"].startswith("derived:bf16cat:e0+1")
        casts = [p for p in m["phases"] if p["kind"] == "cat_cast"]
        views, grew = {binding[0]}, True
        while grew:
            grew = False
            for name, value in ir.values.items():
                if value.alias_of in views and name not in views:
                    views.add(name)
                    grew = True
        readers = [p for p in m["phases"] if views & set(p.get("reads", ()))]
        assert readers, "the product reads the derived copy"
        if update_pieces:
            assert len(casts) == 1 and binding[0] in m["state_writes"]
            assert m["in_place_commits"].get(binding[0]) == "derived_commit_0", "the refresh stores into the copy"
        else:
            assert not casts and binding[0] not in m["state_writes"]


def test_rowchain_runs_are_cut_at_arena_reuse():
    """TINYMOE-FINER-TRANSFER: a merged run must not reuse storage an earlier interval of the run accessed."""
    from types import SimpleNamespace as NS
    lay = lambda root: NS(root=root)
    phases = [dict(names=["a"], reads=["x"], writes=["a"]), dict(names=["b"], reads=["a"], writes=["b"]),
              dict(names=["c"], reads=["b"], writes=["c"]), dict(names=["d"], reads=["c"], writes=["d"])]
    layouts = {n: lay(n) for n in "xabcd"}
    intervals = [[0], [1], [2], [3]]
    # c reuses a's storage (a's last reader is interval 1): the run must stop before interval 2.
    offsets = dict(x=0, a=100, b=200, c=100, d=300)
    sizes = dict(x=50, a=50, b=50, c=50, d=50)
    safe, records, cuts = row_chain.arena_safe_runs({0: [0, 1, 2, 3]}, intervals, phases, layouts, offsets, sizes)
    assert safe == {0: [0, 1], 2: [2, 3]} and cuts == 1 and [r["intervals"] for r in records] == [2, 2]
    # Disjoint storage keeps the whole run.
    offsets = dict(x=0, a=100, b=200, c=400, d=300)
    assert row_chain.arena_safe_runs({0: [0, 1, 2, 3]}, intervals, phases, layouts, offsets, sizes)[0] == {0: [0, 1, 2, 3]}
    # A cut that leaves a single interval drops it from the runs.
    offsets = dict(x=0, a=100, b=100, c=400, d=500)
    assert row_chain.arena_safe_runs({0: [0, 1, 2, 3]}, intervals, phases, layouts, offsets, sizes)[0] == {1: [1, 2, 3]}


def _emitted_arena_overlaps(manifest):
    """LATCH-ROWCHAIN-REPAIR: pairs of arena roots that share storage while live in the same EMITTED interval.

    Independent of row_chain.arena_safe_runs: the planned lifetimes (planning-interval units) are remapped onto the
    barrier intervals actually emitted (row_chain_substages) and the vendored planner check is re-run."""
    from megakernel_lab.operation_compiler.vendor import plan
    nbytes = {"float32": 4, "int32": 4, "int64": 8, "float64": 8, "bfloat16": 2, "float16": 2, "int16": 2, "bool": 1,
              "uint8": 1, "int8": 1}
    words = {}
    for name, lay in manifest["layouts"].items():
        extent = 0 if any(int(s) == 0 for s in lay["shape"]) else int(lay.get("offset", 0)) + 1 + sum(
            (int(s) - 1) * int(t) for s, t in zip(lay["shape"], lay["strides"]))
        root = lay.get("root", name)
        words[root] = max(words.get(root, 0), (extent * nbytes[lay["dtype"]] + 3) // 4)
    runs = {int(k): v for k, v in manifest.get("row_chain_substages", {}).items()}
    emitted = []
    for e, _ in enumerate(manifest["barrier_intervals"]):
        emitted += [e] * len(runs.get(e, [None]))
    end = len(manifest["barrier_intervals"])
    roots = [r for r in manifest["offsets"] if r in words]

    def remap(c):
        return c if not isinstance(c, list) else (c[0], *(emitted[x] if x < len(emitted) else end for x in c[1:]))

    planned = {r: tuple(manifest["lifetimes"][r]) if isinstance(manifest["lifetimes"][r], list)
               else manifest["lifetimes"][r] for r in roots}
    merged = {r: remap(list(planned[r])) if isinstance(planned[r], tuple) else planned[r] for r in roots}
    total = max(int(manifest["offsets"][r]) + plan.a64(words[r]) for r in roots)
    assert not plan.check(roots, words, planned, manifest["offsets"], total)  # the planner's own schedule is valid
    return [e for e in plan.check(roots, words, merged, manifest["offsets"], total) if e.startswith("live overlap")]


@pytest.mark.parametrize("tokens", [1024, 32768])
def test_rowchain_router_never_reuses_live_arena_storage(tokens, monkeypatch):
    """LATCH-ROWCHAIN-REPAIR: end-to-end regression of the TINYMOE-FINER-TRANSFER race through capture and lowering.

    A token-choice router (FP32 softmax over 16 experts, the CUDA small-row top-2 law, renormalised weights, route
    order, one ragged product) lowered with the one-update rule set forms rowchain runs. The historical planning
    (ARENA_SAFE False) emits merged intervals in which two roots share arena storage while both are live (at 32,768
    tokens: the B32-TRANSFER pattern, the top-k input reused by a later where/cat stage); the repair cuts those runs."""
    import torch
    import torch.nn.functional as F
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, route_order, token_topk

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Parameter(torch.randn(16, 16) * 0.1)
            self.w = torch.nn.Parameter(torch.randn(16, 16, 16) * 0.1)

        def forward(self, x):
            weights, selected = token_topk(F.softmax(x @ self.gate.t(), dim=-1), 2)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            order, offsets, inverse = route_order(selected, 16)
            out = ragged_mm(F.embedding(order // 2, x), self.w, offsets)
            return (F.embedding(inverse, out).reshape(tokens, 2, 16) * weights.unsqueeze(-1)).sum(1)

    torch.manual_seed(1729)
    captured = capture_module(Router(), lambda y: y.square().mean(), (torch.randn(tokens, 16),),
                              optimizer=SGDConfig())
    manifests = {}
    for safe in (False, True):
        monkeypatch.setattr(row_chain, "ARENA_SAFE", safe, raising=False)  # also runs (and fails) on unfixed trees
        with ar.scope("fsgather+fswarp+wsoftn+fsdeep+rorder+rowchain+fsmix+fsfence"):  # PRESETS["update"] (B32)
            manifests[safe] = lower(captured.ir, precision_policy="practical", execution="compose",
                                    target_arch="sm_90").manifest
    assert manifests[False].get("row_chain_substages") and _emitted_arena_overlaps(manifests[False])
    fixed = manifests[True]
    assert fixed.get("row_chain_substages"), "the repair keeps the safe parts of the chains"
    assert _emitted_arena_overlaps(fixed) == []
    assert fixed["group_rules"].get("row_chain_arena_cuts", 0) >= 1
