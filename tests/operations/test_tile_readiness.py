"""Per-row-block readiness (``tile_readiness="attention"``, ops/tile_ready.py): planning, emission and lowering.

Synthetic phase lists check which boundaries open, what is emitted and the refusals; one CPU lowering of segmented
attention followed by a claimed output projection checks the elided barrier and that arena storage is never shared
across an open boundary. Native execution is not covered here.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.operation_compiler import tile_ready, tile_readiness

ROWS = 4608                  # 4608 x 2 x 32 elements: above the row-block loop threshold
FORWARD = ("{\nlatch_segmented::Args A{};\nA.q=(v_q + 0LL);\n"
           "latch_segmented::run_forward<32,6>(A, (v_off + 0LL), smem);\n}")
KV = FORWARD.replace("run_forward", "run_kv")
FLAT_LOOP = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {ROWS * 64}LL; "
             "i += (long long)gridDim.x * blockDim.x) {\n  const float r_c = v_o[i] * 2.0f; v_c[i] = r_c;\n}")
PRODUCT = "{\nlatch_ragged::tflat_claim<1,2>(/*XCLAIM*/nullptr, a, b, c);\n}"
PUBLICATION = "// tile-ready per-item row publication\nlatch_segmented::run_"


def _lay(shape, root, **kw):
    return SimpleNamespace(shape=tuple(shape), root=root, contiguous=True, offset=0, **kw)


def _ectaq():
    used = []

    def alloc(n):
        used.append(int(n))
        return f"reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + {sum(used) - int(n)}LL)"
    return alloc, used


def _loop_case(rows=ROWS, code=FLAT_LOOP):
    """prepare | attention forward (publishes o) | elementwise consumer of o | tail."""
    layouts = {"qkv": _lay((rows, 2, 32), "qkv"), "o": _lay((rows, 2, 32), "o"), "lse": _lay((2, rows), "lse"),
               "work": _lay((4, 2, 3), "work"), "c": _lay((rows, 2, 32), "c"), "d": _lay((rows, 2, 32), "d")}
    phases = [dict(kind="elementwise", names=["qkv"], body=["x"], reads=[], writes=["qkv"], code="prepare();"),
              dict(kind="segmented_attention_forward", names=["o", "lse", "work"], reads=["qkv"],
                   writes=["o", "lse", "work"], code=FORWARD),
              dict(kind="elementwise", names=["c"], body=["const float r_c = v_o[i] * 2.0f; v_c[i] = r_c;"],
                   reads=["o"], writes=["c"], code=code),
              dict(kind="elementwise", names=["d"], body=["v_d[i] = v_c[i];"], reads=["c"], writes=["d"], code="d();")]
    return phases, [[0], [1], [2], [3]], layouts


def _product_case(engine="ragged_flat_tma_claim_wgmma"):
    """prepare | attention kv (publishes dk, dv) | claimed product with A = dk viewed [rows, 64] | tail."""
    layouts = {"qkv": _lay((ROWS, 2, 32), "qkv"), "dk": _lay((ROWS, 2, 32), "dk"), "dv": _lay((ROWS, 2, 32), "dv"),
               "dkv": _lay((ROWS, 64), "dk"), "w": _lay((4, 64, 64), "w"), "y": _lay((ROWS, 64), "y"),
               "z": _lay((ROWS, 64), "z")}
    phases = [dict(kind="elementwise", names=["qkv"], body=["x"], reads=[], writes=["qkv"], code="prepare();"),
              dict(kind="segmented_attention_kv", names=["dk", "dv"], reads=["qkv"], writes=["dk", "dv"], code=KV),
              dict(kind="gemm", engine=engine, flat_claim=True, names=["y"], reads=["dkv", "w"], writes=["y"],
                   code=PRODUCT),
              dict(kind="elementwise", names=["z"], body=["q"], reads=["y"], writes=["z"], code="z();")]
    return phases, [[0], [1], [2], [3]], layouts


def _plan(phases, intervals, layouts, **kw):
    with tile_readiness.scope("attention"):
        return tile_ready.plan(phases, intervals, layouts, **kw)


def test_off_by_default():
    phases, intervals, layouts = _loop_case()
    assert not tile_ready.enabled() and tile_ready.mode() == 0
    assert tile_ready.plan(phases, intervals, layouts) is None and tile_ready.LAST_CENSUS == []
    with tile_readiness.scope(None):
        assert tile_ready.plan(phases, intervals, layouts) is None
    with tile_readiness.scope("attention"):
        assert tile_ready.enabled() and tile_ready.mode() == 2
    assert not tile_ready.enabled()


def test_attention_forward_to_row_block_loop():
    phases, intervals, layouts = _loop_case()
    plan = _plan(phases, intervals, layouts)
    assert plan is not None and [b["interval"] for b in plan.boundaries] == [1]
    assert [c["opened"] for c in tile_ready.LAST_CENSUS] == [True]
    assert plan.products == {1: dict(rows=ROWS, cols=64, publish=True)}
    assert plan.loops == {2: dict(rows=ROWS, per_row=64, waits=[(1, 64)], publish=False)}
    assert plan.epoch_of(intervals) == {0: 0, 1: 1, 2: 1, 3: 2}
    assert plan.closes_open([1]) and not plan.closes_open([2])
    alloc, used = _ectaq()
    producer = plan.product_code(1, FORWARD, alloc)
    assert producer == FORWARD.replace(
        "latch_segmented::run_", "A.tr_cnt = reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + 0LL); "
        "A.tr_rb = 128; A.tr_delay = 0;  " + PUBLICATION)
    assert used == [-(-ROWS // 128)]
    loop = plan.consumer_code(2, alloc, FLAT_LOOP)
    rb = plan.rb_of[2]
    assert rb == 16 and used == [-(-ROWS // 128)]          # the consumer allocates no words
    assert loop.startswith(f"// tile-ready row-block loop: {rb}-row blocks, static round robin over the grid in "
                           "ascending order; waits for producer rows\n#pragma unroll 1\n"
                           f"for (long long tr_s_ = (long long)blockIdx.x; tr_s_ < {ROWS // rb}LL; "
                           "tr_s_ += gridDim.x) {\n  const long long tr_b_ = tr_s_;\n")
    assert "latch_tr::wait_rows(reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + 0LL), tr_q_, " in loop
    assert "i < tr_v1_" in loop and "gridDim.x * blockDim.x" not in loop and loop.endswith("\n}")
    assert plan.forms == {2: "emitted_flat_loop"}
    with tile_readiness.scope("attention"):
        record = plan.record()
    assert record["version"] == "tile-ready-v2" and record["mode"] == 2 and record["rows_per_block"] == 128
    assert record["boundaries"][0]["edges"] == [("o", "c")]


def test_row_block_loop_falls_back_to_the_scalar_body():
    phases, intervals, layouts = _loop_case(code="opaque();")
    plan = _plan(phases, intervals, layouts)
    alloc, _ = _ectaq()
    plan.product_code(1, FORWARD, alloc)
    loop = plan.consumer_code(2, alloc, "opaque();")
    assert plan.forms == {2: "scalar_body"}
    assert "i < latch_rc_r1 * 64LL; i += blockDim.x) {\nconst float r_c = v_o[i] * 2.0f; v_c[i] = r_c;\n}" in loop


def test_attention_kv_to_claimed_product():
    phases, intervals, layouts = _product_case()
    plan = _plan(phases, intervals, layouts)
    assert plan is not None and [b["interval"] for b in plan.boundaries] == [1]
    assert plan.products[1] == dict(rows=ROWS, cols=64, publish=True)
    assert plan.products[2] == dict(rows=ROWS, wait=(1, 64), cols=0) and not plan.loops
    assert plan.order([0, 2]) == [0, 2] and plan.order([2, 0]) == [0, 2]
    alloc, used = _ectaq()
    plan.product_code(1, KV, alloc)
    body = plan.product_code(2, PRODUCT, alloc)
    assert body == ("{\nconst latch_tr::ReadyT<false, true> tr_rdy_2{nullptr, 128, nullptr, 0, "
                    "reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + 0LL), 64u, 128, nullptr, 0};  "
                    "// tile-ready readiness (wait A rows)\n\n"
                    "latch_ragged::tflat_claim<1,2>(/*XCLAIM*/nullptr, tr_rdy_2, a, b, c);\n}")
    assert used == [-(-ROWS // 128)]


def test_waiting_product_before_its_producer_fails_closed():
    plan = _plan(*_product_case())
    alloc, _ = _ectaq()
    with pytest.raises(RuntimeError, match="before its producer"):
        plan.product_code(2, PRODUCT, alloc)


def test_k_concatenated_host_is_not_a_waiting_product():
    phases, intervals, layouts = _product_case("ragged_flat_tma_claim_kcat_wgmma")
    assert not tile_ready.product_waits_ok(phases[2])
    assert _plan(phases, intervals, layouts) is None
    assert "unsupported consumer" in tile_ready.LAST_CENSUS[0]["reason"]


def _refusal(case):
    kw = {}
    if case == "narrow_loop":             # at or below the row-block loop threshold
        phases, intervals, layouts = _loop_case(rows=128)
    elif case == "rows":                  # consumer rows differ from the attention rows: not row-local
        phases, intervals, layouts = _loop_case()
        layouts["c"] = _lay((ROWS // 2, 4, 32), "c")
    elif case == "wide":                  # width-64 attention does not publish rows
        phases, intervals, layouts = _loop_case()
        phases[1]["code"] = FORWARD.replace("<32,6>", "<64,0>")
    elif case == "war":                   # a phase after the boundary overwrites storage the producer interval reads
        phases, intervals, layouts = _loop_case()
        phases.append(dict(kind="elementwise", names=["q2"], body=["y"], reads=[], writes=["qkv"]))
        intervals[2].append(4)
    elif case == "full_grid":
        phases, intervals, layouts = _loop_case()
        phases.append(dict(kind="reduction", names=["z"], reads=[], writes=["z"], code="{ a(); grid.sync(); b(); }"))
        intervals[2].append(4)
    elif case == "unsupported_dep":       # a reduction reads the attention result
        phases, intervals, layouts = _loop_case()
        phases.append(dict(kind="reduction", names=["z"], reads=["o"], writes=["z"]))
        intervals[2].append(4)
    elif case == "inplace":               # in-place update of an input the producer interval reads
        phases, intervals, layouts = _loop_case()
        phases.append(dict(kind="elementwise", names=["u"], body=["q"], reads=[], writes=["u"]))
        intervals[2].append(4)
        kw = dict(placement={"u": (0, "inplace:0")}, inputs=["qkv"])
    elif case == "excluded":
        phases, intervals, layouts = _loop_case()
        kw = dict(excluded={2})
    elif case == "outer":                 # claimed weight-gradient products never wait
        phases, intervals, layouts = _product_case("ragged_flat_tma_outer_claim_wgmma")
    elif case == "other_read":            # the product reads another interval-k value besides its A operand
        phases, intervals, layouts = _product_case()
        phases.append(dict(kind="elementwise", names=["w"], body=["w"], reads=[], writes=["w"]))
        intervals[1].append(4)
    else:
        raise AssertionError(case)
    return phases, intervals, layouts, kw


@pytest.mark.parametrize("case", ["narrow_loop", "rows", "wide", "war", "full_grid", "unsupported_dep", "inplace",
                                  "excluded", "outer", "other_read"])
def test_planner_refusals(case):
    phases, intervals, layouts, kw = _refusal(case)
    before = [list(m) for m in intervals]
    assert _plan(phases, intervals, layouts, **kw) is None
    assert intervals == before
    assert all(not c["opened"] for c in tile_ready.LAST_CENSUS)


def test_consecutive_boundaries_form_one_epoch():
    phases, intervals, layouts = _loop_case()
    layouts["o2"] = _lay((ROWS, 2, 32), "o2")
    layouts["c2"] = _lay((ROWS, 2, 32), "c2")
    phases[3] = dict(kind="segmented_attention_forward", names=["o2"], reads=["c"], writes=["o2"], code=FORWARD)
    phases.append(dict(kind="elementwise", names=["c2"], body=["v_c2[i] = v_o2[i];"], reads=["o2"], writes=["c2"]))
    intervals.append([4])
    plan = _plan(phases, intervals, layouts)
    assert [b["interval"] for b in plan.boundaries] == [1, 3]
    assert plan.epoch_of(intervals) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


def test_publication_then_claim_prefix_order():
    """Readiness publication is bound first, then the attention item claim, sharing one per-launch allocator."""
    from megakernel_lab.operation_compiler import attention_claim, segmented_attention_rules
    phases, intervals, layouts = _loop_case()
    plan = _plan(phases, intervals, layouts)
    alloc, used = _ectaq()
    body = plan.product_code(1, FORWARD, alloc)
    with segmented_attention_rules.scope("claim"):
        body, bound = attention_claim.attention_claim(phases[1], body, alloc)
    blocks = -(-ROWS // 128)
    assert bound and used == [blocks, 1]
    assert ("A.tr_cnt = reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + 0LL); A.tr_rb = 128; A.tr_delay = 0;  "
            "// tile-ready per-item row publication\n"
            f"A.tkt_claim = reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + {blocks}LL); A.tkt_ahead = 1; "
            "A.tkt_rank = 1;  // attention items claimed by tickets\nlatch_segmented::run_forward<32,6>(") in body
    assert body.count("latch_segmented::run_") == 1


# --- one CPU lowering: segmented attention followed by a claimed output projection ----------------------------------

def _attention_program(monkeypatch, rules, *, rows=256, heads=2, experts=4, out=64):
    from megakernel_lab.operation_compiler import capture_module, ragged_experts_lower
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    from megakernel_lab.operation_compiler.segmented_attention import sdpa

    class Attend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1031)
            self.q, self.k, self.v = (torch.nn.Parameter(torch.randn(rows, heads, 32) / 4) for _ in range(3))
            self.w = torch.nn.Parameter(torch.randn(experts, heads * 32, out) / 8)

        def forward(self, offsets, projection_offsets):
            lo = torch.bfloat16
            o = sdpa(self.q.to(lo), self.k.to(lo), self.v.to(lo), offsets, max_seqlen=rows, scale=1 / 32)
            return ragged_mm(o.reshape(rows, heads * 32), self.w.to(lo), projection_offsets).float()

    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", 0)      # small test program
    offsets = torch.tensor([0, rows // 3, rows], dtype=torch.int64)
    projection_offsets = torch.arange(0, rows + 1, rows // experts, dtype=torch.int64)
    captured = capture_module(Attend(), lambda y: y.square().mean(), (offsets, projection_offsets),
                              optimizer=SGDConfig())
    with products_scope("flat+tma+rotate+claim+sepi"), tile_readiness.scope(rules):
        return lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")


def _word_ranges(manifest):
    sizes = {"float32": 4, "bfloat16": 2, "int64": 8, "int32": 4, "bool": 1, "float16": 2, "uint8": 1, "float64": 8,
             "int8": 1, "int16": 2}
    out = {}
    for name, layout in manifest["layouts"].items():
        root = layout.get("root", name)
        if root in manifest["offsets"] and root == name:
            words = (int(torch.tensor(layout["shape"]).prod().item()) * sizes[layout["dtype"]] + 3) // 4
            lo = int(manifest["offsets"][root])
            out[root] = (lo, lo + words)
    return out


def test_lowering_elides_attention_barriers_and_plans_epochs(monkeypatch):
    base = _attention_program(monkeypatch, None)
    assert "latch_tr::" not in base.source and "tile_ready" not in (base.manifest.get("group_rules") or {})
    prog = _attention_program(monkeypatch, "attention")
    rules = prog.manifest["group_rules"]
    record = rules.get("tile_ready")
    assert record and record["boundaries"], rules.get("tile_ready_census")
    n = len(record["boundaries"])
    # Each open boundary replaces exactly one grid barrier with a CTA barrier; everything else keeps its barrier.
    assert prog.source.count("latch_tr::elided_barrier();") == n
    assert base.source.count("grid.sync();") - prog.source.count("grid.sync();") == n
    assert prog.source.count(PUBLICATION) == len({p for b in record["boundaries"] for p in b["producers"]})
    assert "const latch_tr::ReadyT<false, true> tr_rdy_" in prog.source
    assert re.search(r"_claim(?:_h)?<[^>]*>\([^;]*, tr_rdy_\d+[,)]", prog.source)
    assert '#include "tile_ready.cuh"' in prog.source
    assert record["counter_words"] >= 1 and not re.search(r"/\*ECTAQ_BASE\*/|/\*XCLAIM\*/", prog.source)
    # Same phases and values: only placement and timing change.
    assert [p["names"] for p in prog.manifest["phases"]] == [p["names"] for p in base.manifest["phases"]]
    # Arena lifetimes are planned in barrier epochs: no storage is shared across an open boundary.
    m = prog.manifest
    phases, intervals, ranges = m["phases"], m["barrier_intervals"], _word_ranges(m)
    root = lambda n_: (m["layouts"].get(n_) or {}).get("root", n_)
    for b in record["boundaries"]:
        touched = {root(x) for i in intervals[b["interval"]] for x in phases[i]["reads"] + phases[i]["writes"]}
        written = {root(x) for i in intervals[b["interval"] + 1] for x in phases[i]["writes"]}
        for w in written & set(ranges):
            for t in touched & set(ranges):
                if w != t:
                    assert ranges[w][1] <= ranges[t][0] or ranges[t][1] <= ranges[w][0], (w, t)
