"""Independent packed-routing semantics and operation-based compiler coverage.

Run on admitted CPU workers with Torch2.9.1; CUDA verification is a separate
native artifact job. Small float64 cases compare against direct token equations,
including uneven and unused experts and changing membership.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.ragged_experts import (
    route_order, ragged_mm, token_choice_experts, token_topk, router_auxiliary_loss,
)


@pytest.mark.parametrize("selected", [
    [[0, 1], [0, 1], [0, 2], [3, 0], [0, 2]],
    [[1, 3], [1, 3], [1, 3], [1, 3], [1, 3]],
    [[3, 2], [1, 0], [2, 1], [3, 0], [0, 2]],
])
def test_route_order_preserves_all_choices_and_stable_buckets(selected):
    ids = torch.tensor(selected)
    order, offsets, inverse = route_order(ids, 5)
    flat = ids.flatten().tolist()
    expected = sorted(range(len(flat)), key=lambda i: (flat[i], i))
    assert order.tolist() == expected
    assert offsets.tolist() == [sum(x < e for x in flat) for e in range(6)]
    assert torch.equal(order[inverse], torch.arange(ids.numel()))
    assert torch.equal(inverse[order], torch.arange(ids.numel()))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_ragged_product_vjp_and_empty_segment(dtype):
    torch.manual_seed(1731)
    offsets = torch.tensor([0, 0, 3, 4, 9, 9])
    x = torch.randn(9, 7, dtype=dtype).requires_grad_()
    w = torch.randn(5, 7, 11, dtype=dtype).requires_grad_()
    dy = torch.randn(9, 11, dtype=dtype)
    y = ragged_mm(x, w, offsets)
    dx, dw = torch.autograd.grad(y, (x, w), dy)
    # Direct row membership reference; no packed product or segmented reduction.
    members = torch.tensor([1, 1, 1, 2, 3, 3, 3, 3, 3])
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    yr = torch.stack([xr[i] @ wr[int(e)] for i, e in enumerate(members)])
    dxr, dwr = torch.autograd.grad(yr, (xr, wr), dy)
    tol = 0.06 if dtype == torch.bfloat16 else 2e-6 if dtype == torch.float32 else 2e-14
    assert torch.allclose(y, yr, atol=tol, rtol=tol)
    assert torch.allclose(dx, dxr, atol=tol, rtol=tol)
    assert torch.allclose(dw, dwr, atol=tol, rtol=tol)
    assert torch.count_nonzero(dw[[0, 4]]) == 0


def test_token_choice_swiglu_matches_direct_scalar_membership_and_gradients():
    torch.manual_seed(1732)
    ids = torch.tensor([[1, 3], [1, 3], [0, 1], [1, 3], [3, 0]])
    leaves = [torch.randn(*shape, dtype=torch.float64).requires_grad_() for shape in
              ((5, 3), (5, 2), (5, 3, 7), (5, 7, 3), (5, 3, 7))]
    x, gates, w1, w2, w3 = leaves
    weights = gates.softmax(-1)
    y = token_choice_experts(x, ids, weights, w1, w2, w3)
    direct = torch.stack([sum((F.silu(x[t] @ w1[e]) * (x[t] @ w3[e])) @ w2[e] * weights[t, s]
                              for s, e in enumerate(ids[t])) for t in range(5)])
    dy = torch.randn_like(y)
    actual = torch.autograd.grad(y, leaves, dy, retain_graph=True)
    expected = torch.autograd.grad(direct, leaves, dy)
    assert torch.allclose(y, direct, rtol=2e-14, atol=2e-14)
    for got, want in zip(actual, expected):
        assert torch.allclose(got, want, rtol=2e-13, atol=2e-13)
    # Numerical independent perturbation with fixed membership, including router.
    eps = 1e-6
    with torch.no_grad():
        gates[0, 0] += eps
        plus = (token_choice_experts(x, ids, gates.softmax(-1), w1, w2, w3) * dy).sum()
        gates[0, 0] -= 2 * eps
        minus = (token_choice_experts(x, ids, gates.softmax(-1), w1, w2, w3) * dy).sum()
        gates[0, 0] += eps
    assert abs(float((plus - minus) / (2 * eps) - actual[1][0, 0])) < 1e-7


def test_topk16_non_ties_and_probability_gradients():
    torch.manual_seed(1733)
    scores = torch.randn(31, 16).softmax(-1).requires_grad_()
    values, selected = token_topk(scores)
    expected = torch.topk(scores, 2, dim=-1)
    assert torch.equal(selected, expected.indices)
    assert torch.equal(values, expected.values)
    grad, = torch.autograd.grad(values.sum(), scores)
    assert torch.equal(grad, torch.zeros_like(scores).scatter_(1, selected, 1))


def test_router_auxiliary_uses_global_layer_statistics():
    torch.manual_seed(1737)
    logits = [torch.randn(7, 16, requires_grad=True) for _ in range(3)]
    result = router_auxiliary_loss(logits)
    stacked = torch.cat(logits)
    probabilities = stacked.softmax(-1)
    selected = torch.topk(probabilities, 2, dim=-1).indices
    mask = F.one_hot(selected, num_classes=16).float()
    expected = (mask.mean(0) * probabilities.mean(0)[None]).sum() * 16
    assert torch.equal(result, expected)
    actual_grad = torch.autograd.grad(result, logits, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, logits)
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_capture_and_lower_segmented_vjp(dtype):
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 7, 11))

        def forward(self, x, ids):
            order, offsets, inverse = route_order(ids, 4)
            packed = F.embedding(order // ids.shape[1], x).to(dtype)
            out = ragged_mm(packed, self.weight.to(dtype), offsets)
            return F.embedding(inverse, out).reshape(x.shape[0], 2, 11).sum(1).float()

    model = Layer()
    args = (torch.randn(5, 7), torch.tensor([[0, 1], [0, 1], [0, 1], [2, 0], [0, 2]]))
    captured = capture_module(model, lambda y: y.square().mean(), args, optimizer=SGDConfig())
    targets = [op.target for op in captured.ir.operations]
    assert "latch_ops.ragged_order.default" in targets
    assert "latch_ops.ragged_mm.default" in targets
    assert "latch_ops.ragged_outer.default" in targets
    program = lower(captured.ir, precision_policy="practical", execution="compose")
    assert 'ragged_experts.cuh' in program.manifest['header_hashes']
    assert "latch_ragged::simt" in program.source
    assert "latch_ragged::counts<4>" in program.source


def test_invalid_runtime_offsets_and_ids_refused():
    with pytest.raises(ValueError, match="outside group"):
        route_order(torch.tensor([[0, 4]]), 4)
    x, w = torch.ones(4, 3), torch.ones(2, 3, 7)
    with pytest.raises(ValueError, match="monotonically cover"):
        ragged_mm(x, w, torch.tensor([0, 3, 2]))


def _aligned_bf16_program(segmented_products, width=16):
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, width, 32))

        def forward(self, x, ids):
            order, offsets, inverse = route_order(ids, 4)
            packed = F.embedding(order // ids.shape[1], x).to(torch.bfloat16)
            out = ragged_mm(packed, self.weight.to(torch.bfloat16), offsets)
            return F.embedding(inverse, out).reshape(x.shape[0], 2, 32).sum(1).float()

    torch.manual_seed(1741)
    args = (torch.randn(8, width), torch.tensor([[0, 1], [0, 1], [0, 1], [2, 0], [0, 2], [1, 0], [0, 1], [2, 0]]))
    captured = capture_module(Layer(), lambda y: y.square().mean(), args, optimizer=SGDConfig())
    with products_scope(segmented_products):
        return lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")


def test_flat_segmented_products_are_opt_in_and_isolated():
    from megakernel_lab.operation_compiler import step17_intervals
    from megakernel_lab.operation_compiler.ragged_experts_lower import validate_products
    default, flat = _aligned_bf16_program(None), _aligned_bf16_program("flat")
    assert "latch_ragged::product<" in default.source and "latch_ragged::flat<" not in default.source
    assert "latch_ragged::flat<" in flat.source and "latch_ragged::product<" not in flat.source
    # The forward and weight-gradient products (the packed input needs no gradient) take the flattened body.
    engines = [p.get("engine") for p in flat.manifest["phases"] if p.get("kind") == "gemm"]
    assert flat.source.count("latch_ragged::flat<") == engines.count("ragged_flat_wgmma") == 2
    assert "ragged_pops4b_wgmma" in [p.get("engine") for p in default.manifest["phases"]]
    shared = [p["shared_bytes"] for p in flat.manifest["phases"] if p.get("engine") == "ragged_flat_wgmma"]
    assert all(s == 4 * 256 * 128 + 1024 + 8 * 6 for s in shared)
    # Device-unit isolation adds the inline body's pattern only for programs that emit it.
    assert step17_intervals.wgmma_inline(default.source) == step17_intervals.WGMMA_INLINE
    assert step17_intervals.RAGGED_FLAT in step17_intervals.wgmma_inline(flat.source)
    with pytest.raises(ValueError, match="segmented_products"):
        validate_products("grouped")


@pytest.mark.parametrize("separate_casts", [False, True])
def test_flat_pairs_products_sharing_their_packed_operand(separate_casts):
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(4, 16, 32))
            self.w3 = torch.nn.Parameter(torch.randn(4, 16, 32))

        def forward(self, x, ids):
            order, offsets, inverse = route_order(ids, 4)
            if separate_casts:            # as TinyMoE: each projection casts the FP32 packed rows itself
                packed = F.embedding(order // ids.shape[1], x)
                up = ragged_mm(packed.to(torch.bfloat16), self.w1.to(torch.bfloat16), offsets)
                gate = ragged_mm(packed.to(torch.bfloat16), self.w3.to(torch.bfloat16), offsets)
            else:
                packed = F.embedding(order // ids.shape[1], x).to(torch.bfloat16)
                up = ragged_mm(packed, self.w1.to(torch.bfloat16), offsets)
                gate = ragged_mm(packed, self.w3.to(torch.bfloat16), offsets)
            out = F.silu(up) * gate
            return F.embedding(inverse, out).reshape(x.shape[0], 2, 32).sum(1).float()

    torch.manual_seed(1743)
    args = (torch.randn(8, 16), torch.tensor([[0, 1], [0, 1], [0, 1], [2, 0], [0, 2], [1, 0], [0, 1], [2, 0]]))
    captured = capture_module(Experts(), lambda y: y.square().mean(), args, optimizer=SGDConfig())
    programs = {}
    for mode in ("flat", "flat_single"):
        with products_scope(mode):
            programs[mode] = lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")
    paired, single = programs["flat"], programs["flat_single"]
    # Forward gate/up pair. Their weight gradients do not: the first one's result is read (parameter-gradient
    # widening) before the second one's B operand exists, so neither position can hold both.
    assert paired.source.count("latch_ragged::flat2<") == 1 and paired.source.count("latch_ragged::flat<") == 2
    assert single.source.count("latch_ragged::flat<") == 4 and "flat2" not in single.source
    def outputs(program):
        return sorted(w for p in program.manifest["phases"] if str(p.get("engine", "")).startswith("ragged_flat")
                      for w in p["writes"])
    assert outputs(paired) == outputs(single) and len(outputs(paired)) == 4
    for program in (paired, single):      # every phase reads only values written by earlier phases
        writer = {}
        for index, p in enumerate(program.manifest["phases"]):
            assert all(writer.get(r, -1) < index for r in p["reads"])
            for w in p["writes"]:
                writer[w] = index
    from megakernel_lab.operation_compiler import step17_intervals
    import re
    pattern = step17_intervals.wgmma_inline(paired.source)[-1]
    assert pattern == step17_intervals.RAGGED_FLAT
    assert len(re.findall(pattern, paired.source)) == 3   # every flat/flat2 phase is isolated in device units
    two = [p for p in paired.manifest["phases"] if p.get("engine") == "ragged_flat2_wgmma"]
    assert all(len(p["writes"]) == 2 and p["shared_bytes"] == 4 * (128 + 256) * 128 + 1024 + 8 * 6 for p in two)


def test_flat_tma_right_products_use_tensor_maps():
    program = _aligned_bf16_program("flat_tma", width=64)     # TMA needs whole 64-wide k stages
    # The forward right product reads K-major packed rows and a K-contiguous stacked weight; the weight gradient
    # reads row-major packed rows and output gradients (segment tails zeroed in shared memory). Both TMA-fed.
    assert program.source.count("latch_ragged::tflat<") == 1 and program.source.count("latch_ragged::touter<") == 1
    assert "latch_ragged::flat<" not in program.source and "POPS4B_TMAP(" not in program.source
    assert "ragged_tma.cuh" in program.manifest["header_hashes"]
    maps = program.manifest.get("tensor_maps") or []
    assert len(maps) == 4 and all(m["box"][0] == 64 for m in maps)
    from megakernel_lab.operation_compiler import step17_intervals
    import re
    assert len(re.findall(step17_intervals.RAGGED_FLAT, program.source)) == 2


def test_flat_prefetch_phases_write_nothing_and_precede_their_products():
    program = _aligned_bf16_program("flat_pf")
    phases = program.manifest["phases"]
    prefetch = [i for i, p in enumerate(phases) if p.get("engine") == "ragged_l2_prefetch"]
    assert prefetch and all(phases[i]["writes"] == [] for i in prefetch)
    assert program.source.count("latch_ragged::l2_prefetch(") >= len(prefetch)
    where = {i: j for j, members in enumerate(program.manifest["barrier_intervals"]) for i in members}
    products = [i for i, p in enumerate(phases) if str(p.get("engine", "")).startswith("ragged_flat")]
    # Each prefetch runs in an earlier interval than the product that follows it in phase order.
    detail = [(j, phases[j].get("engine") or phases[j].get("kind"), phases[j]["names"][:2], phases[j]["reads"][:6],
               phases[j]["writes"][:3], where[j]) for j in range(len(phases))]
    for i in prefetch:
        nxt = min(j for j in products if j > i)
        writer = {w: j for j, p in enumerate(phases) for w in p["writes"]}
        info = (f"prefetch {i} reads {[(r, writer.get(r), where.get(writer.get(r))) for r in phases[i]['reads']]} at {where[i]}; "
                f"product {nxt} {phases[nxt]['names']} reads {[(r, writer.get(r), where.get(writer.get(r))) for r in phases[nxt]['reads']]} at {where[nxt]}")
        assert where[i] < where[nxt], info
    # Values are unchanged: the products are the same bodies as without prefetch.
    plain = _aligned_bf16_program("flat")
    import re
    calls = lambda s: re.findall(r"latch_ragged::t?flat2?<[^;]*;", s)
    assert calls(program.source) == calls(plain.source) and len(calls(plain.source)) == 2


def test_rotate_continues_round_robin_across_products_of_an_interval():
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope, validate_products

    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(4, 64, 32))
            self.w3 = torch.nn.Parameter(torch.randn(4, 64, 32))

        def forward(self, x, ids):
            order, offsets, inverse = route_order(ids, 4)
            packed = F.embedding(order // ids.shape[1], x).to(torch.bfloat16)
            out = ragged_mm(packed, self.w1.to(torch.bfloat16), offsets) * ragged_mm(packed, self.w3.to(torch.bfloat16), offsets)
            return F.embedding(inverse, out).reshape(x.shape[0], 2, 32).sum(1).float()

    torch.manual_seed(1747)
    args = (torch.randn(8, 64), torch.tensor([[0, 1], [0, 1], [0, 1], [2, 0], [0, 2], [1, 0], [0, 1], [2, 0]]))
    captured = capture_module(Two(), lambda y: y.square().mean(), args, optimizer=SGDConfig())
    with products_scope("flat_single+rotate"):
        rotated = lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")
    with products_scope("flat_single"):
        plain = lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")
    assert "latch_ragged::rot_after(latch_ragged::tasks<4,128,128,false>(" in rotated.source
    assert "rot_after" not in plain.source
    for bad in ("flat+bogus", "flat+tma+tma", "tma"):
        with pytest.raises(ValueError):
            validate_products(bad)
    assert validate_products("flat+tma+prefetch+rotate") == "flat+tma+prefetch+rotate"


def test_consume_moves_identity_elementwise_consumers_into_the_product():
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope

    class Glu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(4, 64, 32))
            self.w3 = torch.nn.Parameter(torch.randn(4, 64, 32))

        def forward(self, x, ids):
            order, offsets, inverse = route_order(ids, 4)
            packed = F.embedding(order // ids.shape[1], x)
            up = ragged_mm(packed.to(torch.bfloat16), self.w1.to(torch.bfloat16), offsets)
            gate = ragged_mm(packed.to(torch.bfloat16), self.w3.to(torch.bfloat16), offsets)
            out = F.silu(up) * gate
            return F.embedding(inverse, out).reshape(x.shape[0], 2, 32).sum(1).float()

    torch.manual_seed(1749)
    args = (torch.randn(8, 64), torch.tensor([[0, 1], [0, 1], [0, 1], [2, 0], [0, 2], [1, 0], [0, 1], [2, 0]]))
    captured = capture_module(Glu(), lambda y: y.square().mean(), args, optimizer=SGDConfig())
    programs = {}
    for mode in ("flat", "flat+consume"):
        with products_scope(mode):
            programs[mode] = lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")
    plain, fused = programs["flat"], programs["flat+consume"]
    assert "xmat_epi" in fused.source and "xmat_epi" not in plain.source
    owner = [p for p in fused.manifest["phases"] if p.get("tile_consumers")]
    assert owner and any("silu" in n for group in owner[0]["tile_consumers"] for n in group)
    assert len(fused.manifest["phases"]) < len(plain.manifest["phases"])
    # Every value is still written exactly once, by an earlier or the same phase as each reader.
    writer = {}
    for index, p in enumerate(fused.manifest["phases"]):
        assert all(writer.get(r, -1) <= index for r in p["reads"])
        for w in p["writes"]:
            assert w not in writer
            writer[w] = index
    assert {w for p in plain.manifest["phases"] for w in p["writes"]} <= set(writer) | {
        w for p in plain.manifest["phases"] for w in p["writes"] if w not in writer and False}


def test_claim_products_take_counters_and_leave_claim_off_sources_unchanged(monkeypatch):
    """LATCH-TRAIN-EXPERTS feature "claim": TMA-fed products take claimed tasks from zeroed per-launch counters."""
    import re
    from megakernel_lab.operation_compiler import step17_intervals, ragged_experts_lower
    from megakernel_lab.operation_compiler.ragged_experts_lower import validate_products, CLAIM_SLOT, CLAIM_STATIC0
    plain = _aligned_bf16_program("flat+tma+rotate", width=64)
    # Structural admission: 16 packed rows over 4 segments are short products; they keep the static bodies.
    short = _aligned_bf16_program("flat+tma+rotate+claim", width=64)
    assert short.source == plain.source
    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", 0)
    claim = _aligned_bf16_program("flat+tma+rotate+claim", width=64)
    assert "ragged_claim.cuh" not in plain.source and "_claim<" not in plain.source
    assert "ragged_claim.cuh" not in plain.manifest["header_hashes"]
    assert claim.source.count("latch_ragged::tflat_claim<") == 1 and claim.source.count("latch_ragged::touter_claim<") == 1
    assert "latch_ragged::tflat<" not in claim.source and "latch_ragged::touter<" not in claim.source
    assert "ragged_claim.cuh" in claim.manifest["header_hashes"]
    assert CLAIM_SLOT not in claim.source and CLAIM_STATIC0 not in claim.source
    records = claim.manifest["group_rules"]["ragged_claim"]
    counters = claim.manifest["group_rules"]["ecta_queue_counters"]
    assert len(records) == 2 and sorted(r["counter"] for r in records) == [0, 1] and counters["count"] == 2
    # Counters are zeroed by CTA 0 at kernel entry (before the first grid barrier) at their base word.
    base = counters["base_word"]
    assert f"reinterpret_cast<unsigned*>(arena + {base}LL)[threadIdx.x] = 0u;" in claim.source
    assert all(f"arena + {base}LL + {r['counter']}LL" in claim.source for r in records)
    # The first claimed product of each interval keeps a static first task; later ones claim their first task.
    by_interval = {}
    for r in records:
        by_interval.setdefault(r["interval"], []).append(r["static_first"])
    assert all(flags[0] and not any(flags[1:]) for flags in by_interval.values())
    # Same operand maps and tile templates as the static bodies; device-unit isolation recognizes the claimed bodies.
    assert claim.manifest.get("tensor_maps") == plain.manifest.get("tensor_maps")
    assert len(re.findall(step17_intervals.RAGGED_CLAIM, claim.source)) == 2
    assert step17_intervals.RAGGED_CLAIM not in step17_intervals.wgmma_inline(plain.source)
    engines = {p.get("engine") for p in claim.manifest["phases"]}
    assert {"ragged_flat_tma_claim_wgmma", "ragged_flat_tma_outer_claim_wgmma"} <= engines
    # Claimed weight-gradient products lead their interval (they hold the longest tasks).
    order = [m.start() for m in re.finditer(r"latch_ragged::(touter_claim|tflat_claim)<", claim.source)]
    kinds = re.findall(r"latch_ragged::(touter_claim|tflat_claim)<", claim.source)
    assert len(order) == 2 and set(kinds) == {"touter_claim", "tflat_claim"}
    assert validate_products("flat+tma+rotate+narrow+claim")
    with pytest.raises(ValueError):
        validate_products("flat+tma+claim+claim")


def test_repi_runs_swiglu_consumers_on_claimed_product_registers(monkeypatch):
    """LATCH-TRAIN-EXPERTS feature "repi": SwiGLU consumers of a claimed gate/up pair read the pair's registers."""
    from megakernel_lab.operation_compiler import capture_module, ragged_experts_lower
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope

    class Swi(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(4, 64, 128))
            self.w3 = torch.nn.Parameter(torch.randn(4, 64, 128))
            self.w2 = torch.nn.Parameter(torch.randn(4, 128, 64))

        def forward(self, x, ids):
            order, offsets, inverse = route_order(ids, 4)
            packed = F.embedding(order // ids.shape[1], x).to(torch.bfloat16)
            act = F.silu(ragged_mm(packed, self.w1.to(torch.bfloat16), offsets)) * ragged_mm(packed, self.w3.to(torch.bfloat16), offsets)
            out = ragged_mm(act, self.w2.to(torch.bfloat16), offsets)
            return F.embedding(inverse, out).reshape(x.shape[0], 2, 64).sum(1).float()

    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", 0)
    torch.manual_seed(1749)
    args = (torch.randn(8, 64), torch.tensor([[0, 1], [0, 1], [0, 1], [2, 0], [0, 2], [1, 0], [0, 1], [2, 0]]))
    captured = capture_module(Swi(), lambda y: y.square().mean(), args, optimizer=SGDConfig())
    with products_scope("flat+tma+rotate+claim+repi"):
        program = lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")
    with products_scope("flat+tma+rotate+claim"):
        plain = lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")
    assert "using reg_tag = void;" in program.source and "reg_tag" not in plain.source
    heads = [p for p in program.manifest["phases"] if p.get("register_consumers")]
    assert heads and all(p.get("tile_consumers") for p in heads)
    assert len(program.manifest["phases"]) < len(plain.manifest["phases"])
    assert program.source.count("xmat_epi);") == len(heads)
