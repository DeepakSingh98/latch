"""CPU contracts for the bitwise skinny bodies (operation_compiler/skinny_backward.py; matrix_bodies 'skinny').

Under ``matrix_bodies='skinny'`` the P9-E skinny split / skinny-K bodies of explicit-BF16 products whose shapes and
layouts the skinny predicates admit are replaced, keeping their tasks, K parts, part-order sums and scratch contracts;
everything else (and the default, no token) keeps the P9-E bodies. ``skinnybatch`` additionally runs one interval's
same-geometry split products in one body and puts small batches' skinny-K products on warps 4-7.
"""
from __future__ import annotations

import re
from types import SimpleNamespace as N

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import matrix_bodies
from megakernel_lab.operation_compiler import skinny_backward as sb

NEW = ("skinny_split_ring", "skinny_k_vec")
P9E = ("p9e_skinny_split", "p9e_skinny_k")


class Read(torch.nn.Module):
    """Per-source [T, 1] GEMV keys: s.bf16 @ q.bf16^T, scaled per row."""

    def __init__(self, sources, rows, width):
        super().__init__()
        g = torch.Generator().manual_seed(5101)
        self.n = sources
        for i in range(sources):
            setattr(self, f"s{i}", torch.nn.Parameter(torch.randn(rows, width, generator=g)))
        self.query = torch.nn.Parameter(torch.randn(width, generator=g) / width ** 0.5)

    def forward(self, r):
        q = self.query.to(torch.bfloat16).unsqueeze(0)
        return torch.cat([(F.linear(getattr(self, f"s{i}").to(torch.bfloat16), q) * r).float() for i in range(self.n)], -1)


def _program(bodies, sources=3, rows=2048, width=64, **options):
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    g = torch.Generator().manual_seed(5103)
    r = torch.rand(rows, 1, generator=g) + 0.5
    cot = torch.randn(rows, sources, generator=g)
    captured = capture_module(Read(sources, rows, width), lambda p, c: (p * c).sum(), (r,), (cot,),
                              optimizer=SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0))
    with matrix_bodies.scope(bodies):
        return lower(captured.ir, execution="compose", **options)


def _engines(program):
    return [p.get("engine") for p in program.manifest["phases"] if p.get("kind") == "gemm"]


def test_default_keeps_p9e_bodies():
    program = _program(None)
    engines = _engines(program)
    assert engines.count("p9e_skinny_split") == 3 and engines.count("p9e_skinny_k") == 3, engines
    assert not set(engines) & set(NEW)
    assert "Alt skinny" not in program.source


def test_accessors_follow_the_tokens():
    assert not sb.enabled() and not sb.batching() and not sb.combining()
    with matrix_bodies.scope("exact"):
        assert not sb.enabled()
    with matrix_bodies.scope("skinny"):
        assert sb.enabled() and not sb.batching() and not sb.combining()
    with matrix_bodies.scope("exact+skinny+skinnybatch"):
        assert sb.enabled() and sb.batching() and sb.combining()
    with pytest.raises(ValueError):
        matrix_bodies.rules_of("skinnybatch")


def test_token_selects_skinny_bodies_with_same_split_contract():
    base, cand = _program(None), _program("skinny")
    eb, ec = _engines(base), _engines(cand)
    assert ec.count("skinny_split_ring") == 3 and ec.count("skinny_k_vec") == 3, ec
    assert len(eb) == len(ec) and [P9E[NEW.index(e)] if e in NEW else e for e in ec] == eb
    splits_b = [p for p in base.manifest["phases"] if p.get("engine") == "p9e_skinny_split"]
    splits_c = [p for p in cand.manifest["phases"] if p.get("engine") == "skinny_split_ring"]
    for pb, pc in zip(splits_b, splits_c):
        # Same K parts / part length and the same private partial and counter contract (values bit for bit).
        parts = re.search(r"\((\d+) parts of (\d+)", pb["arithmetic"]).groups()
        assert re.search(r"\((\d+) parts of (\d+)", pc["arithmetic"]).groups() == parts
        assert pc["private_scratch"] and pc["private_scratch"] == pb["private_scratch"]
        assert [e.get("storage") for e in pc.get("counter_effects") or ()] == \
            [e.get("storage") for e in pb.get("counter_effects") or ()]
        assert pc["shared_bytes"] <= 196608
    assert "/*P2_" not in cand.source and "/*GSKC" not in cand.source
    assert "cp.async.cg.shared.global" in cand.source and cand.source.count("Alt skinny split") == 3
    assert cand.source.count("Alt skinny-K product") == 3
    # Barrier structure is unchanged: the bodies replace phase code only.
    assert cand.manifest["barrier_count"] == base.manifest["barrier_count"]


def test_unadmitted_shapes_keep_p9e_split():
    """K * S not a multiple of 8 (odd row count): the skinny split predicate refuses; the outer body still applies."""
    engines = _engines(_program("skinny", sources=2, rows=2049))
    assert engines.count("p9e_skinny_split") == 2 and engines.count("skinny_k_vec") == 2, engines


def test_reference_policy_has_no_split():
    """The reference policy admits no split bodies (old or new); short-K bodies keep the per-element arithmetic."""
    program = _program("skinny", policy="reference")
    engines = _engines(program)
    assert not {"skinny_split_ring", "p9e_skinny_split"} & set(engines), engines
    for p in program.manifest["phases"]:
        if p.get("engine") in ("skinny_k_vec", "p9e_skinny_k"):
            assert p["arithmetic"].startswith("explicit BF16 operands/output; FP32 ascending-K fmaf from +0 per element")


@pytest.mark.parametrize("width", [72, 120])
def test_neighbouring_widths(width):
    """L % 16 == 8 (half last chunk) and a wider L: both bodies admitted, no unresolved markers."""
    program = _program("skinny", sources=2, rows=4096, width=width)
    engines = _engines(program)
    assert engines.count("skinny_split_ring") == 2 and engines.count("skinny_k_vec") == 2, engines
    assert "/*P2_" not in program.source


def test_plan_and_code_shapes():
    """Pure emitter checks (no lowering)."""
    L = lambda off=0: N(offset=off)
    plan = sb.split_plan(1, 384, 8192, 1, 1, 384, 1, L(), L(), 4, 2048)
    assert plan and plan["chunks"] == 24 and plan["S"] == 1 and plan["shared_bytes"] == 69632
    code = sb.split("v_a", "v_b", "v_o", 1, 384, 8192, 4, 2048, plan, lambda x: x)
    assert code.startswith("{ // Alt skinny split (bitwise = P9-E skinny split): 384 x 1 outputs, 4 K parts of 2048")
    assert "task < 96LL" in code and "k0 + 2048" in code and "q < 4; ++q" in code
    assert "fmaf(__bfloat162float(bk[z * 16]), __bfloat162float(sk[z * 1]), acc)" in code
    assert "#pragma unroll 16\n" in code and "cp.async.wait_group 6;" in code
    w2 = sb.split_plan(2, 384, 8192, 1, 2, 384, 1, L(16), L(4096), 4, 2048)
    assert w2 and w2["S"] == 2
    assert sb.split_plan(1, 384, 8001, 1, 1, 384, 1, L(), L(), 4, 2001) is None
    assert sb.split_plan(1, 384, 8192, 1, 1, 384, 1, L(4), L(), 4, 2048) is None      # unaligned small operand
    assert sb.split_plan(1, 384, 8192, 1, 1, 386, 1, L(), L(), 4, 2048) is None      # unaligned big rows
    outer = sb.outer("v_a", "v_b", "v_o", 8192, 384, 1, 1, 1, 384, 1, lambda x: x)
    assert outer.startswith("{ // Alt skinny-K product (bitwise = P9-E skinny-K): 8192x384 = (8192x1)(1x384)")
    assert "393216u" in outer and "& 15ull" in outer and "e < 3145728LL" in outer
    # All A values of the 8 unrolled items load before any 16-byte store; plain stores.
    assert "float avs[8][1];" in outer and "*reinterpret_cast<uint4*>(v_o + (long long)row * 384LL + c) = v;" in outer
    assert "__stcs" not in outer
    assert sb.outer_ok(8192, 384, 2, 384, 1, None) and not sb.outer_ok(8192, 380, 1, 380, 1, None)


def _interval(splits, outers):
    """Synthetic interval phases: ``splits`` same-geometry split products and ``outers`` skinny-K products."""
    L = lambda off=0: N(offset=off)
    m, n, k, parts, kp = 1, 384, 8192, 4, 2048
    plan = sb.split_plan(m, n, k, 1, 1, n, 1, L(), L(), parts, kp)
    phases = []
    for j in range(splits):
        code = f"{{\n  unsigned* latch_cnt = /*GSKC:{plan['chunks']}*/;\n  float* latch_part = v_gsk_p{j};\n}}"
        info = dict(plan, m=m, n=n, k=k, parts=parts, kp=kp, out=f"v_o{j}", big=f"v_b{j}", small=f"v_s{j}")
        phases.append({"names": [f"p{j}"], "code": code, "skinny_split": info, "shared_bytes": plan["shared_bytes"]})
    for j in range(outers):
        phases.append({"names": [f"q{j}"], "code": "", "skinny_outer": dict(
            m=4096, n=384, k=1, sam=1, sak=1, sbk=384, sbn=1, a=f"v_qa{j}", b=f"v_qb{j}", out=f"v_qo{j}")})
    return phases


def test_batch_interval_combines_small_batches():
    """skinnybatch: a batch of <= COMBINE_MAX products carries the interval's skinny-K products on warps 4-7."""
    phases = _interval(3, 3)
    with matrix_bodies.scope("skinny+skinnybatch"):
        out = sb.batch_interval(phases, list(range(6)))
    assert sorted(out) == [0, 1, 2, 3, 4, 5]
    body = out[0]
    assert body.startswith("{ // Alt batched skinny split (3 products: p0, p1, p2)")
    assert "if (threadIdx.x < 128) {" in body and 'asm volatile("bar.sync 1, 128;" ::: "memory");' in body
    assert "warps 4-7: skinny-K products" in body and "both groups leave together" in body
    for j in range(3):
        assert f"unsigned* skb_cnt{j} = /*GSKC:24*/;" in body and f"float* skb_part{j} = v_gsk_p{j};" in body
        assert f"// q{j}: 4096x384 = (4096x1)(1x384)" in body
    assert out[1] == "// Alt: p1 runs in the batched skinny split of p0"
    assert out[3] == "// Alt: q0 runs on warps 4-7 of the batched skinny split of p0"
    assert [p["skinny_batch"] for p in phases[3:]] == [dict(leader="p0", size=3, outer=True)] * 3
    per = phases[0]["skinny_split"]["small_bytes"] + 4 * 128 * 16 * 2
    assert phases[0]["shared_bytes"] == max(phases[0]["skinny_split"]["shared_bytes"], 3 * per)


def test_batch_interval_large_batch_is_not_combined():
    """More than COMBINE_MAX products: one 256-thread batch body; the skinny-K products keep their own phases."""
    assert sb.COMBINE_MAX == 5
    phases = _interval(6, 2)
    with matrix_bodies.scope("skinny+skinnybatch"):
        out = sb.batch_interval(phases, list(range(8)))
    assert sorted(out) == [0, 1, 2, 3, 4, 5]
    assert out[0].startswith("{ // Alt batched skinny split (6 products: ")
    assert "bar.sync 1, 128;" not in out[0] and "warps 4-7" not in out[0]
    assert all("skinny_batch" not in p for p in phases[6:])


def test_combined_mode_puts_outers_on_warps_4_7():
    """skinnybatch: the interval's skinny-K products run on warps 4-7 of the batch body (named barrier 1 for the
    split warps); both groups leave the body together. ``splitctr`` keeps the split products in the skinny-K
    products' interval (no counter zero phase in between)."""
    program = _program("skinny+skinnybatch+splitctr", sources=3)
    phases = program.manifest["phases"]
    outer = [p for p in phases if (p.get("skinny_batch") or {}).get("outer")]
    split = [p for p in phases if p.get("skinny_batch") and not p["skinny_batch"].get("outer")]
    assert len(outer) == 3 and len(split) == 3, [p.get("skinny_batch") for p in phases if p.get("skinny_batch")]
    assert all(p["skinny_batch"]["size"] <= sb.COMBINE_MAX for p in outer)
    assert program.source.count("Alt batched skinny split (3 products") == 1
    assert program.source.count("runs in the batched skinny split of") == 2
    assert program.source.count("runs on warps 4-7 of the batched skinny split of") == 3
    assert "bar.sync 1, 128;" in program.source and "both groups leave together" in program.source
    assert "/*GSKC" not in program.source and "/*P2_" not in program.source
    plain = _program("skinny", sources=3)
    assert "Alt batched" not in plain.source
