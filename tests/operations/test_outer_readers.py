"""CPU contracts for operation_compiler/outer_readers.py (matrix_bodies 'shortk'): short-K BF16 outer products
evaluated at their readers.

``shortk`` removes admitted skinny-K products and substitutes their element expression (ascending-K fmaf chain from
+0, one BF16 rounding) at their reader's identity reads; without the token programs are unchanged.
"""
from __future__ import annotations

import re

import torch
from torch import nn
import torch.nn.functional as F

from megakernel_lab.operation_compiler import matrix_bodies, outer_readers


class Mix(nn.Module):
    """A softmax-weighted mixture of sources (RMS scale, BF16 GEMV keys, softmax x gains, FP32 mixture)."""

    def __init__(self, sources, rows, width):
        super().__init__()
        g = torch.Generator().manual_seed(5201)
        self.n, self.d = sources, width
        for i in range(sources):
            setattr(self, f"s{i}", nn.Parameter(torch.randn(rows, width, generator=g)))
        self.query = nn.Parameter(torch.randn(width, generator=g) / width ** 0.5)
        self.gains = nn.Parameter(1.0 + 0.1 * torch.randn(sources, generator=g))

    def forward(self, r):
        q = self.query.to(torch.bfloat16).unsqueeze(0)
        src = [getattr(self, f"s{i}") for i in range(self.n)]
        scales, logits = [], []
        for s in src:
            rr = torch.rsqrt((s * s).mean(-1, keepdim=True) + 1e-6)
            scales.append(rr)
            logits.append((F.linear(s.to(torch.bfloat16), q) * rr).float())
        alpha = torch.softmax(torch.cat(logits, -1) / self.d ** 0.5, -1) * self.gains * torch.cat(scales, -1) * r
        out = src[0] * alpha[:, 0:1]
        for i in range(1, self.n):
            out = out + src[i] * alpha[:, i:i + 1]
        return out


class Linear2(nn.Module):
    def __init__(self, rows, width):
        super().__init__()
        g = torch.Generator().manual_seed(5202)
        self.x = nn.Parameter(torch.randn(rows, width, generator=g))
        self.w = nn.Parameter(torch.randn(2, width, generator=g) / width ** 0.5)

    def forward(self, r):
        return (F.linear(self.x.to(torch.bfloat16), self.w.to(torch.bfloat16)) * r).float()


def _program(bodies, model="mix", rows=2048, width=64):
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    g = torch.Generator().manual_seed(5203)
    r = torch.rand(rows, 1, generator=g) + 0.5
    module = Mix(3, rows, width) if model == "mix" else Linear2(rows, width)
    cot = torch.randn(rows, width if model == "mix" else 2, generator=g)
    captured = capture_module(module, lambda p, c: (p * c).sum(), (r,), (cot,),
                              optimizer=SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0))
    with matrix_bodies.scope(bodies):
        return lower(captured.ir, execution="compose")


def _records(program):
    return (program.manifest.get("group_rules") or {}).get("short_k_at_readers") or []


def _engines(program):
    return [p.get("engine") for p in program.manifest["phases"] if p.get("kind") == "gemm"]


def test_accessor_follows_the_token():
    assert not outer_readers.enabled()
    with matrix_bodies.scope("skinny"):
        assert not outer_readers.enabled()
    with matrix_bodies.scope("skinny+shortk"):
        assert outer_readers.enabled()
    assert outer_readers.KMAX == 2 and outer_readers.MAX_READERS == 1


def test_default_off_unchanged():
    program = _program("skinny")
    assert not _records(program)
    assert "latch_out_col" not in program.source and "Low-K BF16 outer products" not in program.source
    assert _engines(program).count("skinny_k_vec") == 3


def test_token_places_products_at_readers():
    base, cand = _program("skinny"), _program("skinny+shortk")
    placed = [r for r in _records(cand) if "refused" not in r]
    assert len(placed) == 3, _records(cand)
    assert _engines(cand).count("skinny_k_vec") == 0 and _engines(base).count("skinny_k_vec") == 3
    for r in placed:
        assert r["k"] == 1 and r["elements_not_stored"] == 2048 * 64 and len(r["readers"]) == 1
        out = "v_" + r["value"]
        assert not re.search(rf"(?<![\w.]){re.escape(out)}(?!\w)", cand.source), out   # never stored nor read
    # the producer's arithmetic, verbatim: one fmaf from +0 and one RN rounding per element
    assert cand.source.count("__float2bfloat16_rn(fmaf(") >= 3
    assert ", 0.0f))" in cand.source
    # operand lifetimes now reach the reader: in a tiny graph arena packing may grow slightly
    assert cand.manifest["arena_bytes"] <= 1.05 * base.manifest["arena_bytes"]
    # everything else (the query-gradient split products) is unchanged
    assert [e for e in _engines(cand)] == [e for e in _engines(base) if e != "skinny_k_vec"]
    readers = [p for p in cand.manifest["phases"] if p.get("short_k_inputs")]
    assert {v for p in readers for v in p["short_k_inputs"]} <= {r["value"] for r in placed}


def test_k2_products():
    cand = _program("skinny+shortk", model="linear2")
    recs = _records(cand)
    placed = [r for r in recs if "refused" not in r]
    assert placed and all(r["k"] == 2 for r in placed), recs
    assert "fmaf(" in cand.source and "_rn(fmaf(" in cand.source


def test_helpers_text():
    assert outer_readers.HELPERS == (
        "// Low-K BF16 outer products at readers (ops/outer_readers.py)\n"
        "#ifndef LATCH_OUT_COL\n#define LATCH_OUT_COL\n"
        "#define latch_out_col(P, OFF, N, I) ((P)[(long long)((unsigned)(I) % (N)) + (OFF)])\n#endif\n")


def test_vector_lanes_rewrite():
    text = ("x = __bfloat162float(latch_out_col(v_t_5, 0LL, 384u, ((4 * j + 0)))) + "
            "__bfloat162float(latch_out_col(v_t_5, 384LL, 384u, ((4 * jc1 + 3))));")
    got = outer_readers.vector_lanes(text)
    assert "latch_out_col" not in got
    assert "__uint_as_float(reinterpret_cast<const uint2*>(v_t_5 + 0LL)[(unsigned)((4 * j)) % 384u / 4u].x << 16)" in got
    assert "__uint_as_float(reinterpret_cast<const uint2*>(v_t_5 + 384LL)[(unsigned)((4 * jc1)) % 384u / 4u].y & 0xffff0000u)" in got


def test_element_expression_order():
    o = dict(m=8192, n=384, k=2, sam=2, sak=1, sbk=384, sbn=1, a="v_a", b="v_b", out="v_o")
    e = outer_readers.element_expression(o, False)
    # z = 1 is the outer fmaf; z = 0 the inner from +0 (ascending K)
    assert e.startswith("__float2bfloat16_rn(fmaf(__bfloat162float(v_a[(long long)((unsigned)(i) / 384u) * 2LL + 1LL])")
    assert e.endswith("0LL]), 0.0f)))")
    v = outer_readers.element_expression(o, True)
    assert "__bfloat162float(latch_out_col(v_b, 384LL, 384u, (i)))" in v
    assert "__bfloat162float(latch_out_col(v_b, 0LL, 384u, (i)))" in v
