"""K-concatenated sibling claimed products (segmented_products 'kcat'; ops/kcat_products.py).

* 2-way: two claimed right products whose results are read only by one aten.add become one
  latch_grp::tflat_kcat_claim_h product writing the add's result; the add and both results are not materialized.
* 3-way: three claimed right products summed only inside a 'packsum' fused pack-sum load become one
  latch_grp::tflat_kcat3_claim_h product; the pack gradient reads the fused BF16 sum.
* Off (no 'kcat'): phases unchanged. 'kcat' requires 'bsearch' and precision_policy='practical'.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

FEATURES = "flat+tma+rotate+claim+sepi"
KCAT = FEATURES + "+bsearch+kcat"


def _ident(name):
    return f"v_{name}"


def _layout(name, rows=256, width=128):
    return SimpleNamespace(root=name, contiguous=True, offset=0, numel=rows * width, shape=(rows, width),
                           strides=(width, 1))


def _product(name, a, w, rows=256, width=128, k=128):
    init = (f"(v_{a} + 0LL), {k}LL, 1LL, 0LL,\n                 (v_{w} + 0LL), 1LL, {k}LL, {width * k}LL,\n"
            f"                 (v_{name} + 0LL), {width}LL, 0LL, {rows}, {width}, {k}, 1")
    code = (f"{{\n  popsc::BArgs g{{{init}}};\n"
            f"  latch_grp::tflat_claim_h<4,128,128,2,true,4,/*XCLAIM_S0*/true>(POPS4B_TMAP({a},0,{k},{rows},{2 * k},64,128), "
            f"POPS4B_TMAP({w},0,{k},{4 * width},{2 * k},64,128), g, (v_off + 0LL), {rows}, smem, /*XMAT_ROT*/0, "
            f"/*XCLAIM*/nullptr);\n}}")
    return dict(kind="gemm", engine="ragged_flat_tma_claim_wgmma", flat_claim=True, names=[name], writes=[name],
                reads=[a, w, "off"], targets=["latch_ops.ragged_mm.default"], code=code, arithmetic="product",
                flat_consumable=True, flat_numel=rows * width)


def _pair():
    phases = [_product("pa", "da", "wa"), _product("pb", "db", "wb"),
              dict(kind="elementwise", names=["s"], writes=["s"], reads=["pa", "pb"], targets=["aten.add.Tensor"],
                   code="s = pa + pb")]
    layouts = {n: _layout(n) for n in ("pa", "pb", "s")}
    dtypes = {n: "bfloat16" for n in ("pa", "pb", "s")}
    return phases, layouts, dtypes


def test_two_way_host_form():
    from megakernel_lab.operation_compiler import kcat_products
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    phases, layouts, dtypes = _pair()
    with products_scope(KCAT):
        records, drop, values = kcat_products.fuse(phases, layouts, dtypes, _ident, set())
    assert records == [dict(add="s", products=["pa", "pb"], host="pb")]
    assert drop == {0, 2} and values == {"pa", "pb"}
    host = phases[1]
    assert host["engine"] == "ragged_flat_tma_claim_kcat_wgmma" and host["writes"] == ["s"]
    assert host["kcat"] == dict(products=["pa", "pb"], add="s")
    code = host["code"]
    assert code.count("latch_grp::tflat_kcat_claim_h<4,128,128,2,true,4,/*XCLAIM_S0*/true>(") == 1
    # The later product hosts the call (its maps first); both C fields point at the add's result.
    assert code.index("POPS4B_TMAP(db,") < code.index("POPS4B_TMAP(da,")
    assert code.count("(v_s + 0LL)") == 2 and "v_pa + 0LL" not in code and "v_pb + 0LL" not in code
    assert "flat_consumable" not in host and set(host["reads"]) == {"db", "wb", "off", "da", "wa"}


def test_two_way_refusals_and_off():
    from megakernel_lab.operation_compiler import kcat_products
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    phases, layouts, dtypes = _pair()
    before = copy.deepcopy(phases)
    for products in (None, FEATURES, FEATURES + "+bsearch"):
        with products_scope(products):
            assert kcat_products.fuse(phases, layouts, dtypes, _ident, set()) == ([], set(), set())
            assert kcat_products.fuse_qkv(phases, layouts, dtypes, _ident, set()) == ([], set(), set())
    assert phases == before
    # A second reader of one product (or a kept value) refuses the fusion and leaves the phases unchanged.
    phases.append(dict(kind="elementwise", names=["t"], writes=["t"], reads=["pa"], targets=["aten.neg.default"]))
    with products_scope(KCAT):
        records, drop, values = kcat_products.fuse(phases, layouts, dtypes, _ident, set())
        assert records == [dict(add="s", refused="other readers or kept values")] and not drop and not values
        phases.pop()
        records, drop, _ = kcat_products.fuse(phases, layouts, dtypes, _ident, {"s"})
        assert records[0]["refused"] == "other readers or kept values" and not drop
    assert phases == before


def _pack_sum():
    """Three claimed products whose widened FP32 sum is read only by a routing-pack fixed-k sum."""
    shape = dict(rows=64, width=64, k=64)
    phases = [_product(n, "x" + n, "w" + n, **shape) for n in ("pa", "pb", "pc")]
    phases.append(dict(kind="elementwise", names=["g"], writes=["g"], reads=["pa", "pb", "pc"],
                       targets=["aten.add.Tensor"],
                       body=["const float r_g = __bfloat162float(v_pa[i]) + __bfloat162float(v_pb[i]) + "
                             "__bfloat162float(v_pc[i]); v_g[i] = r_g;"]))
    phases.append(dict(kind="embedding_gradient", engine="permutation_fixed_sum_v1", names=["dx"], writes=["dx"],
                       reads=["g", "ids"], arithmetic="sum",
                       code="for (;;) { v_[s_] = reinterpret_cast<const float4*>(v_g + (q_[s_] * 16LL))[col]; }"))
    layouts = {n: _layout(n, 64, 64) for n in ("pa", "pb", "pc", "g", "dx")}
    dtypes = dict(pa="bfloat16", pb="bfloat16", pc="bfloat16", g="float32", dx="float32", ids="int64")
    return phases, layouts, dtypes


def test_three_way_host_form_after_pack_sum():
    from megakernel_lab.operation_compiler import claimed_products, kcat_products
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    phases, layouts, dtypes = _pack_sum()
    with products_scope(KCAT + "+packsum"):
        fused, drop, values = claimed_products.fuse_pack_sums(phases, layouts, dtypes, _ident, set())
        assert fused == [(["g"], ["dx"])] and drop == {3} and values == {"g"}
        phases = [p for k, p in enumerate(phases) if k not in drop]
        pack = phases[-1]
        assert pack["fused_pack_sum"] == ["g"] and set(pack["reads"]) == {"pa", "pb", "pc", "ids"}
        records, drop, values = kcat_products.fuse_qkv(phases, layouts, dtypes, _ident, set())
    assert len(records) == 1 and records[0]["pack"] == "dx" and "refused" not in records[0]
    assert drop == {0, 1} and values == {"pb", "pc"}
    host = phases[2]
    assert host["engine"] == "ragged_flat_tma_claim_kcat_wgmma" and host["writes"] == ["pa"]
    assert host["code"].count("latch_grp::tflat_kcat3_claim_h<4,128,128,2,true,4,/*XCLAIM_S0*/true>(") == 1
    assert host["code"].count("(v_pa + 0LL), 64LL, 0LL") == 3            # every C field is the fused sum's buffer
    assert pack["code"] == ("for (;;) { v_[s_] = latch_pgrad::widen4(reinterpret_cast<const uint2*>(v_pa + "
                            "((q_[s_] * 16LL)))[col]); }")
    assert set(pack["reads"]) == {"pa", "ids"}


def test_pack_sum_is_inert_without_its_token():
    from megakernel_lab.operation_compiler import claimed_products
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    phases, layouts, dtypes = _pack_sum()
    before = copy.deepcopy(phases)
    for products in (None, FEATURES, KCAT):
        with products_scope(products):
            assert claimed_products.fuse_pack_sums(phases, layouts, dtypes, _ident, set()) == ([], set(), set())
    assert phases == before


def test_pack_sum_alone_fuses_the_widened_sum_into_the_pack():
    from megakernel_lab.operation_compiler import claimed_products
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope, validate_products
    phases, layouts, dtypes = _pack_sum()
    validate_products(FEATURES + "+packsum")
    with products_scope(FEATURES + "+packsum"):
        fused, drop, values = claimed_products.fuse_pack_sums(phases, layouts, dtypes, _ident, set())
    assert fused == [(["g"], ["dx"])] and drop == {3} and values == {"g"}
    pack = phases[4]
    assert pack["fused_pack_sum"] == ["g"] and set(pack["reads"]) == {"pa", "pb", "pc", "ids"}
    code = pack["code"]                       # the pack loads the three BF16 products and adds them in FP32 itself
    assert "v_g" not in code and code.count("const float r_g = ") == 4
    assert all(f"reinterpret_cast<const uint2*>(v_{n} + rb_)[col]" in code for n in ("pa", "pb", "pc"))
    assert [p["names"] for p in phases[:3]] == [["pa"], ["pb"], ["pc"]]      # the products themselves are unchanged
    # A kept sum, or a second reader of it, refuses the fusion.
    phases, layouts, dtypes = _pack_sum()
    with products_scope(FEATURES + "+packsum"):
        assert claimed_products.fuse_pack_sums(phases, layouts, dtypes, _ident, {"g"})[0] == []
        phases.append(dict(kind="elementwise", names=["h"], writes=["h"], reads=["g"], targets=["aten.neg.default"]))
        assert claimed_products.fuse_pack_sums(phases, layouts, dtypes, _ident, set())[0] == []


def test_kcat_requires_bsearch_and_practical_precision(tmp_path):
    from megakernel_lab.operation_compiler.ragged_experts_lower import validate_products
    from .test_option_rules import _refused
    with pytest.raises(ValueError, match="requires 'bsearch'"):
        validate_products(FEATURES + "+kcat")
    assert validate_products(KCAT) == KCAT
    assert "requires precision_policy='practical'" in _refused(tmp_path, segmented_products=KCAT)


def _siblings(tokens=64, rows=256, width=128, hidden=128, groups=4):
    """Two sibling products of the same gathered rows (h1 = R W1, h2 = R W2): the rows' gradient adds both VJP
    products and is read only by the gather's VJP."""
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm

    class Siblings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1033)
            self.x = torch.nn.Parameter(torch.randn(tokens, width))
            self.w1 = torch.nn.Parameter(torch.randn(groups, width, hidden))
            self.w2 = torch.nn.Parameter(torch.randn(groups, width, hidden))

        def forward(self, token, offsets):
            packed = torch.nn.functional.embedding(token, self.x.to(torch.bfloat16))
            h1 = ragged_mm(packed, self.w1.to(torch.bfloat16), offsets)
            h2 = ragged_mm(packed, self.w2.to(torch.bfloat16), offsets)
            return torch.nn.functional.silu(h1.float()) * h2.float()

    token = torch.arange(rows) * 7 % tokens
    offsets = torch.tensor([0, 40, 40, 150, rows])
    return Siblings(), (token, offsets)


def test_lowering_fuses_sibling_input_gradients(monkeypatch):
    from megakernel_lab.operation_compiler import capture_module, ragged_experts_lower
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", 0)
    module, args = _siblings()
    captured = capture_module(module, lambda y: y.square().mean(), args, optimizer=SGDConfig())
    programs = {}
    for products in (FEATURES + "+bsearch", KCAT):
        with products_scope(products):
            programs[products] = lower(captured.ir, precision_policy="practical", execution="compose",
                                       target_arch="sm_90")
    base, fused = programs[FEATURES + "+bsearch"], programs[KCAT]
    assert "kcat" not in base.source
    assert fused.source.count("latch_grp::tflat_kcat_claim_h<") == 1
    (host,) = [p for p in fused.manifest["phases"] if p.get("kcat")]
    assert host["engine"] == "ragged_flat_tma_claim_kcat_wgmma"
    gemms = lambda program: [p for p in program.manifest["phases"] if p.get("kind") == "gemm"]
    assert len(gemms(fused)) == len(gemms(base)) - 1
