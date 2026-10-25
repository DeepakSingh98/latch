"""Portable short-attention intake; capture/lowering tests run on pinned Modal CPU."""
import math

import pytest
import torch

from megakernel_lab.operation_compiler import attention_adapters as AA
from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training


def _graph(*, bf16, t=64, d=128, h=2, causal=True):
    """One pre-norm attention block; explicit BF16 projection casts when bf16 (the BF16-boundary recipe)."""
    g = torch.Generator().manual_seed(7)
    b, hd = 2, d // h
    params = {"wq": torch.randn(d, d, generator=g) / math.sqrt(d), "wk": torch.randn(d, d, generator=g) / math.sqrt(d),
              "wv": torch.randn(d, d, generator=g) / math.sqrt(d), "wo": torch.randn(d, d, generator=g) / math.sqrt(d)}
    x = torch.randn(b, t, d, generator=g)
    mask = torch.triu(torch.full((t, t), -10000.0), diagonal=1)

    def linear(v, w):
        return (v.to(torch.bfloat16) @ w.to(torch.bfloat16)).float() if bf16 else v @ w

    def loss(p, x, mask):
        def heads(w):
            return linear(x.reshape(b * t, d), w).reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)
        q, k, v = heads(p["wq"]), heads(p["wk"]), heads(p["wv"])
        s = torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(hd))
        if causal:
            s = s + mask
        o = torch.bmm(torch.softmax(s, -1), v).reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d)
        return linear(o, p["wo"]).square().mean()
    momentum = {k: torch.zeros_like(v) for k, v in params.items()}
    return capture_training(loss, params, momentum, (x, mask), SGDConfig(lr=0.01, momentum=0.9)).ir


def test_registry_contracts_are_complete():
    registry = AA.registry()
    assert set(registry["bodies"]) == set(AA.BODIES)
    for name, body in AA.BODIES.items():
        assert body.headers and body.phases and body.storage and body.synchronization, name
        assert body.arch in ("sm_90", "sm_90a")
    for names in AA.PREFERENCE.values():
        assert all(name in AA.BODIES for name in names)


def test_short_bf16_boundary_group_prefers_short_body_over_legacy_fp32_tiles():
    ir = _graph(bf16=True)
    survey = AA.survey(ir)
    (row,) = survey["groups"]
    assert row["operand_class"] == "bf16_boundary"
    assert row["candidates"]["bf16b_short"]["admitted"]
    assert not row["candidates"]["bf16b_flat_long"]["admitted"]
    assert row["selected"] == "bf16b_short"
    assert AA.legacy_choice(row, "p6d") == "fp32_tiles"


def test_short_phases_declare_storage_and_satisfy_execution_contract():
    from megakernel_lab.operation_compiler import attention_lower, short_attention
    from megakernel_lab.operation_compiler.lower import _ident
    ir = _graph(bf16=True)
    operations, values, names, ref, field, output_name, protected = AA._graph_parts(ir)
    layouts = AA._layouts(values, field, operations)
    groups, _ = attention_lower.plan(operations, values, names, ref=ref, field=field, output_name=output_name,
                                     protected=protected)
    plans, _ = short_attention.admit(groups, (operations, values, names, ref, field, output_name), layouts, protected)
    (plan,) = plans.values()
    dtypes = {n: str(field(v, "dtype", "float32")).removeprefix("torch.") for n, v in values.items()}
    mask = attention_lower._mask_strides(layouts[plan["group"]["mask"]], plan["bh"], plan["t"], True)
    phases = AA.emit_phases("bf16b_short", plan, layouts=layouts, dtypes=dtypes, values=values, field=field,
                            ident=_ident, mask_strides=mask)
    assert [p["role"] for p in phases] == ["forward", "backward"]
    lifetimes = {name: s["lifetime"] for p in phases for name, s in p["storage"].items()}
    aux = short_attention.aux_names(plan["group"])
    assert lifetimes[aux["o32"]] == lifetimes[aux["lse"]] == "forward_to_backward"
    contract = AA.contract(phases, layouts, dtypes)
    assert contract[1].dependencies == (0,)
    assert all(c.threads == 256 and 0 < c.shared_bytes <= AA.MAX_SHARED for c in contract)


@pytest.mark.parametrize("t,d,h", [(64, 128, 2), (96, 192, 6)])
def test_common_planner_installs_portable_short_body_and_preserves_legacy(t, d, h):
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.execution_plan import current_execution
    graph = _graph(bf16=True, t=t, d=d, h=h)
    legacy = lower(graph)
    composed = lower(graph, execution="compose")
    assert composed.manifest["execution_selection"]["selected"] == "common_static"
    selected = [g for g in composed.manifest["attention_groups"] if g["decision"] == "adapter:bf16b_short"]
    assert len(selected) == 1 and selected[0]["splits"] == 3
    phases = [p for p in composed.manifest["phases"] if p.get("engine") == "bf16b_short"]
    assert [p["kind"] for p in phases] == ["attention_short_forward", "attention_short_backward"]
    assert all(p["threads"] == 256 and p["requires_full_grid"] for p in phases)
    assert "attention_short.cuh" in composed.manifest["header_hashes"]
    assert all(p["kind"] != "attention_forward" for p in composed.manifest["phases"])
    aux = {name for name, desc in phases[0]["storage"].items() if desc["lifetime"] == "forward_to_backward"}
    assert len(aux) == 2 and aux.issubset(phases[1]["reads"])
    assert aux.issubset(composed.manifest["offsets"])
    assert composed.source.count("pshort::forward_item") == 1
    assert current_execution() == "legacy"
    again = lower(graph)
    assert (again.source, again.manifest) == (legacy.source, legacy.manifest)


def test_short_accumulated_final_is_refused_before_suppressing_operations():
    from megakernel_lab.operation_compiler import attention_lower, short_attention
    from megakernel_lab.operation_compiler.compose_attention import ComposedAttention
    from megakernel_lab.operation_compiler.lower import _dtype
    graph = _graph(bf16=True)
    operations, values, names, ref, field, output_name, protected = AA._graph_parts(graph)
    static = AA._layouts(values, field, operations)
    groups, _ = attention_lower.plan(operations, values, names, ref=ref, field=field,
                                     output_name=output_name, protected=protected)
    plans, _ = short_attention.admit(groups, (operations, values, names, ref, field, output_name), static, protected)
    plan = next(iter(plans.values()))
    refusals = []
    adapter = ComposedAttention(operations=operations, values=values, names=names, inputs=[],
        outputs=[], state_writes={}, placement={plan["finals"]["dq"]: (0, "accumulate")},
        layouts=dict(static), dtypes={name: _dtype(value) for name, value in values.items()},
        roots=[], phases=[], producers={}, helper_parts=[], extra_headers=[], decisions=[],
        ref=ref, field=field, output_name=output_name, groups=groups, excluded_members=set(),
        refusals=refusals, allow_short=True)
    assert not adapter.short
    assert any("accumulated output" in item["reason"] for item in refusals)
    assert all(name not in adapter.members for name in plan["skip"])


def test_short_portable_intake_records_peer_scope_and_preserved_failure():
    registry = AA.registry()
    assert registry["portable_source_commit"] == "1ad3e78"
    assert registry["common_phase_intake"] == ["bf16b_short"]
    assert "shared-bf16b scalar-loss failure" in AA.__doc__
