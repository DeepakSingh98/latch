"""Option contracts of the opt-in rule families (CPU; no lowering or native build).

Every rule family is a '+'-joined list of distinct names validated by ``prepare_module``, refused with matrix tuning,
forwarded by ``CapturedModule.build`` only when given, and active only inside the scopes that ``prepare_module`` opens
around ``lower()``. Tokens that reassociate arithmetic or make an addition order timing-dependent need
``precision_policy='practical'``.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import (accumulation_rules, elementwise_rules, graph_rules, matrix_bodies,
                                               ragged_experts_lower, segmented_attention_rules, tile_readiness,
                                               vocab_rules)
from megakernel_lab.operation_compiler.api import CapturedModule, prepare_module

NEW_OPTIONS = {
    "segmented_attention": segmented_attention_rules,
    "tile_readiness": tile_readiness,
    "vocab_rules": vocab_rules,
    "elementwise_rules": elementwise_rules,
    "graph_rules": graph_rules,
}
EXPECTED_RULES = {
    "segmented_attention": ("folded_forward", "fused_backward", "claim", "rotary"),
    "tile_readiness": ("attention",),
    "vocab_rules": ("stream", "views", "rows"),
    "elementwise_rules": ("lanecse", "bf16vec", "rolling", "divc", "indexed", "permrows", "embunroll", "rowcopy",
                          "acopy"),
    "graph_rules": ("cse", "depth_reads", "castshare"),
}
SEGMENTED_PRODUCTS = "flat+tma+rotate+claim+sepi+ktail+fit+bsearch+scatter+route+packsum+kcat"
MATRIX_BODIES = "exact+skinny+skinnybatch+shortk+splitctr"
ACCUMULATION_RULES = ("update+fsload+fsspread+rilp+fsdeepx+rinl+rleaf+rhoist2+rsum1"
                      "+rwarp+rstage+ridx+rearly+rremat+rslot+rnarrow+rmid")


# ----------------------------------------------------------------------------------------------- module API
@pytest.mark.parametrize("option", sorted(NEW_OPTIONS))
def test_rule_modules_validate_their_tokens(option):
    module = NEW_OPTIONS[option]
    assert module.RULES == EXPECTED_RULES[option] and module.VERSION == 1
    assert module.rules_of(None) == frozenset()
    assert module.rules_of("+".join(module.RULES)) == frozenset(module.RULES)
    for rule in module.RULES:
        assert module.validate(rule) == rule
        assert module.rules_of(rule) == frozenset({rule})
    for bad in ("", "bogus", f"{module.RULES[0]}+{module.RULES[0]}", f"{module.RULES[0]}+", "+", 1,
                ["".join(module.RULES)]):
        with pytest.raises(ValueError):
            module.validate(bad)
    with pytest.raises(ValueError):
        module.active("bogus")


@pytest.mark.parametrize("option", sorted(NEW_OPTIONS))
def test_rule_scopes_nest_and_reset(option):
    module = NEW_OPTIONS[option]
    first, last = module.RULES[0], module.RULES[-1]
    assert module.current() is None and not any(module.active(rule) for rule in module.RULES)
    with module.scope(first):
        assert module.current() == first and module.active(first)
        assert len(module.RULES) == 1 or not module.active(last)
        with module.scope(None):
            assert module.current() is None and not module.active(first)
        with module.scope("+".join(module.RULES)):
            assert all(module.active(rule) for rule in module.RULES)
        assert module.current() == first and module.active(first)
    assert module.current() is None and not module.active(first)
    with pytest.raises(ValueError):
        with module.scope("bogus"):
            pass
    assert module.current() is None


def test_matrix_bodies_expand_presets_inside_lists_and_check_dependencies():
    exact = {"tailrot", "taildeep", "fill", "p5c"}
    assert matrix_bodies.rules_of("exact") == exact
    assert matrix_bodies.rules_of("exact+skinny") == exact | {"skinny"}
    assert matrix_bodies.rules_of(MATRIX_BODIES) == exact | {"skinny", "skinnybatch", "shortk", "splitctr"}
    assert matrix_bodies.rules_of("splitctr") == {"splitctr"}
    for bad in ("exact+p5c", "exact+exact", "skinny+skinny", "exact+", "bogus", "", "skinnybatch", "shortk",
                "splitctr+skinnybatch"):
        with pytest.raises(ValueError):
            matrix_bodies.validate(bad)
    assert not matrix_bodies.active("splitctr")
    with matrix_bodies.scope("skinny+shortk"):
        assert matrix_bodies.active("skinny") and matrix_bodies.active("shortk")
        assert not matrix_bodies.active("skinnybatch") and not matrix_bodies.active("splitctr")
    assert not matrix_bodies.active("skinny")
    with pytest.raises(ValueError):
        matrix_bodies.active("bogus")


def test_accumulation_rules_new_names_and_unchanged_feature_string():
    accumulation_rules.validate(ACCUMULATION_RULES)
    for removed in ("rchunk", "rpref", "rhoistp"):
        assert removed not in accumulation_rules.RULES
        with pytest.raises(ValueError):
            accumulation_rules.validate("update+" + removed)
    with pytest.raises(ValueError):
        accumulation_rules.active("bogus")
    for rule in ("rwarp", "rstage", "ridx", "rearly", "rremat", "rslot", "rnarrow", "rmid"):
        assert not accumulation_rules.active(rule)
        with accumulation_rules.scope(rule):
            assert accumulation_rules.active(rule)
            assert accumulation_rules.features() == ""          # read with active(), never group-rule features
    with accumulation_rules.scope(ACCUMULATION_RULES):
        assert accumulation_rules.features() == ("+fsgather+fswarp+rowchain+fsmix+fsfence+fsload+fsspread"
                                                 "+rilp+rinl+rleaf+rhoist2+rsum1")


def test_segmented_product_token_dependencies():
    ragged_experts_lower.validate_products(SEGMENTED_PRODUCTS)
    for good in ("flat+tma+ktail", "flat+tma+claim+fit", "flat+tma+claim+bsearch", "flat+tma+claim+scatter",
                 "flat+tma+claim+bsearch+kcat", "flat+route", "flat+packsum"):
        ragged_experts_lower.validate_products(good)
    for bad in ("flat+ktail", "flat+claim+fit", "flat+tma+fit", "flat+claim+bsearch", "flat+tma+bsearch",
                "flat+tma+scatter", "flat+claim+scatter", "flat+tma+claim+kcat", "flat+tma+claim+ktail+ktail"):
        with pytest.raises(ValueError):
            ragged_experts_lower.validate_products(bad)


# ----------------------------------------------------------------------------------------------- prepare_module
def _refused(tmp_path, **options):
    """prepare_module refuses ``options`` before touching the captured program or the artifact directory."""
    directory = tmp_path / "artifact"
    with pytest.raises(ValueError) as error:
        prepare_module(None, directory, cache_dir=tmp_path / "cache", **options)
    assert not directory.exists()
    return str(error.value)


class _Stop(Exception):
    pass


def _prepared(monkeypatch, tmp_path, **options):
    """Run prepare_module up to the native build with a stub lowering; returns what lower() and build() saw."""
    lower_module = importlib.import_module("megakernel_lab.operation_compiler.lower")
    runtime_module = importlib.import_module("megakernel_lab.operation_compiler.runtime")
    seen = {}

    def fake_lower(ir, **kwargs):
        seen["scopes"] = {name: module.current() for name, module in NEW_OPTIONS.items()}
        seen["matrix_bodies"] = matrix_bodies.active_rules()
        seen["lower"] = kwargs
        return SimpleNamespace(write=lambda directory: None, manifest={})

    def fake_build(program, **kwargs):
        seen["build"] = kwargs
        raise _Stop

    monkeypatch.setattr(lower_module, "lower", fake_lower)
    monkeypatch.setattr(runtime_module, "build", fake_build)
    captured = SimpleNamespace(ir=SimpleNamespace(operations=(), metadata={}), example_args=(), constants={})
    with pytest.raises(_Stop):
        prepare_module(captured, tmp_path / "artifact", cache_dir=tmp_path / "cache", **options)
    return seen


@pytest.mark.parametrize("option", sorted(NEW_OPTIONS))
def test_prepare_module_validates_new_options(tmp_path, option):
    assert "must be None" in _refused(tmp_path, **{option: "bogus"})


@pytest.mark.parametrize("option", sorted(NEW_OPTIONS))
def test_matrix_tuning_refuses_each_new_option(tmp_path, option):
    value = NEW_OPTIONS[option].RULES[0]
    assert _refused(tmp_path, tuning=object(), **{option: value}) == f"{option} is not yet supported with matrix tuning"


@pytest.mark.parametrize("options", [
    dict(segmented_products="flat+tma+claim+bsearch+kcat"),
    dict(segmented_attention="folded_forward"),
    dict(segmented_attention="claim+fused_backward"),
    dict(vocab_rules="stream+rows"),
])
def test_practical_only_tokens_need_practical_precision(tmp_path, monkeypatch, options):
    assert "requires precision_policy='practical'" in _refused(tmp_path, **options)
    assert "requires precision_policy='practical'" in _refused(tmp_path, precision_policy="strict", **options)
    seen = _prepared(monkeypatch, tmp_path, precision_policy="practical", **options)
    assert seen["lower"]["precision_policy"] == "practical"


def test_bitwise_tokens_need_no_practical_precision(tmp_path, monkeypatch):
    seen = _prepared(monkeypatch, tmp_path, segmented_attention="claim+rotary", tile_readiness="attention",
                     vocab_rules="stream+views", elementwise_rules="+".join(elementwise_rules.RULES),
                     graph_rules="depth_reads+castshare", matrix_bodies=MATRIX_BODIES)
    assert seen["scopes"] == {"segmented_attention": "claim+rotary", "tile_readiness": "attention",
                              "vocab_rules": "stream+views", "elementwise_rules": "+".join(elementwise_rules.RULES),
                              "graph_rules": "depth_reads+castshare"}
    assert seen["matrix_bodies"] == sorted(matrix_bodies.rules_of(MATRIX_BODIES))
    # The scopes close with prepare_module.
    assert all(module.current() is None for module in NEW_OPTIONS.values())
    assert matrix_bodies.active_rules() is None


def test_lowering_sees_no_new_rules_by_default(tmp_path, monkeypatch):
    seen = _prepared(monkeypatch, tmp_path)
    assert seen["scopes"] == dict.fromkeys(NEW_OPTIONS)
    assert "device_units" not in seen["build"]


def test_graph_cse_runs_before_lowering_only_when_selected(tmp_path, monkeypatch):
    exact_cse = importlib.import_module("megakernel_lab.operation_compiler.exact_cse")
    calls = []
    monkeypatch.setattr(exact_cse, "apply", lambda graph: (calls.append(graph), (graph, {}))[1])
    _prepared(monkeypatch, tmp_path / "a", graph_rules="castshare")
    assert calls == []
    _prepared(monkeypatch, tmp_path / "b", graph_rules="cse")
    assert len(calls) == 1


def test_isolate_all_and_unit_registers_reach_the_native_build(tmp_path, monkeypatch):
    seen = _prepared(monkeypatch, tmp_path / "a", device_units=4, device_unit_inline="isolate_all",
                     device_unit_max_registers=255)
    assert seen["build"]["device_units"] == 4
    assert seen["build"]["device_unit_inline"] == "isolate_all"
    assert seen["build"]["device_unit_max_registers"] == 255
    seen = _prepared(monkeypatch, tmp_path / "b", device_units=4, device_unit_inline="isolate_all")
    assert seen["build"]["device_unit_inline"] == "isolate_all"
    assert "device_unit_max_registers" not in seen["build"]
    for registers in (64, 128, 248):
        seen = _prepared(monkeypatch, tmp_path / f"r{registers}", device_units=2, device_unit_max_registers=registers)
        assert seen["build"]["device_unit_max_registers"] == registers


@pytest.mark.parametrize("registers", [0, 32, 63, 65, 100, 254, 256, 512, True, 255.0, "255"])
def test_unit_register_limit_values(tmp_path, registers):
    assert "device_unit_max_registers must be" in _refused(tmp_path, device_units=4,
                                                           device_unit_max_registers=registers)


def test_unit_register_limit_requires_device_units(tmp_path):
    assert _refused(tmp_path, device_unit_max_registers=255) == "device_unit_max_registers requires device_units"


# ----------------------------------------------------------------------------------------------- CapturedModule.build
def test_captured_module_build_forwards_new_options_only_when_given(monkeypatch):
    api = importlib.import_module("megakernel_lab.operation_compiler.api")
    seen = {}
    monkeypatch.setattr(api, "prepare_module", lambda *args, **kwargs: seen.update(kwargs) or "receipt")
    module = CapturedModule(captured=SimpleNamespace(), input_indices={}, buffer_aliases={})
    options = dict(segmented_attention="claim", tile_readiness="attention", vocab_rules="views",
                   elementwise_rules="divc", graph_rules="cse", device_unit_max_registers=255,
                   device_units=2, device_unit_inline="isolate_all", matrix_bodies="exact+skinny",
                   segmented_products=SEGMENTED_PRODUCTS, accumulation_rules=ACCUMULATION_RULES)
    assert module.build("artifact", cache_dir="cache", **options) == "receipt"
    for key, value in options.items():
        assert seen[key] == value
    seen.clear()
    module.build("artifact", cache_dir="cache")
    assert not (set(NEW_OPTIONS) | {"device_unit_max_registers"}) & set(seen)
    assert seen["device_unit_inline"] == "wgmma"
