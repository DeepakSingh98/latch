"""Exact generated-source selection while scanning each interval only once."""
import importlib
from pathlib import Path
import random
import re
import time

import pytest

lower = importlib.import_module("megakernel_lab.operation_compiler.lower")


def reference(named, code):
    return [(name, line) for name, line in named if re.search(rf"\b{name}\b", code)]


@pytest.mark.parametrize("code", [
    "v_x[0] + v_xx[1]", "av_x + v_x_suffix + v_v_x", "/* v_x */ + 1",
    '#define USE v_x\n#define JOIN(x) x ## v_xx', '"v_x" + \'v_xx\'',
    "v_x. v_xx:: v__y[0]", "é_v_x + v_xé", "_v_x / v_x; // v_missing",
    "v_xx[0] + v_xx[1] + v__y[2]", "", "v_x\nv_xx\tv__y",
])
def test_whole_identifier_boundaries_comments_and_macros(code):
    named = [(name, name + " = declaration;") for name in ("v_xx", "v_x", "v__y", "v_xé", "v_missing")]
    assert lower._used_tensor_declarations(named, code) == reference(named, code)


def test_many_declarations_match_donor_selection_and_preserve_order():
    generator = random.Random(731)
    named = [(f"v_tensor{i}", f"float* v_tensor{i} = args[{i}];") for i in range(4096)]
    generator.shuffle(named)
    selected = generator.sample(named, 123)
    code = "\n".join(f"{name}[i] += {name}[j]; // {name}r must not match" for name, _ in selected)
    began = time.perf_counter()
    expected = reference(named, code)
    reference_seconds = time.perf_counter() - began
    began = time.perf_counter()
    actual = lower._used_tensor_declarations(named, code)
    scan_seconds = time.perf_counter() - began
    assert actual == expected and len(actual) == len(selected)
    print(dict(declarations=len(named), reference_seconds=reference_seconds, scan_seconds=scan_seconds), flush=True)


def test_complete_lowered_source_is_identical_to_individual_searches(monkeypatch):
    spec = importlib.util.spec_from_file_location("square_sum_fixture",
        Path(__file__).with_name("test_square_sum_association.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    graph = fixture.long_fixture()
    graph["outputs"] = [graph["outputs"]["result"]]
    for op in graph["operations"]:
        op["name"] = op["outputs"][0]
    actual = lower.lower(graph, execution="compose", precision_policy="practical", compose_selection="common")
    calls = []
    def checked_reference(named, code):
        calls.append(len(named))
        return reference(named, code)
    monkeypatch.setattr(lower, "_used_tensor_declarations", checked_reference)
    expected = lower.lower(graph, execution="compose", precision_policy="practical", compose_selection="common")
    assert calls
    assert actual.source == expected.source
    assert actual.manifest == expected.manifest
