"""P6D row attachment cannot create cross-domain same-phase dependencies."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import fusion_p6d, p6d_rules
from megakernel_lab.operation_compiler.lower import TensorLayout


def _fixture(*, commit=True, consumer_phase=0, chain=False):
    shape = (2, 96)
    layout = TensorLayout(shape, (96, 1), "g", 0)
    row_layout = TensorLayout((2, 1), (1, 0), "r", 0)
    def node(name, kind="elementwise", dtype="float32"):
        return SimpleNamespace(name=name, kind=kind, dtype=dtype,
            shape=(2, 1) if kind == "reduction" else shape,
            extra={"engine": "p6d_exact_row_per_warp", "reduction_elements": 96}
                  if kind == "reduction" else {})
    items = [node("r", "reduction"), node("g"), node("u")]
    if chain:
        items.insert(2, node("h"))
    nodes = {n.name: n for n in items}
    loads = {"r": [], "g": [("r", row_layout, shape, "i")],
             "u": [("g", layout, shape, "i")]}
    if chain:
        loads["h"] = [("g", layout, shape, "i")]
    phases = [{"heavy": ["r"], "loops": {192: ["g"] + (["h"] if chain else [])}},
              {"heavy": [], "loops": {}}]
    phases[consumer_phase]["loops"].setdefault(192, []).append("u")
    phase_of = {n.name: (consumer_phase if n.name == "u" else 0) for n in items}
    layouts = {n.name: row_layout if n.name == "r" else layout for n in items}
    return dict(items=items, loads_of=loads, phase_of=phase_of, phases=phases,
                nodes=nodes, layouts=layouts, in_place={"state": "u"} if commit else {},
                inplace_source={"u"} if commit else set(), numel=math.prod,
                index32=lambda lay, shape, idx: "i/96u" if lay.root == "r" else "i")


def test_same_phase_commit_keeps_its_gradient_in_the_global_loop():
    f = _fixture()
    attached, owner = p6d_rules.attach_row_consumers(**f)
    assert not attached and not owner
    assert f["phases"][0]["loops"][192] == ["g", "u"]


def test_rejected_producer_also_keeps_dependent_members_in_the_global_loop():
    f = _fixture(chain=True)
    attached, owner = p6d_rules.attach_row_consumers(**f)
    assert not attached and not owner
    assert f["phases"][0]["loops"][192] == ["g", "h", "u"]


def test_complete_same_phase_consumer_chain_can_still_attach():
    f = _fixture(commit=False)
    attached, owner = p6d_rules.attach_row_consumers(**f)
    assert attached["r"]["elements"] == ["g", "u"]
    assert owner == {"g": "r", "u": "r"}


def test_later_commit_reads_attached_gradient_after_the_grid_barrier():
    f = _fixture(consumer_phase=1)
    attached, owner = p6d_rules.attach_row_consumers(**f)
    assert attached["r"]["elements"] == ["g"] and owner == {"g": "r"}
    assert f["phases"][1]["loops"][192] == ["u"]


def test_independent_schedule_check_rejects_equal_size_different_loop_domains():
    f = _fixture()
    with pytest.raises(ValueError, match="u .* reads g produced in phase 0"):
        fusion_p6d._check_schedule(f["items"], f["loads_of"], f["phase_of"],
            lambda *args: True, {"g", "u"}, {}, f["nodes"], ValueError,
            epilogue_of={"g": "r"})


def test_independent_schedule_check_keeps_same_row_domain_register_reads():
    f = _fixture(commit=False)
    fusion_p6d._check_schedule(f["items"], f["loads_of"], f["phase_of"],
        lambda *args: True, {"g", "u"}, {}, f["nodes"], ValueError,
        epilogue_of={"g": "r", "u": "r"})
