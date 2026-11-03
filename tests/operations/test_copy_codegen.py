"""Generated row-permutation and generic cat/slice_backward copy forms (CPU lowering checks).

``elementwise_rules`` 'rowcopy' selects the 16-byte row form of ``permutation_gather`` (warp per rows for rows of at
most 128 vectors) and 'acopy' the assembled 16-byte form of cat/slice_backward copies. Without the tokens the release
scalar forms are emitted. These tests check admission, fallbacks, the tokens and that only the copy bodies change
(phases, intervals and launch estimates are identical).
"""
from __future__ import annotations

import re

import pytest
import torch

from megakernel_lab.operation_compiler import SGDConfig, capture_module, elementwise_rules, extra_ops
from megakernel_lab.operation_compiler.ir import Ref
from megakernel_lab.operation_compiler.lower import TensorLayout, lower
from megakernel_lab.operation_compiler.row_permutation import permutation_gather

COPY_KINDS = {"indexed_permutation", "cat", "slice_backward"}
ROWS = dict(wide=1031, view=67, narrow=13)
HEADS = (37, 3)


class CopyModel(torch.nn.Module):
    """Row permutations of 1152-wide BF16 rows (144 vectors: grid-stride row form), a 16-byte-offset FP32 view (96
    vectors: warp-per-rows form), a 4-byte-offset view and 44-byte BF16 rows (both keep the scalar loop), the
    rotary-style cat(-x2, x1) of 22-wide heads, a BF16 22 -> 24 pad cat and an FP32 22 -> 24 slice_backward."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(271028)
        p = lambda *shape: torch.nn.Parameter(torch.randn(*shape, generator=g))
        n, h = HEADS
        self.x1, self.w1 = p(ROWS["wide"], 1152), p(ROWS["wide"], 1152)
        self.x2, self.w2, self.w2b = p(ROWS["view"], 392), p(ROWS["view"], 384), p(ROWS["view"], 384)
        self.x3, self.w3 = p(ROWS["narrow"], 22), p(ROWS["narrow"], 22)
        self.a, self.wr, self.wp = p(n, h, 22), p(n, h, 22), p(n, h, 24)
        self.t24, self.ws = p(n, h, 24), p(n, h, 22)

    def forward(self, o1, i1, o2, i2, o3, i3):
        n, h = HEADS
        y1 = permutation_gather(self.x1.bfloat16(), o1, i1)
        y2 = permutation_gather(self.x2[:, 4:388], o2, i2)           # 16-byte offset view: row vectors
        y2b = permutation_gather(self.x2[:, 1:385], o2, i2)          # 4-byte offset view: scalar loop
        y3 = permutation_gather(self.x3.bfloat16(), o3, i3)
        rot = torch.cat((-self.a[..., 11:], self.a[..., :11]), -1)   # halves of 22-wide heads
        ab = self.a.bfloat16()
        pad = torch.cat((ab, ab.new_zeros(n, h, 2)), -1)             # BF16 22 -> 24 pad
        sl = self.t24[..., :22]                                      # backward: FP32 22 -> 24 slice_backward
        return ((y1.float() * self.w1).sum() + (y2 * self.w2).sum() + (y2b * self.w2b).sum()
                + (y3.float() * self.w3).sum() + (rot * self.wr).sum() + (pad.float() * self.wp).sum()
                + (sl * self.ws).sum())


def plans(step=0):
    g = torch.Generator().manual_seed(1000 + step)
    out = []
    for rows in ROWS.values():
        order = torch.randperm(rows, generator=g)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(rows)
        out += [order, inverse]
    return tuple(out)


def _program(rules="rowcopy+acopy", floor=0):
    # The small fixture's copies are below the production size floor; floor=0 admits them here.
    saved = extra_ops.ASSEMBLED_MIN_ELEMENTS
    extra_ops.ASSEMBLED_MIN_ELEMENTS = floor
    try:
        captured = capture_module(CopyModel(), lambda y: y, plans(), optimizer=SGDConfig(lr=0.0))
        with elementwise_rules.scope(rules):
            return lower(captured.ir, execution="compose")
    finally:
        extra_ops.ASSEMBLED_MIN_ELEMENTS = saved


def _copies(program):
    return [p for p in program.manifest["phases"] if p["kind"] in COPY_KINDS]


@pytest.fixture(scope="module")
def default_program():
    return _program()


def test_permutation_row_vectors_only_for_aligned_rows(default_program):
    perms = [p for p in default_program.manifest["phases"] if p["kind"] == "indexed_permutation"]
    by_vector = sorted(bool(p.get("row_vector")) for p in perms)
    # Forward + VJP of the BF16 1152 rows and of the 16-byte-offset FP32 view are row vectors; the 4-byte-offset
    # view and the 44-byte BF16 rows keep the scalar loop (forward and VJP).
    assert by_vector.count(True) >= 3 and by_vector.count(False) >= 3, by_vector
    for p in perms:
        assert p["engine"] == "validated_bijective_row_copy" and p["requires_full_grid"]
        assert p["validation"] == ("once_per_row" if p.get("row_vector") else "per_element")


def test_row_vector_bodies_check_each_row_once_and_keep_scalar_fallback(default_program):
    source = default_program.source
    # rows of more than 128 vectors: grid-stride 16-byte form, one inverse load per row (c_ == 0 thread)
    assert source.count("pgq_[s_]") >= 2
    assert "c_ == 0u ? pgq_[s_] : (int64_t)r_" in source
    assert "if (s_ < 0 || s_ >= " in source                   # range check before the row load
    # rows of at most 128 vectors: one warp per 16 rows, deferred bijection trap
    assert "// Wide-issue row copy: warp per 16 rows" in source
    assert "mlp_bad_ |= pgq_[sl_] != r0_ + mlp_lane_;" in source
    # Run-time misalignment falls back to the original per-element loop inside the same phase.
    rows = source.count("pgq_[s_]") + source.count("pgq_[sl_]")
    assert source.count("if (source < 0 || source >= ") >= rows


def test_without_tokens_release_bodies_with_identical_schedule(default_program):
    old = _program(rules=None)
    assert not any(p.get("row_vector") or p.get("assembled_copy") for p in old.manifest["phases"])
    strip = lambda m: [(p["kind"], p["names"], p.get("work_items"), p.get("threads"), p.get("shared_bytes"))
                       for p in m["phases"]]
    assert strip(old.manifest) == strip(default_program.manifest)
    assert old.manifest["barrier_intervals"] == default_program.manifest["barrier_intervals"]
    assert old.manifest["arena_words"] == default_program.manifest["arena_words"]
    for text in ("pgq_[s_]", "pgq_[sl_]", "ao_[av_", "Wide-issue"):
        assert text not in old.source


def test_each_token_selects_only_its_form():
    rows_only = _program(rules="rowcopy")
    assert any(p.get("row_vector") for p in rows_only.manifest["phases"])
    assert not any(p.get("assembled_copy") for p in rows_only.manifest["phases"])
    copies_only = _program(rules="acopy")
    assert not any(p.get("row_vector") for p in copies_only.manifest["phases"])
    assert any(p.get("assembled_copy") for p in copies_only.manifest["phases"])


def test_pad_and_slice_backward_copies_are_assembled(default_program):
    copies = _copies(default_program)
    assembled = [p for p in copies if p.get("assembled_copy")]
    assert assembled, [(p["kind"], p.get("vector_copy")) for p in copies]
    assert {p["kind"] for p in assembled} <= {"cat", "slice_backward"}
    assert not any(p.get("vector_copy") and p.get("assembled_copy") for p in copies)


def _layouts():
    L = lambda shape, strides, root, offset=0: TensorLayout(tuple(shape), tuple(strides), root, offset)
    layouts = dict(neg=L((6, 5, 11), (55, 11, 1), "neg"), sl=L((6, 5, 11), (120, 24, 1), "q", 12),
                   rot=L((6, 5, 22), (110, 22, 1), "rot"), x22=L((6, 5, 22), (110, 22, 1), "x22"),
                   z2=L((6, 5, 2), (10, 2, 1), "z2"), pad=L((6, 5, 24), (120, 24, 1), "pad"),
                   g=L((6, 5, 11), (110, 22, 1), "gg", 11), sb=L((6, 5, 22), (110, 22, 1), "sb"),
                   a32=L((8, 64), (64, 1), "a32"), b32=L((8, 64), (64, 1), "b32"), c=L((8, 128), (128, 1), "c"))
    dtypes = dict(neg="float32", sl="float32", rot="float32", x22="bfloat16", z2="bfloat16", pad="bfloat16",
                  g="float32", sb="float32", a32="bfloat16", b32="bfloat16", c="bfloat16")
    return layouts, dtypes, set(layouts)


@pytest.mark.parametrize("base,args,name,shape,pieces", [
    ("cat", [[Ref("neg"), Ref("sl")], -1], "rot", (6, 5, 22), ("v_neg", "v_sl")),
    ("cat", [[Ref("x22"), Ref("z2")], -1], "pad", (6, 5, 24), ("v_x22", "v_z2")),
    ("slice_backward", [Ref("g"), [6, 5, 22], 2, 11, 22, 1], "sb", (6, 5, 22), ("v_g",)),
])
def test_assembled_form_reads_raw_bits_of_the_same_sources(base, args, name, shape, pieces):
    layouts, dtypes, names = _layouts()
    assert extra_ops._vector_copy(base, args, {}, name, shape, layouts, dtypes, names) is None
    code = extra_ops._assembled_copy(base, args, {}, name, shape, layouts, dtypes, names)
    assert code is not None
    raw = "unsigned" if dtypes[name] == "float32" else "unsigned short"
    for p in pieces:
        assert f"reinterpret_cast<const {raw}*>({p})" in code
    assert "float" not in code and "bfloat16" not in code        # integer moves only: bits preserved
    assert code.count("ao_[av_") == (2 if dtypes[name] == "float32" else 1)
    # The source offsets are the merged row-major strides of each piece (sliced half: 120 = 5 * 24).
    if name == "rot":
        assert "o_ * 11u" in code and "o_ * 24u + (c_ - 11u)" in code


def test_aligned_runs_keep_the_vector_form_and_misaligned_results_refuse():
    layouts, dtypes, names = _layouts()
    args = [[Ref("a32"), Ref("b32")], -1]
    assert extra_ops._vector_copy("cat", args, {}, "c", (8, 128), layouts, dtypes, names) is not None
    with elementwise_rules.scope("acopy"):
        kind, code, extra, _ = extra_ops.emit("cat.default", args, {}, "c", (8, 128), layouts, dtypes, names, None)
    assert extra.get("vector_copy") and not extra.get("assembled_copy")
    layouts["pad"] = TensorLayout((6, 5, 24), (120, 24, 1), "pad", 3)   # result offset not on a 16-byte boundary
    assert extra_ops._assembled_copy("cat", [[Ref("x22"), Ref("z2")], -1], {}, "pad", (6, 5, 24),
                                     layouts, dtypes, names) is None


def test_small_results_keep_the_scalar_form_under_the_size_floor():
    assert extra_ops.ASSEMBLED_MIN_ELEMENTS == 1 << 20
    program = _program(floor=extra_ops.ASSEMBLED_MIN_ELEMENTS)
    assert not any(p.get("assembled_copy") for p in program.manifest["phases"])
    layouts, dtypes, names = _layouts()
    with elementwise_rules.scope("acopy"):
        kind, code, extra, _ = extra_ops.emit("cat.default", [[Ref("neg"), Ref("sl")], -1], {}, "rot", (6, 5, 22),
                                              layouts, dtypes, names, None)
    assert kind == "cat" and extra == {} and "ao_[av_" not in code


def test_assembled_emit_guards_result_alignment_with_scalar_fallback(monkeypatch):
    monkeypatch.setattr(extra_ops, "ASSEMBLED_MIN_ELEMENTS", 0)
    layouts, dtypes, names = _layouts()
    args = [[Ref("neg"), Ref("sl")], -1]
    kind, plain, extra, _ = extra_ops.emit("cat.default", args, {}, "rot", (6, 5, 22), layouts, dtypes, names, None)
    assert kind == "cat" and extra == {} and "ao_[av_" not in plain          # token absent: release scalar form
    with elementwise_rules.scope("acopy"):
        kind, code, extra, _ = extra_ops.emit("cat.default", args, {}, "rot", (6, 5, 22), layouts, dtypes, names,
                                              None)
    assert kind == "cat" and extra == {"assembled_copy": True}
    assert re.match(r"if \(\(\(reinterpret_cast<uintptr_t>\(v_rot\) & 15u\) == 0\)\) \{", code)
    assert "for (long long i = " in code and plain in code       # the original scalar loop is the else branch
