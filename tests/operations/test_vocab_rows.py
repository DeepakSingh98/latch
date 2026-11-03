"""Fused vocabulary statistics + logits gradient (ops/vocab_rows.py, vocab_rules ``rows``): emission-time terms,
the one-row-pass fusion record, refusals and the practical-precision requirement (CPU; no native build)."""
import pytest

from megakernel_lab.operation_compiler import vocab_rows, vocab_rules
from megakernel_lab.operation_compiler.api import prepare_module
from megakernel_lab.operation_compiler.lower import TensorLayout

ROWS, PADDED, VOCAB = 512, 2048, 2000
COMMENT = "  // Fused vocab: statistics + logits gradient, one staged row per CTA (ops/vocab_rows.py).\n"


def _values(padded=PADDED):
    layouts = {"logits": TensorLayout((ROWS, padded), (padded, 1), "logits"),
               "stats": TensorLayout((ROWS, 2), (2, 1), "stats"),
               "target": TensorLayout((ROWS, 1), (1, 1), "target"),
               "cot": TensorLayout((ROWS, 1), (1, 1), "cot"),
               "dlogits": TensorLayout((ROWS, padded), (padded, 1), "dlogits")}
    dtypes = {"logits": "bfloat16", "stats": "float32", "target": "int64", "cot": "float32", "dlogits": "bfloat16"}
    return layouts, dtypes


def _annotate(padded=PADDED, kind="vocab_gradient"):
    layouts, dtypes = _values(padded)
    extra = {}
    args = [{"ref": "logits"}, {"ref": "stats"}, {"ref": "target"}, {"ref": "cot"}]
    vocab_rows.annotate(kind, "latch_ops.vocab_grad.default", "dlogits", args,
                        {"vocabulary": VOCAB, "chain": []}, None, set(layouts), layouts, dtypes, extra)
    return extra


def test_annotate_is_off_by_default():
    assert not vocab_rows.enabled()
    assert _annotate() == {}
    with vocab_rules.scope("stream+views"):
        assert not vocab_rows.enabled()
        assert _annotate() == {}


def test_annotate_records_the_gradient_terms():
    with vocab_rules.scope("rows"):
        assert vocab_rows.enabled()
        terms = _annotate()["vocab_rows_terms"]
        assert _annotate(kind="vocab_rows") == {}
        assert _annotate(padded=120000) == {}        # one staged row must fit the dynamic shared budget
    assert {key: terms[key] for key in ("product", "stats", "rows", "vp", "vocab")} == dict(
        product="logits", stats="stats", rows=ROWS, vp=PADDED, vocab=VOCAB)
    code = terms["code"]
    assert code.startswith("{\n" + COMMENT)
    assert f"const __nv_bfloat16* src_ = v_logits + (long long)blockIdx.x * {PADDED}LL;" in code
    assert "if (threadIdx.x == 0) { v_stats[2 * row] = maximum; v_stats[2 * row + 1] = logsum; }" in code
    assert f"assert(t >= 0 && t < {VOCAB}LL);" in code
    assert f"__nv_bfloat16* dst_ = v_dlogits + row * {PADDED}LL;" in code


def _phases(terms, cot_reads=("loss_scale",)):
    return [
        {"kind": "gemm", "names": ["logits"], "writes": ["logits"], "reads": ["x", "w"], "targets": ["mm"]},
        {"kind": "vocab_rows", "names": ["stats"], "writes": ["stats"], "reads": ["logits"],
         "reduction_elements": VOCAB, "targets": ["latch_ops.vocab_stats.default"]},
        {"kind": "vocab_pick", "names": ["picked"], "writes": ["picked"], "reads": ["logits", "stats", "target"]},
        {"kind": "elementwise", "names": ["cot"], "writes": ["cot"], "reads": list(cot_reads)},
        {"kind": "vocab_gradient", "names": ["dlogits"], "writes": ["dlogits"],
         "reads": ["logits", "stats", "target", "cot"], "targets": ["latch_ops.vocab_grad.default"],
         "vocab_rows_terms": dict(terms)},
        {"kind": "gemm", "names": ["dw"], "writes": ["dw"], "reads": ["dlogits", "x"]},
    ]


def test_fuse_pass_replaces_statistics_and_gradient_with_one_row_pass():
    with vocab_rules.scope("rows"):
        terms = _annotate()["vocab_rows_terms"]
    layouts, _ = _values()
    phases = _phases(terms)
    assert vocab_rows.fuse_pass(phases, layouts) == ["dlogits"]
    assert [p["names"] for p in phases] == [["logits"], ["cot"], ["stats", "dlogits"], ["picked"], ["dw"]]
    fused = phases[2]
    assert fused["kind"] == "vocab_rows_gradient" and fused["engine"] == "vocab_rows_staged"
    assert fused["writes"] == ["stats", "dlogits"] and fused["reads"] == ["logits", "target", "cot"]
    assert fused["targets"] == ["latch_ops.vocab_stats.default", "latch_ops.vocab_grad.default"]
    assert fused["code"] == terms["code"] and fused["code"].startswith("{\n" + COMMENT)
    assert fused["shared_bytes"] == vocab_rows.shared_bytes(PADDED) == 256 + 2 * PADDED
    assert fused["threads"] == vocab_rows.THREADS and fused["reduction_elements"] == VOCAB
    record = fused["vocab_rows_fused"]
    assert record["rows"] == ROWS and record["padded"] == PADDED and record["vocabulary"] == VOCAB
    assert record["logits_reads_per_row"] == 1 and record["hoisted"] == [["cot"]]
    assert not any("vocab_rows_terms" in p for p in phases)


def test_fuse_pass_refuses_operands_that_depend_on_the_statistics():
    with vocab_rules.scope("rows"):
        terms = _annotate()["vocab_rows_terms"]
    layouts, _ = _values()
    phases = _phases(terms, cot_reads=("stats",))
    expected = [{k: v for k, v in p.items() if k != "vocab_rows_terms"} for p in phases]
    assert vocab_rows.fuse_pass(phases, layouts) == []
    assert phases == expected


def test_fuse_pass_without_terms_is_identity():
    phases = _phases({})
    del phases[4]["vocab_rows_terms"]
    expected = [dict(p) for p in phases]
    assert vocab_rows.fuse_pass(phases, _values()[0]) == []
    assert phases == expected


@pytest.mark.parametrize("precision", [None, "strict"])
def test_rows_requires_practical_precision(tmp_path, precision):
    options = {} if precision is None else {"precision_policy": precision}
    with pytest.raises(ValueError, match="requires precision_policy='practical'"):
        prepare_module(None, tmp_path / "artifact", cache_dir=tmp_path / "cache", vocab_rules="stream+rows",
                       **options)
    assert not (tmp_path / "artifact").exists()
