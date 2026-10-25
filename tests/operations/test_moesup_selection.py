"""Program-scoped batched TMA admission and nearby fallback contracts (cloud CPU)."""
import functools
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import compose
from megakernel_lab.operation_compiler.matrix_adapters import dense, select
from megakernel_lab.operation_compiler.matrix_adapters.bodies import BATCH_TMA_PROGRAM, _stacked


def _graph(count=2, rows=4096, padded=65536, **changes):
    head = {"dtype": "bfloat16", "backward": True, "rows": rows, "padded": padded, **changes}
    return {"metadata": {"vocabulary_head": {"rewritten": [dict(head) for _ in range(count)]}}}


@pytest.mark.parametrize("graph,enabled,large", [
    (_graph(), True, 2), (_graph(count=1), True, 1), (_graph(count=0), True, 0),
    (_graph(rows=4095), True, 0), (_graph(dtype="float32"), True, 0),
    (_graph(backward=False), True, 0), ({}, True, 0)])
def test_batch_policy_uses_readout_structure(graph, enabled, large):
    record = compose.batch_tma_policy(graph)
    assert record["batch_stacked_tma"] is enabled and record["large_training_readouts"] == large
    assert compose.batch_tma_policy(SimpleNamespace(metadata=graph.get("metadata", {}))) == record


def test_batch_tma_scope_preserves_nonstackable_products():
    stacked = dense(256, 1024, 1024, batch=16)
    transposed = dense(1024, 1024, 256, batch=16, ak=False)
    # Accepted c24 admits the M-major expert weight-gradient operand as well.
    assert _stacked(stacked) and _stacked(transposed)
    assert select(stacked)[0].body == "pops4b_tma"
    assert select(transposed)[0].body == "pops4b_tma"
    token = BATCH_TMA_PROGRAM.set(False)
    try:
        assert not _stacked(stacked)
        assert select(stacked)[0].body == "pops4b_wg"
        assert select(transposed)[0].body == "pops4b_wg"
    finally:
        BATCH_TMA_PROGRAM.reset(token)
    assert _stacked(stacked)


@pytest.mark.parametrize("raises", [False, True])
def test_composed_planner_restores_batch_policy_on_every_exit(monkeypatch, raises):
    from megakernel_lab.operation_compiler import lower as lower_module
    # Keep the lowering signature so the normal option validation still runs.
    original = lower_module._lower_serial

    @functools.wraps(original)
    def inspect_scope(graph, **options):
        assert BATCH_TMA_PROGRAM.get() is True
        if raises:
            raise RuntimeError("deliberate lowering failure")
        return SimpleNamespace(manifest={})

    monkeypatch.setattr(lower_module, "_lower_serial", inspect_scope)
    assert BATCH_TMA_PROGRAM.get() is True
    if raises:
        with pytest.raises(RuntimeError, match="deliberate lowering failure"):
            compose.lower_composed_flat(_graph())
    else:
        program, _ = compose.lower_composed_flat(_graph())
        assert program.manifest["batch_tma_selection"]["batch_stacked_tma"] is True
    assert BATCH_TMA_PROGRAM.get() is True
