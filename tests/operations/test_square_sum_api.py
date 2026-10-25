"""Public permission and forwarding boundaries for optional square-sum association."""
import pytest
from megakernel_lab.operation_compiler import api


def test_module_build_forwards_explicit_square_sum_permission(monkeypatch):
    recorded = {}
    def prepare(captured, directory, **kwargs):
        recorded.update(kwargs)
        return "forwarded"
    monkeypatch.setattr(api, "prepare_module", prepare)
    captured = api.CapturedModule(None, {}, {})
    assert captured.build("unused", cache_dir="unused", precision_policy="practical",
                          square_sum_association="balanced") == "forwarded"
    assert recorded["precision_policy"] == "practical"
    assert recorded["square_sum_association"] == "balanced"


@pytest.mark.parametrize("precision,option,tuning", [
    (None, "balanced", None), ("strict", "balanced", None),
    ("practical", "unknown", None), ("practical", "balanced", object()),
])
def test_invalid_square_sum_permission_fails_before_artifact_write(tmp_path, precision, option, tuning):
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        api.prepare_module(None, out, cache_dir=tmp_path / "cache", precision_policy=precision,
                           square_sum_association=option, tuning=tuning)
    assert not out.exists()
