"""Failure evidence must survive the native runner's finite-value gate."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.tinymoe import native, train
from megakernel_lab.tinymoe.data import digest
from megakernel_lab.tinymoe.optimizer import CLIP, EMA_ALPHA, LR, NORM, stage_schedule


def run_fixture(monkeypatch, tmp_path, *, loss=2., norm=3., observe=True):
    events = []
    owners = []

    class Engine:
        def __init__(self, *, recipe, **kwargs):
            self.recipe = recipe
            self.next_step = 0
            self._identity = {"artifacts": {"plain": "frozen-test-artifact"}}
            self.params = {"weight": torch.tensor([1., 2.])}
            self.opt = {LR: torch.tensor(0.), EMA_ALPHA: torch.tensor(0.),
                        NORM: torch.tensor(norm), CLIP: torch.tensor(.5)}
            self.model_state = {"selected.0": torch.tensor([[0, 1], [1, 0]])}
            self.loss = torch.tensor(loss)
            owners.append(self)

        def evaluator(self, _model_dir):
            return object()

        def state_dict(self, *, host):
            assert host
            events.append(("state", self.next_step))
            clone = lambda values: {key: value.clone() for key, value in values.items()}
            return dict(schema="latch-tinymoe-native-state-v1", recipe=asdict(self.recipe),
                identity=self._identity, next_step=self.next_step, parameters=clone(self.params),
                optimizer_state=clone(self.opt), model_state=clone(self.model_state))

        def stage(self, x, y, index):
            assert index == self.next_step == 0
            assert torch.equal(x, torch.tensor([[1, 2]]))
            assert torch.equal(y, torch.tensor([[2, 3]]))
            events.append(("stage", index))
            stage_schedule(self.opt, self.recipe, index)

        def step(self):
            events.append(("step", self.next_step))
            self.params["weight"].add_(1)
            self.next_step += 1
            return dict(loss=self.loss, norm=self.opt[NORM], clip=self.opt[CLIP],
                        learning_rate=self.recipe.lr(0), supervision_weight=0.)

        def snapshot(self, *, host):
            events.append(("snapshot", self.next_step))
            result = self.state_dict(host=host)
            result["diagnostics"] = dict(loss=self.loss.clone(),
                gradient_norm=self.opt[NORM].clone(), clip_coefficient=self.opt[CLIP].clone())
            result["gradients_before_clip"] = {"weight": torch.tensor([4., 5.])}
            return result

        def execution_summary(self):
            return {"test_fixture": True}

    class Batches:
        def __init__(self, *_args):
            pass

        def batch(self, index, device):
            assert index == 0 and device == "cpu"
            return torch.tensor([[1, 2]]), torch.tensor([[2, 3]])

    monkeypatch.setattr(native, "Native", Engine)
    monkeypatch.setattr(train, "TokenBatches", Batches)
    monkeypatch.setattr(train.conventional, "configure_execution", lambda: {"test_fixture": True})
    monkeypatch.setattr(train.conventional, "synchronize", lambda _device: events.append(("sync",)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "CPU fixture")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}\n")
    args = SimpleNamespace(threads=1, out=str(tmp_path / "run"), recipe="unchanged",
        horizon=400, learning_rate=1e-4, weight_decay=.01, plain_dir="frozen",
        auxiliary_dir=None, device="cpu", data_dir=str(data), batch_size=1,
        sequence_length=2, eval_windows=1, resume=None, model_dir="unused",
        eval_every=0, updates=1, observe_at=[1] if observe else [],
        checkpoint_every=0, checkpoint_at=[], save_final=False, timing_warmup=0)
    return args, events, owners


@pytest.mark.parametrize("loss,norm", [(float("nan"), 3.), (2., float("inf")), (2., 3.)])
def test_completed_observation_survives_nonfinite_rejection(monkeypatch, tmp_path, loss, norm):
    args, events, owners = run_fixture(monkeypatch, tmp_path, loss=loss, norm=norm)
    failing = not bool(torch.isfinite(torch.tensor([loss, norm])).all())
    if failing:
        with pytest.raises(FloatingPointError, match="nonfinite generated update: 1"):
            train.native_train(args)
    else:
        train.native_train(args)

    out = tmp_path / "run"
    path = out / "observation-000001.pt"
    receipt = json.loads(path.with_suffix(".json").read_text())
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["sha256"] == digest(path)
    assert receipt["completed_update"] == 1
    observed = torch.load(path, map_location="cpu", weights_only=True)
    assert observed["before"]["next_step"] == 0
    assert observed["after"]["next_step"] == owners[0].next_step == 1
    assert torch.equal(observed["before"]["parameters"]["weight"], torch.tensor([1., 2.]))
    assert torch.equal(observed["after"]["parameters"]["weight"], torch.tensor([2., 3.]))
    assert torch.equal(observed["gradients"]["weight"], torch.tensor([4., 5.]))
    assert torch.equal(observed["fixed_routes"], torch.tensor([[[0, 1], [1, 0]]]))
    assert observed["after"]["optimizer_state"][LR].item() == pytest.approx(owners[0].recipe.lr(0))
    assert observed["after"]["optimizer_state"][EMA_ALPHA].item() == 0.
    assert observed["identity"] == receipt["identity"] == owners[0]._identity
    assert bool(torch.isfinite(observed["loss"])) == bool(torch.isfinite(torch.tensor(loss)))
    assert bool(torch.isfinite(observed["after"]["diagnostics"]["gradient_norm"])) == bool(torch.isfinite(torch.tensor(norm)))
    assert events == [("sync",), ("state", 0), ("stage", 0), ("sync",),
                      ("step", 0), ("sync",), ("snapshot", 1), ("state", 1)]
    rows = [json.loads(line) for line in (out / "updates.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["completed_update"] == 1
    assert (out / "result.json").exists() == (not failing)


def test_nonfinite_unobserved_update_retains_row_and_still_raises(monkeypatch, tmp_path):
    args, events, owners = run_fixture(monkeypatch, tmp_path, norm=float("inf"), observe=False)
    with pytest.raises(FloatingPointError, match="nonfinite generated update: 1"):
        train.native_train(args)
    out = tmp_path / "run"
    assert json.loads((out / "updates.jsonl").read_text())["completed_update"] == 1
    assert not list(out.glob("observation-*"))
    assert not (out / "result.json").exists()
    assert owners[0].next_step == 1
    assert events == [("sync",), ("stage", 0), ("sync",), ("step", 0), ("sync",)]
