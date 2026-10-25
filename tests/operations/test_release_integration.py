"""Contracts at the boundaries of independently qualified release features."""
from pathlib import Path

import pytest
import torch

from megakernel_lab.operation_compiler import CaptureError, DynamicBatch, SGDConfig, capture_module, tensor_update
from megakernel_lab.operation_compiler.lower import LoweringError, lower


class FrozenDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 4)
        self.frozen.requires_grad_(False)
        self.dropout = torch.nn.Dropout(0.25)
        self.output = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.output(self.dropout(self.frozen(x).tanh()))


def mse(prediction, target):
    return (prediction - target).square().mean()


def test_frozen_dropout_and_tensor_optimizer_compose():
    def update(parameters, gradients, state, hyperparameters):
        return {name: value - 0.01 * gradients[name] for name, value in parameters.items()}, state

    captured = capture_module(FrozenDropout(), mse, (torch.ones(2, 3),),
                              (torch.zeros(2, 2),), optimizer=tensor_update(update),
                              dropout_seed=73)
    program = lower(captured.ir, execution="compose")
    api = captured.ir.metadata["module_api"]
    assert set(api["input_indices"]["frozen"]) == {"frozen.weight", "frozen.bias"}
    assert "dropout" in api
    assert "dropout_rng" in captured.ir.metadata["state_names"]
    assert program.manifest["execution_selection"]["requested"] == "compose"


def test_dynamic_batch_preserves_declared_dropout_and_frozen_scope():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.25),
                                torch.nn.Linear(4, 2))
    with pytest.raises(CaptureError, match="explicit Philox dropout"):
        capture_module(model, mse, (torch.ones(4, 3),), (torch.zeros(4, 2),),
                       optimizer=SGDConfig(), dropout_seed=17,
                       dynamic_batch=DynamicBatch(maximum=4))
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    with pytest.raises(CaptureError, match="frozen parameters"):
        capture_module(model, mse, (torch.ones(4, 3),), (torch.zeros(4, 2),),
                       dynamic_batch=DynamicBatch(maximum=4))


def test_dynamic_batch_refuses_user_optimizer():
    def update(parameters, gradients, state, hyperparameters):
        return {name: value - 0.01 * gradients[name] for name, value in parameters.items()}, state

    with pytest.raises(CaptureError, match="momentum SGD only"):
        capture_module(torch.nn.Linear(3, 2), mse, (torch.ones(4, 3),),
                       (torch.zeros(4, 2),), optimizer=tensor_update(update),
                       dynamic_batch=DynamicBatch(maximum=4))


def test_conditional_refuses_ordinary_dropout():
    class GatedDropout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.dropout = torch.nn.Dropout(.25)

        def forward(self, x, flag):
            h = self.linear(x)
            return torch.cond(flag, lambda value: self.dropout(value),
                              lambda value: value * .5, (h,))

    with pytest.raises(CaptureError, match="ordinary nn.Dropout inside torch.cond"):
        capture_module(GatedDropout(), mse, (torch.ones(2, 3), torch.tensor(True)),
                       (torch.zeros(2, 2),), dropout_seed=17)


def test_deferred_gradients_are_explicit_and_reject_flat_programs():
    captured = capture_module(torch.nn.Linear(3, 2), mse,
                              (torch.ones(2, 3),), (torch.zeros(2, 2),))
    with pytest.raises(LoweringError, match="bounded training loop"):
        lower(captured.ir, recurrent_gradients="deferred")
    with pytest.raises(LoweringError, match="inline or deferred"):
        lower(captured.ir, recurrent_gradients="unknown")


def test_public_fast_build_selects_only_qualified_single_unit(tmp_path, monkeypatch):
    from megakernel_lab.operation_compiler import runtime

    captured = capture_module(torch.nn.Linear(3, 2), mse,
                              (torch.ones(2, 3),), (torch.zeros(2, 2),))
    calls = []

    def fake_build(program, **options):
        calls.append(options)
        library = Path(tmp_path) / "test-library.so"
        library.write_bytes(b"test build identity")
        return library

    monkeypatch.setattr(runtime, "build", fake_build)
    receipt = captured.build(tmp_path / "artifact", cache_dir=tmp_path / "cache", fast_build=True)
    assert calls[0]["step17"] == "qualified"
    assert receipt["fast_build"] is True


def test_staged_indexed_body_preserves_loop_barrier_contract():
    from .test_compose import _value
    from megakernel_lab.operation_compiler.indexed_reduction import policy

    graph = {"inputs": ["grad", "ids"], "outputs": ["dw"], "values": {
        "grad": _value([8, 16], "float32"), "ids": _value([8], "int64"),
        "dw": _value([32, 16], "float32")}, "operations": [
        {"name": "dw", "target": "aten.embedding_dense_backward.default",
         "args": [{"ref": "grad"}, {"ref": "ids"}, 32, -1, False]}]}
    ordinary = lower(graph, execution="compose", compose_selection="common")
    assert ordinary.manifest["ordered_index_gradients"][0]["indexed_reduction_policy"] == "atomic_v1"
    options = dict(execution="compose", region_body=True, staged=True,
                   output_bindings=["store"], barrier_schedule="dependency")
    staged = lower(graph, **options)
    assert staged.manifest["ordered_index_gradients"][0]["indexed_reduction_policy"] == "frozen_base"
    assert "grid.sync();" not in staged.kernel_body
    with policy("atomic_v1"), pytest.raises(LoweringError, match="staged indexed bodies"):
        lower(graph, **options)
