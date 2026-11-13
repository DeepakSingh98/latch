"""Shared helpers: train with Latch and with ordinary PyTorch, then compare.

Tests run under host emulation by default. Set LATCH_TEST_TARGET=cuda to run the
same tests on a GPU.
"""
import copy
import os
import shutil

import pytest
import torch

import latch

TARGET = os.environ.get("LATCH_TEST_TARGET", "cpu")


def pytest_collection_modifyitems(items):
    if shutil.which("nvcc" if TARGET == "cuda" else "c++") is None:
        skip = pytest.mark.skip(reason=f"no compiler for target {TARGET}")
        for item in items:
            if "train" in item.fixturenames:
                item.add_marker(skip)


def eager(model, loss_fn, batches, optimizer):
    """The same updates with autograd and the optimizer's own arithmetic, run eagerly."""
    params = {name: p.detach().clone().requires_grad_() for name, p in model.named_parameters() if p.requires_grad}
    state, losses = optimizer.init(params), []
    for inputs, targets in batches:
        loss = loss_fn(torch.func.functional_call(model, params, inputs), *targets)
        grads = dict(zip(params, torch.autograd.grad(loss, list(params.values()))))
        with torch.no_grad():
            new_params, new_state = optimizer.update(params, grads, state)
        params = {name: p.detach().requires_grad_() for name, p in new_params.items()}
        state = {**state, **new_state}
        losses.append(loss.item())
    return losses, params, state


def torch_optim(model, loss_fn, batches, optimizer):
    """The same updates with torch.optim, as an independent check of the optimizer arithmetic."""
    model = copy.deepcopy(model)
    if isinstance(optimizer, latch.AdamWConfig):
        opt = torch.optim.AdamW(model.parameters(), lr=optimizer.lr, betas=optimizer.betas, eps=optimizer.eps,
                                weight_decay=optimizer.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=optimizer.lr, momentum=optimizer.momentum,
                              weight_decay=optimizer.weight_decay)
    losses = []
    for inputs, targets in batches:
        opt.zero_grad()
        loss = loss_fn(model(*inputs), *targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, dict(model.named_parameters())


def close(actual, expected, tolerance):
    scale = expected.abs().max().clamp_min(1e-6)
    return ((actual.cpu() - expected.cpu()).abs().max() / scale).item() <= tolerance


@pytest.fixture
def train(tmp_path):
    """train(model, loss_fn, batches, optimizer) -> the loaded ModuleUpdate, after checking it against PyTorch."""
    def run(model, loss_fn, batches, optimizer, tolerance=2e-5, reference=eager):
        captured = latch.capture_module(model, loss_fn, *batches[0], optimizer=optimizer)
        captured.build(tmp_path, target=TARGET)
        update = latch.ModuleUpdate.load(tmp_path)
        losses = []
        for inputs, targets in batches:
            update.stage(inputs, targets)
            losses.append(update.step().item())
        expected = reference(model, loss_fn, batches, optimizer)
        assert losses == pytest.approx(expected[0], rel=tolerance, abs=tolerance)
        for name, value in expected[1].items():
            assert close(update.parameters[name], value, tolerance), f"parameter {name} differs"
        if len(expected) > 2:
            for name, value in expected[2].items():
                if value.is_floating_point():
                    assert close(update.optimizer_state[name], value, tolerance), f"optimizer state {name} differs"
                else:
                    assert torch.equal(update.optimizer_state[name].cpu(), value), f"optimizer state {name} differs"
        return update
    return run
