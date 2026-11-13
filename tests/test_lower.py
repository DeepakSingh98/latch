"""The lowering's decisions: fusion, in-place state, copies when in-place is unsafe, and refusals."""
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import latch
from latch.lower import lower


def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 64, bias=False), nn.Tanh(), nn.Linear(64, 64, bias=False))


def capture(optimizer, model=None):
    return latch.capture_module(model or mlp(), F.mse_loss, (torch.randn(64, 64),), (torch.randn(64, 64),),
                                optimizer=optimizer)


def test_adamw_mlp_is_eight_phases_with_state_written_in_place():
    program = lower(capture(latch.AdamWConfig()).graph)
    assert len(program.phases) == 8
    kinds = [item.kind for phase in program.phases for item in phase]
    assert kinds.count("gemm") == 5 and kinds.count("reduce") == 1
    names = [op.name for phase in program.phases for item in phase if item.kind == "loop" for op in item.ops]
    assert not any(name.endswith("_commit") for name in names)
    # Seven persistent tensors (two weights, four moments, the step count) are each some loop's direct output.
    bound = [b for b in program.buffers.values() if b.arg is not None and b.name not in program.graph.inputs]
    assert len(bound) == 7


def test_temporaries_share_arena_space():
    program = lower(capture(latch.AdamWConfig()).graph)
    temporaries = [b for b in program.buffers.values() if b.arg is None and b.name in
                   {name for phase in program.phases for item in phase
                    for name in (item.stores if item.kind == "loop" else [item.op.name])}]
    assert program.arena_bytes < sum(b.bytes for b in temporaries)


class Halve:
    """Ignores gradients: the new weights exist before the model has finished reading the old ones."""
    def init(self, params):
        return {}

    def update(self, params, grads, state):
        return {name: p * 0.5 for name, p in params.items()}, {}


class Remember:
    """Keeps the previous weights as state: an update that is a plain copy of bound memory."""
    def init(self, params):
        return {f"previous:{name}": torch.zeros_like(p) for name, p in params.items()}

    def update(self, params, grads, state):
        return ({name: p - 0.1 * grads[name] for name, p in params.items()},
                {f"previous:{name}": p for name, p in params.items()})


def test_early_update_is_copied_in_a_final_phase(train):
    program = lower(capture(Halve()).graph)
    last = program.phases[-1]
    assert all(item.kind == "loop" and all(op.name.endswith("_commit") for op in item.ops) for item in last)
    torch.manual_seed(1)
    batches = [((torch.randn(64, 64),), (torch.randn(64, 64),)) for _ in range(2)]
    train(mlp(), F.mse_loss, batches, Halve())


def test_state_that_copies_old_parameters(train):
    torch.manual_seed(1)
    batches = [((torch.randn(64, 64),), (torch.randn(64, 64),)) for _ in range(3)]
    update = train(mlp(), F.mse_loss, batches, Remember())
    assert set(update.optimizer_state) == {"previous:0.weight", "previous:2.weight"}


class Twice:
    """Two state tensors that are the same computed value."""
    def init(self, params):
        return {f"{kind}:{name}": torch.zeros_like(p) for name, p in params.items() for kind in ("a", "b")}

    def update(self, params, grads, state):
        new_params, new_state = {}, {}
        for name, p in params.items():
            average = new_state[f"a:{name}"] = new_state[f"b:{name}"] = 0.5 * state[f"a:{name}"] + grads[name]
            new_params[name] = p - 0.1 * average
        return new_params, new_state


def test_one_value_for_two_state_tensors(train):
    torch.manual_seed(1)
    batches = [((torch.randn(64, 64),), (torch.randn(64, 64),)) for _ in range(3)]
    train(mlp(), F.mse_loss, batches, Twice())


def test_loss_that_is_a_view(train):
    torch.manual_seed(1)
    batches = [((torch.randn(4, 64),), (torch.randn(4, 64),)) for _ in range(2)]  # few enough for a pointwise mean
    train(mlp(), lambda prediction, target: F.mse_loss(prediction, target).unsqueeze(0)[0], batches,
          latch.SGDConfig(lr=0.1))


def test_unsupported_operation_is_refused():
    model = nn.Sequential(nn.Linear(64, 64), nn.Dropout(0.5), nn.Linear(64, 64))
    with pytest.raises(latch.CompileError, match="no rule for"):
        lower(capture(latch.SGDConfig(), model).graph)


def test_graph_round_trips_through_json():
    import json
    graph = capture(latch.AdamWConfig()).graph
    again = latch.Graph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert latch.generate(lower(again)) == latch.generate(lower(graph))
