"""Small independent per-expert reference for the public-model capture adapter."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from megakernel_lab.tinymoe.compiled_model import CapturedTinyMoE


def test_packed_experts_match_independent_fixed_membership_vjp():
    torch.manual_seed(892)
    module = nn.Module()
    module.gate = nn.Linear(4, 4, bias=False)
    module.experts = nn.ModuleList()
    for _ in range(4):
        expert = nn.Module()
        expert.w1 = nn.Linear(4, 8, bias=False)
        expert.w2 = nn.Linear(8, 4, bias=False)
        expert.w3 = nn.Linear(4, 8, bias=False)
        module.experts.append(expert)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    # Uneven segments, an unused expert, and changing selected-slot ordering.
    selected = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [0, 1]])
    context = SimpleNamespace(top_k=2, num_experts=4)
    actual, _, _, active = CapturedTinyMoE._experts(context, module, hidden, selected)
    x = hidden.reshape(-1, 4)
    logits = F.linear(x.to(torch.bfloat16), module.gate.weight.to(torch.bfloat16))
    probability = F.softmax(logits, -1, dtype=torch.float32)
    weights = probability.gather(1, selected)
    weights = weights / weights.sum(-1, keepdim=True)
    reference = torch.zeros_like(x)
    for number, expert in enumerate(module.experts):
        tokens, slots = torch.where(selected == number)
        if not tokens.numel():
            continue
        # Independent HF spelling: gather FP32 before each linear's BF16 cast,
        # and widen the weighted output before accumulation into the residual.
        rows = x[tokens]
        up = F.linear(rows.to(torch.bfloat16), expert.w1.weight.to(torch.bfloat16))
        gate = F.linear(rows.to(torch.bfloat16), expert.w3.weight.to(torch.bfloat16))
        down = F.linear(F.silu(up) * gate, expert.w2.weight.to(torch.bfloat16))
        reference = reference.index_add(0, tokens, down * weights[tokens, slots, None])
    reference = reference.reshape(hidden.shape)
    cotangent = torch.randn_like(hidden)
    inputs = (hidden, *module.parameters())
    candidate_grads = torch.autograd.grad((actual * cotangent).sum(), inputs, retain_graph=True)
    reference_grads = torch.autograd.grad((reference * cotangent).sum(), inputs, allow_unused=True)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for candidate, oracle, source in zip(candidate_grads, reference_grads, inputs):
        if oracle is None:
            oracle = torch.zeros_like(source)
        torch.testing.assert_close(candidate, oracle, rtol=0, atol=0)
    torch.testing.assert_close(active, torch.tensor([1., 1., 1., 0.]), rtol=0, atol=0)


def test_fp32_dispatch_gradient_cannot_move_cast_before_gather():
    """Discriminating example: BF16 addition would lose a representable contribution."""
    x = torch.ones(1, 1, requires_grad=True)
    ids = torch.tensor([0, 0])
    cotangent = torch.tensor([[1.], [1. / 256]])
    actual = F.embedding(ids, x).to(torch.bfloat16)
    derivative, = torch.autograd.grad((actual.float() * cotangent).sum(), x)
    assert derivative.item() == 1.0 + 1.0 / 256
    incorrectly_moved_cast = F.embedding(ids, x.to(torch.bfloat16))
    wrong, = torch.autograd.grad((incorrectly_moved_cast.float() * cotangent).sum(), x)
    assert wrong.item() == 1.0 and not torch.equal(derivative, wrong)
