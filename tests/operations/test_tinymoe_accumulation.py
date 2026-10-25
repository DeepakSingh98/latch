"""Eager semantics of the accumulating rule against torch AdamW + clip (tiny, CPU)."""
import sys
import types

import torch

from megakernel_lab.tinymoe.learning import ga_rule
from megakernel_lab.tinymoe.learning.ga_common import PUBLISHED


def fake_model():
    params = {"model.embed_tokens.weight": torch.randn(5, 4),
              "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(3, 4),
              "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.randn(3, 4),
              "model.layers.0.block_sparse_moe.experts.2.w1.weight": torch.randn(3, 4),
              "model.norm.weight": torch.randn(4)}
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=1, num_local_experts=3),
                                  named_parameters=lambda: list(params.items()))
    return model, params


def test_accumulate_then_update_matches_torch_adamw():
    torch.manual_seed(0)
    model, params = fake_model()
    rule = ga_rule.make_accumulating_rule(model)
    state = rule.init_state({k: v.clone() for k, v in params.items()}, {})
    state[ga_rule.LR] = torch.tensor(3e-4)
    # expert 2 never participates; expert 1 participates only in microbatch 0
    activity = [torch.tensor([1., 1., 0.]), torch.tensor([1., 0., 0.]), torch.tensor([1., 0., 0.])]
    grads = []
    for act in activity:
        g = {k: torch.randn_like(v) * 3 for k, v in params.items()}
        for e in range(3):
            if act[e] == 0:
                g[f"model.layers.0.block_sparse_moe.experts.{e}.w1.weight"].zero_()
        grads.append(g)
    current = {k: v.clone() for k, v in params.items()}
    for index, (g, act) in enumerate(zip(grads, activity)):
        state[ga_rule.FINAL] = torch.tensor(1.0 if index == len(grads) - 1 else 0.0)
        before = {k: v.clone() for k, v in current.items()}
        current, state = rule.update(current, g, state, {}, {"active.0": act})
        if index < len(grads) - 1:
            assert all(torch.equal(before[k], current[k]) for k in before)
            assert all(int(state["step:" + k]) == 0 for k in before)
    # reference: HF-style accumulation (absent grads stay None), clip once, torch AdamW once
    ref = {k: torch.nn.Parameter(v.clone()) for k, v in params.items()}
    for g in grads:
        for k, p in ref.items():
            if "experts.2" in k:
                continue
            p.grad = g[k].clone() if p.grad is None else p.grad + g[k]
    decay = [p for k, p in ref.items() if p.ndim >= 2]
    nodecay = [p for k, p in ref.items() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=PUBLISHED["weight_decay"]),
                             dict(params=nodecay, weight_decay=0.0)], lr=3e-4,
                            betas=(PUBLISHED["adam_beta1"], PUBLISHED["adam_beta2"]), eps=PUBLISHED["adam_epsilon"],
                            foreach=False)
    norm = torch.nn.utils.clip_grad_norm_(list(ref.values()), 1.0)
    opt.step()
    assert torch.allclose(state[ga_rule.NORM], norm, rtol=1e-6)
    for k, p in ref.items():
        assert torch.allclose(current[k], p.detach(), rtol=0, atol=1e-7), k
        assert torch.equal(state["acc:" + k], torch.zeros_like(p))
    k2 = "model.layers.0.block_sparse_moe.experts.2.w1.weight"
    assert torch.equal(current[k2], params[k2]) and int(state["step:" + k2]) == 0
    assert int(state["step:model.layers.0.block_sparse_moe.experts.1.w1.weight"]) == 1
    assert torch.equal(state["seen:0"], torch.zeros(3))
