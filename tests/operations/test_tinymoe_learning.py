"""Eager semantics of the staged-hyper-parameter rule against torch AdamW + clip (tiny, CPU)."""
import math
import sys
import types

import torch

from megakernel_lab.tinymoe.learning import protocol, sb_rule


def fake_model():
    params = {"model.embed_tokens.weight": torch.randn(5, 4),
              "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(3, 4),
              "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.randn(3, 4),
              "model.norm.weight": torch.randn(4)}
    model = types.SimpleNamespace(named_parameters=lambda: list(params.items()))
    return model, params


def test_protocol_derivations():
    assert abs(protocol.beta2_for(128) - 0.999) < 1e-15
    assert abs(protocol.beta2_for(1) - 0.999 ** (1 / 128)) < 1e-15
    assert abs(protocol.weight_decay(3e-4, 128) - 0.1) < 1e-12
    # same per-token decay for any batch/LR
    for lr, b in ((1e-4, 1), (6e-4, 128), (3e-5, 2)):
        assert abs(lr * protocol.weight_decay(lr, b) / b - 3e-5 / 128) < 1e-15
    # token schedule: same LR at the same token position for every batch
    H = 128 * 1024 * 1000
    assert abs(protocol.lr_at(499, 1.0, 128, H) - protocol.lr_at(499 * 128 + 63, 1.0, 1, H)) < 1e-3
    assert protocol.lr_at(0, 1.0, 1, H) > 0


def test_three_updates_match_torch_adamw_with_absent_expert():
    torch.manual_seed(0)
    model, params = fake_model()
    rule = sb_rule.make_rule(model)
    state = rule.init_state({k: v.clone() for k, v in params.items()}, {})
    hp = protocol.hyperparameters("B1", 2e-4)
    activity = [torch.tensor([1., 1.]), torch.tensor([1., 0.]), torch.tensor([1., 1.])]
    ref = {k: torch.nn.Parameter(v.clone()) for k, v in params.items()}
    decay = [p for p in ref.values() if p.ndim >= 2]
    nodecay = [p for p in ref.values() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=hp["weight_decay"]), dict(params=nodecay, weight_decay=0.0)],
                            lr=1.0, betas=(hp["beta1"], hp["beta2"]), eps=hp["eps"], foreach=False)
    current = {k: v.clone() for k, v in params.items()}
    for index, act in enumerate(activity):
        lr = 2e-4 * (index + 1) / 3
        g = {k: torch.randn_like(v) * 3 for k, v in params.items()}
        if act[1] == 0:
            g["model.layers.0.block_sparse_moe.experts.1.w1.weight"].zero_()
        sb_rule.stage(state, hp, lr)
        current, state = rule.update(current, g, state, {}, {"active.0": act})
        for k, p in ref.items():
            p.grad = None if (act[1] == 0 and "experts.1" in k) else g[k].clone()
        for group in opt.param_groups:
            group["lr"] = lr
        norm = torch.nn.utils.clip_grad_norm_([p for p in ref.values() if p.grad is not None], 1.0)
        opt.step()
        assert torch.allclose(state[sb_rule.NORM], norm, rtol=1e-6)
        for k, p in ref.items():
            assert torch.allclose(current[k], p.detach(), rtol=0, atol=2e-7), (index, k)
    k1 = "model.layers.0.block_sparse_moe.experts.1.w1.weight"
    assert int(state["step:" + k1]) == 2 and int(state["step:model.norm.weight"]) == 3
    assert abs(float(state[sb_rule.LOGB2]) - math.log(hp["beta2"])) < 1e-9
