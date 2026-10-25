"""Ordinary causal transformer expressed entirely with differentiable PyTorch ops.

The distinct-layer model, its intermediate prediction objective, and its shared
runtime-loop body use the same block. No model derivative or CUDA schedule lives
here. Byte indices, position indices and the causal mask are explicit inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.random import dropout, make_rng_state, philox_advance, philox_mask


@dataclass(frozen=True)
class TransformerConfig:
    batch: int = 4
    sequence: int = 64
    width: int = 256
    heads: int = 4
    layers: int = 2
    expansion: int = 4
    vocab: int = 256
    epsilon: float = 1e-5
    seed: int = 20261002
    auxiliary_weight: float = 0.25
    drop_probability: float = 0.1
    matrix_dtype: str = 'float32'

    def validate(self):
        if min(self.batch, self.sequence, self.width, self.heads, self.layers) <= 0:
            raise ValueError('transformer dimensions must be positive')
        if self.width % self.heads or self.vocab != 256 or self.expansion <= 0:
            raise ValueError('require integral head width, byte vocabulary and positive expansion')
        if not 0 <= self.drop_probability < 1 or self.epsilon <= 0:
            raise ValueError('invalid dropout probability or normalization epsilon')
        if self.matrix_dtype not in ('float32', 'bfloat16'):
            raise ValueError('matrix dtype must be explicitly float32 or bfloat16')


RECIPES = {
    'development': TransformerConfig(batch=2, width=128, heads=2, layers=1),
    'primary': TransformerConfig(),
    'heldout_width': TransformerConfig(width=384, heads=6),
    'scale': TransformerConfig(width=512, heads=8, layers=8),
}
RECIPES.update({name + '_bf16': replace(RECIPES[name], matrix_dtype='bfloat16')
                for name in ('development', 'primary', 'heldout_width', 'scale')})


def parameters(config=RECIPES['primary'], *, device='cpu', dtype=torch.float32):
    config.validate()
    g = torch.Generator().manual_seed(config.seed)
    d, f = config.width, config.expansion * config.width

    def normal(shape, scale):
        return (torch.randn(shape, generator=g) * scale).to(device=device, dtype=dtype)

    p = {'token_embedding': normal((config.vocab, d), 0.02),
         'position_embedding': normal((config.sequence, d), 0.02)}
    for layer in range(config.layers):
        prefix = f'block{layer}.'
        p[prefix + 'attention_norm'] = torch.ones(d, device=device, dtype=dtype)
        for name in ('query', 'key', 'value'):
            p[prefix + name] = normal((d, d), 1.0 / math.sqrt(d))
        p[prefix + 'attention_out'] = normal((d, d), 1.0 / math.sqrt(2 * config.layers * d))
        p[prefix + 'feedforward_norm'] = torch.ones(d, device=device, dtype=dtype)
        p[prefix + 'up'] = normal((d, f), 1.0 / math.sqrt(d))
        p[prefix + 'down'] = normal((f, d), 1.0 / math.sqrt(2 * config.layers * f))
    p['final_norm'] = torch.ones(d, device=device, dtype=dtype)
    p['head'] = normal((d, config.vocab), 0.02)
    return p


def fixture(config=RECIPES['primary'], *, device='cpu'):
    """Synthetic bytes only for capture/numerical qualification, never training claims."""
    g = torch.Generator().manual_seed(config.seed + 1)
    tokens = torch.randint(0, config.vocab, (config.batch, config.sequence), generator=g).to(device)
    targets = torch.randint(0, config.vocab, (config.batch, config.sequence), generator=g).to(device)
    return tokens, targets, *static_inputs(config, device=device)


def static_inputs(config=RECIPES['primary'], *, device='cpu', dtype=torch.float32):
    positions = torch.arange(config.sequence, dtype=torch.int64, device=device)
    # Finite mask avoids nonfinite diagnostic intermediates. exp(-10000) is zero
    # in the admitted FP32/FP64 policies; each row still includes its diagonal.
    mask = torch.triu(torch.full((config.sequence, config.sequence), -10000.0,
                                dtype=dtype, device=device), diagonal=1)
    return positions, mask


def rms_norm(hidden, weight, epsilon):
    return hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + epsilon) * weight


def embed(params, tokens, positions):
    return F.embedding(tokens, params['token_embedding']) + F.embedding(positions, params['position_embedding'])


def linear(value, weight, config):
    if config.matrix_dtype == 'bfloat16':
        # An explicit changed arithmetic recipe: master weights, residual carry,
        # normalization, attention probabilities and objective stay FP32. Dense
        # operands/results round to BF16; dot accumulation is FP32.
        return (value.to(torch.bfloat16) @ weight.to(torch.bfloat16)).float()
    return value @ weight


def block(params, hidden, mask, config, layer):
    b, t, d, h = config.batch, config.sequence, config.width, config.heads
    hd = d // h
    prefix = f'block{layer}.'
    normalized = rms_norm(hidden, params[prefix + 'attention_norm'], config.epsilon)

    def project(name):
        projected = linear(normalized.reshape(b * t, d), params[prefix + name], config)
        return projected.reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)

    query, key, value = project('query'), project('key'), project('value')
    scores = torch.bmm(query, key.transpose(1, 2)) * (1.0 / math.sqrt(hd)) + mask
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.bmm(probabilities, value).reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d)
    hidden = hidden + linear(attended, params[prefix + 'attention_out'], config).reshape(b, t, d)
    normalized = rms_norm(hidden, params[prefix + 'feedforward_norm'], config.epsilon)
    wide = F.silu(linear(normalized.reshape(b * t, d), params[prefix + 'up'], config))
    return hidden + linear(wide, params[prefix + 'down'], config).reshape(b, t, d)


def logits(params, hidden, config):
    normalized = rms_norm(hidden, params['final_norm'], config.epsilon)
    return linear(normalized.reshape(-1, config.width), params['head'], config).reshape(
        config.batch, config.sequence, config.vocab)


def cross_entropy(params, hidden, targets, config):
    return -torch.log_softmax(logits(params, hidden, config), dim=-1).gather(
        -1, targets.unsqueeze(-1)).mean()


def make_loss(config=RECIPES['primary'], *, intermediate=False):
    """Distinct layers; optional prediction loss after the first block."""
    config.validate()

    def loss(params, tokens, targets, positions, mask):
        hidden = embed(params, tokens, positions)
        auxiliary = None
        for layer in range(config.layers):
            hidden = block(params, hidden, mask, config, layer)
            if intermediate and layer == 0:
                auxiliary = cross_entropy(params, hidden, targets, config)
        terminal = cross_entropy(params, hidden, targets, config)
        return terminal if auxiliary is None else terminal + config.auxiliary_weight * auxiliary

    return loss


def make_loop_functions(config=RECIPES['primary'], *, intermediate=False, stochastic=True):
    """Repeat the same stack with runtime count; all layers share across iterations.

    A two-block primary stack preserves the ordinary model's parameter count.
    Intermediate supervision predicts after every stack and sums (not averages)
    its loss with a declared fixed coefficient. Count zero is embedding + head.
    """
    config.validate()

    def init(params, tokens, targets, positions, mask):
        hidden = embed(params, tokens, positions)
        return {'hidden': hidden, 'aux_loss': hidden.sum() * 0.0}

    def body(params, carry, state, tokens, targets, positions, mask):
        hidden, rng = carry['hidden'], state['rng']
        for layer in range(config.layers):
            hidden = block(params, hidden, mask, config, layer)
            if stochastic:
                if hidden.dtype == torch.float64:
                    # Independent high-precision arithmetic uses the exact same
                    # FP32 mask bits, widened before multiplying FP64 activations.
                    mask_value = philox_mask(rng, list(hidden.shape),
                                            math.floor(config.drop_probability * (1 << 32)),
                                            1.0 / (1.0 - config.drop_probability))
                    hidden = hidden * mask_value.to(torch.float64)
                    rng = philox_advance(rng, (hidden.numel() + 3) // 4)
                else:
                    hidden, rng = dropout(hidden, rng, config.drop_probability)
        auxiliary = carry['aux_loss']
        if intermediate:
            auxiliary = auxiliary + cross_entropy(params, hidden, targets, config)
        return {'hidden': hidden, 'aux_loss': auxiliary}, {'rng': rng}

    def terminal(params, carry, tokens, targets, positions, mask):
        return (cross_entropy(params, carry['hidden'], targets, config)
                + config.auxiliary_weight * carry['aux_loss'])

    return init, body, terminal


def initial_state(*, device='cpu'):
    return {'rng': make_rng_state(0x13579BDF2468ACE0, (1 << 40) + 73, device=device)}


def loop_loss(config, *, count, intermediate=False, stochastic=True):
    """Eager reference only: native capture retains one body and a device count."""
    init, body, terminal = make_loop_functions(config, intermediate=intermediate, stochastic=stochastic)

    def loss(params, state, *inputs):
        carry = init(params, *inputs)
        for _ in range(count):
            carry, state = body(params, carry, state, *inputs)
        return terminal(params, carry, *inputs), state

    return loss
