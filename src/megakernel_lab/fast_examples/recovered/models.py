"""Ordinary PyTorch computations used by the operation-compiler checkpoint.

These functions contain no compiler declarations, generator phases or backwards.
Targets have 64 columns so the first pinned matrix engine can execute every GEMM.
For MNIST, ten columns are one-hot labels and the remaining columns are zero.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Case:
    name: str
    batch: int = 64
    inputs: int = 64
    width: int = 64
    depth: int = 3
    auxiliary: float = 0.0
    activation: str = 'tanh'
    residual: float = 0.4
    skip: bool = False
    seed: int = 41


CASES = {
    'shared': Case('shared'),
    'auxiliary': Case('auxiliary', auxiliary=0.25),
    # Reserved composition: different activation, skip, depth, and shape.
    'heldout': Case('heldout', batch=128, width=128, depth=4,
                    auxiliary=0.15, activation='sigmoid', residual=0.3, skip=True, seed=93),
    'mnist': Case('mnist', inputs=1024, width=128, auxiliary=0.25, seed=17),
}


def parameters(case, device='cpu', dtype=torch.float32):
    rng = torch.Generator(device='cpu').manual_seed(case.seed)
    def matrix(a, b, gain=1.0):
        return (torch.randn(a, b, generator=rng) * (gain / math.sqrt(a))).to(device=device, dtype=dtype)
    p = {'input': matrix(case.inputs, case.width),
         'input_bias': torch.zeros(case.width, device=device, dtype=dtype),
         'shared': matrix(case.width, case.width, 0.6),
         'shared_bias': torch.zeros(case.width, device=device, dtype=dtype),
         'head': matrix(case.width, 64, 0.2),
         'head_bias': torch.zeros(64, device=device, dtype=dtype)}
    if case.auxiliary:
        p['aux_head'] = matrix(case.width, 64, 0.2)
    return p


def fixture(case, device='cpu', dtype=torch.float32, step=0):
    rng = torch.Generator().manual_seed(case.seed + 1000 + step)
    x = (torch.randn(case.batch, case.inputs, generator=rng) * 0.4).to(device=device, dtype=dtype)
    y = torch.zeros(case.batch, 64, dtype=dtype, device=device)
    labels = torch.randint(0, 10, (case.batch,), generator=rng).to(device)
    y.scatter_(1, labels[:, None], 1.0)
    return x, y


def make_loss(case, *, untied=False):
    def loss(params, x, target):
        activate = torch.tanh if case.activation == 'tanh' else torch.sigmoid
        h0 = activate(x @ params['input'] + params['input_bias'])
        h = h0
        aux = None
        for i in range(case.depth):
            weight = params[f'shared_{i}'] if untied else params['shared']
            h = h + case.residual * activate(h @ weight + params['shared_bias'])
            if i == 0 and case.auxiliary:
                aux = h @ params['aux_head']
        if case.skip:
            h = h + 0.2 * h0
        prediction = h @ params['head'] + params['head_bias']
        result = ((prediction - target) ** 2).sum(dim=1).mean()
        if case.auxiliary:
            result = result + case.auxiliary * ((aux - target) ** 2).sum(dim=1).mean()
        return result
    return loss


def predict(case, params, x):
    activate = torch.tanh if case.activation == 'tanh' else torch.sigmoid
    h0 = activate(x @ params['input'] + params['input_bias'])
    h = h0
    for _ in range(case.depth):
        h = h + case.residual * activate(h @ params['shared'] + params['shared_bias'])
    if case.skip:
        h = h + 0.2 * h0
    return h @ params['head'] + params['head_bias']
