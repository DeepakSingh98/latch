"""Step-05 programs: equivalent expressions of the same products (explicit BF16 casts, FP32 master state, SGD).

Each family fixes its parameters, inputs, loss and arithmetic; only the PyTorch expression of the products
changes between forms. Every product is ``(a.bfloat16() <op> b.bfloat16()).float()``: BF16 operands, FP32
accumulation and one BF16 result rounding, as in LATCH-STEP16 workloads (37361c773 ComposedMLP recipe).

Families
  linear  - nn.Linear storage W[out, in]; canonical ``F.linear``; alternates ``(W @ x.T).T`` and einsum.
  proj3d  - activations [B, T, K], W[in, out]; canonical flat ``x.reshape(B*T, K) @ W``; alternates 3-D
            ``matmul``, einsum and ``bmm`` with an expanded weight.
  pair    - grouped pairwise scores of a hidden projection (like LATCH-STEP16 held_mixed); canonical ``bmm``;
            alternates ``@ .mT``, einsum, a 4-D unsqueezed matmul and a harmless view chain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch
import torch.nn.functional as F

LR, MOMENTUM = 0.01, 0.9
BF16 = torch.bfloat16


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    form: str
    batch: int = 8          # B (proj3d) or groups G (pair)
    rows: int = 128         # T (proj3d) or rows per group (pair); linear uses batch*rows samples
    inputs: int = 512
    width: int = 1024
    outputs: int = 256
    seed: int = 5051
    # 'explicit': BF16 casts written in the model (all products); 'policy': FP32 model captured with
    # precision='bf16_projections' (covered linear/matmul products BF16, batched activation products FP32);
    # 'fp32': FP32 model, no policy.
    precision: str = 'explicit'

    @property
    def samples(self):
        return self.batch * self.rows


FORMS = {
    'linear': ('linear', 'matmul_t', 'tt', 'einsum'),
    'proj3d': ('flat', 'matmul3d', 'einsum', 'bmm_expand', 'viewchain'),
    'pair': ('bmm', 'mT', 'einsum', 'unsq4d', 'viewchain'),
}
CANONICAL = {'linear': 'linear', 'proj3d': 'flat', 'pair': 'bmm'}


def _round(t):
    return t.to(BF16)


def product_linear(form, x, w, cast=_round, wide=lambda t: t.float()):
    """x[N, K], w[H, K] -> [N, H]."""
    xb, wb = cast(x), cast(w)
    if form == 'linear':
        return wide(F.linear(xb, wb))
    if form == 'matmul_t':
        return wide(xb @ wb.t())
    if form == 'tt':
        return wide((wb @ xb.t()).t())
    if form == 'einsum':
        return wide(torch.einsum('nk,hk->nh', xb, wb))
    raise ValueError(form)


def product_proj3d(form, x, w, cast=_round, wide=lambda t: t.float()):
    """x[B, T, K], w[K, H] -> [B, T, H]."""
    b, t, k = x.shape
    xb, wb = cast(x), cast(w)
    if form == 'flat':
        return wide((xb.reshape(b * t, k) @ wb).reshape(b, t, -1))
    if form == 'matmul3d':
        return wide(xb @ wb)
    if form == 'einsum':
        return wide(torch.einsum('btk,kh->bth', xb, wb))
    if form == 'bmm_expand':
        return wide(torch.bmm(xb, wb.expand(b, *wb.shape)))
    if form == 'viewchain':
        return wide(xb.view(b * t, k).unsqueeze(0).squeeze(0).view(b, t, k) @ wb)
    raise ValueError(form)


def pair_scores(form, h, groups, cast=_round, wide=lambda t: t.float()):
    """h[G*n, H] -> per-group h_g @ h_g^T [G, n, n]."""
    n = h.shape[0] // groups
    if form == 'viewchain':
        rows = cast(h).unflatten(0, (groups, n)).permute(0, 1, 2).contiguous()
        return wide(torch.bmm(rows, rows.transpose(1, 2)))
    rows = cast(h).reshape(groups, n, h.shape[1])
    if form == 'bmm':
        return wide(torch.bmm(rows, rows.transpose(1, 2)))
    if form == 'mT':
        return wide(rows @ rows.mT)
    if form == 'einsum':
        return wide(torch.einsum('gik,gjk->gij', rows, rows))
    if form == 'unsq4d':
        r4 = rows.unsqueeze(1)
        return wide((r4 @ r4.transpose(-1, -2)).squeeze(1))
    raise ValueError(form)


IDENTITY = (lambda t: t, lambda t: t)
ROUNDED = (_round, lambda t: t.float())


def recipe(case):
    """(projection, pairwise) cast/widen pairs of the conventional explicit recipe for this precision."""
    if case.precision == 'explicit':
        return ROUNDED, ROUNDED
    if case.precision == 'policy':
        return ROUNDED, IDENTITY
    return IDENTITY, IDENTITY


def forward(case, params, x, *, proj=None, pair=None):
    """Two products and a SiLU; ``pair`` adds the grouped pairwise branch on the hidden layer.

    The captured model writes casts only for ``explicit``; ``policy`` relies on the capture policy."""
    if proj is None:
        proj, pair = recipe(case) if case.precision == 'explicit' else (IDENTITY, IDENTITY)
    cast, wide = proj
    if case.family == 'linear':
        hidden = F.silu(product_linear(case.form, x, params['fc1'], cast, wide))
        return product_linear(case.form, hidden, params['fc2'], cast, wide), None
    if case.family == 'proj3d':
        hidden = F.silu(product_proj3d(case.form, x, params['fc1'], cast, wide))
        return product_proj3d(case.form, hidden, params['fc2'], cast, wide), None
    if case.family == 'pair':
        hidden = F.silu(wide(cast(x) @ cast(params['fc1'])))
        scores = pair_scores(case.form, hidden, case.batch, *pair) / math.sqrt(case.width)
        return wide(cast(hidden) @ cast(params['fc2'])), scores
    raise ValueError(case.family)


class Model(torch.nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        g = torch.Generator().manual_seed(case.seed)

        def weight(shape, fan_in, gain=0.6):
            return torch.nn.Parameter(torch.randn(*shape, generator=g) * (gain / math.sqrt(fan_in)))
        if case.family == 'linear':
            self.fc1 = weight((case.width, case.inputs), case.inputs)
            self.fc2 = weight((case.outputs, case.width), case.width, 0.3)
        else:
            self.fc1 = weight((case.inputs, case.width), case.inputs)
            self.fc2 = weight((case.width, case.outputs), case.width, 0.3)

    def forward(self, x):
        return forward(self.case, {'fc1': self.fc1, 'fc2': self.fc2}, x)


def fixture(case, seed_offset=1):
    g = torch.Generator().manual_seed(case.seed + seed_offset)
    if case.family == 'proj3d':
        x = torch.randn(case.batch, case.rows, case.inputs, generator=g) * 0.5
        y = torch.randn(case.batch, case.rows, case.outputs, generator=g) * 0.1
    else:
        x = torch.randn(case.samples, case.inputs, generator=g) * 0.5
        y = torch.randn(case.samples, case.outputs, generator=g) * 0.1
    if case.family == 'pair':
        s = torch.randn(case.batch, case.rows, case.rows, generator=g) * 0.1
        return (x,), (y, s)
    return (x,), (y,)


def loss_fn(case):
    def loss(prediction, target, pairwise_target=None):
        terminal, scores = prediction
        value = (terminal - target).square().mean()
        if scores is not None:
            value = value + 0.2 * (scores - pairwise_target).square().mean()
        return value
    return loss


def capture(case, *, lr=LR):
    from megakernel_lab.operation_compiler import SGDConfig, capture_module
    model = Model(case)
    inputs, targets = fixture(case)
    captured = capture_module(model, loss_fn(case), inputs, targets,
                              optimizer=SGDConfig(lr=lr, momentum=MOMENTUM, weight_decay=0.0),
                              precision='bf16_projections' if case.precision == 'policy' else None)
    return model, captured, inputs, targets


def reference_loss(case, params, inputs, targets, *, fp64=False, form=None):
    """Conventional explicit-cast recipe (cuBLAS BF16) or FP64 with the same roundings. ``form`` overrides."""
    case = replace(case, form=form) if form else case
    proj, pair = recipe(case)
    if fp64:
        r = lambda t: t.to(BF16).to(torch.float64)
        lift = lambda c: (r, r) if c is ROUNDED else IDENTITY
        params = {k: v.double() for k, v in params.items()}
        inputs = tuple(t.double() for t in inputs)
        targets = tuple(t.double() for t in targets)
        return loss_fn(case)(forward(case, params, inputs[0], proj=lift(proj), pair=lift(pair)), *targets)
    return loss_fn(case)(forward(case, params, inputs[0], proj=proj, pair=pair), *targets)


def describe(case):
    return asdict(case)


# Shapes. ``dev`` is used to find and develop the mechanism; reserved shapes are declared in RESERVED.json
# before any implementation measurement and are never used for development.
SHAPES = {
    'dev': dict(batch=8, rows=128, inputs=512, width=1024, outputs=256, seed=5051),
    # Reserved 2026-10-07 before implementation (RESERVED.json); not used for development or tuning.
    'r1': dict(batch=6, rows=160, inputs=640, width=1536, outputs=192, seed=5053),
    'r2': dict(batch=12, rows=96, inputs=384, width=896, outputs=320, seed=5057),
}
RESERVED_SHAPES = ('r1', 'r2')


PRECISIONS = {'': 'explicit', 'p': 'policy', 'f': 'fp32'}
# Forms whose products the bf16_projections policy treats identically to the canonical form. The policy
# refuses einsum, and treats bmm/matmul of two >=3-D operands as FP32 batched (attention-class) products, so
# einsum and bmm_expand are not equal-precision policy forms; the pairwise branch is FP32 in every policy form.
POLICY_FORMS = {'linear': ('linear', 'matmul_t', 'tt'), 'proj3d': ('flat', 'matmul3d', 'viewchain'),
                'pair': ('bmm', 'mT', 'unsq4d', 'viewchain')}


def case(name):
    """``family-form-shape[-p|-f]`` (e.g. ``linear-tt-dev``, ``proj3d-matmul3d-r1-p``)."""
    family, form, shape, *rest = name.split('-')
    precision = PRECISIONS.get(rest[0] if rest else '')
    if (family not in FORMS or form not in FORMS[family] or shape not in SHAPES or len(rest) > 1
            or precision is None or precision == 'policy' and form not in POLICY_FORMS[family]):
        raise KeyError(name)
    return Case(name, family, form, **SHAPES[shape], precision=precision)


def pair_cases(family, shape, suffix=''):
    forms = POLICY_FORMS[family] if suffix == 'p' else FORMS[family]
    return [f'{family}-{form}-{shape}' + (f'-{suffix}' if suffix else '') for form in forms]
