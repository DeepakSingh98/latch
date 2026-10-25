"""Explicit training precision policies applied to the forward program before AD (C-OPS3-A).

A policy is a declared arithmetic recipe, not a type annotation. It rewrites the
user's FP32 forward at the PyTorch function level, *above* automatic
differentiation, by inserting ordinary casts around selected matrix products.
PyTorch then differentiates the transformed forward, so every derivative is the
VJP of the declared casts and products. Nothing downcasts an already
differentiated graph.

Covered operations (the first policies; everything else computes as written):

``linear``   ``F.linear(x, W, b)`` -> ``F.linear(bf16(x), bf16(W)).float() + b``
``matmul``   ``@``/``torch.matmul``/``mm``/``bmm`` (rank >= 2 operands)
             -> ``op(bf16(a), bf16(b)).float()``
``addmm``    ``beta * input + alpha * mm(bf16(a), bf16(b)).float()``
``conv2d``   ``F.conv2d(bf16(x), bf16(W)).float() + b`` (the capture admits only
             the non-overlapping patch class; others are refused there)
``attention`` (policy field) attention-class products: ``F.scaled_dot_product_attention``
             and batched products (``bmm``, ``matmul`` with both operands rank >= 3):
             ``float32``  unchanged FP32 composition/kernel/product;
             ``bfloat16`` both products take BF16 operands and round once to BF16;
             scores are widened before FP32 scale/mask/softmax, probabilities
             round to BF16 only as the value-product operand.
             ``bfloat16_online`` (P8-C) the same casts and products at the function
             level; where the compiler fuses an attention (ops/window_attention.py)
             its softmax is evaluated by the declared online arithmetic (running
             max, hardware exp, BF16 unnormalized probabilities as the value-product
             operand, final 1/l scaling, delta = dO . O in the backward) instead of
             the rounding points of the separate FP32 softmax.
             ``bf16_operands`` (P8-B) scores and score gradients stay FP32;
             only the FP32 operands of the products round to BF16: the
             probabilities as the value / value-gradient product operand and
             the scaled score gradient as the query/key-gradient operand
             (FlashAttention-class arithmetic: exact score products of
             BF16-exact query/key, FP32 accumulation, FP32 dP). Other
             products are left exactly as written (explicit casts allowed).

Every product accumulates in FP32 and rounds its result once to BF16 (ordinary
BF16 ATen semantics), then widens exactly to FP32. Biases, residual additions,
normalization, softmax, activations, losses, loop carry, master parameters,
parameter-gradient accumulation and optimizer state stay FP32; the policy is
entered only around the user's forward/loss functions, never the optimizer.

Gradient semantics are per use. Each covered call casts its own operands, so a
parameter used by several products (tied/shared weights, recurrent iterations)
receives one FP32-widened contribution per use and AD sums them in FP32. A
single shared differentiable BF16 cast would instead sum cotangents in BF16
before widening (Pro review counterexample: 1.00390625 versus 1.0). Reusing a
cast *buffer* after differentiation without changing those VJPs is a separate
optimization owned elsewhere and is not performed here.

Refused, with ``CaptureError("PRECISION_POLICY", ...)``:
* covered operations whose floating operands are not FP32 (mixing explicit casts
  with a policy, FP16/FP64 programs), integer/rank-1 products and ``out=``;
* matrix-product operations the policy does not cover (multi-operand einsum, baddbmm, addbmm,
  mv/dot/inner/outer, tensordot, bilinear, multi-dot, other convolutions,
  ``F.multi_head_attention_forward``) -- they would otherwise silently stay FP32;
* an active ``torch.autocast`` region (two competing recipes);
* SDPA attention masks/dropout/GQA are refused by the capture composition itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from .ir import CaptureError

BF16 = torch.bfloat16
COVERED = ('linear', 'matmul', 'addmm', 'conv2d')
ATTENTION_MODES = ('float32', 'bfloat16', 'bfloat16_online', 'bf16_operands')


@dataclass(frozen=True)
class PrecisionPolicy:
    """A declared forward precision recipe; ``None`` everywhere means FP32 (default)."""
    name: str
    products: tuple[str, ...] = COVERED
    attention: str = 'float32'
    compute_dtype: str = 'bfloat16'
    # P8-B: 'fast' declares approximate transcendental evaluation (ex2.approx exponential, approximate division)
    # in compiler-generated activation epilogues; values stay FP32 (relative differences ~1e-6, below the BF16
    # rounding of the activation outputs). 'exact' (default) keeps correctly rounded/IEEE evaluation.
    activation: str = 'exact'

    def validate(self):
        if self.compute_dtype != 'bfloat16':
            raise CaptureError('PRECISION_POLICY', 'only BF16 product operands are implemented (no FP16/FP8)')
        if not set(self.products) <= set(COVERED):
            raise CaptureError('PRECISION_POLICY', f'unknown covered products {sorted(set(self.products) - set(COVERED))}')
        if self.attention not in ATTENTION_MODES:
            raise CaptureError('PRECISION_POLICY', f'attention must be one of {ATTENTION_MODES}')
        if self.activation not in ('exact', 'fast'):
            raise CaptureError('PRECISION_POLICY', "activation must be 'exact' or 'fast'")
        return self

    def describe(self):
        return {'name': self.name, 'products': list(self.products), 'attention': self.attention,
                'activation': self.activation,
                'compute_dtype': self.compute_dtype,
                'product_arithmetic': 'BF16 operands, FP32 accumulation, one BF16 result rounding, exact widening',
                'fp32': ['master parameters', 'optimizer state', 'parameter-gradient accumulation (per-use VJPs)',
                         'biases', 'residual/loop carry', 'normalization', 'softmax', 'activations', 'loss'],
                'applied': 'PyTorch function level, before automatic differentiation'}


# Named policies. Each corresponds to a previously written explicit-cast recipe
# (examples/mixed_recurrent.py), which is the reference they are tested against.
BF16_PROJECTIONS = PrecisionPolicy('bf16_projections', attention='float32')   # recipe 'bfloat16'
MIXED_BF16 = PrecisionPolicy('mixed_bf16', attention='bfloat16')             # recipe 'mixed'
# P8-C: declared 2026-10-05 before its validation; eager reference of the fused attention arithmetic:
# ops/window_attention.py (_online_forward/_online_backward); function-level casts identical to mixed_bf16.
MIXED_BF16_ONLINE = PrecisionPolicy('mixed_bf16_online', attention='bfloat16_online')
POLICIES = {policy.name: policy for policy in (BF16_PROJECTIONS, MIXED_BF16, MIXED_BF16_ONLINE)}
# P8-B: attention operand rounding only; composes with explicitly cast (or FP32) dense products, which it leaves
# unchanged. Reference: the same policy-wrapped eager computation (ops.precision.bf16_operand_attention).
BF16_ATTENTION_OPERANDS = PrecisionPolicy('bf16_attention_operands', products=(), attention='bf16_operands')
BF16_ATTENTION_FAST = PrecisionPolicy('bf16_attention_operands_fast_activation', products=(),
                                      attention='bf16_operands', activation='fast')
# P8-B attention-operand policies compose with explicitly cast products (they cover no product); they have no
# product explicit-cast recipe counterpart, so they are kept out of POLICIES (whose entries the product-policy
# contracts enumerate) and resolved from this separate registry.
ATTENTION_POLICIES = {policy.name: policy for policy in (BF16_ATTENTION_OPERANDS, BF16_ATTENTION_FAST)}


def resolve(policy):
    """None/'float32' -> None (unchanged default); a name or PrecisionPolicy -> validated policy."""
    if policy is None or policy == 'float32':
        return None
    if isinstance(policy, str):
        known = {**POLICIES, **ATTENTION_POLICIES}
        if policy not in known:
            raise CaptureError('PRECISION_POLICY', f'unknown precision policy {policy!r}; known: {sorted(known)}')
        policy = known[policy]
    if not isinstance(policy, PrecisionPolicy):
        raise CaptureError('PRECISION_POLICY', 'precision must be None, a policy name or a PrecisionPolicy')
    return policy.validate()


@dataclass
class PolicyRecord:
    """Covered calls observed during one trace (shapes only; no values)."""
    calls: list = field(default_factory=list)

    def summary(self):
        counts = {}
        for call in self.calls:
            counts[call['op']] = counts.get(call['op'], 0) + 1
        return {'covered_calls': counts, 'calls': list(self.calls)}


# ----------------------------------------------------------------------------- handlers
def _refuse(message):
    raise CaptureError('PRECISION_POLICY', message)


def _check_fp32(label, *tensors):
    for value in tensors:
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            _refuse(f'{label}: tensor operands required')
        if value.dtype != torch.float32:
            _refuse(f'{label}: policy operands must be FP32 (got {value.dtype}); do not combine explicit casts '
                    'or other floating types with a precision policy')


def _bf16(value):
    return value.to(BF16)


def _wide(value):
    return value.float()


def _linear(policy, record, input, weight, bias=None):
    _check_fp32('linear', input, weight, bias)
    record.calls.append({'op': 'linear', 'input': list(input.shape), 'weight': list(weight.shape),
                         'bias': bias is not None})
    out = _wide(F.linear(_bf16(input), _bf16(weight)))
    return out if bias is None else out + bias


def _product(kind, function):
    def handler(policy, record, a, b, *, out=None):
        if out is not None:
            _refuse(f'{kind}: out= is not admitted under a precision policy')
        _check_fp32(kind, a, b)
        if a.dim() < 2 or b.dim() < 2:
            _refuse(f'{kind}: rank-1 operands (dot/mv) are not covered by the policy')
        if a.dim() >= 3 and b.dim() >= 3 and policy.attention == 'float32':
            # Batched (activation x activation) products are attention-class products.
            record.calls.append({'op': 'batched', 'mode': 'float32', 'a': list(a.shape), 'b': list(b.shape)})
            return function(a, b)
        record.calls.append({'op': 'matmul', 'function': kind, 'a': list(a.shape), 'b': list(b.shape)})
        return _wide(function(_bf16(a), _bf16(b)))
    return handler


def _addmm(policy, record, input, mat1, mat2, *, beta=1, alpha=1, out=None):
    if out is not None:
        _refuse('addmm: out= is not admitted under a precision policy')
    _check_fp32('addmm', input, mat1, mat2)
    record.calls.append({'op': 'addmm', 'a': list(mat1.shape), 'b': list(mat2.shape)})
    product = _wide(torch.mm(_bf16(mat1), _bf16(mat2)))
    if alpha != 1:
        product = product * alpha
    if beta == 0:
        return product
    return (input if beta == 1 else input * beta) + product


def _conv2d(policy, record, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    _check_fp32('conv2d', input, weight, bias)
    record.calls.append({'op': 'conv2d', 'input': list(input.shape), 'weight': list(weight.shape),
                         'bias': bias is not None})
    out = _wide(F.conv2d(_bf16(input), _bf16(weight), None, stride, padding, dilation, groups))
    return out if bias is None else out + bias.reshape(1, -1, 1, 1)


def mixed_attention(query, key, value, *, is_causal=False, scale=None):
    """BF16-product attention composition (the explicit 'mixed' recipe's order)."""
    if query.dim() < 3 or key.shape[-1] != query.shape[-1] or key.shape[-2] != value.shape[-2]:
        _refuse('attention: query/key/value shapes are inconsistent')
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        _refuse('attention: grouped/broadcast heads are not admitted')
    lead = tuple(query.shape[:-2])
    tq, tk, d, dv = query.shape[-2], key.shape[-2], query.shape[-1], value.shape[-1]
    batch = math.prod(lead)
    factor = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    q = _bf16(query).reshape(batch, tq, d)
    k = _bf16(key).reshape(batch, tk, d)
    v = _bf16(value).reshape(batch, tk, dv)
    scores = _wide(torch.bmm(q, k.transpose(1, 2))) * factor
    if is_causal:
        # Same exact finite mask as the FP32 capture composition (standard_ops).
        above = (torch.arange(tk, device=scores.device)[None, :] > torch.arange(tq, device=scores.device)[:, None])
        scores = scores + above.to(scores.dtype) * torch.finfo(scores.dtype).min
    probabilities = torch.softmax(scores, dim=-1)
    return _wide(torch.bmm(_bf16(probabilities), v)).reshape(*lead, tq, dv)


class _RoundOperand(torch.autograd.Function):
    """Value of ``x`` rounded once to BF16 and widened back (a product operand); its derivative is the identity, so
    the cotangent of ``x`` keeps the product's FP32 accumulation (no BF16 rounding of dP)."""

    @staticmethod
    def forward(x):
        return x.to(BF16).to(x.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class _RoundCotangent(torch.autograd.Function):
    """Identity on values; the cotangent is rounded once to BF16 and widened (the operand of the gradient
    products that consume it)."""

    @staticmethod
    def forward(x):
        return x.view_as(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to(BF16).to(gradient.dtype)


def bf16_operand_score(query, key_t, product=torch.bmm):
    """Scores bmm(query, key^T) in the input dtype (exact for BF16-exact operands); cotangent rounded to BF16."""
    return _RoundCotangent.apply(product(query, key_t))


def bf16_operand_value(probabilities, value, product=torch.bmm):
    """bmm(round_bf16(probabilities), value) with an identity probability derivative."""
    return product(_RoundOperand.apply(probabilities), value)


def bf16_operand_attention(query, key, value, *, is_causal=False, scale=None):
    """SDPA composition of the ``bf16_operands`` recipe (same finite causal mask as the FP32 composition)."""
    if query.dim() < 3 or key.shape[-1] != query.shape[-1] or key.shape[-2] != value.shape[-2]:
        _refuse('attention: query/key/value shapes are inconsistent')
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        _refuse('attention: grouped/broadcast heads are not admitted')
    lead = tuple(query.shape[:-2])
    tq, tk, d, dv = query.shape[-2], key.shape[-2], query.shape[-1], value.shape[-1]
    batch = math.prod(lead)
    factor = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    q, k, v = query.reshape(batch, tq, d), key.reshape(batch, tk, d), value.reshape(batch, tk, dv)
    scores = bf16_operand_score(q, k.transpose(1, 2)) * factor
    if is_causal:
        above = (torch.arange(tk, device=scores.device)[None, :] > torch.arange(tq, device=scores.device)[:, None])
        scores = scores + above.to(scores.dtype) * torch.finfo(scores.dtype).min
    probabilities = torch.softmax(scores, dim=-1)
    return bf16_operand_value(probabilities, v).reshape(*lead, tq, dv)


def _operand_handlers(policy):
    """'bf16_operands': batched (rank >= 3 x rank >= 3) products whose first operand is a softmax result are value
    products; other batched products are score products. Lower-rank products pass through unchanged."""
    import weakref
    probabilities = {}   # id -> weak reference (identity checked: ids of freed tensors can be reused)

    def softmax_of(function):
        def handler(policy, record, *args, **kwargs):
            result = function(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                probabilities[id(result)] = weakref.ref(result)
            return result
        return handler

    def is_probability(a):
        ref = probabilities.get(id(a))
        return ref is not None and ref() is a

    def product(kind, function):
        base = torch.bmm if kind == 'bmm' else torch.matmul

        def handler(policy, record, a, b, *, out=None):
            if out is not None or not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) \
                    or a.dim() < 3 or b.dim() < 3:
                return None
            if a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype:
                _refuse(f'{kind}: BF16-operand attention requires FP32 attention operands (got {a.dtype}, {b.dtype})')
            if is_probability(a):
                record.calls.append({'op': 'attention_value', 'mode': 'bf16_operands', 'a': list(a.shape),
                                     'b': list(b.shape)})
                return bf16_operand_value(a, b, base)
            record.calls.append({'op': 'attention_score', 'mode': 'bf16_operands', 'a': list(a.shape),
                                 'b': list(b.shape)})
            return bf16_operand_score(a, b, base)
        return handler

    table = {}
    for function in (torch.softmax, torch.Tensor.softmax, F.softmax):
        table[function] = softmax_of(function)
    for function, kind in ((torch.bmm, 'bmm'), (torch.Tensor.bmm, 'bmm'), (torch.matmul, 'matmul'),
                           (torch.Tensor.matmul, 'matmul'), (torch.Tensor.__matmul__, 'matmul')):
        table[function] = product(kind, function)
    return table


def _attention(policy, record, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
               enable_gqa=False):
    _check_fp32('attention', query, key, value, attn_mask)
    if policy.attention == 'float32':
        record.calls.append({'op': 'attention', 'mode': 'float32', 'query': list(query.shape)})
        return None   # caller passes the call through unchanged
    if attn_mask is not None or dropout_p != 0.0 or enable_gqa:
        _refuse('attention: BF16 attention admits no explicit mask, dropout or grouped heads')
    record.calls.append({'op': 'attention', 'mode': policy.attention, 'query': list(query.shape),
                         'key': list(key.shape)})
    if policy.attention == 'bf16_operands':
        return bf16_operand_attention(query, key, value, is_causal=is_causal, scale=scale)
    return mixed_attention(query, key, value, is_causal=is_causal, scale=scale)


def _handlers(policy):
    table = {}
    if 'linear' in policy.products:
        table[F.linear] = _linear
    if 'matmul' in policy.products:
        for function, kind in ((torch.matmul, 'matmul'), (torch.Tensor.matmul, 'matmul'),
                               (torch.Tensor.__matmul__, 'matmul'), (torch.mm, 'mm'), (torch.Tensor.mm, 'mm'),
                               (torch.bmm, 'bmm'), (torch.Tensor.bmm, 'bmm')):
            base = {'matmul': torch.matmul, 'mm': torch.mm, 'bmm': torch.bmm}[kind]
            table[function] = _product(kind, base)
        table[torch.Tensor.__rmatmul__] = lambda policy, record, b, a: _product('matmul', torch.matmul)(policy, record, a, b)
    if 'addmm' in policy.products:
        table[torch.addmm] = _addmm
        table[torch.Tensor.addmm] = _addmm
    if 'conv2d' in policy.products:
        table[F.conv2d] = _conv2d
    if policy.attention == 'bf16_operands':
        table.update({k: v for k, v in _operand_handlers(policy).items() if k not in table})
    table[F.scaled_dot_product_attention] = _attention
    return table


def _uncovered():
    """Product-like functions refused under every policy (they would silently stay FP32)."""
    names = {
        torch: ('einsum', 'baddbmm', 'addbmm', 'addmv', 'mv', 'dot', 'vdot', 'inner', 'outer', 'ger',
                'tensordot', 'chain_matmul', 'matrix_power', 'kron', 'conv1d', 'conv2d', 'conv3d',
                'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'cdist'),
        torch.Tensor: ('baddbmm', 'addbmm', 'addmv', 'mv', 'dot', 'inner', 'outer', 'ger', 'addmm_', 'baddbmm_',
                       'addbmm_', 'matrix_power', 'kron'),
        F: ('bilinear', 'conv1d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
            'multi_head_attention_forward'),
        torch.linalg: ('multi_dot', 'matmul', 'vecdot'),
    }
    found = {}
    for owner, members in names.items():
        for member in members:
            function = getattr(owner, member, None)
            if function is not None:
                found[function] = f'{getattr(owner, "__name__", "Tensor")}.{member}'
    return found


_UNCOVERED = _uncovered()


class _PolicyMode(torch.overrides.TorchFunctionMode):
    def __init__(self, policy, record):
        super().__init__()
        self.policy, self.record = policy, record
        self.table = _handlers(policy)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        handler = self.table.get(func)
        if handler is not None:
            if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
                _refuse('torch.autocast is active; a precision policy and autocast are competing recipes')
            result = handler(self.policy, self.record, *args, **kwargs)
            if result is not None:
                return result
            return func(*args, **kwargs)
        if func is torch.einsum:
            operands = args[1:]
            if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
                operands = tuple(operands[0])
            if len(operands) == 1:
                return func(*args, **kwargs)   # one operand: a permutation/reduction, not a product
        if func in _UNCOVERED:
            _refuse(f'{_UNCOVERED[func]} is a matrix product not covered by policy {self.policy.name!r}; '
                    'rewrite it with a covered operation or capture without a precision policy')
        return func(*args, **kwargs)


@contextmanager
def precision_policy(policy, record=None):
    """Apply ``policy`` to forward code executed in this context (eager or traced).

    ``None`` is a no-op. Use the same context for conventional reference/control
    arms so that native and conventional execute one declared arithmetic.
    """
    policy = resolve(policy)
    if policy is None:
        yield record
        return
    record = record if record is not None else PolicyRecord()
    with _PolicyMode(policy, record):
        yield record


def wrap(function, policy, record=None):
    """Return ``function`` whose every invocation runs under ``policy`` (identity for None)."""
    policy = resolve(policy)
    if policy is None:
        return function

    def wrapped(*args, **kwargs):
        with precision_policy(policy, record):
            return function(*args, **kwargs)
    wrapped.__wrapped__ = function
    return wrapped


__all__ = ['ATTENTION_MODES', 'ATTENTION_POLICIES', 'BF16_ATTENTION_FAST', 'BF16_ATTENTION_OPERANDS', 'BF16_PROJECTIONS', 'COVERED', 'MIXED_BF16', 'POLICIES',
           'PolicyRecord', 'PrecisionPolicy', 'bf16_operand_attention', 'bf16_operand_score', 'bf16_operand_value',
           'mixed_attention', 'precision_policy', 'resolve', 'wrap']


# ----------------------------------------------------------------------------- exact round trips
_MOVES = {'view', '_unsafe_view', 'reshape', 'permute', 'transpose', 't', 'squeeze', 'unsqueeze', 'clone',
          'alias', 'detach', 'expand', 'slice', 'select'}


def _aten_base(node):
    return str(node.target).removeprefix('torch.ops.').removeprefix('aten.').split('.')[0]


def _float_cast(node, dtype):
    if node.op != 'call_function' or _aten_base(node) != '_to_copy':
        return False
    value = node.meta.get('val')
    return isinstance(value, torch.Tensor) and value.dtype == dtype and set(node.kwargs) <= {'dtype', 'memory_format'}


def cancel_round_trips(gm):
    """Remove exact BF16 -> FP32 -> (data movement) -> BF16 round trips from a captured graph.

    Pattern: ``narrow(m_k(...m_1(widen(s))))`` with ``s`` BF16, ``widen`` an exact
    FP32 cast, ``m_i`` pure data movement (views, clones) and ``narrow`` a BF16
    cast. Widening is exact and rounding an exactly representable value is the
    identity, so the result equals ``m_k(...m_1(s))`` bit for bit; the moves are
    replayed on the BF16 source. Every rounding point, every VJP and every
    accumulation dtype of the differentiated program is unchanged (the forward
    and its derivative were formed before this pass; it only removes value
    identities, it never introduces a cast or shares one). FP32 nodes that keep
    other users remain. Returns the number of narrowing casts removed.
    """
    removed = 0
    for node in list(gm.graph.nodes):
        if not _float_cast(node, BF16):
            continue
        chain, current = [], node.args[0]
        while (isinstance(current, torch.fx.Node) and current.op == 'call_function'
               and _aten_base(current) in _MOVES):
            chain.append(current)
            current = current.args[0]
        if not (isinstance(current, torch.fx.Node) and _float_cast(current, torch.float32)):
            continue
        source = current.args[0]
        value = source.meta.get('val') if isinstance(source, torch.fx.Node) else None
        if not isinstance(value, torch.Tensor) or value.dtype != BF16:
            continue
        replacement = source
        with gm.graph.inserting_before(node):
            for move in reversed(chain):
                replacement_value = move.target(replacement.meta['val'], *move.args[1:], **move.kwargs)
                created = gm.graph.call_function(move.target, (replacement, *move.args[1:]), dict(move.kwargs))
                created.meta = {'val': replacement_value}
                replacement = created
        expected = node.meta['val']
        if (tuple(replacement.meta['val'].shape) != tuple(expected.shape)
                or replacement.meta['val'].dtype != expected.dtype):
            raise CaptureError('PRECISION_POLICY', f'round-trip cancellation changed {node.name} metadata')
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)
        removed += 1
    if removed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return removed


_PERMUTES = {'permute', 'transpose', 't'}


def _plain_float_cast(node):
    if node.op != 'call_function' or _aten_base(node) != '_to_copy' or set(node.kwargs) - {'dtype', 'memory_format'}:
        return False
    if node.kwargs.get('memory_format', torch.preserve_format) is not torch.preserve_format:
        return False
    source = node.args[0] if node.args else None
    out = node.meta.get('val')
    src = source.meta.get('val') if isinstance(source, torch.fx.Node) else None
    return (isinstance(out, torch.Tensor) and isinstance(src, torch.Tensor)
            and out.is_floating_point() and src.is_floating_point())


def _signature(value):
    return tuple(value.shape), tuple(value.stride()), value.dtype


def share_float_casts(gm):
    """Merge float casts that convert the same source, applied to an already differentiated graph (P9-C).

    1. ``cast(p(x))`` with ``p`` a permutation view (permute/transpose/t) becomes ``p(cast(x))`` when the result's
       shape, strides and dtype are unchanged (a dense ``x`` keeps its strides under preserve_format), so casts
       of one tensor and of its transposes share one source form.
    2. Identical casts of one source, and identical permutation views of one cast, merge.

    A float cast is elementwise and deterministic, so every merged node computed the same value bit for bit.
    The graph is differentiated before this pass: every VJP, rounding point and accumulation dtype is
    unchanged; only duplicate conversions (and the storage they occupied) are removed. Graph outputs are not
    merged. Returns {'commuted': n, 'merged': [names]}.
    """
    from torch.fx.node import map_arg
    commuted = 0
    for node in list(gm.graph.nodes):
        if not _plain_float_cast(node):
            continue
        view = node.args[0]
        if view.op != 'call_function' or _aten_base(view) not in _PERMUTES:
            continue
        source = view.args[0]
        value = source.meta.get('val') if isinstance(source, torch.fx.Node) else None
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            continue
        try:
            cast_value = node.target(value, *node.args[1:], **node.kwargs)
            view_value = view.target(cast_value, *view.args[1:], **view.kwargs)
        except Exception:
            continue
        if _signature(view_value) != _signature(node.meta['val']):
            continue
        with gm.graph.inserting_before(node):
            cast = gm.graph.call_function(node.target, (source, *node.args[1:]), dict(node.kwargs))
            cast.meta = {'val': cast_value}
            moved = gm.graph.call_function(view.target, (cast, *view.args[1:]), dict(view.kwargs))
            moved.meta = {'val': view_value}
        node.replace_all_uses_with(moved)
        gm.graph.erase_node(node)
        commuted += 1
    seen, merged = {}, []
    for node in list(gm.graph.nodes):
        if node.op != 'call_function' or any(user.op == 'output' for user in node.users):
            continue
        if not (_plain_float_cast(node) or (_aten_base(node) in _PERMUTES and isinstance(node.args[0], torch.fx.Node)
                                            and _plain_float_cast(node.args[0]))):
            continue
        key = (str(node.target), repr(map_arg(node.args, lambda n: n.name)), repr(map_arg(node.kwargs, lambda n: n.name)))
        if key in seen:
            if _signature(seen[key].meta['val']) != _signature(node.meta['val']):
                continue
            node.replace_all_uses_with(seen[key])
            gm.graph.erase_node(node)
            merged.append(node.name)
        else:
            seen[key] = node
    if commuted or merged:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return {'commuted': commuted, 'merged': merged}
