"""Explicit Philox4x32-10 state and differentiable inverted dropout.

This is an independently written implementation of the algorithm documented by
Random123, not PyTorch's seed/offset mapping or random stream:
https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h

State is a contiguous int64[4]: key low/high words, then counter low/high words.
Each word is interpreted modulo 2**32. A counter identifies one four-word Philox
block, using (counter_low, counter_high, 0, 0). Calls consume ceil(numel/4) blocks,
including p=0 calls, discard unused trailing words, and wrap the counter modulo
2**64. There is no global generator or implicit state mutation. Recomputing with
the same old state returns exactly the same mask and must not commit twice.

The two custom operators accept only integer tensors, hence are nondifferentiable.
AD differentiates the ordinary x*mask multiplication. Their fake implementations
keep each operation opaque in make_fx; no custom backward schedule is required.
"""
from __future__ import annotations

import math

import torch

U32 = (1 << 32) - 1
U64 = (1 << 64) - 1
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK_TARGET = 'latch_ops.philox_mask.default'
ADVANCE_TARGET = 'latch_ops.philox_advance.default'


def _validate_state(state):
    if state.dtype != torch.int64 or tuple(state.shape) != (4,) or not state.is_contiguous():
        raise ValueError('Philox state must be a contiguous int64[4] tensor')
    if state.device.type not in ('cpu', 'cuda'):
        raise ValueError('Philox state supports CPU/CUDA only')


def _validate_mask(state, shape, threshold, scale):
    _validate_state(state)
    if any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError('Philox mask shape must have static positive integer dimensions')
    if math.prod(shape) > (1 << 63) - 1:
        raise ValueError('Philox mask has too many elements')
    if not 0 <= threshold <= 1 << 32:
        raise ValueError('Philox mask threshold must be in [0, 2**32]')
    if not math.isfinite(scale) or not 0 <= scale <= torch.finfo(torch.float32).max:
        raise ValueError('Philox mask scale must be finite nonnegative FP32')


def make_rng_state(seed: int, counter: int = 0, *, device='cpu') -> torch.Tensor:
    """Construct the explicit state from unsigned 64-bit Python integers."""
    if type(seed) is not int or type(counter) is not int or not 0 <= seed <= U64 or not 0 <= counter <= U64:
        raise ValueError('seed and counter must be unsigned 64-bit integers')
    return torch.tensor([seed & U32, seed >> 32, counter & U32, counter >> 32],
                        dtype=torch.int64, device=device)


def _mulhilo(constant, value):
    # 16-bit limbs keep every intermediate below 2**34, including on CPU where
    # relying on signed int64 multiplication overflow would obscure the contract.
    lo, hi = value & 0xffff, value >> 16
    c_lo, c_hi = constant & 0xffff, constant >> 16
    middle = c_lo * hi + c_hi * lo
    low = c_lo * lo + ((middle & 0xffff) << 16)
    high = c_hi * hi + (middle >> 16) + (low >> 32)
    return high & U32, low & U32


def _philox4x32(counter, key):
    """Tensor reference of the full four-word bijection (also used by KAT tests)."""
    c0, c1, c2, c3 = (x & U32 for x in counter)
    k0, k1 = (x & U32 for x in key)
    for round_index in range(10):
        hi0, lo0 = _mulhilo(M0, c0)
        hi1, lo1 = _mulhilo(M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if round_index != 9:
            k0, k1 = (k0 + W0) & U32, (k1 + W1) & U32
    return c0, c1, c2, c3


def _words(state, count):
    blocks = (count + 3) // 4
    indices = torch.arange(blocks, dtype=torch.int64, device=state.device)
    low = (state[2] & U32) + indices
    high = ((state[3] & U32) + (low >> 32)) & U32
    zero = torch.zeros_like(low)
    words = _philox4x32((low & U32, high, zero, zero), (state[0], state[1]))
    return torch.stack(words, dim=-1).flatten()[:count]


@torch.library.custom_op('latch_ops::philox_mask', mutates_args=())
def philox_mask(state: torch.Tensor, shape: list[int], threshold: int, scale: float) -> torch.Tensor:
    """Return an opaque deterministic mask; threshold compares unsigned words."""
    _validate_mask(state, shape, threshold, scale)
    return (_words(state, math.prod(shape)) >= threshold).to(torch.float32).reshape(shape) * scale


@philox_mask.register_fake
def _fake_mask(state, shape, threshold, scale):
    _validate_mask(state, shape, threshold, scale)
    return torch.empty(shape, dtype=torch.float32, device=state.device)


@torch.library.custom_op('latch_ops::philox_advance', mutates_args=())
def philox_advance(state: torch.Tensor, blocks: int) -> torch.Tensor:
    """Return a fresh canonical state; only a caller's final commit mutates state."""
    _validate_state(state)
    if not 0 <= blocks <= (1 << 63) - 1:
        raise ValueError('Philox advance requires a nonnegative signed-64-bit block count')
    low = (state[2] & U32) + (blocks & U32)
    high = ((state[3] & U32) + (blocks >> 32) + (low >> 32)) & U32
    return torch.stack((state[0] & U32, state[1] & U32, low & U32, high))


@philox_advance.register_fake
def _fake_advance(state, blocks):
    _validate_state(state)
    if not 0 <= blocks <= (1 << 63) - 1:
        raise ValueError('Philox advance requires a nonnegative signed-64-bit block count')
    return torch.empty((4,), dtype=torch.int64, device=state.device)


def dropout(x: torch.Tensor, state: torch.Tensor, p: float = 0.5):
    """Return (x * inverted_mask, new_state), with a static dropout probability.

    Drop probability is floor(p*2**32)/2**32; the scale is FP32(1/(1-p)). The
    quantization differs from p by less than 2**-32. This API is always training
    mode; callers express evaluation/control explicitly. A scalar x is admitted.
    """
    if x.dtype != torch.float32 or x.device != state.device:
        raise ValueError('dropout requires FP32 x and state on the same device')
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p < 1:
        raise ValueError('dropout probability must be a static finite number in [0, 1)')
    shape = list(x.shape)
    mask = philox_mask(state, shape, math.floor(p * (1 << 32)), 1.0 / (1.0 - p))
    next_state = philox_advance(state, (x.numel() + 3) // 4)
    return x * mask, next_state


# ---------------------------------------------------------------------------------------------------------
# C-OPS4-B: counter-grid random fields.
#
# A two-dimensional field indexed by (row i, column j) draws Philox4x32-10 block (j >> 2, i, c2, c3) with
# key (key[0], key[1]) and returns word j & 3, transformed. c2 and c3 are runtime int64 scalars (mod 2**32),
# so a loop iteration or update counter selects a fresh, independent field without any state transition.
# This is the counter layout of the S100L energy-model RNG (`s100l-rng-v1`, research/s100/S100L/SPEC.md §5);
# the transforms are its uniform and Box-Muller definitions:
#   'bits'    int64 word value in [0, 2**32)
#   'uniform' FP32 (w >> 8) * 2**-24 (exact)
#   'normal'  pairs (w0, w1) -> (n0, n1), (w2, w3) -> (n2, n3): a = ((w_even >> 8) + 1) 2**-24,
#             b = (w_odd >> 8) 2**-24, r = sqrt_rn(-2 logf(a)), n_even = r cospif(2b), n_odd = r sinpif(2b),
#             evaluated in FP32 with the CUDA libdevice functions on the device.
# The CPU implementation computes words exactly and the normal transform in FP64 (then rounds to FP32): it is
# an independent reference within a few ulps of the device transform, not a bitwise oracle for normals.
# CUDA tensors (conventional controls) use the Triton kernel in ops/random_triton.py, whose libdevice
# transform is bitwise equal to the generated device code.
FIELD_TARGET = 'latch_ops.philox_field.default'
FIELD_TRANSFORMS = ('bits', 'uniform', 'normal')


def _validate_field(key, c2, c3, shape, transform):
    if key.dtype != torch.int64 or tuple(key.shape) != (2,) or not key.is_contiguous():
        raise ValueError('Philox field key must be a contiguous int64[2] tensor')
    for label, value in (('c2', c2), ('c3', c3)):
        if value.dtype != torch.int64 or value.numel() != 1 or value.device != key.device:
            raise ValueError(f'Philox field counter {label} must be one int64 element on the key device')
    if len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError('Philox field shape must be [rows, columns] with static positive integers')
    if shape[0] > 1 << 32 or (shape[1] + 3) // 4 > 1 << 32:
        raise ValueError('Philox field rows and column blocks must fit 32-bit counter words')
    if transform not in FIELD_TRANSFORMS:
        raise ValueError(f'unknown Philox field transform {transform!r}')


def make_key(seed: int, *, device='cpu') -> torch.Tensor:
    """int64[2] key (low, high 32-bit words) of an unsigned 64-bit seed."""
    if type(seed) is not int or not 0 <= seed <= U64:
        raise ValueError('seed must be an unsigned 64-bit integer')
    return torch.tensor([seed & U32, seed >> 32], dtype=torch.int64, device=device)


def field_words(key, c2, c3, rows, cols):
    """Exact int64 words [rows, cols] (CPU reference arithmetic; any device)."""
    i = torch.arange(rows, dtype=torch.int64, device=key.device)[:, None].expand(rows, cols)
    j = torch.arange(cols, dtype=torch.int64, device=key.device)[None, :].expand(rows, cols)
    c2v = (c2.reshape(()) & U32).expand(rows, cols)
    c3v = (c3.reshape(()) & U32).expand(rows, cols)
    words = _philox4x32((j >> 2, i, c2v, c3v), (key[0], key[1]))
    stacked = torch.stack(words, dim=-1)
    return torch.gather(stacked, -1, (j & 3)[..., None])[..., 0]


def _field_reference(key, c2, c3, shape, transform):
    rows, cols = shape
    if transform == 'bits':
        return field_words(key, c2, c3, rows, cols)
    if transform == 'uniform':
        return ((field_words(key, c2, c3, rows, cols) >> 8).to(torch.float64) * 2.0 ** -24).to(torch.float32)
    padded = (cols + 3) // 4 * 4
    words = field_words(key, c2, c3, rows, padded).reshape(rows, padded // 4, 4)
    even, odd = words[..., 0::2], words[..., 1::2]
    a = ((even >> 8) + 1).to(torch.float64) * 2.0 ** -24
    b = (odd >> 8).to(torch.float64) * 2.0 ** -24
    r = torch.sqrt(-2.0 * torch.log(a))
    n = torch.stack((r * torch.cos(2.0 * math.pi * b), r * torch.sin(2.0 * math.pi * b)), dim=-1)
    return n.reshape(rows, padded)[:, :cols].to(torch.float32)


@torch.library.custom_op('latch_ops::philox_field', mutates_args=())
def philox_field(key: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, shape: list[int],
                 transform: str) -> torch.Tensor:
    """Opaque nondifferentiable random field; see the module comment for the counter layout."""
    _validate_field(key, c2, c3, shape, transform)
    if key.device.type == 'cuda':
        from .random_triton import field as triton_field
        return triton_field(key, c2, c3, list(shape), transform)
    return _field_reference(key, c2, c3, list(shape), transform)


@philox_field.register_fake
def _fake_field(key, c2, c3, shape, transform):
    _validate_field(key, c2, c3, shape, transform)
    return torch.empty(list(shape), dtype=torch.int64 if transform == 'bits' else torch.float32, device=key.device)
