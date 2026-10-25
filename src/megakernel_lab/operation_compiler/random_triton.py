"""Triton CUDA implementation of ops.random.philox_field for conventional (eager/compiled) execution.

Adapted from the S100L conventional controls (src/s100l_ebm/conventional.py at 0936b034e, released family
s100l): tl.philox is Philox4x32-10 with key (seed low, seed high) and counter (c0, c1, c2, c3); the Box-Muller
transform uses libdevice log / cospi / sinpi and sqrt_rn, bitwise equal to the generated device helper
(ops/random_cuda.FIELD_HELPERS, compiled without fast math). Only imported for CUDA tensors.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _bm(wa, wb):
    a = ((wa >> 8) + 1).to(tl.float32) * 5.9604644775390625e-08
    b2 = (wb >> 8).to(tl.float32) * 5.9604644775390625e-08 * 2.0
    r = tl.sqrt_rn(-2.0 * libdevice.log(a))
    return r * libdevice.cospi(b2), r * libdevice.sinpi(b2)


@triton.jit
def _field_kernel(out, key_ptr, c2_ptr, c3_ptr, rows, cols, blocks, MODE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    q = pid * BLOCK + tl.arange(0, BLOCK)
    m = q < rows * blocks
    row = q // blocks
    j4 = q % blocks
    k0 = tl.load(key_ptr).to(tl.uint32)
    k1 = tl.load(key_ptr + 1).to(tl.uint32)
    z = tl.zeros([BLOCK], dtype=tl.uint32)
    c2 = z + tl.load(c2_ptr).to(tl.uint32)
    c3 = z + tl.load(c3_ptr).to(tl.uint32)
    w0, w1, w2, w3 = tl.philox((k0.to(tl.uint64) | (k1.to(tl.uint64) << 32)), j4.to(tl.uint32), row.to(tl.uint32),
                               c2, c3)
    base = out + row * cols + 4 * j4
    col = 4 * j4
    if MODE == 0:
        v0, v1, v2, v3 = w0.to(tl.int64), w1.to(tl.int64), w2.to(tl.int64), w3.to(tl.int64)
    elif MODE == 1:
        s = 5.9604644775390625e-08
        v0, v1 = (w0 >> 8).to(tl.float32) * s, (w1 >> 8).to(tl.float32) * s
        v2, v3 = (w2 >> 8).to(tl.float32) * s, (w3 >> 8).to(tl.float32) * s
    else:
        v0, v1 = _bm(w0, w1)
        v2, v3 = _bm(w2, w3)
    tl.store(base, v0, mask=m & (col < cols))
    tl.store(base + 1, v1, mask=m & (col + 1 < cols))
    tl.store(base + 2, v2, mask=m & (col + 2 < cols))
    tl.store(base + 3, v3, mask=m & (col + 3 < cols))


def field(key, c2, c3, shape, transform):
    rows, cols = shape
    out = torch.empty((rows, cols), dtype=torch.int64 if transform == 'bits' else torch.float32, device=key.device)
    blocks = (cols + 3) // 4
    grid = (triton.cdiv(rows * blocks, 256),)
    _field_kernel[grid](out, key, c2, c3, rows, cols, blocks, MODE=('bits', 'uniform', 'normal').index(transform),
                        BLOCK=256)
    return out
