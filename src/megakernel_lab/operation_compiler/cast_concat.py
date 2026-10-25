"""Concatenation and float cast in one pass: ``cat(xs, dim).to(dtype)`` (LATCH-TINYMOE-EGRAD).

The BF16 product operand of stacked per-expert FP32 weights is captured as ``cast(view(cat(w_0..w_E-1)))``,
which lowers to an FP32 concatenation copy and a second pass that reads it back and casts it. Capture rewrites
the pattern (``layout_canonical.fuse_cat_casts``) to this operation: every element is read once from its own
piece and rounded with the same ``__float2bfloat16_rn`` as the separate cast, so the stored values are
identical and the FP32 intermediate (its store and reread) disappears. Data movement only.
"""
from __future__ import annotations

import math
from typing import List

import torch

TARGET = "latch_ops.cat_cast.default"
TARGETS = (TARGET,)


@torch.library.custom_op("latch_ops::cat_cast", mutates_args=())
def _cat_cast(xs: List[torch.Tensor], dim: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.cat(xs, dim).to(dtype)


@_cat_cast.register_fake
def _cat_cast_fake(xs, dim, dtype):
    return torch.cat(xs, dim).to(dtype)


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define, target_arch):
    """One grid-stride phase: FP32 pieces concatenated along dim 0 into contiguous BF16 storage."""
    from .lower import LoweringError
    args = list(field(op, "args", ()))
    outputs = tuple(field(op, "outputs", ())) or (field(op, "output", field(op, "name")),)
    if len(args) != 3 or len(outputs) != 1 or not isinstance(args[0], (list, tuple)):
        raise LoweringError("cat_cast: invalid operation ABI")
    name = outputs[0]
    shape = tuple(field(values[name], "shape"))
    define(name, shape, None, dtypes[name])
    sources = [ref(a, names) for a in args[0]]
    if (any(s is None for s in sources) or int(args[1]) % len(shape) != 0 or dtypes[name] != "bfloat16"
            or any(dtypes[s] != "float32" for s in sources)):
        raise LoweringError("cat_cast lowers FP32 pieces concatenated along dim 0 to BF16")
    pieces, start = [], 0
    for s in sources:
        lay = layouts[s]
        if len(lay.shape) != len(shape) or tuple(lay.shape[1:]) != shape[1:] or not lay.contiguous:
            raise LoweringError("cat_cast pieces must be contiguous with the result's trailing extents")
        n = math.prod(lay.shape)
        if n:
            pieces.append((s, start, n))
        start += n
    total = math.prod(shape)
    if start != total or total >= 2**31:
        raise LoweringError("cat_cast extent mismatch")

    def ptr(s):
        lay = layouts[s]
        return f"({ident(lay.root)} + {lay.offset}LL)"
    out = f"({ident(layouts[name].root)} + {layouts[name].offset}LL)"
    # Scalar form: every element from its piece, one rounding (the separate cast's expression).
    chain = "".join(f"i < {lo + n}LL ? {ptr(s)}[i - {lo}LL] : " for s, lo, n in pieces[:-1])
    last_s, last_lo, _ = pieces[-1]
    scalar = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {total}LL; "
              f"i += (long long)gridDim.x * blockDim.x) {{\n"
              f"  {out}[i] = __float2bfloat16_rn({chain}{ptr(last_s)}[i - {last_lo}LL]);\n}}")
    vector_ok = all(lo % 8 == 0 and n % 8 == 0 for _, lo, n in pieces) and layouts[name].offset % 8 == 0
    if not vector_ok:
        code = scalar
    else:
        # Groups of 8 elements: two float4 loads, one 16-byte store of 8 BF16 values; four groups in flight per
        # thread (loads first). Piece boundaries are multiples of 8, so a group never straddles two pieces.
        chain8 = "".join(f"g < {(lo + n) // 8}LL ? reinterpret_cast<const float4*>({ptr(s)}) + 2 * (g - {lo // 8}LL) : "
                         for s, lo, n in pieces[:-1])
        src = f"{chain8}reinterpret_cast<const float4*>({ptr(last_s)}) + 2 * (g - {last_lo // 8}LL)"
        groups = total // 8
        aligned = " && ".join(f"((reinterpret_cast<uintptr_t>({p}) & 15u) == 0)"
                              for p in [out] + [ptr(s) for s, _, _ in pieces])
        vector = f"""{{
  const long long s_ = (long long)gridDim.x * blockDim.x;
  auto src_ = [&](long long g) -> const float4* {{ return {src}; }};
  for (long long b_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; b_ < {groups}LL; b_ += 4LL * s_) {{
    float4 lo_[4], hi_[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {{
      const long long g = b_ + u * s_;
      const float4* p = src_(g < {groups}LL ? g : b_);
      lo_[u] = p[0]; hi_[u] = p[1];
    }}
#pragma unroll
    for (int u = 0; u < 4; ++u) {{
      const long long g = b_ + u * s_;
      if (g < {groups}LL) {{
        const __nv_bfloat162 w0 = __floats2bfloat162_rn(lo_[u].x, lo_[u].y), w1 = __floats2bfloat162_rn(lo_[u].z, lo_[u].w);
        const __nv_bfloat162 w2 = __floats2bfloat162_rn(hi_[u].x, hi_[u].y), w3 = __floats2bfloat162_rn(hi_[u].z, hi_[u].w);
        reinterpret_cast<uint4*>({out})[g] = make_uint4(*reinterpret_cast<const unsigned*>(&w0),
            *reinterpret_cast<const unsigned*>(&w1), *reinterpret_cast<const unsigned*>(&w2),
            *reinterpret_cast<const unsigned*>(&w3));
      }}
    }}
  }}
}}"""
        code = f"if ({aligned}) {vector} else {{\n{scalar}\n}}"
    return [dict(kind="cat_cast", names=[name], targets=[TARGET], reads=list(dict.fromkeys(sources)),
                 writes=[name], code=code, threads=256, requires_full_grid=True, work_items=max(1, total // 8),
                 arithmetic="data movement and one round-to-nearest-even BF16 rounding per element (as the cast)",
                 pieces=len(pieces))]
