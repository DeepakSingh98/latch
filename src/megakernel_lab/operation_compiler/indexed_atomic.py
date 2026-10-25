"""LATCH-STEP-04 `atomic_v1`: FP32 embedding gradients by global atomic accumulation (practical atomic native).

Generic lowering of `aten.embedding_dense_backward`, selected from shape, dtype and layout only. One phase with
two internal grid barriers, the same structure as conventional atomic scatter implementations:
1. the dense [V, D] output is zeroed, grid-strided over every CTA (inactive rows stay +0.0);
2. grid barrier; every (occurrence, column vector) pair is added into its row with one global float4 (or float)
   atomicAdd, grid-strided; padding-index occurrences contribute nothing; hot rows are spread over all SMs;
3. grid barrier; with a fused clip norm, each thread squares the output elements of its grid-stride range, the
   CTA tree combines them, and the last CTA to arrive sums the CTA partials in CTA order.
Declared arithmetic policy `atomic_fp32_unordered_v1`: FP32 addition in hardware arrival order, so repeated runs
are not bitwise identical (admitted by the step-04 numerical policy for natives and controls alike). Work:
O(V*D + N*D); no index preparation, sorting or scratch.
"""
from __future__ import annotations

from .indexed_owner import _unravel

HELPERS = r"""#include <cassert>
// LATCH-STEP-04 atomic_v1 helpers.
__device__ __forceinline__ float at_sq(const float a) { return __fmul_rn(a, a); }
__device__ __forceinline__ float at_sq(const float4 a) {
  return __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(a.x, a.x), __fmul_rn(a.y, a.y)), __fmul_rn(a.z, a.z)), __fmul_rn(a.w, a.w));
}
"""


def supported(n, rows, width):
    if n < 1 or rows < 1 or width < 1:
        return False, "empty index or row dimension"
    if n * width >= (1 << 40) or rows * width >= (1 << 40):
        return False, "element counts exceed the admitted range"
    return True, "atomic engine admitted"


# LATCH-STEP-04: with a fused clip norm, the norm reads only rows that received occurrences (first occurrence of a
# row claims a flag and appends the row to a list) instead of the whole dense output. Off: dense read.
DISTINCT_ROW_NORM = False   # measured mixed: embedding -0.6 us, reserved_embedding +1.5 us, expert neutral; option


def emit(name, grad, ids, lg, li, rows, padding, width, ident, norm=None, flags=None, read=None):
    """read(position, lanes): optional gradient element reader (static-emitter storage tokens; see indexed_owner)."""
    if flags:
        raise ValueError("atomic_v1 does not produce row-activity flags")
    n, V, D = li.numel, rows, width
    G, I, OUT = ident(grad), ident(ids), ident(name)
    id_at = _unravel(li.shape, li.strides, "o")
    grad_base = _unravel(lg.shape[:-1], lg.strides[:-1], "o")
    sc = lg.strides[-1]
    vector_ok = sc == 1 and D % 4 == 0 and all(s % 4 == 0 for s, e in zip(lg.strides[:-1], lg.shape[:-1]) if e != 1)
    distinct = bool(norm) and DISTINCT_ROW_NORM
    flag_v, list_v, count_v = name + "__rowflag", name + "__rowlist", name + "__rowcount"
    auxiliary = [(flag_v, (V,), "float32"), (list_v, (n,), "float32"), (count_v, (1,), "float32")] if distinct else []
    FL, LS, CT = (f"reinterpret_cast<unsigned*>({ident(flag_v)})", f"reinterpret_cast<int*>({ident(list_v)})",
                  f"reinterpret_cast<unsigned*>({ident(count_v)})")

    def body(w):
        T = "float4" if w == 4 else "float"
        DV = D // w
        zero = "make_float4(0.0f, 0.0f, 0.0f, 0.0f)" if w == 4 else "0.0f"
        if read is not None:
            load = read("gb_ + 4LL * (long long)c_", 4) if w == 4 else read(f"gb_ + (long long)c_ * {sc}LL", 1)
        else:
            load = (f"reinterpret_cast<const float4*>({G} + gb_)[c_]" if w == 4
                    else f"{G}[gb_ + (long long)c_ * {sc}LL]")
        claim = (f"""
      if (c_ == 0 && atomicExch({FL} + r_, 1u) == 0u) {LS}[atomicAdd({CT}, 1u)] = (int)r_;""" if distinct else "")
        zero_aux = (f"""
  for (long long e = tid_; e < {V}LL; e += gs_) {FL}[e] = 0u;
  if (tid_ == 0) {CT}[0] = 0u;""" if distinct else "")
        if distinct:
            squares = f"""    const long long rows_ = (long long)__ldcg({CT});
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < rows_ * {DV}LL;
         e += (long long)gridDim.x * blockDim.x)
      sq_ = __fadd_rn(sq_, at_sq(__ldcg(out_ + (long long)__ldcg({LS} + e / {DV}LL) * {DV}LL + e % {DV}LL)));"""
        else:
            squares = f"""    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < {V * DV}LL;
         e += (long long)gridDim.x * blockDim.x) sq_ = __fadd_rn(sq_, at_sq(__ldcg(out_ + e)));"""
        norm_code = ""
        if norm:
            norm_code = f"""
  grid.sync();
  {{
    __shared__ float at_wsq_[8];
    __shared__ unsigned at_last_;
    float sq_ = 0.0f;
{squares}
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sq_ = __fadd_rn(sq_, __shfl_xor_sync(0xffffffffu, sq_, o));
    if ((threadIdx.x & 31) == 0) at_wsq_[threadIdx.x >> 5] = sq_;
    __syncthreads();
    if (threadIdx.x == 0) {{
      float s = 0.0f;
      for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s = __fadd_rn(s, at_wsq_[w]);
      (/*P2_PART*/)[blockIdx.x] = s;
      __threadfence();
      at_last_ = atomicAdd(/*P2_CNT*/, 1u) == gridDim.x - 1u;
    }}
    __syncthreads();
    if (at_last_ && threadIdx.x < 32) {{
      __threadfence();
      float s = 0.0f;
      for (int b = threadIdx.x; b < (int)gridDim.x; b += 32) s = __fadd_rn(s, __ldcg((/*P2_PART*/) + b));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, o));
      if (threadIdx.x == 0) {{ {ident(norm)}[0] = s; *(/*P2_CNT*/) = 0u; }}
    }}
    __syncthreads();
  }}"""
        return f"""{{
  {T}* out_ = reinterpret_cast<{T}*>({OUT});
  const long long tid_ = (long long)blockIdx.x * blockDim.x + threadIdx.x, gs_ = (long long)gridDim.x * blockDim.x;
  /*AT_ZERO_BEGIN*/for (long long e = tid_; e < {V * DV}LL; e += gs_) out_[e] = {zero};{zero_aux}
  grid.sync();/*AT_ZERO_END*/
  for (long long t = tid_; t < {n * DV}LL; t += gs_) {{
    const int o = (int)(t / {DV}LL), c_ = (int)(t % {DV}LL);
    const long long r_ = {I}[{id_at}];
    assert(r_ >= 0 && r_ < {V}LL);
    if (r_ != {padding}) {{
      const long long gb_ = {grad_base};
      atomicAdd(out_ + r_ * {DV}LL + c_, {load});{claim}
    }}
  }}{norm_code}
}}"""

    if vector_ok:
        g_addr = read("0", 0) if read is not None else f"(unsigned long long)({G})"
        code = (f"if ((({g_addr} | (unsigned long long)({OUT})) & 15ULL) == 0ULL) {body(4)} "
                f"else {body(1)}")
    else:
        code = body(1)
    prezero = [{"value": name, "elements": V * D}]
    if distinct:
        prezero += [{"value": flag_v, "elements": V}, {"value": count_v, "elements": 1}]
    phase = dict(kind="embedding_gradient",
                 writes=[name] + ([norm] if norm else []) + ([flag_v, list_v, count_v] if distinct else []),
                 reads=[grad, ids], code=code, shared_bytes=0, internal_grid_barriers=2 if norm else 1,
                 # The static emitter may zero these values at the end of the previous phase instead (their storage
                 # then lives from that phase) and drop the zero stage and its barrier from this body.
                 prezero={"values": prezero, "begin": "/*AT_ZERO_BEGIN*/", "end": "/*AT_ZERO_END*/"},
                 **({"p2_split": {"counter_words": 1, "partial_words": 1024}, "fused_norm": norm,
                     "norm_arithmetic": ("per-thread squares of the output rows that received occurrences (claimed row "
                                         "list) " if distinct else "per-thread squares of grid-strided output elements ")
                                        + "; CTA warp butterflies, warp partials in order; last-arriver sum of CTA "
                                          "partials in CTA order"}
                    if norm else {}),
                 reduction_elements="global atomic accumulation", accumulation_dtype="float32",
                 indexed_engine="atomic_v1", indexed_reduction_policy="atomic_v1",
                 arithmetic_policy="atomic_fp32_unordered_v1")
    decision = dict(engine="atomic_v1", vector_columns=vector_ok, shared_bytes=0, reused_occurrences=False,
                    arithmetic_policy="atomic_fp32_unordered_v1", deterministic=False, distinct_row_norm=distinct,
                    accumulation_order="hardware arrival order (float atomics); not bitwise repeatable",
                    auxiliary=auxiliary)
    return [phase], decision
