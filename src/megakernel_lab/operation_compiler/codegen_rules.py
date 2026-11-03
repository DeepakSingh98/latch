"""Shared exact reduction rules from P-OPS-D (775505d79b1d).

Used by both flat fusion and staged region lowering. Replays the original CTA
butterfly, including each leaf's addition to +0; no change to accumulation dtype.
"""


def _butterfly0(lanes):
    """Lane 0 of the xor butterfly (s101h::warp_sum) over 32 lane expressions; None lanes hold +0.0f.

    a + (+0) == a exactly for every a that is not -0; leaves enter as (0.0f + leaf) and sums of such values are
    never -0, so pruning the zero lanes keeps the frozen tree's bits."""
    lanes = list(lanes) + [None] * (32 - len(lanes))
    for offset in (16, 8, 4, 2, 1):
        lanes = [a if b is None else b if a is None else f'({a} + {b})'
                 for a, b in zip(lanes, (lanes[i ^ offset] for i in range(32)))]
    return lanes[0]


def _butterfly_sub(values):
    """xor butterfly (lane 0) over 8 values = lanes q, q+4, ..., q+28 of a 32-lane butterfly (offsets 16, 8, 4)."""
    lanes = list(values)
    for offset in (4, 2, 1):
        lanes = [a if b is None else b if a is None else f'({a} + {b})'
                 for a, b in zip(lanes, (lanes[i ^ offset] for i in range(8)))]
    return lanes[0]


# LATCH-STEP-04: per-warp engines finish on lane 0 in registers. s101h::warp_sum is an xor butterfly, so
# lane 0 already holds each canonical warp partial; the frozen second stage (partials on lanes 0..7, +0
# elsewhere, xor butterfly, lane 0) is _butterfly0 over them. Partials of warps past the extent are +0 and
# are pruned exactly (no partial is -0), so their butterflies are not evaluated. Bitwise identical.
PRUNED_WARP_TREE = True


def _warp_second_stage(live, out, rounded, scale):
    if not PRUNED_WARP_TREE:
        return f'''#pragma unroll
  for (int w = 0; w < 8; ++w) v[w] = s101h::warp_sum(v[w]);
  float second = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) {{
    const float partial = __shfl_sync(0xffffffffu, v[w], 0);
    if (lane == w) second = partial;
  }}
  second = s101h::warp_sum(second);
  if (lane == 0) {out}[o] = {rounded('second' + scale)};'''
    second = _butterfly0([f"v[{w}]" for w in range(live)])
    return f'''#pragma unroll
  for (int w = 0; w < {live}; ++w) v[w] = s101h::warp_sum(v[w]);
  if (lane == 0) {{ const float second = {second}; {out}[o] = {rounded('second' + scale)}; /*RED_EPILOGUE*/ }}'''


def batched_results(extent, results, threads=132 * 256):
    """accumulation_rules 'rilp': results per thread and grid-stride round for the small-extent thread-per-result tree.

    With one 256-thread CTA per SM, a thread that loads only its own result's leaves before storing keeps 1-8 loads
    in flight (latency-bound). Aim for roughly 16-32 independent leaf loads per round."""
    if not 0 < extent <= 16:
        return 1
    batch = 8 if extent <= 3 else 4 if extent <= 8 else 2
    while batch > 1 and results < batch * threads:
        batch //= 2
    return batch


def exact_reduction(*, out, position, extent, results, read, rounded, scale='', wide=False,
                    coalesced_threads=False, quad_threads=False, batch=1, vector_read=None):
    if quad_threads and coalesced_threads and 32 < extent <= 256 and results >= 1024 and results % 8 == 0:
        # P6-B: four threads per result (a lane quad). Lane q of the quad evaluates, for each warp partial, the
        # sub-butterfly over the leaves of lanes q, q+4, ..., q+28 (offsets 16, 8, 4 of the frozen butterfly); two
        # xor shuffles (offsets 2, 1) finish each partial exactly as the frozen tree does ((A0 + A2) + (A1 + A3) on
        # lane 0), then the second-stage butterfly over the partials. Absent leaves are +0 (exact: no partial is
        # -0); out-of-range rows are not read.
        lines, partials = [], []
        for w in range(8):
            values = []
            for k in range(8):
                row = 32 * w + 4 * k
                if row >= extent:
                    values.append(None)
                    continue
                leaf = read(position.replace("(r / ", f"(({row}LL + (long long)q) / "))
                values.append(f'(0.0f + {leaf})' if row + 3 < extent else
                              f'({row}u + q < {extent}u ? (0.0f + {leaf}) : 0.0f)')
            if all(v is None for v in values):
                partials.append(None)
                continue
            lines.append(f'float acc_w{w} = {_butterfly_sub(values)};')
            lines.append(f'acc_w{w} = acc_w{w} + __shfl_xor_sync(0xffffffffu, acc_w{w}, 2);')
            lines.append(f'acc_w{w} = acc_w{w} + __shfl_xor_sync(0xffffffffu, acc_w{w}, 1);')
            partials.append(f'acc_w{w}')
        body = "\n  ".join(lines)
        code = f'''for (long long latch_t = (long long)blockIdx.x * blockDim.x + threadIdx.x; latch_t < {4 * results}LL; latch_t += (long long)gridDim.x * blockDim.x) {{
  const long long o = latch_t >> 2; const unsigned q = (unsigned)latch_t & 3u;
/*P6B_BODY*/
  {body}
  const float acc = {_butterfly0(partials)};
  if (q == 0u) {{ {out}[o] = {rounded('acc' + scale)}; /*RED_EPILOGUE*/ }}
}}'''
        return code, {'engine': 'exact_cta_tree_per_thread_column', 'reduction_elements': extent}
    if coalesced_threads and 32 < extent <= 256 and results >= 1024:
        # P6-B: one thread per result, consecutive threads on consecutive (unit-stride) results, replaying the
        # frozen 256-thread CTA tree symbolically: per-warp xor butterflies over leaves 32w..32w+31, then the
        # second-stage butterfly over the eight warp partials. Bitwise identical to exact_cta_tree_per_warp, but
        # every leaf load is coalesced across the warp (the reduced axis is strided, the kept axis contiguous).
        partials = []
        for w in range(8):
            leaves = [f'(0.0f + {read(position.replace("(r / ", f"({r}LL / ").replace("(unsigned)(r)", f"(unsigned)({r}LL)"))})'
                      for r in range(32 * w, min(32 * w + 32, extent))]
            partials.append(_butterfly0(leaves) if leaves else None)
        acc = _butterfly0(partials)
        code = f'''for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x; o < {results}LL; o += (long long)gridDim.x * blockDim.x) {{
  const float acc = {acc};
  {out}[o] = {rounded('acc' + scale)}; /*RED_EPILOGUE*/
}}'''
        return code, {'engine': 'exact_cta_tree_per_thread_column', 'reduction_elements': extent}
    if wide and 256 < extent <= 2048 and results >= 128:
        # P6-C: one warp per result replays the 256-thread CTA fallback exactly: CTA thread
        # t = 32 w + lane accumulates acc = 0.0f; acc += x[t]; acc += x[t + 256]; ... in that
        # order, then the identical warp butterflies and second stage. Only which threads
        # evaluate the tree changes (one warp instead of one CTA per result).
        leaf = read(position.replace('(r / ', '(rr / ').replace('(unsigned)(r)', '(unsigned)(rr)'))
        passes = -(-extent // 256)
        folds = '\n'.join(f'    {{ const long long rr = {256 * p}LL + 32 * w + lane; if (rr < {extent}LL) v[w] = v[w] + {leaf}; }}'
                          for p in range(1, passes))
        code = f'''for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; o < {results}LL; o += ((long long)gridDim.x * blockDim.x) >> 5) {{
  const int lane = threadIdx.x & 31;
  float v[8];
#pragma unroll
  for (int w = 0; w < 8; ++w) {{
    {{ const long long rr = 32 * w + lane; v[w] = 0.0f + {leaf}; }}
{folds}
  }}
{_warp_second_stage(8 if passes else -(-extent // 32), out, rounded, scale)}
}}'''
        return code, {'engine': 'exact_cta_tree_per_warp', 'reduction_elements': extent, 'folded_passes': passes}
    if 32 < extent <= 256 and results >= 128:
        passes = 0
        leaf = read(position.replace('(r / ', '(rr / ').replace('(unsigned)(r)', '(unsigned)(rr)'))
        code = f'''for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; o < {results}LL; o += ((long long)gridDim.x * blockDim.x) >> 5) {{
  const int lane = threadIdx.x & 31;
  float v[8];
#pragma unroll
  for (int w = 0; w < 8; ++w) {{
    const long long rr = 32 * w + lane;
    v[w] = rr < {extent}LL ? 0.0f + {leaf} : 0.0f;
  }}
{_warp_second_stage(8 if passes else -(-extent // 32), out, rounded, scale)}
}}'''
        return code, {'engine': 'exact_cta_tree_per_warp', 'reduction_elements': extent}
    if 0 < extent <= 32:
        lanes = [f'(0.0f + {read(position.replace("(r / ", f"({i}LL / ").replace("(unsigned)(r)", f"(unsigned)({i}LL)"))})'
                 for i in range(extent)] + [None] * (32 - extent)
        for offset in (16, 8, 4, 2, 1):
            lanes = [a if b is None else b if a is None else f'({a} + {b})'
                     for a, b in zip(lanes, (lanes[i ^ offset] for i in range(32)))]
        if batch > 1 and vector_read is not None and results % 4 == 0:
            # accumulation_rules 'rilp' float4 form: lower.py admits it only when four consecutive results read four
            # consecutive, 16-byte aligned FP32 leaves (unit-stride innermost kept axis of extent % 4 == 0, every other
            # stride and the offsets multiples of 4) and store to a contiguous aligned FP32 result. Component c of
            # leaf load r is exactly the scalar leaf r of result o + c; each component evaluates the unchanged tree
            # and rounding. It avoids the scalar batch's issue cost (64-bit index arithmetic per leaf).
            groups = max(1, batch // 2)
            q4 = results // 4
            pos_of = lambda i: position.replace("(r / ", f"({i}LL / ").replace("(unsigned)(r)", f"(unsigned)({i}LL)")
            loads = "\n".join(f"    const float4 l{i}_ = {vector_read(pos_of(i))};" for i in range(extent))

            def tree(c):
                lanes_c = [f'(0.0f + l{i}_.{c})' for i in range(extent)] + [None] * (32 - extent)
                for offset in (16, 8, 4, 2, 1):
                    lanes_c = [a if b is None else b if a is None else f'({a} + {b})'
                               for a, b in zip(lanes_c, (lanes_c[i ^ offset] for i in range(32)))]
                return lanes_c[0]
            stores = ", ".join(rounded(f"acc_[u_].{c}" + scale) for c in "xyzw")
            code = f'''for (long long gb_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; gb_ < {q4}LL; gb_ += {groups}LL * (long long)gridDim.x * blockDim.x) {{
  const long long s_ = (long long)gridDim.x * blockDim.x;
  float4 acc_[{groups}];
#pragma unroll
  for (int u_ = 0; u_ < {groups}; ++u_) {{
    const long long o = 4 * (gb_ + u_ * s_ < {q4}LL ? gb_ + u_ * s_ : gb_);
{loads}
    acc_[u_] = make_float4({tree("x")}, {tree("y")}, {tree("z")}, {tree("w")});
  }}
#pragma unroll
  for (int u_ = 0; u_ < {groups}; ++u_) {{
    const long long g_ = gb_ + u_ * s_;
    if (g_ < {q4}LL) reinterpret_cast<float4*>({out})[g_] = make_float4({stores});
  }}
}}'''
            return code, {'engine': 'exact_cta_tree_per_thread', 'reduction_elements': extent, 'batched_results': 4 * groups,
                          'vector4': True}
        if batch > 1:
            # accumulation_rules 'rilp': each thread evaluates the identical tree for `batch` grid-stride results
            # (o, o + s, ...) with every leaf load issued before any store. Same leaves, same tree, same rounding and
            # the same single store per result (bitwise equal). Out-of-range slots recompute a valid result and are
            # never stored. Loading later results before storing earlier ones requires that no stored element is a
            # leaf of this phase; lower.py restores the unbatched body for any phase whose result storage root is
            # also one of its read roots after storage aliasing.
            code = f'''for (long long ob_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; ob_ < {results}LL; ob_ += {batch}LL * (long long)gridDim.x * blockDim.x) {{
  const long long s_ = (long long)gridDim.x * blockDim.x;
  float acc_[{batch}];
#pragma unroll
  for (int u_ = 0; u_ < {batch}; ++u_) {{
    const long long o = ob_ + u_ * s_ < {results}LL ? ob_ + u_ * s_ : ob_;
    acc_[u_] = {lanes[0]};
  }}
#pragma unroll
  for (int u_ = 0; u_ < {batch}; ++u_) {{
    const long long o = ob_ + u_ * s_;
    if (o < {results}LL) {{ const float acc = acc_[u_]; {out}[o] = {rounded('acc' + scale)}; }}
  }}
}}'''
            return code, {'engine': 'exact_cta_tree_per_thread', 'reduction_elements': extent, 'batched_results': batch}
        code = f'''for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x; o < {results}LL; o += (long long)gridDim.x * blockDim.x) {{
  const float acc = {lanes[0]};
  {out}[o] = {rounded('acc' + scale)}; /*RED_EPILOGUE*/
}}'''
        return code, {'engine': 'exact_cta_tree_per_thread', 'reduction_elements': extent}
    return None
