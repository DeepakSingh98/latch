"""Bounded active extents over capacity storage (one binary, one launch ABI).

The capture contract names axes by value identity, never by integer coincidence.
Static allocation and strides remain in the ordinary planner. Only live iteration
spaces and reduction denominators depend on the device scalar.
"""
from __future__ import annotations

import math


def scalar(value, *, integer=False):
    from .lower import LoweringError
    if not isinstance(value, dict) or set(value) != {"dynamic_batch_scalar"}:
        return None
    spec = value["dynamic_batch_scalar"]
    coefficient, power = spec["coefficient"], spec["power"]
    if not math.isfinite(coefficient) or power not in (-1, 1):
        raise LoweringError("invalid bounded batch scalar")
    if integer or "ceil_divisor" in spec:
        if int(coefficient) != coefficient or power != 1:
            raise LoweringError("integer batch scalar must be an integral multiple of B")
        expression = f"({int(coefficient)}ULL * (unsigned long long)latch_active_batch)"
        divisor = spec.get("ceil_divisor", 1)
        if type(divisor) is not int or divisor < 1:
            raise LoweringError("invalid dynamic ceil divisor")
        return f"(({expression} + {divisor - 1}ULL) / {divisor}ULL)" if divisor != 1 else expression
    # Python shape-derived scalars are folded in double precision before ATen's
    # FP32 scalar conversion. Round once after the runtime extent arithmetic;
    # float(C) * float(B) would change, for example, the literal 0.1 * 7.
    return f"((float)({float(coefficient)!r} {'*' if power == 1 else '/'} (double)latch_active_batch))"


class ActiveExtents:
    def __init__(self, declaration, values):
        from .lower import LoweringError, _field
        self.declaration = declaration
        self.axes = declaration["value_axes"]
        self.storage_axes = {}  # compiler-owned temporaries, separate from captured value provenance
        self.maximum = declaration["maximum"]
        self.input = declaration["input"]
        if type(self.maximum) is not int or not 2 <= self.maximum <= 32:
            raise LoweringError("invalid bounded batch capacity")
        for name, axis in self.axes.items():
            shape = _field(values[name], "shape")
            if type(axis) is not int or not 0 <= axis < len(shape) or shape[axis] != self.maximum:
                raise LoweringError(f"{name}: active axis must have the declared capacity")

    def numel(self, name, shape):
        axis = self.axes.get(name)
        if axis is None:
            return f"{math.prod(shape)}LL"
        return f"((long long)latch_active_batch * {math.prod(shape) // shape[axis]}LL)"

    def material(self, name, base):
        from .lower import LoweringError, _VIEW_BASES
        if name in self.axes and self.axes[name] != 0 and base not in _VIEW_BASES:
            raise LoweringError(f"{name}: bounded batch supports leading-axis material tensors; "
                                "nonleading batch views are admitted as matrix/reduction operands")

    def matrix(self, a, b, name, layouts, dtypes, *, split_k=True, target_arch="sm_90"):
        """Retain the existing masked BF16 MMA body and bounded FP32 tile body."""
        from .lower import LoweringError, _ident
        la, lb, out = layouts[a], layouts[b], layouts[name]
        if len(la.shape) != 2 or len(lb.shape) != 2:
            raise LoweringError("bounded batch matrix operands must be rank two")
        m, k = la.shape
        kb, n = lb.shape
        if k != kb or out.shape != (m, n):
            raise LoweringError("bounded batch matrix shape mismatch")
        aa, ba, oa = self.axes.get(a), self.axes.get(b), self.axes.get(name)
        if (aa == 1) != (ba == 0):
            raise LoweringError("both matrix reduction operands must share their active K extent")
        if aa == 0 and ba == 1 or (aa == 0 and oa != 0) or (ba == 1 and oa != 1):
            raise LoweringError("matrix output active-axis contract mismatch")
        runtime = lambda d, active: "latch_active_batch" if active else str(d)
        M, N, K = runtime(m, aa == 0), runtime(n, ba == 1), runtime(k, aa == 1)
        sam, sak = la.strides
        sbk, sbn = lb.strides
        dtype = dtypes[name]
        if dtypes[a] != dtype or dtypes[b] != dtype or dtype not in ("float32", "bfloat16"):
            raise LoweringError("bounded batch matrices need matching FP32/BF16 operands and result")
        align = 8 if dtype == "bfloat16" else 4
        aligned = ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)
                   and all(x % align == 0 for x in (sam if sak == 1 else sak,
                       sbn if sbk == 1 else sbk, la.offset, lb.offset)))
        split = None
        if split_k and aligned and dtype == "float32" and aa == 0 and ba is None:
            # K and the feature geometry are static. Reuse the established
            # capacity planner's fixed split count; there is no plan per B.
            from .matrix_select import plan
            selected, _ = plan(a, b, name, la, lb, out.shape, ident=_ident,
                               bodies=("popsb",), arch=target_arch)
            if selected and selected["metadata"]["ksplit"] > 1:
                split = selected["metadata"]["matrix_plan"]
        stage_metadata = {}
        if aligned:
            # An active K has at most 32 elements. The existing BF16 edge body
            # supports one k16 stage, so B<=16 must not execute a second,
            # entirely zero MMA block merely because capacity is 32.
            bm, bn, depth, stages = (16 if aa == 0 else 32), 64, 16 if dtype == "float32" or aa == 1 else 32, 3
            ak, bkc = str(sak == 1).lower(), str(sbk == 1).lower()
            if dtype == "bfloat16":
                wm, wn = bm // 16, 128 // bm
                tile_metadata = {}
                if aa == 1 and target_arch == "sm_90" and not (m % 32 or n % 32 or k % 32):
                    # M/N and capacity K are static. Preserve the existing
                    # whole-grid tile choice; a partition-specific larger tile
                    # would require a separate CTA grouping contract. Keep the
                    # runtime K16 stage law independently of that geometry.
                    from .matrix_rules import bf16_plan, TARGET_GRID
                    plan, _ = bf16_plan(m, n, k, 1, sak == 1, sbk == 1, grid=TARGET_GRID)
                    bm, bn, wm, wn = (plan[key] for key in ("bm", "bn", "wm", "wn"))
                    tile_metadata = {"capacity_tile_selection": {
                        key: plan[key] for key in ("bm", "bn", "wm", "wn", "tiles", "grid")},
                        "tile_policy": "existing whole-grid BF16 capacity planner; active KS16/NST3 retained"}
                # Existing P7-B/P6-E element-granular masked cp.async and MMA.
                call = f"popsc::bmm_edge_phase<{bm},{bn},{wm},{wn},{ak},{bkc},{depth},{stages},{str(ba != 1 and n % 2 == 0).lower()}>"
                struct, header, engine = "popsc::BArgs", "popsc_mm.cuh", "popsc_bf16_edge_dynamic"
                pad, width = 8, 2
                stage_metadata = {**tile_metadata, "stage_depth": depth, "pipeline_stages": stages,
                    "hardware_k16_blocks": "ceil(active_batch / 16)" if aa == 1 else math.ceil(k / depth) * (depth // 16)}
            else:
                call = f"latch_dynamic::smm_phase<{bm},{bn},{ak},{bkc},{depth},{stages}>"
                struct, header, engine = "popsc::SArgs", "dynamic_batch.cuh", "popsc_fp32_edge_dynamic"
                pad, width = 4, 4
            shared = stages * width * ((bm * (depth + pad) if sak == 1 else depth * (bm + pad))
                                      + (bn * (depth + pad) if sbk == 1 else depth * (bn + pad)))
            code = f"""{{
  {struct} g{{{_ident(a)}, {sam}LL, {sak}LL, 0LL, {_ident(b)}, {sbk}LL, {sbn}LL, 0LL,
               {_ident(name)}, {n}LL, {m*n}LL, {M}, {N}, {K}, 1}};
  {call}(g, smem);
}}"""
            headers = ["tgemm.cuh", "bgemm.cuh", "popsc_mm.cuh"] + ([header] if dtype == "float32" else [])
            if split:
                splits = split["ksplit"]
                scratch = name + "__dynamic_partials"
                self.storage_axes[scratch] = 0
                # Both row tiles use the same ascending-K calculation. The
                # larger tile reuses weights for two live row groups; small B
                # keeps one 16-row tile. Neither branch changes the split law.
                code = f"""{{
  popsc::SArgs g{{{_ident(a)}, {sam}LL, {sak}LL, 0LL, {_ident(b)}, {sbk}LL, {sbn}LL, 0LL,
                  {_ident(name)}, {n}LL, {m*n}LL, {M}, {N}, {K}, 1}};
  if (latch_active_batch <= 16)
    latch_dynamic::smm_split_phase<16,{bn},{ak},{bkc},{depth},{stages},{splits}>(g, {_ident(scratch)}, smem);
  else
    latch_dynamic::smm_split_phase<32,{bn},{ak},{bkc},{depth},{stages},{splits}>(g, {_ident(scratch)}, smem);
}}"""
                # The scratch is [capacity, split * N], with each slice's N
                # columns adjacent. Only the active leading rows are written
                # and read. A normal dependency edge publishes the partials.
                reduce_code = f"""for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
     i < (long long)latch_active_batch * {n}LL; i += (long long)gridDim.x * blockDim.x) {{
  const long long base = (i / {n}LL) * {splits*n}LL + i % {n}LL;
  float acc = {_ident(scratch)}[base];
  #pragma unroll
  for (int s = 1; s < {splits}; ++s) acc += {_ident(scratch)}[base + s * {n}LL];
  {_ident(name)}[i] = acc;
}}"""
                shared = stages * width * ((32 * (depth + pad) if sak == 1 else depth * (32 + pad))
                                          + (bn * (depth + pad) if sbk == 1 else depth * (bn + pad)))
                return code, headers, {
                    "engine": "popsc_fp32_split_edge_dynamic", "tiles": math.ceil(m / 32) * math.ceil(n / bn) * splits,
                    "shared_bytes": shared, "matrix_shape": [m, n, k], "batch": 1, "ksplit": splits,
                    "static_capacity_plan": split, "row_tile_policy": "16 rows for B<=16, otherwise 32 rows",
                    "active_extents": {"m": True, "n": False, "k": False}, "work_extent": [M, N, K],
                    "arithmetic": "ascending-K FP32 fmaf within fixed slices; ascending slice sum; no TF32/data atomics",
                    "bounded_edges": "inactive operands not loaded; inactive partials and results not stored",
                    "split_storage": {"name": scratch, "shape": (m, splits*n), "active_axis": 0},
                    "split_reduce_code": reduce_code}
        else:
            # The existing ragged 16x16 SIMT body with runtime loop limits.
            # Fixed ascending K, including the partial last K tile; no padding reads.
            cvt = lambda text: f"__bfloat162float({text})" if dtype == "bfloat16" else text
            result = "__float2bfloat16_rn(acc)" if dtype == "bfloat16" else "acc"
            code = f"""{{
 const int mt = ({M} + 15) / 16, nt = ({N} + 15) / 16;
 for (int tile = blockIdx.x; tile < mt * nt; tile += gridDim.x) {{
  const int row = (tile / nt) * 16 + threadIdx.x / 16;
  const int col = (tile % nt) * 16 + threadIdx.x % 16;
  float acc = 0.0f;
  for (int kk = 0; kk < {K}; kk += 16) {{
   const int ak = kk + threadIdx.x % 16, bk = kk + threadIdx.x / 16;
   smem[threadIdx.x] = row < {M} && ak < {K} ? {cvt(f'{_ident(a)}[row * {sam}LL + ak * {sak}LL]')} : 0.0f;
   smem[256 + threadIdx.x] = bk < {K} && col < {N} ? {cvt(f'{_ident(b)}[bk * {sbk}LL + col * {sbn}LL]')} : 0.0f;
   __syncthreads();
   for (int z = 0; z < 16 && kk + z < {K}; ++z)
    acc = fmaf(smem[(threadIdx.x / 16) * 16 + z], smem[256 + z * 16 + threadIdx.x % 16], acc);
   __syncthreads();
  }}
  if (row < {M} && col < {N}) {_ident(name)}[row * {n}LL + col] = {result};
 }}
}}"""
            bm = bn = 16
            shared, headers, engine = 2048, [], "generic_simt_16_dynamic"
        return code, headers, {"engine": engine, "tiles": math.ceil(m / bm) * math.ceil(n / bn), **stage_metadata,
            "shared_bytes": shared, "matrix_shape": [m, n, k], "batch": 1,
            "active_extents": {"m": aa == 0, "n": ba == 1, "k": aa == 1},
            "arithmetic": ("explicit BF16 MMA; FP32 accumulation; one BF16 rounding" if dtype == "bfloat16" and aligned
                           else "exactly widened BF16 operands; ascending-K FP32 fmaf; one BF16 rounding" if dtype == "bfloat16"
                           else "ascending-K FP32 fmaf; no TF32"),
            "work_extent": [M, N, K], "bounded_edges": "inactive operands not loaded; inactive results not stored"}

    def reduction(self, name, source, src, dims, out_shape, dtype, output_dtype, mean, accumulate=False):
        from .lower import _ident
        axis = self.axes.get(source)
        kept = [i for i in range(len(src.shape)) if i not in dims]
        extent = lambda axes: (f"((long long)latch_active_batch * {math.prod(src.shape[i] for i in axes if i != axis)}LL)"
                              if axis in axes else f"{math.prod(src.shape[i] for i in axes)}LL")
        terms = []
        for axes, index in ((dims, "r"), (kept, "o")):
            for j, a in enumerate(axes):
                if src.strides[a] == 0:
                    continue
                size = "latch_active_batch" if a == axis else str(src.shape[a])
                tail = extent(axes[j+1:])
                coordinate = index if tail == "1LL" else f"({index} / {tail})"
                # The index is bounded by the product of these axes, so its
                # first coordinate already lies below the first extent. In
                # particular, no runtime 64-bit remainder by B is necessary.
                if j:
                    coordinate = f"({coordinate} % {size})"
                terms.append(coordinate if src.strides[a] == 1 else f"{coordinate} * {src.strides[a]}LL")
        # A contiguous full reduction visits its leaves in flat order. Keep
        # that same order without reconstructing coordinates (and % B) for
        # every leaf. Transposed views deliberately retain their logical order.
        address = "r" if src.contiguous and list(dims) == list(range(len(src.shape))) else " + ".join(terms) or "0"
        read = f"{_ident(source)}[{address}]"
        if dtype == "bfloat16":
            read = f"__bfloat162float({read})"
        scale = f" / (float){extent(dims)}" if mean else ""
        result = f"acc{scale}"
        if accumulate:
            result = f"{_ident(name)}[o] + ({result})"
        if output_dtype == "bfloat16":
            result = f"__float2bfloat16_rn({result})"
        return f"""for (long long o = blockIdx.x; o < {extent(kept)}; o += gridDim.x) {{
  float acc = 0.0f;
  for (long long r = threadIdx.x; r < {extent(dims)}; r += blockDim.x) acc += {read};
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {_ident(name)}[o] = {result};
  }}
  __syncthreads();
}}"""
