# P-OPS2-D adaptation from 04ca670a17bb51ca6123a5c0d9a1002a27fbc83c.
"""Generic ATen indexing and row-normalization phases used by transformer graphs.

These are operation emitters, not model or backward-schedule recognizers. ATen AD
selects the backward operations. Indexed reductions use fixed-order gathers rather
than floating atomics, including repeated embedding IDs and scatter destinations.
"""
from __future__ import annotations

import os
import math


P6E_VEC_GATHER = False      # P6-E lowering option p6e_gather (set per lowering by fusion.lower_fused)
P9_EMB_PAD = False          # P9-A: zero-padded 32-load batches in the ordered embedding gradient (bitwise)
P9_EMB_TAIL8 = False        # P9-A: zero-padded 8-load batches for the remainder of the ordered embedding gradient (bitwise)
P9_EMB_TAIL = False         # P9-A: predicated tail batch in the ordered embedding gradient (bitwise)
P9C_VEC_GATHER = True       # P9-C: batched exact 16-byte row gathers
P6E_EMB_PIPE = False        # P6-E lowering option p6e_emb_pipe: next occurrence batch loaded while one is summed
P6E_SCATTER_VEC = False     # P6-E lowering option p6e_scatter_vec: 4 outputs per thread for last-axis scalar scatter
# LATCH-MOESUP-ROW: 32-wide rows (e.g. router softmax over 32 experts) run one row per thread: the thread replays,
# in registers, lane 0's xor-butterfly of the warp-per-row form (pairs (i, i + o) for o = 16 .. 1, own value first,
# as s101h::warp_sum/warp_max) and the second stage's identity steps, so every row's statistic and output are
# bitwise those of the warp form, without 45 shuffles per row. Option: bitwise but neutral on the 25M MoE+sup update
# (router softmax rows are latency-cheap; aux +0.26 / plain -0.09 ms, scr-sm), so off.
SOFTMAX_LANE_ROWS = False
P6E_SOFTMAX_ROWS = 0        # P6-E lowering option p6e_softmax_rows: rows per warp iteration of the warp-row softmax


# LATCH-MOESUP: warp-per-row argmax for short rows (exact; False restores one CTA per row).
ARGMAX_WARP_ROWS = True
ARGMAX_WARP_MAX_WIDTH = 256

def emit_operation(target, args, kwargs, name, shape, layouts, dtypes, names, read, warp_rows=False, inputs=()):
    from .lower import LoweringError, _ident, _index, _ref

    if target.startswith('latch_ops.vocab_'):
        from .vocab_head import emit as emit_vocabulary
        return emit_vocabulary(target, args, kwargs, name, shape, layouts, dtypes, names, read)
    if target == 'latch_ops.topk_rounds.default':
        from .topk_rounds import emit as emit_topk_rounds
        return emit_topk_rounds(target, args, kwargs, name, shape, layouts, dtypes, names, read)

    from .extra_ops import emit as emit_extra
    extra = emit_extra(target, args, kwargs, name, shape, layouts, dtypes, names, read, input_names=inputs)
    if extra is not None:
        return extra

    from .conv_ops import emit as emit_conv
    conv = emit_conv(target, args, kwargs, name, shape, layouts, dtypes, names)
    if conv is not None:
        return conv

    base = target.split('.')[0]
    from .standard_ops import emit as emit_standard
    emitted = emit_standard(target, args, kwargs, name, shape, layouts, dtypes, names, read)
    if emitted is not None:
        return emitted
    supported = {'embedding', 'embedding_dense_backward', 'gather', 'scatter_add', 'scatter',
                 '_softmax', '_log_softmax', '_softmax_backward_data',
                 '_log_softmax_backward_data', 'argmax', 'logsumexp', '_safe_softmax'}
    if base not in supported:
        return None
    ref = lambda i: _ref(args[i], names)
    out = _ident(name)
    numel = math.prod(shape)
    stride_loop = f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {numel}LL; i += (long long)gridDim.x * blockDim.x)'
    helpers = ''

    def require_float(*values):
        if any(dtypes[v] != 'float32' for v in values):
            raise LoweringError(f'{target}: FP32 operands/results required')

    def require_index(value):
        if dtypes[value] != 'int64':
            raise LoweringError(f'{target}: int64 indices required')

    if base == 'embedding':
        weight, ids = ref(0), ref(1)
        lw, li = layouts[weight], layouts[ids]
        # LATCH-MOESUP: BF16 tables (e.g. after ops/gather_cast.py) gather BF16 rows; FP32 as before.
        if dtypes[weight] not in ('float32', 'bfloat16') or dtypes[name] != dtypes[weight]:
            raise LoweringError(f'{target}: FP32 or BF16 table and matching result required')
        require_index(ids)
        per = 4 if dtypes[weight] == 'float32' else 8   # elements per 16-byte chunk
        if len(lw.shape) != 2 or shape != (*li.shape, lw.shape[1]):
            raise LoweringError('embedding requires rank-2 weights and indices-shaped rows')
        if bool(args[3] if len(args) > 3 else kwargs.get('scale_grad_by_freq', False)):
            raise LoweringError('embedding frequency-scaled gradients are not admitted')
        if bool(args[4] if len(args) > 4 else kwargs.get('sparse', False)):
            raise LoweringError('sparse embedding gradients are not admitted')
        width = lw.shape[1]
        numel = 1
        for d in shape:
            numel *= d
        if ((P6E_VEC_GATHER or (P9C_VEC_GATHER and os.environ.get('P9C_EMB', '1') != '0')) and tuple(lw.strides) == (width, 1) and width % per == 0 and li.contiguous
                and not li.offset and numel < 2 ** 31):
            # P6-E: 16-byte row chunks with 32-bit chunk arithmetic (the same element copies, index check per chunk).
            # P9-C: eight chunks per thread per iteration, every index and row load issued before the stores (the
            # same copies; one index -> row dependent latency per eight chunks instead of per chunk).
            code = f'''for (unsigned c0 = blockIdx.x * blockDim.x + threadIdx.x; c0 < {numel // per}u; c0 += 8u * gridDim.x * blockDim.x) {{
  int64_t row_[8]; uint4 val_[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) {{
    const unsigned c = c0 + (unsigned)u * gridDim.x * blockDim.x;
    row_[u] = c < {numel // per}u ? {_ident(ids)}[c / {width // per}u] : 0;
  }}
#pragma unroll
  for (int u = 0; u < 8; ++u) {{
    const unsigned c = c0 + (unsigned)u * gridDim.x * blockDim.x;
    assert(row_[u] >= 0 && row_[u] < {lw.shape[0]}LL);
    if (c < {numel // per}u) val_[u] = *reinterpret_cast<const uint4*>({_ident(weight)} + row_[u] * {width}LL + (c % {width // per}u) * {per}u);
  }}
#pragma unroll
  for (int u = 0; u < 8; ++u) {{
    const unsigned c = c0 + (unsigned)u * gridDim.x * blockDim.x;
    if (c < {numel // per}u) *reinterpret_cast<uint4*>({out} + {per}ull * c) = val_[u];
  }}
}}'''
            return 'embedding', code, {}, '#include <cassert>\n'
        index = f'{_ident(ids)}[{_index(li, li.shape, f"(i / {width}LL)")}]'
        code = f'''{stride_loop} {{
  const int64_t row = {index};
  assert(row >= 0 && row < {lw.shape[0]}LL);
  {out}[i] = {_ident(weight)}[row * {lw.strides[0]}LL + (i % {width}LL) * {lw.strides[1]}LL];
}}'''
        helpers = '#include <cassert>\n'
        return 'embedding', code, {}, helpers

    if base == 'embedding_dense_backward':
        grad, ids = ref(0), ref(1)
        lg, li = layouts[grad], layouts[ids]
        require_float(grad, name)
        require_index(ids)
        rows = int(args[2])
        padding = int(args[3] if len(args) > 3 else kwargs.get('padding_idx', -1))
        if bool(args[4] if len(args) > 4 else kwargs.get('scale_grad_by_freq', False)):
            raise LoweringError('embedding frequency-scaled gradients are not admitted')
        if len(shape) != 2 or shape[0] != rows or lg.shape != (*li.shape, shape[1]):
            raise LoweringError('embedding backward shape mismatch')
        width = shape[1]
        index = f'{_ident(ids)}[{_index(li, li.shape, "j")}]'
        grad_index = _index(lg, lg.shape, f'(j * {width}LL + col)')
        if P6E_VEC_GATHER and lg.contiguous and not lg.offset:
            grad_index = f'(j * {width}LL + col)'     # P6-E: same element, no generic div/mod index
        code = f'''{stride_loop} {{
  const long long row = i / {width}LL, col = i % {width}LL;
  float acc = 0.0f;
  if (row != {padding}LL) {{
    for (long long j = 0; j < {li.numel}LL; ++j)
      if ({index} == row) acc += {_ident(grad)}[{grad_index}];
  }}
  {out}[i] = acc;
}}'''
        return 'embedding_gradient', code, {'reduction_elements': li.numel}, helpers

    if base in {'gather', 'scatter_add', 'scatter'}:
        source, ids = ref(0), ref(2)
        ls, li = layouts[source], layouts[ids]
        require_float(source, name)
        require_index(ids)
        dim = int(args[1]) % len(shape)
        if len(ls.shape) != len(shape) or len(li.shape) != len(shape):
            raise LoweringError(f'{target}: operand ranks must agree')
        if base == 'gather':
            if shape != li.shape or any(shape[d] > ls.shape[d] for d in range(len(shape)) if d != dim):
                raise LoweringError('gather shape mismatch')
            if bool(args[3] if len(args) > 3 else kwargs.get('sparse_grad', False)):
                raise LoweringError('sparse gather gradients are not admitted')
            terms = [f'((i / {math.prod(shape[d+1:])}LL) % {shape[d]}LL) * {ls.strides[d]}LL'
                     for d in range(len(shape)) if d != dim]
            terms.append(f'index * {ls.strides[dim]}LL')
            code = f'''{stride_loop} {{
  const int64_t index = {_ident(ids)}[{_index(li, shape)}];
  assert(index >= 0 && index < {ls.shape[dim]}LL);
  {out}[i] = {_ident(source)}[{" + ".join(terms)}];
}}'''
            return 'gather', code, {}, '#include <cassert>\n'
        scalar_scatter = target == 'scatter.value'
        if base == 'scatter' and not scalar_scatter:
            raise LoweringError('only scalar-value scatter is admitted; duplicate tensor updates need an explicit deterministic contract')
        updates = None if scalar_scatter else ref(3)
        lu = li if scalar_scatter else layouts[updates]
        if not scalar_scatter:
            require_float(updates)
        if shape != ls.shape or li.shape != lu.shape or any(li.shape[d] > shape[d] for d in range(len(shape)) if d != dim):
            raise LoweringError('scatter_add requires equal index/update shapes and valid destination dimensions')
        terms_index, terms_update, guards = [], [], []
        for d in range(len(shape)):
            coord = 'j' if d == dim else f'((i / {math.prod(shape[d+1:])}LL) % {shape[d]}LL)'
            terms_index.append(f'{coord} * {li.strides[d]}LL')
            terms_update.append(f'{coord} * {lu.strides[d]}LL')
            if d != dim:
                guards.append(f'{coord} < {li.shape[d]}LL')
        destination = f'((i / {math.prod(shape[dim+1:])}LL) % {shape[dim]}LL)'
        if (P6E_SCATTER_VEC and scalar_scatter and dim == len(shape) - 1 and li.shape[dim] == 1
                and shape[dim] % 4 == 0 and numel % 4 == 0):
            # P6-E: last-axis scalar scatter, one index per row: four consecutive outputs of one row per thread
            # (shared row coordinates and index), one 16-byte store. Same values per element.
            W = shape[dim]
            index_at = " + ".join(t.replace('(i /', '(i0 /') for t in terms_index).replace('j *', '0LL *')
            guard = " && ".join(g.replace('(i /', '(i0 /') for g in guards) or "true"
            srcs = "\n  ".join(f'{{ const long long i = i0 + {k}; acc[{k}] = {read(args[0], shape)}; }}' for k in range(4))
            sets = " ".join(f'if (index == c0 + {k}LL) acc[{k}] = {read(args[3], shape)};' for k in range(4))
            code = f'''for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < {numel // 4}LL; q += (long long)gridDim.x * blockDim.x) {{
  const long long i0 = 4LL * q, c0 = i0 % {W}LL;
  float acc[4];
  {srcs}
  if ({guard}) {{
    const int64_t index = {_ident(ids)}[{index_at}];
    assert(index >= 0 && index < {W}LL);
    {{ const long long i = i0; (void)i; {sets} }}
  }}
  *reinterpret_cast<float4*>({out} + i0) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}}'''
            return 'scatter', code, {'reduction_elements': 1, 'vector': 4}, '#include <cassert>\n'
        value = read(args[3], shape) if scalar_scatter else f'{_ident(updates)}[{" + ".join(terms_update)}]'
        update = f'acc = {value};' if scalar_scatter else f'acc += {value};'
        code = f'''{stride_loop} {{
  float acc = {read(args[0], shape)};
  if ({" && ".join(guards) or "true"}) {{
    for (long long j = 0; j < {li.shape[dim]}LL; ++j) {{
      const int64_t index = {_ident(ids)}[{" + ".join(terms_index)}];
      assert(index >= 0 && index < {shape[dim]}LL);
      if (index == {destination}) {update}
    }}
  }}
  {out}[i] = acc;
}}'''
        return ('scatter' if scalar_scatter else 'scatter_add'), code, {'reduction_elements': li.shape[dim]}, '#include <cassert>\n'

    # C-OPS-I f121882295ecdcf094f8727e8d72f15fc51199ba: generic inference argmax.
    if base == 'argmax':
        # Exact row selection (no arithmetic): torch.argmax semantics, where NaN
        # is maximal and ties keep the first index. One CTA per output row.
        source = ref(0)
        ls = layouts[source]
        require_float(source)
        if dtypes[name] != 'int64':
            raise LoweringError('argmax requires an int64 result')
        dim = args[1] if len(args) > 1 else kwargs.get('dim')
        keep = bool(args[2] if len(args) > 2 else kwargs.get('keepdim', False))
        if dim is None or not ls.shape or int(dim) % len(ls.shape) != len(ls.shape) - 1:
            raise LoweringError('argmax currently requires an explicit last tensor dimension')
        if shape != (ls.shape[:-1] + (1,) if keep else ls.shape[:-1]):
            raise LoweringError('argmax result shape mismatch')
        width = ls.shape[-1]
        rows = math.prod(ls.shape[:-1])
        x = f'{_ident(source)}[{_index(ls, ls.shape, "(row * %dLL + k)" % width)}]'
        better = 'latch_argmax_better'
        helper = f'''#ifndef LATCH_ARGMAX_HELPERS
#define LATCH_ARGMAX_HELPERS
__device__ __forceinline__ bool {better}(float v, long long i, float b, long long j) {{
  if (isnan(v)) return !isnan(b) || i < j;
  if (isnan(b)) return false;
  return v > b || (v == b && i < j);
}}
#endif
'''
        if ARGMAX_WARP_ROWS and width <= ARGMAX_WARP_MAX_WIDTH and rows >= 256:
            # LATCH-MOESUP: short rows (e.g. expert-choice selection over 32 sequences) take one warp per row
            # instead of one CTA with two block barriers. Lane l scans k = l, l + 32, ...; an xor butterfly combines
            # lanes. The result is the unique maximum of the same total order (NaN maximal, then value, then the
            # lowest index), so the selected index is identical to the CTA form for every input.
            code = f'''for (long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < {rows}LL; row += ((long long)gridDim.x * blockDim.x) >> 5) {{
  const int lane = threadIdx.x & 31;
  float best = -INFINITY; long long index = 0x7fffffffffffffffLL;
  for (long long k = lane; k < {width}LL; k += 32) {{
    const float v = {x};
    if ({better}(v, k, best, index)) {{ best = v; index = k; }}
  }}
  for (int delta = 16; delta > 0; delta /= 2) {{
    const float v = __shfl_xor_sync(0xffffffffU, best, delta);
    const long long k = __shfl_xor_sync(0xffffffffU, index, delta);
    if ({better}(v, k, best, index)) {{ best = v; index = k; }}
  }}
  if (lane == 0) {out}[row] = static_cast<int64_t>(index);
}}'''
            return 'reduction', code, {'reduction_elements': width, 'engine': 'argmax_warp_row',
                                       'arithmetic': 'exact comparison; NaN maximal; first index on ties'}, helper
        code = f'''for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
  float best = -INFINITY; long long index = 0x7fffffffffffffffLL;
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x) {{
    const float v = {x};
    if ({better}(v, k, best, index)) {{ best = v; index = k; }}
  }}
  for (int delta = 16; delta > 0; delta /= 2) {{
    const float v = __shfl_down_sync(0xffffffffU, best, delta);
    const long long k = __shfl_down_sync(0xffffffffU, index, delta);
    if ({better}(v, k, best, index)) {{ best = v; index = k; }}
  }}
  long long* indices = reinterpret_cast<long long*>(smem + 16);
  if ((threadIdx.x & 31) == 0) {{ smem[threadIdx.x >> 5] = best; indices[threadIdx.x >> 5] = index; }}
  __syncthreads();
  if (threadIdx.x == 0) {{
    for (int w = 1; w < 8; ++w)
      if ({better}(smem[w], indices[w], best, index)) {{ best = smem[w]; index = indices[w]; }}
    {out}[row] = static_cast<int64_t>(index);
  }}
  __syncthreads();
}}'''
        return 'reduction', code, {'reduction_elements': width,
                                   'arithmetic': 'exact comparison; NaN maximal; first index on ties'}, helper

    # C-OPS4-A: generic last-dimension logsumexp (ATen semantics: an infinite row
    # maximum is replaced by 0 before the shifted exponential sum). One CTA per
    # row; fixed-order CTA trees as in the softmax rows below. AD supplies the
    # backward (grad * exp(x - result)) as ordinary pointwise operations.
    if base == 'logsumexp':
        source = ref(0)
        ls = layouts[source]
        require_float(source, name)
        dims = args[1] if len(args) > 1 else kwargs.get('dim')
        dims = [dims] if isinstance(dims, int) else list(dims or [])
        keep = bool(args[2] if len(args) > 2 else kwargs.get('keepdim', False))
        if len(dims) != 1 or not ls.shape or int(dims[0]) % len(ls.shape) != len(ls.shape) - 1:
            raise LoweringError('logsumexp currently requires one explicit last tensor dimension')
        if shape != (ls.shape[:-1] + (1,) if keep else ls.shape[:-1]):
            raise LoweringError('logsumexp result shape mismatch')
        width = ls.shape[-1]
        rows = math.prod(ls.shape[:-1])
        x = f'{_ident(source)}[{_index(ls, ls.shape, "(row * %dLL + k)" % width)}]'

        def tree(var, operation, identity):
            return f'''{var} = s101h::warp_{operation}({var});
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = {var};
  __syncthreads();
  if (threadIdx.x < 32) {{
    float partial = threadIdx.x < 8 ? smem[threadIdx.x] : {identity};
    partial = s101h::warp_{operation}(partial);
    if (threadIdx.x == 0) smem[0] = partial;
  }}
  __syncthreads();
  {var} = smem[0];
  __syncthreads();'''
        code = f'''for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
  float maximum = -INFINITY;
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x) maximum = fmaxf(maximum, {x});
  {tree('maximum', 'max', '-INFINITY')}
  const float shift = isinf(maximum) ? 0.0f : maximum;
  float total = 0.0f;
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x) total += expf({x} - shift);
  {tree('total', 'sum', '0.0f')}
  if (threadIdx.x == 0) {out}[row] = logf(total) + shift;
  __syncthreads();
}}'''
        return 'reduction', code, {'reduction_elements': width,
                                   'arithmetic': 'FP32 CTA-tree max and shifted exp-sum; ATen infinite-max rule'}, helpers

    backward = base.endswith('_backward_data')
    logged = 'log_softmax' in base
    # Release step 09: PyTorch's boolean-mask softmax; a row whose maximum is -inf (no allowed key) is 0.
    # Its derivative is the ordinary _softmax_backward_data of this result (rows of zeros give zeros).
    safe = base == '_safe_softmax'
    if safe and (args[2] if len(args) > 2 else kwargs.get('dtype')) is not None:
        raise LoweringError('_safe_softmax with a dtype argument is not admitted')
    source = ref(1) if backward else ref(0)
    ls = layouts[source]
    require_float(source, name)
    dim = int(args[2] if backward else args[1]) % len(shape)
    if ls.shape != shape or dim != len(shape) - 1:
        raise LoweringError('softmax currently requires the last tensor dimension')
    if not backward and not safe and bool(args[2] if len(args) > 2 else kwargs.get('half_to_float', False)):
        raise LoweringError('softmax half_to_float is not admitted')
    width = shape[-1]
    rows = numel // width
    position = '(row * %dLL + k)' % width
    x = f'{_ident(source)}[{_index(ls, shape, position)}]'

    def reduce(var, operation, identity):
        return f'''{var} = s101h::warp_{operation}({var});
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = {var};
  __syncthreads();
  if (threadIdx.x < 32) {{
    float partial = threadIdx.x < 8 ? smem[threadIdx.x] : {identity};
    partial = s101h::warp_{operation}(partial);
    if (threadIdx.x == 0) smem[0] = partial;
  }}
  __syncthreads();
  {var} = smem[0];
  __syncthreads();'''

    if (warp_rows == "pops2a" and width <= 1024) or (warp_rows and 256 < width <= 1024):
        return _warp_row_softmax(backward, logged, x, ref, layouts, shape, width, rows, out, position, _index,
                                 _ident, require_float, LoweringError, helpers, safe=safe)
    if SOFTMAX_LANE_ROWS and warp_rows and width == 32 and rows >= 128:
        return _lane_row_softmax(backward, logged, x, ref, layouts, shape, rows, out, position, _index, _ident,
                                 require_float, LoweringError, helpers, safe=safe)
    from . import accumulation_rules as _acc_rules
    if warp_rows and 1 <= width < 32 and rows >= 128 and _acc_rules.active("wsoftn"):
        # LATCH-SELECTIVE-10 ``wsoftn``: the P-OPS2-D exact warp-per-row form for rows narrower than a warp. Lane l
        # of the row's warp holds CTA thread l's element (l < W); every other CTA thread of the tree held only the
        # identity (0.0f / -INFINITY start, no element), as do lanes l >= W here and the seven empty warps of the
        # second level, so each butterfly adds/compares the same values in the same pairing.
        grad = ref(0) if backward else None
        if backward:
            require_float(grad)
            if layouts[grad].shape != shape:
                raise LoweringError('softmax backward gradient shape mismatch')
            g = f'{_ident(grad)}[{_index(layouts[grad], shape, position)}]'

        def narrow(var, operation, identity, leaf):
            return f'''  float {var};
  {{
    const float first = lane < {width} ? {leaf} : {identity};
    const float level = __shfl_sync(0xffffffffu, s101h::warp_{operation}(first), 0);
    {var} = lane == 0 ? level : {identity};
    {var} = s101h::warp_{operation}({var});
  }}'''
        lines = [f'for (long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < {rows}LL; '
                 f'row += ((long long)gridDim.x * blockDim.x) >> 5) {{', '  const int lane = threadIdx.x & 31;',
                 '  const long long k = lane;',
                 f'  const float xv = lane < {width} ? {x} : 0.0f;'
                 + (f' const float gv = lane < {width} ? {g} : 0.0f;' if backward else '')]
        if backward:
            summand = 'gv' if logged else '(gv * xv)'
            lines.append(narrow('total', 'sum', '0.0f', f'0.0f + {summand}'))
            result = '(gv - expf(xv) * total)' if logged else '(xv * (gv - total))'
        else:
            lines.append(narrow('maximum', 'max', '-INFINITY', 'fmaxf(-INFINITY, xv)'))
            lines.append(narrow('total', 'sum', '0.0f', '0.0f + expf(xv - maximum)'))
            result = '(xv - maximum - logf(total))' if logged else '(expf(xv - maximum) / total)'
            if safe:
                result = f'(maximum == -INFINITY ? 0.0f : {result})'
        lines += [f'  if (lane < {width}) {out}[row * {width}LL + k] = {result};', '}']
        return ('log_softmax' if logged else 'softmax', '\n'.join(lines),
                {'reduction_elements': width, 'engine': 'exact_cta_tree_per_warp_narrow' + ('_safe' if safe else ''),
                 'work_items': (rows + 7) // 8}, helpers)
    if warp_rows and width % 32 == 0 and 32 <= width <= 256 and rows >= 128:
        # P-OPS2-D exact warp-per-row form: lane l of a row's warp holds the
        # element of CTA thread 32*w + l (one element per thread when W <= 256),
        # so every warp butterfly, the second level (lanes w < 8, identities
        # elsewhere) and each 0.0f/-INFINITY start replay the CTA tree exactly.
        nw = width // 32
        grad = ref(0) if backward else None
        if backward:
            require_float(grad)
            if layouts[grad].shape != shape:
                raise LoweringError('softmax backward gradient shape mismatch')
            g = f'{_ident(grad)}[{_index(layouts[grad], shape, position)}]'

        def tree(var, operation, identity, leaf):
            return f'''  float {var};
  {{
    float v[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) v[w] = w < {nw} ? {leaf} : {identity};
#pragma unroll
    for (int w = 0; w < 8; ++w) v[w] = s101h::warp_{operation}(v[w]);
    {var} = {identity};
#pragma unroll
    for (int w = 0; w < 8; ++w) {{
      const float partial = __shfl_sync(0xffffffffu, v[w], 0);
      if (lane == w) {var} = partial;
    }}
    {var} = s101h::warp_{operation}({var});
  }}'''
        lines = [f'for (long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < {rows}LL; '
                 f'row += ((long long)gridDim.x * blockDim.x) >> 5) {{', '  const int lane = threadIdx.x & 31;',
                 f'  float xv[{nw}];' + (f' float gv[{nw}];' if backward else ''), '#pragma unroll',
                 f'  for (int w = 0; w < {nw}; ++w) {{ const long long k = 32 * w + lane; xv[w] = {x};'
                 + (f' gv[w] = {g};' if backward else '') + ' }']
        load_line = lines[-1]
        pick = f'(w < {nw} ? w : 0)'
        if backward:
            summand = f'gv[{pick}]' if logged else f'(gv[{pick}] * xv[{pick}])'
            lines.append(tree('total', 'sum', '0.0f', f'0.0f + {summand}'))
            result = '(gv[w] - expf(xv[w]) * total)' if logged else '(xv[w] * (gv[w] - total))'
        else:
            lines.append(tree('maximum', 'max', '-INFINITY', f'fmaxf(-INFINITY, xv[{pick}])'))
            lines.append(tree('total', 'sum', '0.0f', f'0.0f + expf(xv[{pick}] - maximum)'))
            result = '(xv[w] - maximum - logf(total))' if logged else '(expf(xv[w] - maximum) / total)'
            if safe:
                result = f'(maximum == -INFINITY ? 0.0f : {result})'
        lines += ['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{ const long long k = 32 * w + lane; '
                  f'{out}[row * {width}LL + k] = {result}; }}', '}']
        return ('log_softmax' if logged else 'softmax', '\n'.join(lines),
                {'reduction_elements': width, 'engine': 'exact_cta_tree_per_warp' + ('_safe' if safe else ''),
                 'work_items': (rows + 7) // 8,
                 **({} if backward or safe or not ls.contiguous else
                    {'row_softmax': {'source': source, 'width': width, 'rows': rows, 'load_line': load_line}})},
                helpers)

    if backward:
        grad = ref(0)
        require_float(grad)
        if layouts[grad].shape != shape:
            raise LoweringError('softmax backward gradient shape mismatch')
        g = f'{_ident(grad)}[{_index(layouts[grad], shape, position)}]'
        summand = g if logged else f'({g} * {x})'
        result = f'({g} - expf({x}) * total)' if logged else f'({x} * ({g} - total))'
        code = f'''for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
  float total = 0.0f;
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x) total += {summand};
  {reduce('total', 'sum', '0.0f')}
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x)
    {out}[row * {width}LL + k] = {result};
  __syncthreads();
}}'''
    else:
        result = f'({x} - maximum - logf(total))' if logged else f'(expf({x} - maximum) / total)'
        if safe:
            result = f'(maximum == -INFINITY ? 0.0f : {result})'
        code = f'''for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
  float maximum = -INFINITY;
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x) maximum = fmaxf(maximum, {x});
  {reduce('maximum', 'max', '-INFINITY')}
  float total = 0.0f;
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x) total += expf({x} - maximum);
  {reduce('total', 'sum', '0.0f')}
  for (long long k = threadIdx.x; k < {width}LL; k += blockDim.x)
    {out}[row * {width}LL + k] = {result};
  __syncthreads();
}}'''
    return 'log_softmax' if logged else 'softmax', code, {'reduction_elements': width}, helpers
def _lane_row_softmax(backward, logged, x, ref, layouts, shape, rows, out, position, _index, _ident,
                      require_float, LoweringError, helpers, safe=False):
    """LATCH-MOESUP-ROW SOFTMAX_LANE_ROWS: the P-OPS2-D warp-per-row form for W == 32 evaluated by one thread per
    row. That form's statistic is lane 0's value after (1) s101h's xor butterfly over the 32 leaves (offsets
    16, 8, 4, 2, 1; v = op(v, partner)), (2) the second stage: lane 0 holds it, every other lane the identity, and
    the same butterfly runs again, i.e. five op(r, identity) steps at lane 0 (every lane ends with that value).
    The thread computes exactly those operations in that order."""
    grad = ref(0) if backward else None
    if backward:
        require_float(grad)
        if layouts[grad].shape != shape:
            raise LoweringError('softmax backward gradient shape mismatch')
        g = f'{_ident(grad)}[{_index(layouts[grad], shape, position)}]'

    def tree(var, op, identity, leaf):
        step = (lambda a, b: f'fmaxf({a}, {b})') if op == 'max' else (lambda a, b: f'({a} + {b})')
        lines = [f'  float {var}_t[32];', '#pragma unroll', f'  for (int k = 0; k < 32; ++k) {var}_t[k] = {leaf};']
        for o in (16, 8, 4, 2, 1):
            lines += ['#pragma unroll', f'  for (int k = 0; k < {o}; ++k) {var}_t[k] = {step(f"{var}_t[k]", f"{var}_t[k + {o}]")};']
        lines.append(f'  float {var} = {var}_t[0];')
        lines += [f'  {var} = {step(var, identity)};'] * 5
        return lines

    lines = [f'for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x; row < {rows}LL; '
             f'row += (long long)gridDim.x * blockDim.x) {{',
             '  float xv[32];' + (' float gv[32];' if backward else ''), '#pragma unroll',
             f'  for (int k = 0; k < 32; ++k) {{ xv[k] = {x};' + (f' gv[k] = {g};' if backward else '') + ' }']
    if backward:
        summand = 'gv[k]' if logged else '(gv[k] * xv[k])'
        lines += tree('total', 'sum', '0.0f', f'0.0f + {summand}')
        result = '(gv[k] - expf(xv[k]) * total)' if logged else '(xv[k] * (gv[k] - total))'
    else:
        lines += tree('maximum', 'max', '-INFINITY', 'fmaxf(-INFINITY, xv[k])')
        lines += tree('total', 'sum', '0.0f', '0.0f + expf(xv[k] - maximum)')
        result = '(xv[k] - maximum - logf(total))' if logged else '(expf(xv[k] - maximum) / total)'
        if safe:
            result = f'(maximum == -INFINITY ? 0.0f : {result})'
    lines += ['#pragma unroll', f'  for (int k = 0; k < 32; ++k) {out}[row * 32LL + k] = {result};', '}']
    return ('log_softmax' if logged else 'softmax', '\n'.join(lines),
            {'reduction_elements': 32, 'engine': 'exact_cta_tree_lane_rows' + ('_safe' if safe else ''),
             'work_items': (rows + 255) // 256}, helpers)


def _warp_row_softmax(backward, logged, x, ref, layouts, shape, width, rows, out, position, _index, _ident,
                      require_float, LoweringError, helpers, safe=False):
    """P-OPS2-A: one warp per row, bitwise identical to the 256-thread block-per-row emitter above.

    Lane l emulates block threads t = 32c + l (c < 8): each accumulates its k = t, t+256, ... in the same
    order; each emulated warp c is reduced by the same s101h butterfly; the eight warp partials are then
    placed in lanes 0..7 (identity elsewhere) and reduced by the same butterfly, exactly as the block's
    shared-memory second stage. No CTA barrier; eight rows per CTA in flight.
    """
    chunks = (min(width, 256) + 31) // 32

    def reduce(var, operation, identity, term):
        lines = [f'float {var}_w = {identity};']
        for c in range(chunks):
            lines.append(f'{{ float v = {identity};\n'
                         f'    for (long long k = {32 * c}LL + lane; k < {width}LL; k += 256LL) v = {term("v")};\n'
                         f'    v = s101h::warp_{operation}(v); if (lane == {c}) {var}_w = v; }}')
        lines.append(f'const float {var} = s101h::warp_{operation}({var}_w);')
        return '\n    '.join(lines)

    if P6E_SOFTMAX_ROWS > 1 and width % 32 == 0 and width <= 256:
        return _warp_rows_multi(backward, logged, x, ref, layouts, shape, width, rows, out, position, _index,
                                _ident, require_float, LoweringError, helpers, P6E_SOFTMAX_ROWS, safe=safe)
    head = (f'const int lane = threadIdx.x & 31;\n'
            f'for (long long row = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row < {rows}LL; '
            f'row += (long long)gridDim.x * (blockDim.x >> 5)) {{\n    ')
    if backward:
        grad = ref(0)
        require_float(grad)
        if layouts[grad].shape != shape:
            raise LoweringError('softmax backward gradient shape mismatch')
        g = f'{_ident(grad)}[{_index(layouts[grad], shape, position)}]'
        summand = g if logged else f'({g} * {x})'
        result = f'({g} - expf({x}) * total)' if logged else f'({x} * ({g} - total))'
        code = (head + reduce('total', 'sum', '0.0f', lambda v: f'{v} + {summand}') +
                f'\n    for (long long k = lane; k < {width}LL; k += 32LL) {out}[row * {width}LL + k] = {result};\n}}')
    else:
        result = f'({x} - maximum - logf(total))' if logged else f'(expf({x} - maximum) / total)'
        if safe:
            result = f'(maximum == -INFINITY ? 0.0f : {result})'
        code = (head + reduce('maximum', 'max', '-INFINITY', lambda v: f'fmaxf({v}, {x})') + '\n    '
                + reduce('total', 'sum', '0.0f', lambda v: f'{v} + expf({x} - maximum)') +
                f'\n    for (long long k = lane; k < {width}LL; k += 32LL) {out}[row * {width}LL + k] = {result};\n}}')
    return 'log_softmax' if logged else 'softmax', code, {'reduction_elements': width, 'row_mapping': 'warp'}, helpers


def _warp_rows_multi(backward, logged, x, ref, layouts, shape, width, rows, out, position, _index, _ident,
                     require_float, LoweringError, helpers, R, safe=False):
    """P6-E: the P-OPS2-A warp-row softmax with R rows per warp iteration (W <= 256, W % 32 == 0: one element per
    lane per emulated warp). Every operand of the R rows is loaded first, then each row runs the unchanged tree
    (same identities, same butterflies, same lane placement) on the loaded values and stores its results: per-row
    arithmetic and order identical, bitwise; the rows' memory latencies overlap instead of following each other."""
    C = width // 32
    grad = ref(0) if backward else None
    if backward:
        require_float(grad)
        if layouts[grad].shape != shape:
            raise LoweringError('softmax backward gradient shape mismatch')
        g = f'{_ident(grad)}[{_index(layouts[grad], shape, position)}]'

    def reduce(var, operation, identity, term):
        lines = [f'float {var}_w = {identity};']
        for c in range(C):
            lines.append(f'{{ float v = {identity}; v = {term("v", c)};\n'
                         f'      v = s101h::warp_{operation}(v); if (lane == {c}) {var}_w = v; }}')
        lines.append(f'const float {var} = s101h::warp_{operation}({var}_w);')
        return '\n      '.join(lines)

    X = lambda c: f'xv[r][{c}]'
    G = lambda c: f'gv[r][{c}]'
    loads = (f'#pragma unroll\n    for (int c = 0; c < {C}; ++c) {{ const long long k = 32LL * c + lane; '
             f'xv[r][c] = {x};' + (f' gv[r][c] = {g};' if backward else '') + ' }')
    if backward:
        summand = (lambda c: G(c)) if logged else (lambda c: f'({G(c)} * {X(c)})')
        body = reduce('total', 'sum', '0.0f', lambda v, c: f'{v} + {summand(c)}')
        result = (lambda c: f'({G(c)} - expf({X(c)}) * total)') if logged else (lambda c: f'({X(c)} * ({G(c)} - total))')
    else:
        body = (reduce('maximum', 'max', '-INFINITY', lambda v, c: f'fmaxf({v}, {X(c)})') + '\n      '
                + reduce('total', 'sum', '0.0f', lambda v, c: f'{v} + expf({X(c)} - maximum)'))
        result = (lambda c: f'({X(c)} - maximum - logf(total))') if logged else (lambda c: f'(expf({X(c)} - maximum) / total)')
        if safe:
            plain = result
            result = lambda c: f'(maximum == -INFINITY ? 0.0f : {plain(c)})'
    stores = '\n      '.join(f'{out}[row * {width}LL + {32 * c}LL + lane] = {result(c)};' for c in range(C))
    code = (f'const int lane = threadIdx.x & 31;\n'
            f'const long long latch_nwt = (long long)gridDim.x * (blockDim.x >> 5);\n'
            f'for (long long row0 = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row0 < {rows}LL; '
            f'row0 += {R}LL * latch_nwt) {{\n'
            f'  float xv[{R}][{C}];' + (f' float gv[{R}][{C}];' if backward else '') + '\n'
            f'#pragma unroll\n  for (int r = 0; r < {R}; ++r) {{\n    const long long row = row0 + r * latch_nwt;\n'
            f'    if (row < {rows}LL) {{\n    {loads}\n    }}\n  }}\n'
            f'#pragma unroll\n  for (int r = 0; r < {R}; ++r) {{\n    const long long row = row0 + r * latch_nwt;\n'
            f'    if (row < {rows}LL) {{\n      {body}\n      {stores}\n    }}\n  }}\n}}')
    return 'log_softmax' if logged else 'softmax', code, {'reduction_elements': width, 'row_mapping': 'warp',
                                                         'rows_per_iteration': R}, helpers


def ordered_embedding_backward(args, kwargs, name, shape, layouts, dtypes, names):
    """P-OPS-D: embedding gradient from ordered occurrence lists.

    Ported from P-OPS-D 775505d79b1d7debc9b2865e092e30405eee8455,
    preserving the Stage4 operation emitters above. Select automatically when
    eligible; the caller's reference option must retain the ordinary emitter.

    The frozen rule scans every index for every output element (work V*d*N).
    Here one CTA per vocabulary row ranks its occurrences in ascending position
    (deterministic block scan) into a list, then each output element adds its
    row's gradient rows in that ascending order starting from 0.0f: exactly the
    frozen accumulation sequence, so results are bitwise identical. Returns None
    (frozen rule) outside the admitted shapes.
    """
    from .lower import LoweringError, _ident, _index, _ref

    grad, ids = _ref(args[0], names), _ref(args[1], names)
    lg, li = layouts[grad], layouts[ids]
    if dtypes[grad] != 'float32' or dtypes[name] != 'float32' or dtypes[ids] != 'int64':
        raise LoweringError('embedding backward requires FP32 gradients and int64 indices')
    rows = int(args[2])
    padding = int(args[3] if len(args) > 3 else kwargs.get('padding_idx', -1))
    if bool(args[4] if len(args) > 4 else kwargs.get('scale_grad_by_freq', False)):
        raise LoweringError('embedding frequency-scaled gradients are not admitted')
    if len(shape) != 2 or shape[0] != rows or lg.shape != (*li.shape, shape[1]):
        raise LoweringError('embedding backward shape mismatch')
    n = li.numel
    if n > (1 << 24) or rows > (1 << 24):
        return None
    width = shape[1]
    lst, start, count = name + '__occ', name + '__start', name + '__count'
    generated = {lst, start, count, name + '__phase_embedding_gradient_rank'}
    if generated.intersection(names) or generated.intersection(layouts):
        return None  # Ordinary lowering preserves graphs with colliding names.
    index = f'{_ident(ids)}[{_index(li, li.shape, "j")}]'
    grad_index = _index(lg, lg.shape, f'(pos * {width}LL + col)')
    if P6E_VEC_GATHER and lg.contiguous and not lg.offset:
        grad_index = f'(pos * {width}LL + col)'       # P6-E: same element, no generic div/mod index
    rank = f'''for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
  // Each thread owns a contiguous chunk of positions; chunks are in ascending order.
  const long long chunk = ({n}LL + blockDim.x - 1) / blockDim.x;
  const long long lo = threadIdx.x * chunk, hi = lo + chunk < {n}LL ? lo + chunk : {n}LL;
  int below = 0, equal = 0;
  for (long long j = lo; j < hi; ++j) {{
    const int64_t id = {index};
    below += id < row;
    equal += id == row;
  }}
  // Exclusive block scan of equal counts and block sum of below counts (integers: exact).
  int scan = equal, total_below = below;
  for (int o = 1; o < 32; o <<= 1) {{
    const int up = __shfl_up_sync(0xffffffffu, scan, o);
    if ((threadIdx.x & 31) >= o) scan += up;
  }}
  for (int o = 16; o > 0; o >>= 1) total_below += __shfl_xor_sync(0xffffffffu, total_below, o);
  int* words = reinterpret_cast<int*>(smem);
  if ((threadIdx.x & 31) == 31) words[threadIdx.x >> 5] = scan;
  if ((threadIdx.x & 31) == 0) words[8 + (threadIdx.x >> 5)] = total_below;
  __syncthreads();
  int warp_offset = 0, start = 0, occurrences = 0;
  for (int w = 0; w < 8; ++w) {{
    if (w < (int)(threadIdx.x >> 5)) warp_offset += words[w];
    start += words[8 + w];
    occurrences += words[w];
  }}
  long long slot = start + warp_offset + scan - equal;
  for (long long j = lo; j < hi && equal; ++j)
    if ({index} == row) {{ {_ident(lst)}[slot++] = j; --equal; }}
  if (threadIdx.x == 0) {{ {_ident(start)}[row] = start; {_ident(count)}[row] = occurrences; }}
  __syncthreads();
}}'''
    def load(buf, at):
        return (f'#pragma unroll\n      for (int u = 0; u < 32; ++u) {{ const long long pos = {_ident(lst)}[{at} + u]; '
                f'{buf}[u] = {_ident(grad)}[{grad_index}]; }}')
    add = lambda buf: f'#pragma unroll\n      for (int u = 0; u < 32; ++u) acc += {buf}[u];'
    if P6E_EMB_PIPE:
        # P6-E: two register batches; batch b+1's loads are issued before batch b is summed (same ascending order)
        batches = f'''    float g[32], h[32];
    if (k + 32 <= last) {{
      {load('g', 'k')}
    }}
    while (k + 32 <= last) {{
      const bool m1 = k + 64 <= last;
      if (m1) {{
      {load('h', '(k + 32)')}
      }}
      {add('g')}
      k += 32;
      if (!m1) break;
      const bool m2 = k + 64 <= last;
      if (m2) {{
      {load('g', '(k + 32)')}
      }}
      {add('h')}
      k += 32;
      if (!m2) break;
    }}'''
    else:
        batches = f'''    for (; k + 32 <= last; k += 32) {{
      float g[32];
#pragma unroll
      for (int u = 0; u < 32; ++u) {{
        const long long pos = {_ident(lst)}[k + u];
        g[u] = {_ident(grad)}[{grad_index}];
      }}
#pragma unroll
      for (int u = 0; u < 32; ++u) acc += g[u];
    }}'''
    if P9_EMB_TAIL:
        # P9-A: the occurrences after the last full batch load as one predicated batch (all loads in flight), then
        # add in the same ascending order (identical additions; the serial tail paid two dependent loads each).
        tail = f'''    if (k < last) {{
      float g[32];
#pragma unroll
      for (int u = 0; u < 32; ++u) {{
        if (k + u < last) {{
          const long long pos = {_ident(lst)}[k + u];
          g[u] = {_ident(grad)}[{grad_index}];
        }}
      }}
#pragma unroll
      for (int u = 0; u < 32; ++u) if (k + u < last) acc += g[u];
    }}'''
    else:
        tail = f'''    for (; k < last; ++k) {{
      const long long pos = {_ident(lst)}[k];
      acc += {_ident(grad)}[{grad_index}];
    }}'''
    if P9_EMB_PAD:
        # P9-A: every batch (tail included) issues all 32 loads; slots past the row's last occurrence re-read a valid
        # position and contribute +0.0f. acc starts at +0.0f and a round-to-nearest sum is never -0.0, so adding +0.0f
        # is the identity: the ascending addition sequence and every result are unchanged (bitwise).
        batches = f'''    for (; k < last; k += 32) {{
      float g[32];
#pragma unroll
      for (int u = 0; u < 32; ++u) {{
        const long long pos = {_ident(lst)}[k + u < last ? k + u : k];
        const float x = {_ident(grad)}[{grad_index}];
        g[u] = k + u < last ? x : 0.0f;
      }}
#pragma unroll
      for (int u = 0; u < 32; ++u) acc += g[u];
    }}'''
        tail = ''
    if P9_EMB_TAIL8 and not P9_EMB_PAD:
        # P9-A: the remainder (< 32 occurrences) runs in zero-padded batches of 8 loads instead of one dependent load
        # per occurrence; padded slots re-read a valid position and add +0.0f (identity: acc is never -0.0). Bitwise.
        tail = f'''    for (; k < last; k += 8) {{
      float g[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {{
        const long long pos = {_ident(lst)}[k + u < last ? k + u : k];
        const float x = {_ident(grad)}[{grad_index}];
        g[u] = k + u < last ? x : 0.0f;
      }}
#pragma unroll
      for (int u = 0; u < 8; ++u) acc += g[u];
    }}'''
    gather = f'''for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {rows * width}LL; i += (long long)gridDim.x * blockDim.x) {{
  const long long row = i / {width}LL, col = i % {width}LL;
  float acc = 0.0f;
  if (row != {padding}LL) {{
    const long long first = {_ident(start)}[row], last = first + {_ident(count)}[row];
    long long k = first;
    // Batched loads, unchanged ascending addition order.
{batches}
{tail}
  }}
  {_ident(name)}[i] = acc;
}}'''
    return {'auxiliary': [(lst, (n,), 'int64'), (start, (rows,), 'int64'), (count, (rows,), 'int64')],
            'internal': [lst, start, count],
            'phases': [('embedding_gradient_rank', [lst, start, count], rank,
                        {'rows': rows, 'positions': n, 'work_items': rows}),
                       ('embedding_gradient', [name], gather, {'reduction_elements': 'ordered occurrence list'})],
            'source_commit': '775505d79b1d7debc9b2865e092e30405eee8455',
            'threads': 256, 'accumulation_order': 'ascending input position, starting from FP32 zero'}
