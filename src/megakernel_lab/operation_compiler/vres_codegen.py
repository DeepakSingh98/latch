"""CUDA generation for the vector-resident loop schedule (see vres_loop.py)."""
from __future__ import annotations

import hashlib
import re
import json
import math

from .lower import GeneratedProgram, LoweringError, _header_path, _translation_unit
from .vres_loop import (BINARY, BYTES, CTYPE, PRODUCTS, REDUCTIONS, SMEM_LOCAL_LIMIT, THREADS, UNARY,
                        VIEW_TARGETS, WARPS, Exchange, Step, View, VresLowering, _Region, _dt, _numel, _ref)

HELPERS = r'''
#ifndef LATCH_VRES_HELPERS
#define LATCH_VRES_HELPERS
#include <cuda_bf16.h>
namespace vres {
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// Every thread returns the CTA total; fixed warp order.
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) r += red[w];
  return r;
}
__device__ __forceinline__ float block_max(float v, float* red) {
  v = warp_max(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = -INFINITY;
#pragma unroll
  for (int w = 0; w < 8; ++w) r = fmaxf(r, red[w]);
  return r;
}
__device__ __forceinline__ uint4 ld16h(const uint4* p, unsigned long long pol) {
  uint4 r;
  asm("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ float4 ld16h(const float4* p, unsigned long long pol) {
  float4 r;
  asm("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
      : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
// Same values as bf8, but opaque to CSE: recomputed from the raw words at each use, so a consumer
// after a long dependency (e.g. a warp reduction) keeps 4 raw registers live instead of 8 floats.
__device__ __forceinline__ void bf8_late(const uint4 q, float* f) {
  const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    unsigned lo, hi;
    asm volatile("shl.b32 %0, %1, 16;" : "=r"(lo) : "r"(w[i]));
    asm volatile("and.b32 %0, %1, 0xffff0000;" : "=r"(hi) : "r"(w[i]));
    f[2 * i] = __uint_as_float(lo);
    f[2 * i + 1] = __uint_as_float(hi);
  }
}
__device__ __forceinline__ void bf8(const uint4 q, float* f) {
  const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
}  // namespace vres
#endif
'''


def _lit(x, dtype):
    if isinstance(x, bool):
        return 'true' if x else 'false'
    if dtype == 'float32':
        v = float(x)
        if math.isinf(v):
            return 'INFINITY' if v > 0 else '(-INFINITY)'
        return repr(v) + 'f' if ('e' in repr(v) or '.' in repr(v)) else repr(v) + '.0f'
    if dtype == 'int64':
        return f'{int(x)}LL'
    return repr(x)


def _rowmajor(shape):
    st, s = [], 1
    for e in reversed(shape):
        st.append(s)
        s *= e
    return list(reversed(st))


class Gen:
    def __init__(self, L: VresLowering):
        self.L = L
        self.G = L.G
        self.groups = {}             # N -> chunk
        self._carry_which = 'cur'    # carry buffer read by carry-placed values
        # warp-tree reduce-scatter: measured ~0.6 ms/block faster at GPT-2 small but reorders the fixed sum;
        # off by default so the schedule stays bitwise equal to the GRADLEV-LATCH baseline
        self.rs_warp = bool(L.options.get('vres_rs_warp', False))
        self.state_vec = bool(L.options.get('vres_state_vec', False))   # measured: no gain at GPT-2 small
        self._state_sub = None
        self._cap = {'acc': [], 'emap': None}    # accesses / element map of the step being generated
        self._ctx = None                         # current CTA-strided loop context
        self._regs = {}                          # (region, root) -> register holding a thread-replicated scalar
        self.align = bool(L.options.get('vres_align_fusion', True))
        self.rows_pipeline = bool(L.options.get('vres_rows_pipeline', True))
        self._prefetch_parity = 'par'
        self.lines_decl = []
        self.region_code = {}

    @staticmethod
    def warp_only(code):
        """Rewrite a CTA-strided element loop (`vi = tid; ...; vi += 256`) to warp 0's lanes."""
        if not code.startswith('for (int vi = tid; vi < '):
            if code.startswith('if (tid == 0) {'):
                return 'if (warp == 0 && lane == 0) {' + code[len('if (tid == 0) {'):]
            raise LoweringError('vector_resident: unexpected loop form for a warp-only step')
        head, _, rest = code.partition('; vi += 256) {')
        return 'if (warp == 0) ' + head.replace('vi = tid;', 'vi = lane;') + '; vi += 32) {' + rest

    # ---------------------------------------------------------------- scratch
    def scratch(self, nbytes):
        """Step-transient shared scratch: one pool (every user ends with a CTA barrier)."""
        self.scratch_bytes = max(getattr(self, 'scratch_bytes', 0), nbytes)
        return '@@VRES_SCRATCH@@'

    # ---------------------------------------------------------------- partitions
    def group(self, N):
        if N not in self.groups:
            self.groups[N] = -(-N // self.G)
        return f'vlo_{N}', f'vhi_{N}', self.groups[N]

    # ---------------------------------------------------------------- addressing
    def transient(self, key, nbytes):
        """Shared-memory value whose lifetime is one region pass; offset assigned after scheduling."""
        if key not in self.tkeys:
            self.tkeys[key] = len(self.tsizes)
            self.tsizes.append((nbytes + 15) // 16 * 16)
        return f'@@SM{self.tkeys[key]}@@'

    def smem_ptr(self, R, root):
        key = (R.label, root)
        dtype = R.dtype(root)
        shape = R.shape(root)
        place = R.place[root]
        if place == 'resident':
            n = self.L._local_numel(shape, R.owner[root])
        elif place == 'pcache':
            n = _numel(shape)
        else:
            n = self.L._local_numel(shape, R.shard.get(root)) if place == 'local' else _numel(shape)
        if place == 'prefetch':
            nb = (max(1, n) * BYTES[dtype] + 15) // 16 * 16
            off = self.L.alloc_smem(('prefetch', root), 2 * nb)
            return f'reinterpret_cast<{CTYPE[dtype]}*>(vres_smem + {off} + {self._prefetch_parity} * {nb})'
        if place in ('resident', 'pcache'):
            off = self.L.alloc_smem(key if place == 'resident' else ('pcache', root), max(1, n) * BYTES[dtype])
        else:
            off = self.transient(key, max(1, n) * BYTES[dtype])
        return f'reinterpret_cast<{CTYPE[dtype]}*>(vres_smem + {off})'

    def carry_ptr(self, j, which=None):
        off, _ = self.L.smem[('carry_' + (which or self._carry_which), j)]
        dtype = self.carry_dtypes[j]
        return f'reinterpret_cast<{CTYPE[dtype]}*>(vres_smem + {off})'

    def load_root(self, R, root, ridx):
        reg = self._regs.get((R.label, root))
        if reg is not None:
            self._cap['acc'].append((root, 'reg'))
            return reg
        if R.place.get(root) in ('smem', 'local', 'prefetch', 'resident', 'carry') or (
                R.place.get(root) == 'global' and R.external.get(root, {}).get('role') == 'state'):
            self._cap['acc'].append((root, self._aligned(R, root, ridx)))
        return self._load_root_impl(R, root, ridx)

    def _aligned(self, R, root, ridx):
        """True when this access reads the element the current thread itself handles in a CTA-strided loop
        (thread tid owns linear elements tid + 256 k) of a value with the same shape and sharding."""
        ctx = self._ctx
        if ctx is None:
            return False
        shape = list(R.shape(root))
        shard = R.shard.get(root) if R.place.get(root) == 'local' else None
        if _numel(shape) != _numel(ctx['shape']) or shard != ctx['shard']:
            return False
        a = [i for i, e in zip(ridx, shape) if e != 1]
        b = [i for i, e in zip(ctx['idx'], ctx['shape']) if e != 1]
        return a == b and [e for e in shape if e != 1] == [e for e in ctx['shape'] if e != 1]

    def _load_root_impl(self, R, root, ridx):
        place = R.place[root]
        shape = R.shape(root)
        dtype = R.dtype(root)
        if place == 'global':
            info = R.external[root]
            st = R.values[root].strides
            off = ' + '.join(f'(long long)({i}) * {s}LL' for i, s in zip(ridx, st) if s) or '0'
            return f'reinterpret_cast<const {CTYPE[dtype]}*>({info["ptr"]})[{off}]'
        if place == 'pcache':
            st = _rowmajor(shape)
            off = ' + '.join(f'({i}) * {s}' for i, s in zip(ridx, st)) or '0'
            return f'{self.smem_ptr(R, root)}[{off}]'
        if place == 'carry':
            j = R.external[root]['slot']
            st = _rowmajor(shape)
            off = ' + '.join(f'({i}) * {s}' for i, s in zip(ridx, st)) or '0'
            return f'{self.carry_ptr(j)}[{off}]'
        if place in ('smem', 'prefetch'):
            st = _rowmajor(shape)
            off = ' + '.join(f'({i}) * {s}' for i, s in zip(ridx, st)) or '0'
            return f'{self.smem_ptr(R, root)}[{off}]'
        if place == 'resident' and self._state_sub and self._state_sub[0] == root:
            hit = self._state_sub[1].get(tuple(i for i, e in zip(ridx, shape) if e != 1))
            if hit is not None:
                return hit
        if place in ('local', 'resident'):
            axis, N = R.shard[root] if place == 'local' else R.owner[root]
            lo, _, ch = self.group(N)
            lshape = list(shape)
            lshape[axis] = ch
            st = _rowmajor(lshape)
            terms = []
            for d, (i, s) in enumerate(zip(ridx, st)):
                terms.append(f'(({i}) - {lo}) * {s}' if d == axis else f'({i}) * {s}')
            return f'{self.smem_ptr(R, root)}[{" + ".join(terms) or "0"}]'
        if place == 'virtual':
            op = R.op_of[root]
            if op.target != 'aten.arange.default' or len(op.args) != 1:
                raise LoweringError('vector_resident: only arange(end) is supported')
            return f'((long long)({ridx[0]}))'
        if place == 'inline':
            return self.op_expr(R, R.op_of[root], ridx)
        raise LoweringError(f'vector_resident: no placement for {root}')

    def view_expr(self, R, name, idx):
        v = R.views[name]
        rshape = R.shape(v.root)
        ridx = ['0'] * len(rshape)
        for j, rd in enumerate(v.dims):
            if rd is not None:
                ridx[rd] = idx[j]
        for rd, c in v.fixed.items():
            ridx[rd] = str(c)
        return self.load_root(R, v.root, ridx)

    def bcast(self, R, name, out_idx):
        s = R.shape(name)
        r, q = len(out_idx), len(s)
        return [('0' if s[j] == 1 else out_idx[j + r - q]) for j in range(q)]

    def arg_expr(self, R, a, out_idx, dtype):
        r = _ref(a)
        if r is not None:
            e = self.view_expr(R, r, self.bcast(R, r, out_idx))
            if R.dtype(r) == 'bfloat16' and dtype != 'bfloat16':
                e = f'__bfloat162float({e})'
            return e
        return _lit(a, dtype)

    def op_expr(self, R, op, idx):
        t = op.target
        out = op.outputs[0]
        odt = R.dtype(out)
        ins = op.args
        if t == 'aten._to_copy.default':
            src = _ref(ins[0])
            e = self.view_expr(R, src, self.bcast(R, src, idx))
            sdt = R.dtype(src)
            if odt == sdt:
                return e
            if sdt == 'bfloat16':
                e = f'__bfloat162float({e})'
                return e if odt == 'float32' else f'({CTYPE[odt]})({e})'
            if odt == 'bfloat16':
                return f'__float2bfloat16_rn((float)({e}))'
            return f'({CTYPE[odt]})({e})'
        if t in UNARY:
            return UNARY[t].format(self.arg_expr(R, ins[0], idx, odt))
        if t in BINARY:
            cdt = odt if odt != 'bool' else (R.dtype(_ref(ins[0])) if _ref(ins[0]) else 'float32')
            a = self.arg_expr(R, ins[0], idx, cdt)
            b = self.arg_expr(R, ins[1], idx, cdt)
            if t in ('aten.add.Tensor', 'aten.sub.Tensor'):
                alpha = op.kwargs.get('alpha', 1)
                if alpha != 1:
                    b = f'({_lit(alpha, cdt)} * {b})'
            return BINARY[t].format(a, b)
        if t == 'aten.where.self':
            c = self.arg_expr(R, ins[0], idx, 'bool')
            return f'(({c}) ? ({self.arg_expr(R, ins[1], idx, odt)}) : ({self.arg_expr(R, ins[2], idx, odt)}))'
        raise LoweringError(f'vector_resident: {t} is not an elementwise expression')

    # ---------------------------------------------------------------- loops over a value
    def loop_over(self, R, root, body_fn, *, shard=None, shape=None, split_fn=None):
        """Code visiting every local element of a value (sharded axis restricted to this CTA)."""
        shape = list(R.shape(root) if shape is None else shape)
        shard = R.shard.get(root) if shard is None else shard
        lshape = list(shape)
        guard = ''
        if shard is not None:
            axis, N = shard
            lo, hi, ch = self.group(N)
            lshape[axis] = ch
        n = _numel(lshape)
        idx = [None] * len(shape)
        last = len(shape) - 1
        vec4 = bool(shape) and lshape[last] % 4 == 0 and n >= 4 * THREADS
        if vec4 and shard is not None and shard[0] == last:
            N_ = shard[1]
            vec4 = N_ % 4 == 0 and self.group(N_)[2] % 4 == 0     # a 4-group is entirely in or out of range
        if vec4 and split_fn is not None:
            # two-phase: evaluate four consecutive innermost elements (all loads in flight), then store four
            code = [self._vloop(n // 4), '  int vr = vi;',
                    f'  const int vq{last} = vr % {lshape[last] // 4}; vr /= {lshape[last] // 4};']
            for d in reversed(range(last)):
                code.append(f'  const int vd{d} = vr % {lshape[d]}; vr /= {lshape[d]};')
                idx[d] = f'vd{d}'
            if shard is not None:
                axis, N = shard
                lo, hi, ch = self.group(N)
                base = f'vq{last} * 4' if axis == last else f'vd{axis}'
                code.append(f'  const int vg{axis} = {lo} + {base}; if (vg{axis} >= {hi}) continue;')
                if axis != last:
                    idx[axis] = f'vg{axis}'
            vals, stores = [], []
            for u in range(4):
                ix = list(idx)
                if shard is not None and shard[0] == last:
                    ix[last] = f'(vg{last} + {u})'
                else:
                    ix[last] = f'(vq{last} * 4 + {u})'
                v_, s_ = split_fn(ix)
                vals.append(f'  const auto vres_v{u} = {v_};')
                stores.append(f'  {s_} = vres_v{u};')
            code += vals + stores
            code.append('}')
            return '\n'.join(code)
        vec4 = vec4 and split_fn is None
        if vec4:
            # four consecutive innermost elements per thread: one index decomposition per four elements
            code = [self._vloop(n // 4), '  int vr = vi;',
                    f'  const int vq{last} = vr % {lshape[last] // 4}; vr /= {lshape[last] // 4};']
            for d in reversed(range(last)):
                code.append(f'  const int vd{d} = vr % {lshape[d]}; vr /= {lshape[d]};')
                idx[d] = f'vd{d}'
            code.append('#pragma unroll')
            code.append('  for (int vu = 0; vu < 4; ++vu) {')
            code.append(f'  const int vd{last} = vq{last} * 4 + vu;')
            idx[last] = f'vd{last}'
        else:
            code = [self._vloop(n), '  int vr = vi;']
            for d in reversed(range(len(shape))):
                code.append(f'  const int vd{d} = vr % {lshape[d]}; vr /= {lshape[d]};')
                idx[d] = f'vd{d}'
        if shard is not None:
            axis, N = shard
            lo, hi, ch = self.group(N)
            code.append(f'  const int vg{axis} = {lo} + vd{axis}; if (vg{axis} >= {hi}) continue;')
            idx[axis] = f'vg{axis}'
        if not shape:
            code = ['if (tid == 0) {']
            idx = []
        if not vec4:
            # thread tid handles linear elements tid + 256 k of the (local) value
            self._ctx = {'idx': list(idx), 'shape': list(shape), 'shard': shard}
            self._cap['emap'] = ('s1', _numel(shape), shard)
        try:
            code.append('  ' + body_fn(idx))
        finally:
            self._ctx = None
        code.append('}')
        if vec4:
            code.append('}')
        return '\n'.join(code)

    def _vloop(self, n):
        """Header of a CTA-strided element loop; with several iterations per thread the trip count is made
        static (unrollable, several iterations' loads in flight). Element order per thread is unchanged."""
        U = int(self.L.options.get('vres_loop_unroll', 4))
        if U <= 1 or n <= THREADS or self.small_warp:   # (warp_only() rewrites the plain form)
            return f'for (int vi = tid; vi < {n}; vi += {THREADS}) {{'
        trips = -(-n // THREADS)
        check = '' if n % THREADS == 0 else f' if (vi >= {n}) break;'
        return f'#pragma unroll {min(U, trips)}\nfor (int vk = 0; vk < {trips}; ++vk) {{ const int vi = tid + vk * {THREADS};{check}'

    def state_vec_loop(self, R, name, sroot, shard):
        """In-place owner update with 16-byte read-modify-write of four consecutive innermost state
        elements; every element keeps its scalar expression (reads of the updated state element itself
        are taken from the loaded vector). None when the layout does not admit it."""
        shape = list(R.shape(name))
        if (R.place.get(sroot) != 'resident' or R.dtype(sroot) != 'float32' or shape != list(R.shape(sroot))
                or not shape or R.views[name].root == sroot):
            return None
        last = len(shape) - 1
        axis, N = shard
        lo, hi, ch = self.group(N)
        lshape = list(shape)
        lshape[axis] = ch
        if lshape[last] % 4 or (axis == last and (N % 4 or ch % 4)):
            return None
        q = lshape[last] // 4
        code = [self._vloop(_numel(lshape) // 4), '  int vr = vi;',
                f'  const int vq = vr % {q}; vr /= {q};']
        idx = [None] * len(shape)
        for d in reversed(range(last)):
            code.append(f'  const int vd{d} = vr % {lshape[d]}; vr /= {lshape[d]};')
            idx[d] = f'vd{d}'
        base = 'vq * 4' if axis == last else f'vd{axis}'
        code.append(f'  const int vg = {lo} + {base}; if (vg >= {hi}) continue;')
        if axis != last:
            idx[axis] = 'vg'
        ix = []
        for u in range(4):
            i_ = list(idx)
            i_[last] = f'(vg + {u})' if axis == last else f'(vq * 4 + {u})'
            ix.append(i_)
        ptr = f'reinterpret_cast<float4*>(&{self.load_root(R, sroot, ix[0])})'
        code.append(f'  float4* vsp = {ptr};')
        code.append('  const float4 vsv = *vsp;')
        self._state_sub = (sroot, {tuple(i for i, e in zip(i_, shape) if e != 1): f'vsv.{c}'
                                   for i_, c in zip(ix, 'xyzw')})
        try:
            vals = [self.view_expr(R, name, i_) for i_ in ix]
        finally:
            self._state_sub = None
        code += [f'  const float vres_v{u} = {v};' for u, v in enumerate(vals)]
        code.append('  *vsp = make_float4(vres_v0, vres_v1, vres_v2, vres_v3);')
        code.append('}')
        return '\n'.join(code)

    def store_root(self, R, root, idx, value):
        place = R.place[root]
        if place in ('smem', 'local'):
            ld = self.load_root(R, root, idx)
            return f'{ld} = {value};'
        raise LoweringError(f'vector_resident: cannot store to {place} value {root}')

    # ---------------------------------------------------------------- steps
    def steps_for(self, R, outputs, out_targets):
        """Build Steps for a region. out_targets: output name -> ('carry', j) | ('state', root) | ('pred', ptr)."""
        steps, producer = [], {}
        order = [0]

        def add(name, code, reads=(), xdeps=(), writes=()):
            s = Step(name, code, order=order[0])
            s.access, s.emap, s.writes = list(self._cap['acc']), self._cap['emap'], set(writes)
            s.reg = self._cap.get('reg', False)
            self._cap = {'acc': [], 'emap': None}
            order[0] += 1
            for r in reads:
                if r in producer:
                    s.deps.add(producer[r])
            for x in xdeps:
                s.xdeps.add(x)
            for w in writes:
                producer[w] = name
            steps.append(s)
            return name

        def leaves(name, seen=None):
            """Materialized/external roots read by the expression of a value."""
            seen = set() if seen is None else seen
            root = R.views[name].root
            if root in seen:
                return seen
            seen.add(root)
            if R.place.get(root) == 'inline':
                for r in R._input_refs(R.op_of[root]):
                    leaves(r, seen)
            return seen

        def deps_of(*names):
            out = set()
            for n in names:
                out |= leaves(n)
            return out

        exchanges = {e.name: e for e in self.L.exchanges}
        pending_prefetch, carry_out = [], {}
        for op in R.ops:
            t = op.target
            if t in VIEW_TARGETS:
                continue
            out = op.outputs[0]
            plan = R.plans.get(op.name, {})
            kind = plan.get('kind')
            place = R.place.get(out)
            ins = R._input_refs(op)
            if place in ('inline', 'virtual'):
                continue
            self._cap = {'acc': [], 'emap': None}
            if (self.align and kind == 'elementwise' and R.label == 'body' and place == 'smem'
                    and _numel(R.shape(out)) == 1 and R.dtype(out) in ('float32', 'int64', 'int32', 'bool')):
                # thread-replicated scalar: every thread evaluates it into a register (no shared copy, no barrier)
                var = f'vrs_{out}'
                code = f'const {CTYPE[R.dtype(out)]} {var} = {self.op_expr(R, op, ["0"] * len(R.shape(out)))};'
                self._cap['reg'] = True
                add(op.name, code, reads=deps_of(*ins), writes=[out])
                self._regs[(R.label, out)] = var
                continue
            if kind in ('elementwise', 'fill'):
                if kind == 'fill':
                    val = _lit(op.args[1] if t == 'aten.full.default' else (1 if t == 'aten.ones.default' else 0),
                               R.dtype(out))
                    code = self.loop_over(R, out, lambda idx: self.store_root(R, out, idx, val))
                else:
                    code = self.loop_over(R, out, lambda idx: self.store_root(R, out, idx, self.op_expr(R, op, idx)),
                                          split_fn=lambda idx: (self.op_expr(R, op, idx), self.load_root(R, out, idx)))
                if self.small_warp and place == 'smem' and _numel(R.shape(out)) <= 1024 and kind == 'elementwise':
                    code = self.warp_only(code)
                    add(op.name, code, reads=deps_of(*ins), writes=[out])
                    steps[-1].warp0 = True
                    continue
                add(op.name, code, reads=deps_of(*ins), writes=[out])
            elif kind == 'index_select':
                src, idxv = ins[0], ins[1]
                dim = plan['dim']

                def body(idx, src=src, idxv=idxv, dim=dim):
                    iv = self.view_expr(R, idxv, [idx[dim]])
                    sidx = list(idx)
                    sidx[dim] = 'vsel'
                    n = R.shape(src)[dim]
                    return (f'const long long vsel = {iv}; if (vsel < 0 || vsel >= {n}) __trap(); '
                            + self.store_root(R, out, idx, self.view_expr(R, src, sidx)))
                if place == 'prefetch':
                    # rows arrive asynchronously one iteration ahead; the index is still validated at use
                    n = R.shape(src)[dim]
                    iv = self.view_expr(R, idxv, ['0'] * len(R.shape(idxv)))
                    add(op.name + '_check', f'if (tid == 0) {{ const long long vsel = {iv}; '
                        f'if (vsel < 0 || vsel >= {n}) __trap(); }}', reads=deps_of(idxv))
                    pending_prefetch.append((op, src, idxv, dim))
                    continue
                add(op.name, self.loop_over(R, out, body), reads=deps_of(idxv), writes=[out])
            elif kind == 'product':
                self.emit_product(R, op, plan, add, deps_of, exchanges)
            elif kind in ('reduce_local', 'reduce_sharded'):
                self.emit_reduce(R, op, plan, add, deps_of, exchanges)
            elif kind in ('log_softmax_sharded', 'log_softmax_local'):
                self.emit_log_softmax(R, op, plan, add, deps_of, exchanges)
            else:
                raise LoweringError(f'vector_resident: no emitter for {t} ({kind}, {place})')
        # region outputs
        final = []

        def partition(name):
            """(view axis, N) of a value whose root is partitioned over the CTAs, else None (replicated)."""
            v = R.views[name]
            shard = R.shard.get(v.root)
            if shard is None:
                return None
            if shard[0] not in v.dims:
                raise LoweringError(f'vector_resident: output {name} drops its partitioned dimension')
            return v.dims.index(shard[0]), shard[1]
        for name in outputs:
            tgt = out_targets[name]
            if tgt[0] == 'carry':
                # LATCH-TRAIN-CORRECTNESS: a carry is read whole by every CTA of the next iteration (and by the
                # output region) without an exchange; a partitioned value has only its own shard in each CTA, so
                # storing it gave other CTAs' elements from unwritten storage (NaN predictions on a GradLev loop that
                # carries x W1). Refused before execution: carried values must be replicated.
                if partition(name) is not None:
                    raise LoweringError(f'vector_resident: loop-carried value {name} is partitioned over CTAs '
                                        f'{partition(name)}; carried values must be replicated in every CTA')
                j = tgt[1]
                shape = R.shape(name)
                ptr = self.carry_ptr(j, 'next' if R.label == 'body' else 'cur')
                st = _rowmajor(shape)

                def cbody(idx, name=name, ptr=ptr, st=st):
                    off = ' + '.join(f'({i}) * {s}' for i, s in zip(idx, st)) or '0'
                    return f'{ptr}[{off}] = {self.view_expr(R, name, idx)};'
                croot = next((n for n, e in R.external.items() if e.get('kind') == 'carry' and e['slot'] == j), None)
                carry_out[j] = add(f'out_{name}', self.loop_over(R, name, cbody), reads=deps_of(name),
                                   writes=[croot] if self.align and croot else ())
                final.append(carry_out[j])
            elif tgt[0] == 'state':
                sroot = tgt[1]
                owner = R.owner.get(sroot)
                shard = R.shard.get(R.views[name].root)
                vshard = None
                if shard is not None:
                    v = R.views[name]
                    vshard = (v.dims.index(shard[0]), shard[1])
                if owner is None or vshard is None or vshard != owner:
                    raise LoweringError(f'vector_resident: state {sroot} update partition {vshard} differs from its '
                                        f'read partition {owner}; in-place ownership not provable')
                info = R.external[sroot]
                st = R.values[sroot].strides
                dt = R.dtype(sroot)

                def sbody(idx, name=name, info=info, st=st, dt=dt, sroot=sroot):
                    if R.place[sroot] == 'resident':
                        return f'{self.load_root(R, sroot, idx)} = {self.view_expr(R, name, idx)};'
                    off = ' + '.join(f'(long long)({i}) * {s}LL' for i, s in zip(idx, st) if s) or '0'
                    return f'reinterpret_cast<{CTYPE[dt]}*>({info["ptr"]})[{off}] = {self.view_expr(R, name, idx)};'
                def ssplit(idx, name=name, info=info, st=st, dt=dt, sroot=sroot):
                    if R.place[sroot] == 'resident':
                        tgt = self.load_root(R, sroot, idx)
                    else:
                        off = ' + '.join(f'(long long)({i}) * {s}LL' for i, s in zip(idx, st) if s) or '0'
                        tgt = f'reinterpret_cast<{CTYPE[dt]}*>({info["ptr"]})[{off}]'
                    return self.view_expr(R, name, idx), tgt
                code = (self.state_vec_loop(R, name, sroot, vshard) if self.state_vec else None) \
                    or self.loop_over(R, name, sbody, shard=vshard, split_fn=ssplit)
                # write-after-read: in-place owner writes follow every other step of the iteration
                prior = [s.name for s in steps]
                sname = add(f'out_{name}', code, reads=deps_of(name), writes=[sroot] if self.align else ())
                steps[-1].deps |= set(prior)
                final.append(sname)
            elif tgt[0] == 'pred':
                ptr, dt = tgt[1], tgt[2]
                shape = R.shape(name)
                st = _rowmajor(shape)

                vshard = partition(name)
                if vshard is None:
                    def pbody(idx, name=name, ptr=ptr, st=st, dt=dt):
                        off = ' + '.join(f'({i}) * {s}' for i, s in zip(idx, st)) or '0'
                        return f'if (cta == 0) reinterpret_cast<{CTYPE[dt]}*>({ptr})[{off}] = {self.view_expr(R, name, idx)};'
                    final.append(add(f'out_{name}', self.loop_over(R, name, pbody), reads=deps_of(name)))
                else:
                    # LATCH-TRAIN-CORRECTNESS: a partitioned prediction (e.g. head logits over the vocabulary shards)
                    # is stored by its owners, each CTA its own shard; CTA 0 alone held only its shard, so the other
                    # elements were never written. The output region is the program's last step (no later reader).
                    def obody(idx, name=name, ptr=ptr, st=st, dt=dt):
                        off = ' + '.join(f'({i}) * {s}' for i, s in zip(idx, st)) or '0'
                        return f'reinterpret_cast<{CTYPE[dt]}*>({ptr})[{off}] = {self.view_expr(R, name, idx)};'
                    final.append(add(f'out_{name}', self.loop_over(R, name, obody, shard=vshard), reads=deps_of(name)))
        for op, src, idxv, dim in pending_prefetch:
            slots = {R.external[r]['slot'] for r in leaves(idxv) if R.place.get(r) == 'carry'}
            name = add(op.name + '_prefetch', self.prefetch_code(R, op.outputs[0], src, idxv, dim))
            steps[-1].deps |= {carry_out[j] for j in slots}
            if self.L.options.get('vres_prefetch_place', 'early') == 'head':
                # issue together with the bandwidth-bound head pass (its vector staging precedes it)
                steps[-1].deps |= {s.name for s in steps if s.name.endswith('_vec') and
                                   R.plans.get(s.name[:-4], {}).get('fuse_grad')}
        return steps

    def _prefetch_vec16(self, R, out, src, idxv, dim, n):
        """16-byte asynchronous copies when the selected rows are contiguous and 16-byte aligned."""
        oshape, sv = list(R.shape(out)), R.views[src]
        size = BYTES[R.dtype(out)]
        if (not oshape or dim == len(oshape) - 1 or sv.fixed or sv.dims != list(range(len(oshape)))
                or list(R.shape(sv.root)[:dim]) + [1] + list(R.shape(sv.root)[dim + 1:]) != oshape[:dim] + [1] + oshape[dim + 1:]):
            return None
        st = R.values[sv.root].strides
        last = len(oshape) - 1
        if st[last] != 1 or (oshape[last] * size) % 16 or any((st[d] * size) % 16 for d in range(last) if oshape[d] != 1 or d == dim):
            return None
        epc = 16 // size
        nch = _numel(oshape) // epc
        code = [f'for (int vc = tid; vc < {nch}; vc += {THREADS}) {{', f'  int vr = vc * {epc};']
        idx = [None] * len(oshape)
        for d in reversed(range(len(oshape))):
            code.append(f'  const int vd{d} = vr % {oshape[d]}; vr /= {oshape[d]};')
            idx[d] = f'vd{d}'
        iv = self.view_expr(R, idxv, [idx[dim]])
        sidx = list(idx)
        sidx[dim] = 'vsel'
        code.append(f'  const long long vsel = {iv};')
        code.append(f'  if (vsel >= 0 && vsel < {n}) vres::cp_async16(&{self.load_root(R, out, idx)}, '
                    f'&{self.view_expr(R, src, sidx)});')
        code.append('}')
        return '\n'.join(code)

    def prefetch_code(self, R, out, src, idxv, dim, which='next', parity='(par ^ 1)', sync=False):
        """Copy the rows selected by the index evaluated on the given carry buffer into one parity buffer."""
        n = R.shape(src)[dim]
        dt = R.dtype(out)
        size = BYTES[dt]
        saved = self._carry_which, self._prefetch_parity
        self._carry_which, self._prefetch_parity = which, parity
        try:
            if not sync:
                vec = self._prefetch_vec16(R, out, src, idxv, dim, n)
                if vec:
                    return vec + '\nvres::cp_async_commit();'
            def body(idx):
                iv = self.view_expr(R, idxv, [idx[dim]])
                sidx = list(idx)
                sidx[dim] = 'vsel'
                srcx = self.view_expr(R, src, sidx)
                dst = self.load_root(R, out, idx)
                if sync:
                    return f'const long long vsel = {iv}; if (vsel < 0 || vsel >= {n}) __trap(); {dst} = {srcx};'
                if size not in (4, 8, 16):
                    return f'const long long vsel = {iv}; if (vsel >= 0 && vsel < {n}) {dst} = {srcx};'
                return (f'const long long vsel = {iv}; if (vsel >= 0 && vsel < {n}) '
                        f'vres::cp_async{size}(&{dst}, &{srcx});')
            code = self.loop_over(R, out, body)
        finally:
            self._carry_which, self._prefetch_parity = saved
        return code if sync else code + '\nvres::cp_async_commit();'

    # products ------------------------------------------------------------
    def emit_product(self, R, op, plan, add, deps_of, exchanges):
        if plan.get('fuse_grad'):
            return self.emit_fused_head(R, op, plan, add, deps_of, exchanges)
        if plan.get('fused_from'):
            return self.emit_fused_grad_reduce(R, op, plan, add, deps_of, exchanges)
        out = op.outputs[0]
        K, F = plan['K'], plan['F']
        mdt = plan['matrix_dtype']
        mroot = plan['matrix_root']
        info = R.external[mroot]
        mptr = f'(reinterpret_cast<const {CTYPE[mdt]}*>({info["ptr"]}) + {plan["base"]}LL)'
        sK, sF = plan['sK'], plan['sF']
        if R.place[mroot] == 'resident':
            # owner-local partition in shared memory: local row-major strides, shard axis offset by -lo
            axis, N = R.owner[mroot]
            lo_name, _, ch = self.group(N)
            lshape = list(R.shape(mroot))
            lshape[axis] = ch
            lst = _rowmajor(lshape)
            mv = R.views[plan['vec_view_matrix']]
            sK, sF = lst[plan['kroot']], lst[plan['froot']]
            if plan['form'] == 'dot' and sK != 1 or plan['form'] == 'axpy' and sF != 1:
                raise LoweringError('vector_resident: resident partition changes the product form')
            base = sum(lst[d] * i for d, i in mv.fixed.items())
            mptr = (f'(reinterpret_cast<const {CTYPE[mdt]}*>({self.smem_ptr(R, mroot)}) + {base}LL '
                    f'- (long long){lo_name} * {lst[axis]}LL)')
        vec = plan['vec']
        vshape = R.shape(vec)
        kpos = len(vshape) - 1 if plan['vec_is_a'] else len(vshape) - 2

        def vec_at(kexpr):
            idx = ['0'] * len(vshape)
            idx[kpos] = kexpr
            e = self.view_expr(R, vec, idx)
            return f'__bfloat162float({e})' if R.dtype(vec) == 'bfloat16' else f'(float)({e})'
        self._step_reads.setdefault(op.name, set()).add(plan['matrix_root'])
        reads = deps_of(vec)
        if plan['split'] == 'F':
            lo, hi, ch = self.group(F)
            # stage the full vector (K) as FP32 in shared memory
            vb = self.transient((R.label, 'vb_' + op.name), K * 4)
            vptr = f'reinterpret_cast<float*>(vres_smem + {vb})'
            self._ctx = {'idx': [('k' if d == kpos else '0') for d in range(len(vshape))], 'shape': list(vshape),
                         'shard': None}
            try:
                vk = vec_at('k')
            finally:
                self._ctx = None
            self._cap['emap'] = ('s1', K, None)
            stage = f'for (int k = tid; k < {K}; k += {THREADS}) {vptr}[k] = {vk};'
            s_stage = add(op.name + '_vec', stage, reads=reads)
            optr = self.smem_ptr(R, out)
            if plan['form'] == 'dot':
                glob = self.row_hints(R, mroot)
                code = (self._dot_rows_pipelined(mptr, mdt, sF, K, lo, hi, vptr, lambda f: f'{optr}[({f}) - {lo}]', F,
                                                 plan['matrix_root'])
                        if glob and self.rows_pipeline and self._vec_width(mdt, K, sF) > 1
                        and K % (32 * self._vec_width(mdt, K, sF)) == 0 else None) or \
                    self._dot_rows(mptr, mdt, sF, K, lo, hi, vptr, lambda f: f'{optr}[({f}) - {lo}]', rows_per_cta=ch)
            else:
                code = self._axpy_cols(mptr, mdt, sK, K, lo, hi, ch, vptr, optr)
            add(op.name, code, reads={(R.label, 'vb_' + op.name)}, writes=[out])
            self._fix_dep(op.name, s_stage)
        else:
            lo, hi, ch = self.group(K)
            ex = exchanges[plan['exchange']]
            if not ex.offset:
                ex.offset = self.L.alloc_arena(2 * self.G * F)
                ex.reduced_offset = self.L.alloc_arena(2 * F)
            vb = self.transient((R.label, 'vb_' + op.name), ch * 4)
            vptr = f'reinterpret_cast<float*>(vres_smem + {vb})'
            stage = f'for (int k = tid; k < {ch}; k += {THREADS}) {{ const int kg = {lo} + k; {vptr}[k] = kg < {hi} ? {vec_at("kg")} : 0.0f; }}'
            s_stage = add(op.name + '_vec', stage, reads=reads)
            part = f'(arena + {ex.offset}LL + (long long)par * {self.G * F}LL + (long long)cta * {F}LL)'
            if plan['form'] == 'axpy':
                code = self._axpy_rows_partial(mptr, mdt, sK, F, lo, hi, vptr, part,
                                               glob=self.row_hints(R, mroot), K=K, root=plan['matrix_root'])
            else:
                code = self._dot_partial(mptr, mdt, sF, F, lo, hi, vptr, part)
            s_part = add(op.name + '_part', code)
            self._fix_dep(s_part, s_stage)
            red = f'(arena + {ex.reduced_offset}LL + (long long)par * {F}LL)'
            flo, fhi, fch = self.group(F)
            pbase = f'(arena + {ex.offset}LL + (long long)par * {self.G * F}LL)'
            ng = max(1, THREADS // fch)
            rsb = self.scratch(fch * ng * 4)
            rsp = f'reinterpret_cast<float*>(vres_smem + {rsb})'
            if self.rs_warp:
                rs = self._rs_warp(fch, flo, fhi, red, lambda pexpr: f's += {pbase}[(long long)({pexpr}) * {F}LL + f];')
            else:
              rs = (f'{{ const int c = tid % {fch}, gq = tid / {fch}; '
                  f'if (gq < {ng}) {{ float s = 0.0f; const int f = {flo} + c; '
                  f'if (f < {fhi}) for (int p = gq; p < {self.G}; p += {ng}) s += {pbase}[(long long)p * {F}LL + f]; '
                  f'{rsp}[gq * {fch} + c] = s; }} __syncthreads(); '
                  f'if (tid < {fch} && {flo} + tid < {fhi}) {{ float s = 0.0f; for (int q = 0; q < {ng}; ++q) s += {rsp}[q * {fch} + tid]; '
                  f'{red}[{flo} + tid] = s; }} __syncthreads(); }}')
            s_rs = add(op.name + '_rs', rs, xdeps=[s_part])
            optr = self.smem_ptr(R, out)
            ag = f'for (int f = tid; f < {F}; f += {THREADS}) {optr}[f] = {red}[f];'
            self._cap['emap'] = ('s1', F, None) if _numel(R.shape(out)) == F else None
            add(op.name, ag, xdeps=[s_rs], writes=[out])

    def emit_fused_head(self, R, op, plan, add, deps_of, exchanges):
        """One pass over the head matrix: logits (dot form), per-warp online (max, sum exp) and
        sum_v exp(l_v - m) M_v and sum_v c_v M_v accumulators; CTA partials for the lse and gradient exchanges."""
        out = op.outputs[0]
        gop = next(o for o in R.ops if o.name == plan['fuse_grad'])
        gplan = R.plans[gop.name]
        K, F = plan['K'], plan['F']
        mdt = plan['matrix_dtype']
        info = R.external[plan['matrix_root']]
        mptr = f'(reinterpret_cast<const {CTYPE[mdt]}*>({info["ptr"]}) + {plan["base"]}LL)'
        sF = plan['sF']
        vw = self._vec_width(mdt, K, sF)
        if vw == 1 or K % (32 * vw):
            raise LoweringError('vector_resident: head fusion needs vectorizable rows with K % (32*vec) == 0')
        nk = K // (32 * vw)
        lo, hi, ch = self.group(F)
        vec = plan['vec']
        vshape = R.shape(vec)
        kpos = len(vshape) - 1 if plan['vec_is_a'] else len(vshape) - 2
        idx = ['0'] * len(vshape)
        idx[kpos] = 'k'
        self._ctx = {'idx': list(idx), 'shape': list(vshape), 'shard': None}
        try:
            ve = self.view_expr(R, vec, idx)
        finally:
            self._ctx = None
        ve = f'__bfloat162float({ve})' if R.dtype(vec) == 'bfloat16' else f'(float)({ve})'
        vb = self.transient((R.label, 'vb_' + op.name), K * 4)
        vptr = f'reinterpret_cast<float*>(vres_smem + {vb})'
        self._cap['emap'] = ('s1', K, None)
        s_stage = add(op.name + '_vec', f'for (int k = tid; k < {K}; k += {THREADS}) {vptr}[k] = {ve};',
                      reads=deps_of(vec))
        # c_v expression (independent of the softmax)
        cname = gplan['fused_c']
        onehot = gplan.get('fused_onehot')
        cshape = R.shape(cname)
        cidx = ['0'] * len(cshape)
        cidx[-1] = 'vrow'
        cexpr = self.view_expr(R, cname, cidx)
        UR = 4   # UR=8 spills (measured 82.2 vs 68.7 ms/block)
        # exchanges: lse pairs (shared with the log_softmax op) and the gradient vector partials (2 x K)
        lex = Exchange(f'{R.label}_fusedlse_{op.name}', 2, 'lse')
        lex.offset = self.L.alloc_arena(2 * self.G * 2)
        gex = Exchange(f'{R.label}_fusedgrad_{gop.name}', 2 * K, 'vector')
        gex.offset = self.L.alloc_arena(2 * self.G * 2 * K)
        gex.reduced_offset = self.L.alloc_arena(2 * K)
        self.L.exchanges += [lex, gex]
        self.fused = getattr(self, 'fused', {})
        self.fused[op.name] = {'lse': lex, 'grad': gex, 'K': K}
        lpart = f'(arena + {lex.offset}LL + (long long)par * {2 * self.G}LL)'
        gpart = f'(arena + {gex.offset}LL + (long long)par * {self.G * 2 * K}LL + (long long)cta * {2 * K}LL)'
        optr = self.smem_ptr(R, out)
        red = self.scratch((2 * K + 2 * WARPS) * 4)
        rptr = f'reinterpret_cast<float*>(vres_smem + {red})'
        if mdt == 'bfloat16':
            ld = 'vres::bf8(*reinterpret_cast<const uint4*>(row + i * {s}), e[u][i]);'.format(s=32 * vw)
        else:
            ld = ('{ const float4 q = *reinterpret_cast<const float4*>(row + i * %d); '
                  'e[u][i][0] = q.x; e[u][i][1] = q.y; e[u][i][2] = q.z; e[u][i][3] = q.w; }' % (32 * vw))
        if onehot:
            cterm, cfma = '', ''
        else:
            cterm = f'        const int vrow = f0 + u;\n        const float cu = (float)({cexpr});'
            cfma = ' cacc[i][j] = fmaf(cu, e[u][i][j], cacc[i][j]);'
        pipe = self.L.options.get('vres_head_pipeline', True)
        if pipe is True and int(self.L.options.get('vres_head_rows', 4)) * nk > 12:
            pipe = False      # double buffer would exceed the register budget (e.g. K = 1024): historical loop
            self.L.notes.append(f'head pipeline off for {plan["matrix_root"]}: {nk} chunks per lane per row')
        rowloop = (self._fused_head_rows_pipelined(plan, mptr, mdt, sF, nk, vw, lo, hi, optr, cterm, onehot, F)
                   if pipe else f'''  for (int f0 = {lo} + warp * {UR}; f0 < {hi}; f0 += {WARPS * UR}) {{
    float e[{UR}][{nk}][{vw}];
    float l[{UR}];
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      l[u] = -INFINITY;
      if (f0 + u < {hi}) {{
        const auto* row = {mptr} + (long long)(f0 + u) * {sF}LL + lane * {vw};
#pragma unroll
        for (int i = 0; i < {nk}; ++i) {ld}
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < {nk}; ++i)
#pragma unroll
          for (int j = 0; j < {vw}; ++j) d = fmaf(e[u][i][j], a[i][j], d);
        l[u] = vres::warp_sum(d);
        if (lane == 0) {optr}[(f0 + u) - {lo}] = l[u];
      }}
    }}
    float mn = mw;
#pragma unroll
    for (int u = 0; u < {UR}; ++u) mn = fmaxf(mn, l[u]);
    const float sc = mw == -INFINITY ? 0.0f : expf(mw - mn);
    sw *= sc;
#pragma unroll
    for (int i = 0; i < {nk}; ++i)
#pragma unroll
      for (int j = 0; j < {vw}; ++j) acc[i][j] *= sc;
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      if (f0 + u < {hi}) {{
        const float pu = expf(l[u] - mn);
        sw += pu;
{cterm}
#pragma unroll
        for (int i = 0; i < {nk}; ++i)
#pragma unroll
          for (int j = 0; j < {vw}; ++j) {{ acc[i][j] = fmaf(pu, e[u][i][j], acc[i][j]);{cfma} }}
      }}
    }}
    mw = mn;
  }}
''')
        code = f'''{{
  float a[{nk}][{vw}], acc[{nk}][{vw}], cacc[{nk}][{vw}];
#pragma unroll
  for (int i = 0; i < {nk}; ++i)
#pragma unroll
    for (int j = 0; j < {vw}; ++j) {{ a[i][j] = {vptr}[lane * {vw} + i * {32 * vw} + j]; acc[i][j] = 0.0f; cacc[i][j] = 0.0f; }}
  float mw = -INFINITY, sw = 0.0f;
{rowloop}  // CTA combine (fixed warp order): rescale warps to the CTA maximum
  if (lane == 0) {{ {rptr}[warp] = mw; {rptr}[{WARPS} + warp] = sw; }}
  __syncthreads();
  float M = -INFINITY;
#pragma unroll
  for (int w = 0; w < {WARPS}; ++w) M = fmaxf(M, {rptr}[w]);
  const float myscale = mw == -INFINITY ? 0.0f : expf(mw - M);
  float* wacc = {rptr} + {2 * WARPS};
  for (int k = tid; k < {2 * K}; k += {THREADS}) wacc[k] = 0.0f;
  __syncthreads();
  for (int w = 0; w < {WARPS}; ++w) {{          // fixed warp order, one warp at a time
    if (warp == w) {{
#pragma unroll
      for (int i = 0; i < {nk}; ++i)
#pragma unroll
        for (int j = 0; j < {vw}; ++j) {{ wacc[lane * {vw} + i * {32 * vw} + j] += acc[i][j] * myscale;
          wacc[{K} + lane * {vw} + i * {32 * vw} + j] += cacc[i][j]; }}
    }}
    __syncthreads();
  }}
  for (int k = tid; k < {2 * K}; k += {THREADS}) {gpart}[k] = wacc[k];
  if (tid == 0) {{
    float S = 0.0f;
    for (int w = 0; w < {WARPS}; ++w) {{ const float mw_ = {rptr}[w]; if (mw_ != -INFINITY) S += {rptr}[{WARPS} + w] * expf(mw_ - M); }}
    {lpart}[2 * cta] = M; {lpart}[2 * cta + 1] = S;
  }}
  __syncthreads();
}}'''
        self._step_reads.setdefault(op.name, set()).add(plan['matrix_root'])
        s_main = add(op.name, code, reads=deps_of(cname), writes=[out])
        self._fix_dep(s_main, s_stage)
        self.fused[op.name]['step'] = s_main

    def l2_plan(self):
        """One L2 residency budget shared by every global matrix the body streams each iteration:
        explicit (global, owner-partitioned) state first, then loop-invariant parameters, each matrix
        once however many products read it. Returns root -> resident rows per 64."""
        if getattr(self, '_l2_plan', None) is not None:
            return self._l2_plan
        budget = self.L.options.get('vres_l2_resident_bytes', 'auto')
        if budget == 'auto':
            budget = int(0.55 * 50 * 2**20)
        left = int(budget or 0)
        Rb = self._body
        mats = {}
        for plan in Rb.plans.values():
            hinted = (((plan.get('form'), plan.get('split')) in (('dot', 'F'), ('axpy', 'K'))
                       and self.row_hints(Rb, plan['matrix_root']))
                      or plan.get('fuse_grad') or plan.get('fused_from'))      # emitters that apply the policy
            if plan.get('kind') == 'product' and hinted and Rb.place.get(plan['matrix_root']) == 'global':
                root = plan['matrix_root']
                mats[root] = (Rb.external.get(root, {}).get('role') != 'state',
                              _numel(Rb.shape(root)) * BYTES[Rb.dtype(root)])
        self._l2_plan = {}
        for root, (_, nbytes) in sorted(mats.items(), key=lambda kv: kv[1]):
            q = max(0, min(64, (64 * left) // max(1, nbytes)))
            self._l2_plan[root] = q
            left -= q * nbytes // 64
        return self._l2_plan

    def row_hints(self, R, root):
        """Pipelined / L2-hinted row loops apply to loop-invariant global matrices only: global explicit
        state is rewritten every iteration and keeps the historical emitters (measured: the hinted loops
        lost on the 1024 x 4096 global-state shape)."""
        return self.rows_pipeline and R.place.get(root) == 'global' and R.external.get(root, {}).get('role') != 'state'

    def l2_policy(self, F, row_bytes, root=None):
        """Rows of a loop-invariant streamed matrix kept L2-resident: (numerator p of p/64 groups, mode).

        vres_l2_resident_bytes: total bytes of the matrix to keep resident across iterations (0: plain loads
        everywhere, the historical behavior); the remaining rows are loaded with the evict-first streaming
        hint so they do not displace the resident rows. Cache hints never change values.
        """
        budget = self.L.options.get('vres_l2_resident_bytes', 'auto')
        if budget == 'auto':
            # H100 SXM: 50 MiB L2; measured best near 0.55 of it for the GPT-2 head (24-28 MiB, GRADLEV-BLOCK-60)
            budget = int(0.55 * 50 * 2**20)
        budget = int(budget or 0)
        mode = self.L.options.get('vres_l2_policy', 'hint')
        if mode not in ('cs', 'hint'):
            raise LoweringError("vres_l2_policy must be 'cs' or 'hint'")
        if budget <= 0:
            return 0, 'plain'
        if root is not None and root in self.l2_plan():
            return self._l2_plan[root], mode
        return max(0, min(64, round(64 * budget / (F * row_bytes)))), mode

    def _fused_head_rows_pipelined(self, plan, mptr, mdt, sF, nk, vw, lo, hi, optr, cterm, onehot, F):
        """Same per-warp row assignment and arithmetic order as the historical loop, with the next row
        group's raw 16-byte loads issued before the current group is consumed (register double buffer),
        raw values unpacked at each use, the rescale skipped when the running maximum is unchanged
        (x * 1.0f == x exactly) and an optional per-row L2 residency policy."""
        UR = int(self.L.options.get('vres_head_rows', 4))
        step = WARPS * UR
        raw = 'uint4' if mdt == 'bfloat16' else 'float4'
        unpack = ('vres::bf8(q, e);' if mdt == 'bfloat16' else 'e[0] = q.x; e[1] = q.y; e[2] = q.z; e[3] = q.w;')
        late = self.L.options.get('vres_head_late_unpack', True)
        unpack2 = unpack.replace('vres::bf8(', 'vres::bf8_late(') if late else unpack
        p, mode = self.l2_policy(F, sF * BYTES[mdt], plan['matrix_root'])
        self.L.notes.append(f'head pipeline: {plan["matrix_root"]} next-group loads in flight; '
                            f'L2 policy {mode}, resident row groups {p}/64')
        addr = f'reinterpret_cast<const {raw}*>({mptr} + (long long)(F_) * {sF}LL + lane * {vw} + i * {32 * vw})'
        if mode == 'plain':
            pre, ld = '', f'*{addr}'
        elif mode == 'cs':
            pre, ld = '', f'(RES_ ? *{addr} : __ldcs({addr}))'
        else:
            pre = ('  unsigned long long vpol_keep, vpol_strm;\n'
                   '  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(vpol_keep));\n'
                   '  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(vpol_strm));\n')
            ld = f'vres::ld16h({addr}, RES_ ? vpol_keep : vpol_strm)'

        def loads(dst, fexpr, res):
            body = ld.replace('F_', f'{fexpr} + u').replace('RES_', res)
            return (f'#pragma unroll\n    for (int u = 0; u < {UR}; ++u)\n#pragma unroll\n'
                    f'      for (int i = 0; i < {nk}; ++i) if ({fexpr} + u < {hi}) {dst}[u][i] = {body};')

        def resident(fexpr):
            if mode == 'plain':
                return 'false'
            g = f'(({fexpr}) - {lo}) / {UR}'
            return f'((({g}) + 1) * {p} / 64 != ({g}) * {p} / 64)'
        cfma = '' if onehot else ' cacc[i][j] = fmaf(cu, e[j], cacc[i][j]);'
        # optional L2 prefetch (after this CTA's pass) of the first streamed row groups of the next pass
        pf_bytes = int(self.L.options.get('vres_l2_prefetch_bytes', 0) or 0)
        pf = ''
        if pf_bytes > 0 and mode == 'hint':
            row_bytes = sF * BYTES[mdt]
            n_pf = pf_bytes // (self.G * UR * row_bytes)
            gpf = min(-(-F // (self.G * UR)) + 1, round(n_pf * 64 / max(1, 64 - p)))
            self.L.notes.append(f'head L2 prefetch: first {gpf} row groups per CTA (streamed ones) after each pass')
            pf = (f'''  for (int g = tid; g < {gpf}; g += {THREADS}) {{
    if (((g + 1) * {p} / 64) != (g * {p} / 64)) continue;
    const int fr = {lo} + g * {UR};
    if (fr >= {hi}) continue;
    const unsigned nb = (unsigned)(min({UR}, {hi} - fr) * {row_bytes});
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"({mptr} + (long long)fr * {sF}LL), "r"(nb) : "memory");
  }}
''')
        pmode = self.L.options.get('vres_head_pipeline', True)
        if pmode == 'l2':
            # register-free pipelining: each warp bulk-prefetches its row group D steps ahead into L2
            D = int(self.L.options.get('vres_head_prefetch_distance', 2))
            rowb = nk * 32 * vw * BYTES[mdt]

            def pfrow(fexpr):
                return (f'if (lane < {UR} && ({fexpr}) + lane < {hi}) asm volatile('
                        f'"cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"({mptr} + (long long)(({fexpr}) + lane) * '
                        f'{sF}LL), "r"({rowb}u) : "memory");')
            self.L.notes.append(f'head pipeline l2: row groups bulk-prefetched {D} steps ahead')
            prologue = (f'{pre}  int f0 = {lo} + warp * {UR};\n'
                        + ''.join(f'  {pfrow(f"f0 + {d * step}")}\n' for d in range(1, D)))
            per_iter = (f'    {pfrow(f"f0 + {D * step}")}\n    {raw} qa[{UR}][{nk}];\n'
                        f'    {{ const bool vres_res0 = {resident("f0")}; (void)vres_res0;\n    {loads("qa", "f0", "vres_res0")} }}')
            tail_copy = ''
        else:
            prologue = (f'{pre}  int f0 = {lo} + warp * {UR};\n  {raw} qa[{UR}][{nk}];\n'
                        f'  {{ const bool vres_res0 = {resident("f0")}; (void)vres_res0;\n    {loads("qa", "f0", "vres_res0")} }}\n')
            per_iter = (f'    {raw} qb[{UR}][{nk}];\n    const int f1 = f0 + {step};\n'
                        f'    {{ const bool vres_res1 = {resident("f1")}; (void)vres_res1;\n    {loads("qb", "f1", "vres_res1")} }}')
            tail_copy = (f'#pragma unroll\n    for (int u = 0; u < {UR}; ++u)\n#pragma unroll\n'
                         f'      for (int i = 0; i < {nk}; ++i) qa[u][i] = qb[u][i];\n')
        return f'''{prologue}  for (; f0 < {hi}; f0 += {step}) {{
{per_iter}
    float l[{UR}];
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      l[u] = -INFINITY;
      if (f0 + u < {hi}) {{
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < {nk}; ++i) {{
          const {raw} q = qa[u][i]; float e[{vw}]; {unpack}
#pragma unroll
          for (int j = 0; j < {vw}; ++j) d = fmaf(e[j], a[i][j], d);
        }}
        l[u] = vres::warp_sum(d);
        if (lane == 0) {optr}[(f0 + u) - {lo}] = l[u];
      }}
    }}
    float mn = mw;
#pragma unroll
    for (int u = 0; u < {UR}; ++u) mn = fmaxf(mn, l[u]);
    const float sc = mw == -INFINITY ? 0.0f : expf(mw - mn);
    sw *= sc;
    if (sc != 1.0f) {{
#pragma unroll
      for (int i = 0; i < {nk}; ++i)
#pragma unroll
        for (int j = 0; j < {vw}; ++j) acc[i][j] *= sc;
    }}
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      if (f0 + u < {hi}) {{
        const float pu = expf(l[u] - mn);
        sw += pu;
{cterm}
#pragma unroll
        for (int i = 0; i < {nk}; ++i) {{
          const {raw} q = qa[u][i]; float e[{vw}]; {unpack2}
#pragma unroll
          for (int j = 0; j < {vw}; ++j) {{ acc[i][j] = fmaf(pu, e[j], acc[i][j]);{cfma} }}
        }}
      }}
    }}
    mw = mn;
{tail_copy}  }}
{pf}'''

    def emit_fused_grad_reduce(self, R, op, plan, add, deps_of, exchanges):
        """gn[f] = sum_p exp(m_p - lse) acc_p[f] - sum_p cacc_p[f]  (reduce-scatter), then all-gather."""
        out = op.outputs[0]
        fz = self.fused[plan['fused_from']]
        K = fz['K']
        lex, gex = fz['lse'], fz['grad']
        lse_step, lse_ptr = self.fused_lse[plan['fused_from']]
        F = K
        flo, fhi, fch = self.group(F)
        ng = max(1, THREADS // fch)
        rsb = self.scratch(fch * ng * 4)
        rsp = f'reinterpret_cast<float*>(vres_smem + {rsb})'
        lpart = f'(arena + {lex.offset}LL + (long long)par * {2 * self.G}LL)'
        pbase = f'(arena + {gex.offset}LL + (long long)par * {self.G * 2 * K}LL)'
        red = f'(arena + {gex.reduced_offset}LL + (long long)par * {K}LL)'
        onehot = plan.get('fused_onehot')
        csub = '' if onehot else f' - {pbase}[(long long)p * {2 * K}LL + {K} + f]'
        csub_w = '' if onehot else f' - {pbase}[(long long)(pp) * {2 * K}LL + {K} + f]'
        corr = ''
        if onehot:
            mplan = R.plans[plan['fused_from']]
            mdt = mplan['matrix_dtype']
            minfo = R.external[mplan['matrix_root']]
            iv = self.view_expr(R, onehot, ['0'] * len(R.shape(onehot)))
            elem = f'reinterpret_cast<const {CTYPE[mdt]}*>({minfo["ptr"]})[{mplan["base"]}LL + (long long)({iv}) * {mplan["sF"]}LL + {flo} + tid]'
            elem = f'__bfloat162float({elem})' if mdt == 'bfloat16' else f'(float)({elem})'
            corr = f' s -= {elem};'
            corr_c = f' s -= {elem.replace(f"{flo} + tid]", f"{flo} + c]")};'
        else:
            corr_c = ''
        if self.rs_warp:
            def term(pexpr):
                return (f'const float mp = {lpart}[2 * {pexpr}]; '
                        f'const float w = mp == -INFINITY ? 0.0f : expf(mp - lse_); '
                        f's += w * {pbase}[(long long)({pexpr}) * {2 * K}LL + f]{csub_w.replace("pp", pexpr)};')
            rs = self._rs_warp(fch, flo, fhi, red, term, pre=f'const float lse_ = {lse_ptr}; ',
                               post=corr_c)
        else:
          rs = (f'{{ const int c = tid % {fch}, gq = tid / {fch}; const float lse_ = {lse_ptr}; '
              f'if (gq < {ng}) {{ float s = 0.0f; const int f = {flo} + c; '
              f'if (f < {fhi}) for (int p = gq; p < {self.G}; p += {ng}) {{ const float mp = {lpart}[2 * p]; '
              f'const float w = mp == -INFINITY ? 0.0f : expf(mp - lse_); '
              f's += w * {pbase}[(long long)p * {2 * K}LL + f]{csub}; }} '
              f'{rsp}[gq * {fch} + c] = s; }} __syncthreads(); '
              f'if (tid < {fch} && {flo} + tid < {fhi}) {{ float s = 0.0f; for (int q = 0; q < {ng}; ++q) s += {rsp}[q * {fch} + tid];{corr} '
              f'{red}[{flo} + tid] = s; }} __syncthreads(); }}')
        s_rs = add(op.name + '_rs', rs, xdeps=[fz['step']])
        self._fix_dep(s_rs, lse_step)
        optr = self.smem_ptr(R, out)
        self._cap['emap'] = ('s1', F, None) if _numel(R.shape(out)) == F else None
        add(op.name, f'for (int f = tid; f < {F}; f += {THREADS}) {optr}[f] = {red}[f];', xdeps=[s_rs], writes=[out])

    def _rs_warp(self, fch, flo, fhi, red, term, pre='', post=''):
        """Reduce-scatter of this CTA's slice: one warp per column, lanes stride the G partials,
        fixed butterfly combine (deterministic; no shared scratch, no CTA barrier)."""
        return (f'{{ {pre}for (int c = warp; c < {fch}; c += {WARPS}) {{ const int f = {flo} + c;\n'
                f'  if (f < {fhi}) {{ float s = 0.0f;\n#pragma unroll\n'
                f'    for (int pq = 0; pq < {-(-self.G // 32)}; ++pq) {{ const int p = lane + 32 * pq; '
                f'if (p < {self.G}) {{ {term("p")} }} }}\n'
                f'    s = vres::warp_sum(s); if (lane == 0) {{ {post} {red}[f] = s; }} }} }} }}')

    def _fix_dep(self, step, dep):
        self._extra_deps.setdefault(step, set()).add(dep)

    @staticmethod
    def _vec_width(dt, K, stride, base_ok=True):
        if dt == 'bfloat16' and K % 8 == 0 and stride % 8 == 0 and base_ok:
            return 8
        if dt == 'float32' and K % 4 == 0 and stride % 4 == 0 and base_ok:
            return 4
        return 1

    def _load_vec(self, dt, vw, ptr_expr, dest):
        if vw == 8:
            return f'vres::bf8(*reinterpret_cast<const uint4*>({ptr_expr}), {dest});'
        if vw == 4:
            return f'{{ const float4 q = *reinterpret_cast<const float4*>({ptr_expr}); {dest}[0] = q.x; {dest}[1] = q.y; {dest}[2] = q.z; {dest}[3] = q.w; }}'
        cv = f'__bfloat162float(*({ptr_expr}))' if dt == 'bfloat16' else f'(float)(*({ptr_expr}))'
        return f'{dest}[0] = {cv};'

    def _dot_rows(self, mptr, dt, sF, K, lo, hi, vptr, out_at, UR=4, rows_per_cta=None):
        vw = self._vec_width(dt, K, sF)
        if vw > 1:
            return self._dot_rows_reg(mptr, dt, sF, K, lo, hi, vptr, out_at, vw, rows_per_cta)
        return f'''{{
  for (int f0 = {lo} + warp * {UR}; f0 < {hi}; f0 += {WARPS * UR}) {{
    float acc[{UR}];
#pragma unroll
    for (int u = 0; u < {UR}; ++u) acc[u] = 0.0f;
    for (int k = lane * {vw}; k < {K}; k += {32 * vw}) {{
      float a[{vw}];
#pragma unroll
      for (int j = 0; j < {vw}; ++j) a[j] = {vptr}[k + j];
#pragma unroll
      for (int u = 0; u < {UR}; ++u) {{
        if (f0 + u < {hi}) {{
          float e[{vw}];
          {self._load_vec(dt, vw, f"{mptr} + (long long)(f0 + u) * {sF}LL + k", "e")}
#pragma unroll
          for (int j = 0; j < {vw}; ++j) acc[u] = fmaf(e[j], a[j], acc[u]);
        }}
      }}
    }}
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      const float s = vres::warp_sum(acc[u]);
      if (lane == 0 && f0 + u < {hi}) {out_at("f0 + u")} = s;
    }}
  }}
}}'''

    def _dot_rows_reg(self, mptr, dt, sF, K, lo, hi, vptr, out_at, vw, rows_per_cta):
        """Warp-per-row dot form with the vector held in registers and all UR rows' loads in flight."""
        nk = -(-K // (32 * vw))
        exact = K % (32 * vw) == 0
        per_warp = -(-(rows_per_cta or 8 * WARPS) // WARPS)
        UR = max(1, min(8 if dt == 'bfloat16' else 4, per_warp))
        if dt == 'bfloat16':
            ld = 'const uint4 q = *reinterpret_cast<const uint4*>(row + kk); vres::bf8(q, e);'
        else:
            ld = 'const float4 q = *reinterpret_cast<const float4*>(row + kk); e[0] = q.x; e[1] = q.y; e[2] = q.z; e[3] = q.w;'
        guard = '' if exact else f'if (lane * {vw} + kk >= {K}) break; '
        return f'''{{
  float a[{nk}][{vw}];
#pragma unroll
  for (int i = 0; i < {nk}; ++i)
#pragma unroll
    for (int j = 0; j < {vw}; ++j) {{ const int kk = lane * {vw} + i * {32 * vw} + j; a[i][j] = kk < {K} ? {vptr}[kk] : 0.0f; }}
  for (int f0 = {lo} + warp * {UR}; f0 < {hi}; f0 += {WARPS * UR}) {{
    float acc[{UR}];
#pragma unroll
    for (int u = 0; u < {UR}; ++u) acc[u] = 0.0f;
    if (f0 + {UR} <= {hi}) {{
#pragma unroll
      for (int u = 0; u < {UR}; ++u) {{
        const auto* row = {mptr} + (long long)(f0 + u) * {sF}LL + lane * {vw};
#pragma unroll
        for (int i = 0; i < {nk}; ++i) {{
          const int kk = i * {32 * vw};
          float e[{vw}];
          {guard}{ld}
#pragma unroll
          for (int j = 0; j < {vw}; ++j) acc[u] = fmaf(e[j], a[i][j], acc[u]);
        }}
      }}
    }} else {{
      for (int u = 0; u < {UR}; ++u) {{
        if (f0 + u >= {hi}) break;
        const auto* row = {mptr} + (long long)(f0 + u) * {sF}LL + lane * {vw};
#pragma unroll
        for (int i = 0; i < {nk}; ++i) {{
          const int kk = i * {32 * vw};
          float e[{vw}];
          {guard}{ld}
#pragma unroll
          for (int j = 0; j < {vw}; ++j) acc[u] = fmaf(e[j], a[i][j], acc[u]);
        }}
      }}
    }}
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      const float s = vres::warp_sum(acc[u]);
      if (lane == 0 && f0 + u < {hi}) {out_at("f0 + u")} = s;
    }}
  }}
}}'''

    def _axpy_cols(self, mptr, dt, sK, K, lo, hi, ch, vptr, optr):
        vw = 4 if (dt == 'float32' and ch % 4 == 0 and sK % 4 == 0) else (8 if (dt == 'bfloat16' and ch % 8 == 0 and sK % 8 == 0) else 1)
        ct = -(-ch // vw)
        R_ = max(1, THREADS // ct)
        red = self.scratch(R_ * ch * 4)
        rptr = f'reinterpret_cast<float*>(vres_smem + {red})'
        return f'''{{
  const int g = tid / {ct}, cv = tid % {ct};
  float acc[{vw}];
#pragma unroll
  for (int j = 0; j < {vw}; ++j) acc[j] = 0.0f;
  const int f0 = {lo} + cv * {vw};
  if (g < {R_} && f0 < {hi}) {{
#pragma unroll 4
    for (int k = g; k < {K}; k += {R_}) {{
      float e[{vw}];
      {self._load_vec(dt, vw, f"{mptr} + (long long)k * {sK}LL + f0", "e")}
      const float a = {vptr}[k];
#pragma unroll
      for (int j = 0; j < {vw}; ++j) acc[j] = fmaf(e[j], a, acc[j]);
    }}
  }}
  if (g < {R_}) {{
#pragma unroll
    for (int j = 0; j < {vw}; ++j) if (cv * {vw} + j < {ch}) {rptr}[g * {ch} + cv * {vw} + j] = acc[j];
  }}
  __syncthreads();
  for (int c = tid; c < {ch}; c += {THREADS}) {{
    if ({lo} + c < {hi}) {{
      float s = 0.0f;
      for (int r = 0; r < {R_}; ++r) s += {rptr}[r * {ch} + c];
      {optr}[c] = s;
    }}
  }}
  __syncthreads();
}}'''

    def _row_policy(self, rows, row_bytes, root):
        """(resident-row predicate builder, policy prologue, load builder) for rows of a global matrix."""
        p, mode = self.l2_policy(rows, row_bytes, root)
        self.L.notes.append(f'row loads of {root}: L2 policy {mode}, resident rows {p}/64')
        if mode == 'hint':
            pre = ('unsigned long long vpol_keep, vpol_strm;\n'
                   'asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(vpol_keep));\n'
                   'asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(vpol_strm));\n')
            return (lambda r: f'((({r}) + 1) * {p} / 64 != ({r}) * {p} / 64)'), pre, \
                (lambda addr, res: f'vres::ld16h({addr}, ({res}) ? vpol_keep : vpol_strm)')
        if mode == 'cs':
            return (lambda r: f'((({r}) + 1) * {p} / 64 != ({r}) * {p} / 64)'), '', \
                (lambda addr, res: f'(({res}) ? *{addr} : __ldcs({addr}))')
        return (lambda r: 'false'), '', (lambda addr, res: f'*{addr}')

    def _dot_rows_pipelined(self, mptr, dt, sF, K, lo, hi, vptr, out_at, F, root):
        """Warp-per-row dot form over a global matrix with the next row group's 16-byte loads in flight
        (register double buffer) and the L2 row policy; per-row arithmetic identical to _dot_rows_reg."""
        vw = self._vec_width(dt, K, sF)
        nk = K // (32 * vw)
        # register budget: at most 12 raw 16-byte chunks per buffer (two buffers in flight); per-row
        # arithmetic does not depend on the number of rows per group
        if nk > 12:
            return None
        UR = max(1, min(4, 12 // nk))
        step = WARPS * UR
        raw = 'uint4' if dt == 'bfloat16' else 'float4'
        unpack = 'vres::bf8(q, e);' if dt == 'bfloat16' else 'e[0] = q.x; e[1] = q.y; e[2] = q.z; e[3] = q.w;'
        resident, pre, ld = self._row_policy(F, sF * BYTES[dt], root)

        def loads(dst, fexpr):
            addr = f'reinterpret_cast<const {raw}*>({mptr} + (long long)({fexpr} + u) * {sF}LL + lane * {vw} + i * {32 * vw})'
            return (f'#pragma unroll\n    for (int u = 0; u < {UR}; ++u)\n#pragma unroll\n      for (int i = 0; i < {nk}; ++i) '
                    f'if ({fexpr} + u < {hi}) {dst}[u][i] = {ld(addr, resident(f"{fexpr} + u - {lo}"))};')
        return f'''{{
  {pre}float a[{nk}][{vw}];
#pragma unroll
  for (int i = 0; i < {nk}; ++i)
#pragma unroll
    for (int j = 0; j < {vw}; ++j) a[i][j] = {vptr}[lane * {vw} + i * {32 * vw} + j];
  int f0 = {lo} + warp * {UR};
  {raw} qa[{UR}][{nk}];
  {loads('qa', 'f0')}
  for (; f0 < {hi}; f0 += {step}) {{
    {raw} qb[{UR}][{nk}];
    {loads('qb', f'f0 + {step}')}
    float acc[{UR}];
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      acc[u] = 0.0f;
      if (f0 + u < {hi}) {{
#pragma unroll
        for (int i = 0; i < {nk}; ++i) {{
          const {raw} q = qa[u][i]; float e[{vw}]; {unpack}
#pragma unroll
          for (int j = 0; j < {vw}; ++j) acc[u] = fmaf(e[j], a[i][j], acc[u]);
        }}
      }}
    }}
#pragma unroll
    for (int u = 0; u < {UR}; ++u) {{
      const float s = vres::warp_sum(acc[u]);
      if (lane == 0 && f0 + u < {hi}) {out_at("f0 + u")} = s;
    }}
#pragma unroll
    for (int u = 0; u < {UR}; ++u)
#pragma unroll
      for (int i = 0; i < {nk}; ++i) qa[u][i] = qb[u][i];
  }}
}}'''

    def _axpy_rows_partial(self, mptr, dt, sK, F, lo, hi, vptr, part, glob=False, K=None, root=''):
        vw = self._vec_width(dt, F, sK)
        if glob and self.rows_pipeline and vw > 1 and F % vw == 0 and F // vw <= THREADS:
            return self._axpy_rows_partial_batched(mptr, dt, sK, F, lo, hi, vptr, part, vw, K, root)
        ct = F // vw if F % vw == 0 else F
        if ct > THREADS:
            raise LoweringError('vector_resident: split-K product output too wide for one CTA pass')
        R_ = max(1, THREADS // ct)
        red = self.scratch(R_ * F * 4)
        rptr = f'reinterpret_cast<float*>(vres_smem + {red})'
        return f'''{{
  const int g = tid / {ct}, cv = tid % {ct};
  float acc[{vw}];
#pragma unroll
  for (int j = 0; j < {vw}; ++j) acc[j] = 0.0f;
  if (g < {R_}) {{
#pragma unroll 8
    for (int k = {lo} + g; k < {hi}; k += {R_}) {{
      float e[{vw}];
      {self._load_vec(dt, vw, f"{mptr} + (long long)k * {sK}LL + cv * {vw}", "e")}
      const float a = {vptr}[k - {lo}];
#pragma unroll
      for (int j = 0; j < {vw}; ++j) acc[j] = fmaf(e[j], a, acc[j]);
    }}
#pragma unroll
    for (int j = 0; j < {vw}; ++j) {rptr}[g * {F} + cv * {vw} + j] = acc[j];
  }}
  __syncthreads();
  for (int f = tid; f < {F}; f += {THREADS}) {{
    float s = 0.0f;
    for (int r = 0; r < {R_}; ++r) s += {rptr}[r * {F} + f];
    {part}[f] = s;
  }}
  __syncthreads();
}}'''

    def _axpy_rows_partial_batched(self, mptr, dt, sK, F, lo, hi, vptr, part, vw, K, root):
        """Same per-thread row order and accumulation as _axpy_rows_partial, with B rows' 16-byte loads
        issued as one explicit batch and the L2 row policy (rows of a global matrix)."""
        ct = F // vw
        R_ = max(1, THREADS // ct)
        B = 8
        raw = 'uint4' if dt == 'bfloat16' else 'float4'
        unpack = 'vres::bf8(q[b], e);' if dt == 'bfloat16' else 'e[0] = q[b].x; e[1] = q[b].y; e[2] = q[b].z; e[3] = q[b].w;'
        red = self.scratch(R_ * F * 4)
        rptr = f'reinterpret_cast<float*>(vres_smem + {red})'
        resident, pre, ld = self._row_policy(K, sK * BYTES[dt], root)
        addr = f'reinterpret_cast<const {raw}*>({mptr} + (long long)k * {sK}LL + cv * {vw})'
        return f'''{{
  {pre}const int g = tid / {ct}, cv = tid % {ct};
  float acc[{vw}];
#pragma unroll
  for (int j = 0; j < {vw}; ++j) acc[j] = 0.0f;
  if (g < {R_}) {{
    for (int k0 = {lo} + g; k0 < {hi}; k0 += {R_ * B}) {{
      {raw} q[{B}]; float av[{B}];
#pragma unroll
      for (int b = 0; b < {B}; ++b) {{ const int k = k0 + b * {R_}; if (k < {hi}) {{ q[b] = {ld(addr, resident(f"k - {lo}"))}; av[b] = {vptr}[k - {lo}]; }} }}
#pragma unroll
      for (int b = 0; b < {B}; ++b) {{
        if (k0 + b * {R_} < {hi}) {{
          float e[{vw}]; {unpack}
#pragma unroll
          for (int j = 0; j < {vw}; ++j) acc[j] = fmaf(e[j], av[b], acc[j]);
        }}
      }}
    }}
#pragma unroll
    for (int j = 0; j < {vw}; ++j) {rptr}[g * {F} + cv * {vw} + j] = acc[j];
  }}
  __syncthreads();
  for (int f = tid; f < {F}; f += {THREADS}) {{
    float s = 0.0f;
    for (int r = 0; r < {R_}; ++r) s += {rptr}[r * {F} + f];
    {part}[f] = s;
  }}
  __syncthreads();
}}'''

    def _dot_partial(self, *a, **k):
        raise LoweringError('vector_resident: split-K dot-form products are not implemented')

    # reductions ------------------------------------------------------------
    def emit_reduce(self, R, op, plan, add, deps_of, exchanges):
        src = R._input_refs(op)[0]
        out = op.outputs[0]
        sshape = R.shape(src)
        n = sshape[-1]
        rows = _numel(sshape[:-1])
        if rows != 1:
            raise LoweringError('vector_resident: reductions of more than one row are not implemented')
        red = self.L.alloc_smem(('block_red',), 32 * 4)
        rptr = f'reinterpret_cast<float*>(vres_smem + {red})'
        optr = self.smem_ptr(R, out)
        idx = ['0'] * (len(sshape) - 1)
        scale = f' * {_lit(1.0 / n, "float32")}' if plan['mean'] else ''
        e = self.view_expr(R, src, idx + ['j'])
        if R.dtype(src) != 'float32':
            e = f'(float)({e})'
        if plan['kind'] == 'reduce_local':
            if self.small_warp and n <= 1024:
                code = (f'if (warp == 0) {{ float s = 0.0f; for (int j = lane; j < {n}; j += 32) s += {e}; '
                        f's = vres::warp_sum(s); if (lane == 0) {optr}[0] = s{scale}; }}')
                add(op.name, code, reads=deps_of(src), writes=[out])
                self._warp0_steps.add(op.name)
                return
            if self.align and R.label == 'body':
                self._cap = {'acc': [], 'emap': None}
                self._ctx = {'idx': idx + ['j'], 'shape': list(sshape), 'shard': None}
                try:
                    e = self.view_expr(R, src, idx + ['j'])
                finally:
                    self._ctx = None
                if R.dtype(src) != 'float32':
                    e = f'(float)({e})'
                var = f'vrs_{out}'
                code = (f'float {var}; {{ float s = 0.0f; for (int j = tid; j < {n}; j += {THREADS}) s += {e}; '
                        f's = vres::block_sum(s, {rptr}); {var} = s{scale}; }}')
                self._cap['emap'], self._cap['reg'] = ('s1', _numel(sshape), None), True
                add(op.name, code, reads=deps_of(src), writes=[out])
                self._regs[(R.label, out)] = var
                return
            code = (f'{{ float s = 0.0f; for (int j = tid; j < {n}; j += {THREADS}) s += {e}; '
                    f's = vres::block_sum(s, {rptr}); if (tid == 0) {optr}[0] = s{scale}; }}')
            add(op.name, code, reads=deps_of(src), writes=[out])
            return
        lo, hi, ch = self.group(n)
        ex = Exchange(f'{R.label}_{op.name}', 1, 'scalar')
        ex.offset = self.L.alloc_arena(2 * self.G)
        self.L.exchanges.append(ex)
        part = f'(arena + {ex.offset}LL + (long long)par * {self.G}LL)'
        code = (f'{{ float s = 0.0f; for (int j = {lo} + tid; j < {hi}; j += {THREADS}) s += {e}; '
                f's = vres::block_sum(s, {rptr}); if (tid == 0) {part}[cta] = s; }}')
        s_part = add(op.name + '_part', code, reads=deps_of(src))
        if self.align and R.label == 'body':
            # every warp combines the partials; lane 0's sum is broadcast (identical value in every thread)
            var = f'vrs_{out}'
            comb = (f'float {var}; {{ float s = 0.0f; for (int p = lane; p < {self.G}; p += 32) s += {part}[p]; '
                    f's = vres::warp_sum(s); s = __shfl_sync(0xffffffffu, s, 0); {var} = s{scale}; }}')
            self._cap['reg'] = True
            add(op.name, comb, xdeps=[s_part], writes=[out])
            self._regs[(R.label, out)] = var
            return
        comb = (f'{{ if (warp == 0) {{ float s = 0.0f; for (int p = lane; p < {self.G}; p += 32) s += {part}[p]; '
                f's = vres::warp_sum(s); if (lane == 0) {optr}[0] = s{scale}; }} }}')
        add(op.name, comb, xdeps=[s_part], writes=[out])

    def emit_log_softmax(self, R, op, plan, add, deps_of, exchanges):
        src = R._input_refs(op)[0]
        out = op.outputs[0]
        sshape = R.shape(src)
        n = sshape[-1]
        if _numel(sshape[:-1]) != 1:
            raise LoweringError('vector_resident: log_softmax of more than one row is not implemented')
        red = self.L.alloc_smem(('block_red',), 32 * 4)
        rptr = f'reinterpret_cast<float*>(vres_smem + {red})'
        lse = self.transient((R.label, 'lse_' + op.name), 4)
        lptr = f'reinterpret_cast<float*>(vres_smem + {lse})'
        idx = ['0'] * (len(sshape) - 1)
        e = self.view_expr(R, src, idx + ['j'])
        reg = self.align and R.label == 'body' and (plan['kind'] == 'log_softmax_local'
                                                     or self.L.options.get('vres_lse_registers', True))
        lse_expr = f'vrs_lse_{op.name}' if reg else f'{lptr}[0]'
        if plan['kind'] == 'log_softmax_local':
            if reg:
                code = (f'float {lse_expr}; {{ float m = -INFINITY; for (int j = tid; j < {n}; j += {THREADS}) m = fmaxf(m, {e}); '
                        f'm = vres::block_max(m, {rptr}); float s = 0.0f; '
                        f'for (int j = tid; j < {n}; j += {THREADS}) s += expf({e} - m); '
                        f's = vres::block_sum(s, {rptr}); {lse_expr} = m + logf(s); }}')
                self._cap['reg'] = True
            else:
                code = (f'{{ float m = -INFINITY; for (int j = tid; j < {n}; j += {THREADS}) m = fmaxf(m, {e}); '
                        f'm = vres::block_max(m, {rptr}); float s = 0.0f; '
                        f'for (int j = tid; j < {n}; j += {THREADS}) s += expf({e} - m); '
                        f's = vres::block_sum(s, {rptr}); if (tid == 0) {lptr}[0] = m + logf(s); }}')
            s1 = add(op.name + '_lse', code, reads=deps_of(src))
        elif plan.get('pairs_from'):
            fz = self.fused[plan['pairs_from']]
            part = f'(arena + {fz["lse"].offset}LL + (long long)par * {2 * self.G}LL)'
            s_part = fz['step']
        else:
            lo, hi, ch = self.group(n)
            ex = Exchange(f'{R.label}_{op.name}', 2, 'lse')
            ex.offset = self.L.alloc_arena(2 * self.G * 2)
            self.L.exchanges.append(ex)
            part = f'(arena + {ex.offset}LL + (long long)par * {2 * self.G}LL)'
            code = (f'{{ float m = -INFINITY; for (int j = {lo} + tid; j < {hi}; j += {THREADS}) m = fmaxf(m, {e}); '
                    f'm = vres::block_max(m, {rptr}); float s = 0.0f; '
                    f'for (int j = {lo} + tid; j < {hi}; j += {THREADS}) s += expf({e} - m); '
                    f's = vres::block_sum(s, {rptr}); if (tid == 0) {{ {part}[2 * cta] = m; {part}[2 * cta + 1] = s; }} }}')
            s_part = add(op.name + '_part', code, reads=deps_of(src))
        if plan['kind'] != 'log_softmax_local':
            if self.L.options.get('vres_lse_registers', True):
                # same arithmetic order; every (max, sum) pair is loaded once into registers
                npl = -(-self.G // 32)
                self._cap = {'acc': [], 'emap': None}
                if reg:
                    # every warp combines; lane 0's sum broadcast -> the same value in every thread's register
                    self._cap['reg'] = True
                    comb = (f'float {lse_expr}; {{ float pm[{npl}], ps[{npl}];\n#pragma unroll\n'
                            f'  for (int q = 0; q < {npl}; ++q) {{ const int p = lane + 32 * q; pm[q] = p < {self.G} ? {part}[2 * p] : -INFINITY; '
                            f'ps[q] = p < {self.G} ? {part}[2 * p + 1] : 0.0f; }}\n'
                            f'  float M = -INFINITY;\n#pragma unroll\n  for (int q = 0; q < {npl}; ++q) M = fmaxf(M, pm[q]);\n'
                            f'  M = vres::warp_max(M); float S = 0.0f;\n#pragma unroll\n'
                            f'  for (int q = 0; q < {npl}; ++q) if (lane + 32 * q < {self.G} && ps[q] > 0.0f) S += ps[q] * expf(pm[q] - M);\n'
                            f'  S = vres::warp_sum(S); S = __shfl_sync(0xffffffffu, S, 0); {lse_expr} = M + logf(S); }}')
                else:
                  comb = (f'{{ if (warp == 0) {{ float pm[{npl}], ps[{npl}];\n#pragma unroll\n'
                        f'  for (int q = 0; q < {npl}; ++q) {{ const int p = lane + 32 * q; pm[q] = p < {self.G} ? {part}[2 * p] : -INFINITY; '
                        f'ps[q] = p < {self.G} ? {part}[2 * p + 1] : 0.0f; }}\n'
                        f'  float M = -INFINITY;\n#pragma unroll\n  for (int q = 0; q < {npl}; ++q) M = fmaxf(M, pm[q]);\n'
                        f'  M = vres::warp_max(M); float S = 0.0f;\n#pragma unroll\n'
                        f'  for (int q = 0; q < {npl}; ++q) if (lane + 32 * q < {self.G} && ps[q] > 0.0f) S += ps[q] * expf(pm[q] - M);\n'
                        f'  S = vres::warp_sum(S); if (lane == 0) {lptr}[0] = M + logf(S); }} }}')
            else:
              comb = (f'{{ if (warp == 0) {{ float M = -INFINITY; for (int p = lane; p < {self.G}; p += 32) M = fmaxf(M, {part}[2 * p]); '
                    f'M = vres::warp_max(M); float S = 0.0f; for (int p = lane; p < {self.G}; p += 32) {{ const float sp = {part}[2 * p + 1]; '
                    f'if (sp > 0.0f) S += sp * expf({part}[2 * p] - M); }} S = vres::warp_sum(S); '
                    f'if (lane == 0) {lptr}[0] = M + logf(S); }} }}')
            s1 = add(op.name + '_lse', comb, xdeps=[s_part])
            if plan.get('pairs_from'):
                self.fused_lse = getattr(self, 'fused_lse', {})
                self.fused_lse[plan['pairs_from']] = (s1, lse_expr)
        body = self.loop_over(R, out, lambda ix: self.store_root(
            R, out, ix, f'({self.view_expr(R, src, ix)} - {lse_expr})'))
        s2 = add(op.name, body, reads=deps_of(src), writes=[out])
        self._fix_dep(s2, s1)

    # ---------------------------------------------------------------- scheduling
    def schedule(self, steps):
        """Group ready steps; __syncthreads between groups, grid.sync where a cross-CTA dependency waits."""
        for s in steps:
            s.deps |= self._extra_deps.get(s.name, set())
        names = {s.name for s in steps}
        for s in steps:
            s.deps &= names
            s.xdeps &= names
        emitted, epoch = {}, 0
        pending = list(steps)
        out, barriers = [], 0
        labels = self._tick_labels
        byname = {s.name: s for s in steps}

        def fusable(d, c, group):
            """c may run in d's group without a CTA barrier: every value c reads from d is either a
            thread-replicated register or the element c's own thread wrote in the same strided element map;
            and c neither overwrites nor is overwritten by anything else in the group."""
            if not self.align:
                return False
            mine = [(r, a) for r, a in getattr(c, 'access', []) if r in d.writes]
            if getattr(d, 'reg', False):
                ok = all(a == 'reg' for _, a in mine)
            else:
                ok = (d.emap is not None and d.emap[0] == 's1' and d.emap == getattr(c, 'emap', None)
                      and all(a is True or a == 'reg' for _, a in mine))
            if not ok:
                return False
            for m in group:
                mreads = {r for r, a in getattr(m, 'access', []) if a != 'reg'}
                if c.writes & m.writes:
                    return False
                if c.writes & mreads and not (m.emap == c.emap and c.emap is not None and all(
                        a is True for r, a in m.access if r in c.writes)):
                    return False
            return True

        def tick(label):
            labels.append(label)
            k = 3 + len(labels)
            return (f'if (ticks && cta == 0 && tid == 0) {{ const unsigned long long vt_ = stamp(); '
                    f'ticks[{k}] += vt_ - vt_last; vt_last = vt_; }}')
        while pending:
            group = []
            done_now = set()
            warp_chain = set()
            for s in pending:
                if all(d in emitted or d in done_now for d in s.deps) and all(
                        d in emitted and emitted[d] < epoch for d in s.xdeps):
                    inside = [d for d in s.deps if d in done_now]
                    w0 = s.warp0 or s.name in self._warp0_steps
                    if inside and not (w0 and all(d in warp_chain for d in inside)) and not all(
                            fusable(byname[d], s, group) for d in inside):
                        continue      # depends on a step of this group: next group
                    group.append(s)
                    done_now.add(s.name)
                    if w0:
                        warp_chain.add(s.name)
            if not group:
                out.append('grid.sync();')
                out.append(tick('grid.sync'))
                barriers += 1
                epoch += 1
                continue
            # asynchronous issues first, but never ahead of a step of this group they depend on
            members = {g.name for g in group}
            group.sort(key=lambda s: not (s.name.endswith('_prefetch') and not (s.deps & members)))
            for s in group:
                emitted[s.name] = epoch
                if s.name in warp_chain and any(d in warp_chain for d in s.deps):
                    out.append('__syncwarp();')
                out.append(f'// step {s.name}\n{s.code}')
            out.append('__syncthreads();')
            out.append(tick(','.join(s.name for s in group)))
            pending = [s for s in pending if s.name not in emitted]
        return '\n'.join(out), barriers


def assign_transients(gen, code):
    """Group-granular liveness: tokens sharing no scheduler group may share shared memory."""
    seps = [m.end() for m in re.finditer(r'__syncthreads\(\);\n|grid\.sync\(\);\n', code + '\n')]

    def group(pos):
        lo, hi = 0, len(seps)
        while lo < hi:
            mid = (lo + hi) // 2
            if seps[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        return lo
    spans = {}
    for m in re.finditer(r'@@SM(\d+)@@', code):
        k = int(m.group(1))
        g = group(m.start())
        a, b = spans.get(k, (g, g))
        spans[k] = (min(a, g), max(b, g))
    placed = []          # (start_group, end_group, offset, size)
    offsets, top = {}, 0
    for k in sorted(spans, key=lambda k: (spans[k][0], -gen.tsizes[k])):
        a, b = spans[k]
        size = gen.tsizes[k]
        busy = sorted((o, o + s) for (pa, pb, o, s) in placed if not (pb < a or pa > b))
        off = 0
        for lo_, hi_ in busy:
            if off + size <= lo_:
                break
            off = max(off, hi_)
        offsets[k] = off
        placed.append((a, b, off, size))
        top = max(top, off + size)
    return offsets, top


def emit_program(L: VresLowering):
    graph = L.graph
    np_, ns, ni, nc, npred = L.np_, L.ns, L.ni, L.nc, L.npred
    G = L.G
    gen = Gen(L)
    gen._step_reads = {}
    gen._extra_deps = {}
    gen._tick_labels = []
    gen.tkeys, gen.tsizes = {}, []
    gen.small_warp = bool(L.options.get('vres_small_warp', False))
    gen._warp0_steps = set()
    li = L.loop_inputs
    values = graph.values
    count_name = li[-1]
    param_ptr = [f'args[{i}]' for i in range(np_)]
    state_ptr = [f'args[{np_ + i}]' for i in range(ns)]
    input_ptr = [f'args[{np_ + ns + i}]' for i in range(ni)]
    init, body, outr = graph.init, graph.body, graph.output
    carry_shapes = [tuple(init.values[n].shape) for n in init.outputs]
    gen.carry_dtypes = [_dt(init.values[n]) for n in init.outputs]
    for j, (shape, dt) in enumerate(zip(carry_shapes, gen.carry_dtypes)):
        if _numel(shape) > SMEM_LOCAL_LIMIT:
            raise LoweringError('vector_resident: carries must be small (replicated in shared memory)')
        L.alloc_smem(('carry_cur', j), max(1, _numel(shape)) * BYTES[dt])
        L.alloc_smem(('carry_next', j), max(1, _numel(shape)) * BYTES[dt])

    def ext(region, names_by_kind):
        e = {}
        pos = 0
        for kind, ptrs in names_by_kind:
            for k, p in enumerate(ptrs):
                name = region.inputs[pos]
                pos += 1
                if kind == 'carry':
                    e[name] = {'kind': 'carry', 'slot': k}
                else:
                    e[name] = {'kind': 'global', 'ptr': p, 'role': kind}
        if pos != len(region.inputs):
            raise LoweringError('vector_resident: region ABI mismatch')
        return e

    regions = {}
    regions['init'] = _Region(init, 'init', G, ext(init, [('param', param_ptr), ('input', input_ptr)]))
    regions['body'] = _Region(body, 'body', G, ext(body, [('param', param_ptr), ('carry', [None] * nc),
                                                          ('state', state_ptr), ('input', input_ptr)]))
    regions['output'] = _Region(outr, 'output', G, ext(outr, [('param', param_ptr), ('carry', [None] * nc),
                                                              ('input', input_ptr)]))
    for R in regions.values():
        L.analyze(R)
    gen._body = regions['body']
    if L.options.get('vres_resident_state', True):
        Rb_ = regions['body']
        for root, (axis, N) in Rb_.owner.items():
            Rb_.place[root] = 'resident'
    L.resident = {root for root, pl in regions['body'].place.items() if pl == 'resident'}
    # loop-input rows selected by a carry-only index are fetched asynchronously one iteration ahead
    prefetch_ops = []
    if L.options.get('vres_input_prefetch', True):
        Rb_ = regions['body']
        for op in Rb_.ops:
            if Rb_.plans.get(op.name, {}).get('kind') != 'index_select':
                continue
            out = op.outputs[0]
            src, idxv = Rb_._input_refs(op)[:2]
            roots = L._expr_roots(Rb_, idxv)
            places = {Rb_.place.get(r) for r in roots}
            if (Rb_.external.get(Rb_.views[src].root, {}).get('role') != 'input' or Rb_.place.get(out) != 'smem'
                    or 'carry' not in places or not places <= {'carry', 'inline', 'virtual'}):
                continue
            Rb_.place[out] = 'prefetch'
            prefetch_ops.append((op, src, idxv, Rb_.plans[op.name]['dim']))
        if prefetch_ops:
            L.notes.append('input prefetch: ' + ', '.join(o.name for o, *_ in prefetch_ops)
                           + ' rows copied asynchronously one iteration ahead (parity double buffer)')
    # small loop-invariant FP32 parameters live in shared memory for the whole launch
    pc_limit = int(L.options.get('vres_param_cache_elements', 4096))
    pcached = set()
    for R in regions.values():
        for name, info in R.external.items():
            if info.get('role') == 'param' and R.dtype(name) == 'float32' and _numel(R.shape(name)) <= pc_limit:
                R.place[name] = 'pcache'
                pcached.add((info['ptr'], name, R.label))
    L.pcached = pcached
    # loop-level outputs: predictions (arena), state outputs alias the state inputs, executed/status
    outputs = L.loop_outputs
    layouts, offsets = {}, {}
    for name in li:
        v = values[name]
        layouts[name] = {'shape': list(v.shape), 'strides': list(v.strides), 'root': name, 'offset': 0,
                         'dtype': _dt(v)}
    pred_ptrs = []
    for k, name in enumerate(outputs):
        v = values[name]
        dt = _dt(v)
        if npred <= k < npred + ns:
            sname = li[np_ + (k - npred)]
            layouts[name] = dict(layouts[sname])
            continue
        words = max(1, (_numel(v.shape) * BYTES[dt] + 3) // 4)
        off = L.alloc_arena(words)
        offsets[name] = off
        layouts[name] = {'shape': list(v.shape), 'strides': list(v.strides), 'root': name, 'offset': 0, 'dtype': dt}
        if k < npred:
            pred_ptrs.append((f'(arena + {off}LL)', dt))
    executed_ptr = f'(arena + {offsets[outputs[-2]]}LL)'
    status_ptr = f'(arena + {offsets[outputs[-1]]}LL)'
    # steps
    st_init = gen.steps_for(regions['init'], list(init.outputs), {n: ('carry', j) for j, n in enumerate(init.outputs)})
    body_targets = {n: ('carry', j) for j, n in enumerate(body.outputs[:nc])}
    for i, n in enumerate(body.outputs[nc:]):
        body_targets[n] = ('state', body.inputs[np_ + nc + i])
    # every step reading a state root (for write-after-read ordering)
    Rb = regions['body']
    state_roots = set(body.inputs[np_ + nc:np_ + nc + ns])
    for op in Rb.ops:
        if op.target in VIEW_TARGETS:
            continue
        roots = set()
        for r in Rb._input_refs(op):
            roots.add(Rb.views[r].root)
        if roots & state_roots:
            gen._step_reads.setdefault(op.name, set()).update(roots & state_roots)
    st_body = gen.steps_for(Rb, list(body.outputs), body_targets)
    # state reads by inline expressions consumed in later steps are covered: stores read their own partition only
    st_out = gen.steps_for(regions['output'], list(outr.outputs),
                           {n: ('pred',) + pred_ptrs[k] for k, n in enumerate(outr.outputs)})
    code_init, b0 = gen.schedule(st_init)
    code_body, b1 = gen.schedule(st_body)
    code_out, b2 = gen.schedule(st_out)
    if b0 or b2:
        raise LoweringError('vector_resident: init/output regions must not need grid barriers')
    pc_load = []
    for R in regions.values():
        for name, info in R.external.items():
            if R.place.get(name) == 'pcache':
                n = _numel(R.shape(name))
                pc_load.append(f'for (int k = tid; k < {n}; k += {THREADS}) {gen.smem_ptr(R, name)}[k] = '
                               f'reinterpret_cast<const float*>({info["ptr"]})[k];')
    pf_load = [gen.prefetch_code(Rb, op.outputs[0], src, idxv, dim, which='cur', parity='0', sync=True)
               for op, src, idxv, dim in prefetch_ops]
    res_load, res_store = [], []
    for sroot in sorted(state_roots):
        if Rb.place.get(sroot) != 'resident':
            continue
        info = Rb.external[sroot]
        st = Rb.values[sroot].strides
        dt = Rb.dtype(sroot)

        def gaddr(idx, info=info, st=st, dt=dt):
            off = ' + '.join(f'(long long)({i}) * {s}LL' for i, s in zip(idx, st) if s) or '0'
            return f'reinterpret_cast<{CTYPE[dt]}*>({info["ptr"]})[{off}]'
        res_load.append(gen.loop_over(Rb, sroot, lambda idx, s=sroot, ga=gaddr: f'{gen.load_root(Rb, s, idx)} = {ga(idx)};',
                                      shard=Rb.owner[sroot]))
        res_store.append(gen.loop_over(Rb, sroot, lambda idx, s=sroot, ga=gaddr: f'{ga(idx)} = {gen.load_root(Rb, s, idx)};',
                                       shard=Rb.owner[sroot]))
    carry_copy = []
    for j, shape in enumerate(carry_shapes):
        n = max(1, _numel(shape))
        carry_copy.append(f'for (int k = tid; k < {n}; k += {THREADS}) {gen.carry_ptr(j, "cur")}[k] = {gen.carry_ptr(j, "next")}[k];')
    groups = '\n'.join(f'const int vlo_{N} = cta * {ch}; const int vhi_{N} = min({N}, vlo_{N} + {ch});'
                       for N, ch in sorted(gen.groups.items()))
    toffs, ttop = {}, 0
    for code in (code_init, code_body, code_out):
        o, t = assign_transients(gen, code)
        for k, v in o.items():
            if k in toffs and toffs[k] != v:
                raise LoweringError('vector_resident: transient shared value used in two regions')
            toffs[k] = v
        ttop = max(ttop, t)
    tbase = L.alloc_smem(('transient_pool',), max(16, ttop))
    pool = L.alloc_smem(('scratch_pool',), max(16, getattr(gen, 'scratch_bytes', 0)))
    kernel_parts = None
    smem_bytes = (L.smem_top + 127) // 128 * 128
    if smem_bytes > 232448 - 1024:
        raise LoweringError(f'vector_resident: shared memory plan {smem_bytes} B exceeds the CTA limit')
    kernel = f'''cg::grid_group grid = cg::this_grid();
extern __shared__ __align__(16) unsigned char vres_smem[];
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const int cta = blockIdx.x;
(void)lane; (void)warp;
if (gridDim.x != {G}) __trap();
unsigned long long vt_last = 0; (void)vt_last;
if (ticks && cta == 0 && tid == 0) {{ for (int k = 4; k < {4 + len(gen._tick_labels)}; ++k) ticks[k] = 0; ticks[0] = stamp(); vt_last = ticks[0]; }}
const long long requested_count = *reinterpret_cast<const long long*>(args[{len(li) - 1}]);
const int count = int(requested_count < 0 ? 0 : (requested_count > {graph.max_steps}LL ? {graph.max_steps}LL : requested_count));
if (cta == 0 && tid == 0) {{ *reinterpret_cast<long long*>({executed_ptr}) = count; *reinterpret_cast<long long*>({status_ptr}) = requested_count == count ? 0 : 1; }}
{groups}
int par = 0; (void)par;
// ---- small loop-invariant parameters cached in shared memory
{chr(10).join(pc_load)}
__syncthreads();
// ---- init region
{code_init}
__syncthreads();
// ---- owner-resident explicit state: load this CTA's partition once per launch
{chr(10).join(res_load)}
// ---- first iteration's prefetched loop-input rows (synchronous)
if (count > 0) {{
{chr(10).join(pf_load)}
}}
__syncthreads();
if (ticks && cta == 0 && tid == 0) {{ ticks[1] = stamp(); vt_last = ticks[1]; }}
for (int iteration = 0; iteration < count; ++iteration) {{
par = iteration & 1;
// ---- body region ({b1} grid barriers per iteration)
{code_body}
{'vres::cp_async_wait_all();' if prefetch_ops else ''}
{chr(10).join(carry_copy)}
__syncthreads();
}}
// ---- write back owner-resident state
{chr(10).join(res_store)}
if (ticks && cta == 0 && tid == 0) ticks[2] = stamp();
// ---- output region (CTA 0 stores predictions; explicit state was updated in place by its owners)
{code_out}
__syncthreads();
grid.sync();
if (ticks && cta == 0 && tid == 0) ticks[3] = stamp();
'''
    kernel = kernel.replace('@@VRES_SCRATCH@@', str(pool))
    kernel = re.sub(r'@@SM(\d+)@@', lambda m: str(tbase + toffs[int(m.group(1))]), kernel)
    if '@@' in kernel:
        raise LoweringError('vector_resident: unresolved shared-memory token')
    source = _translation_unit(kernel, helpers=HELPERS, functions='', shared_bytes=smem_bytes,
                               resident_blocks_per_sm=1)
    header_hashes = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest()
                     for name in ('core.cuh', 'gemm.cuh', 'epi.cuh', 'plan.py')}
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    arena_words = (L.arena_words + 63) // 64 * 64
    ownership = {name: {'axis': ax, 'N': N} for name, (ax, N) in regions['body'].owner.items()}
    manifest = {
        'format': 'latch-ops-cuda-inference-loop-v1', 'source_hash': source_hash, 'header_hashes': header_hashes,
        'inputs': li, 'outputs': outputs, 'structured_outputs': outputs,
        'input_roles': {name: {'role': values[name].role, 'parameter': values[name].parameter} for name in li},
        'state_writes': dict(zip(li[np_:np_ + ns], outputs[npred:npred + ns])),
        'layouts': layouts, 'offsets': offsets, 'arena_words': arena_words, 'arena_bytes': 4 * arena_words,
        'region_scratch_bytes': 0, 'shared_bytes': smem_bytes, 'phase_count': 3,
        'target_arch': L.options.get('target_arch', 'sm_90'),
        'resource_policy': {'resident_blocks_per_sm': 1, 'shared_bytes': smem_bytes, 'target_arch': 'sm_90',
                            'version': 'vector-resident-v1', 'runtime_admission': 'CUDA occupancy query; grid == vres_grid'},
        'full_grid': True, 'max_grid_blocks': G, 'tick_words': 4 + len(gen._tick_labels),
        'vres_tick_labels': list(gen._tick_labels),
        'phases': [{'kind': 'vres_loop', 'names': ['vector_resident'], 'work_items': G * THREADS}],
        'max_gemm_tiles': 0, 'regions': {},
        'lower_options': {'loop_schedule': 'vector_resident', 'vres_grid': G, **{k: v for k, v in L.options.items() if k.startswith('vres_')}},
        'runtime_loop': {'max_steps': graph.max_steps, 'count_input': count_name,
                         'count_policy': 'clamp to [0,max_steps]; nonzero status exposes out-of-range request',
                         'boundaries': 'carry replicated in shared memory (restarts from init each launch); '
                                       'explicit state updated in place by owner CTAs'},
        'vector_resident': {'grid': G, 'barriers_per_iteration': b1, 'partition_groups': gen.groups,
                            'state_ownership': ownership, 'resident_state': sorted(L.resident), 'notes': list(L.notes),
                            'exchanges': [{'name': e.name, 'kind': e.kind, 'n': e.n} for e in L.exchanges],
                            'shared_bytes': smem_bytes, 'body_steps': len(st_body),
                            'placements': {k: v for k, v in regions['body'].place.items()},
                            'products': {k: {kk: vv for kk, vv in p.items() if kk in ('form', 'split', 'K', 'F', 'matrix_dtype', 'cast_to', 'matrix_root')}
                                         for k, p in regions['body'].plans.items() if p.get('kind') == 'product'}},
        'numerical_policy': ('FP32 arithmetic for FP32 values; BF16 operands converted exactly on load; FP32 '
                             'accumulation; fixed-order deterministic cross-CTA reductions (no atomics); '
                             'not bitwise equal to ATen/cuBLAS reduction orders'),
        'execution': 'one cooperative kernel: init, runtime-count body loop (vector-resident SPMD), output',
        'memory_plan': 'replicated small values and carries in shared memory; owner-partitioned in-place state; '
                       'parity-double-buffered exchange partials in the arena',
    }
    return GeneratedProgram(source, manifest, kernel, HELPERS, '')
