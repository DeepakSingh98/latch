"""Vector-resident schedule for forward-only runtime-count loops (GRADLEV-LATCH, 2026-10-17).

``lower(loop_ir, loop_schedule='vector_resident', vres_grid=G)`` lowers a
``capture_loop_inference`` loop whose body is a decode / online-learning step:
a few "large" tensors (parameters or explicit state, e.g. weight matrices and a
vocabulary head) touched only by vector-matrix products and elementwise writes,
and many "small" vectors. The schedule is an intra-GPU SPMD sharding of the
body over the G CTAs of one cooperative kernel:

* every large tensor is partitioned along one dimension (its largest), and the
  partition is propagated through products, elementwise ops and reductions
  (output sharded on a free dimension; contraction over a sharded dimension
  becomes CTA partial sums plus a deterministic cross-CTA all-reduce);
* small values are replicated: every CTA computes them redundantly in shared
  memory, so chains of small operations need no grid barrier;
* products with one vector operand are bandwidth-shaped: warp-per-row dot form
  when the matrix is contiguous along the contraction, column-vector "axpy"
  form when it is contiguous along the free dimension; operand casts
  (e.g. BF16 -> FP32) are evaluated while loading, never materialized;
* explicit state whose every access in the body is owner-local is updated in
  place by its owner CTA (no ping-pong boundary, no final commit copy);
* all-reduces write fixed per-CTA partial slots (iteration-parity double
  buffered) and combine them in CTA order: deterministic, no atomics.
  Vectors use reduce-scatter + all-gather (two barriers), scalars one barrier;
  a sharded log-softmax combines (max, sum-exp) pairs online (one barrier).

Grid barriers are placed by a list scheduler only where a value produced by
other CTAs is consumed. Unsupported operations or sharding conflicts raise
LoweringError; nothing silently falls back. Numerics: FP32 arithmetic for
FP32 values, BF16 values converted exactly on load, FP32 accumulation, fixed
reduction order (not bitwise equal to cuBLAS/ATen reduction orders).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

from .lower import GeneratedProgram, LoweringError, _header_path, _translation_unit

THREADS = 256
WARPS = THREADS // 32
LARGE = 1 << 16            # elements: tensors at least this large are partitioned, never replicated
SMEM_LOCAL_LIMIT = 8192    # elements: per-CTA materialization limit for small/local values
CTYPE = {'float32': 'float', 'bfloat16': '__nv_bfloat16', 'int64': 'long long', 'bool': 'bool', 'int32': 'int'}
BYTES = {'float32': 4, 'bfloat16': 2, 'int64': 8, 'bool': 1, 'int32': 4}
VIEW_TARGETS = {'aten.select.int', 'aten.unsqueeze.default', 'aten.view.default', 'aten.t.default',
                'aten.reshape.default', 'aten.squeeze.dim', 'aten.alias.default'}
UNARY = {'aten.tanh.default': 'tanhf({0})', 'aten.exp.default': 'expf({0})', 'aten.log.default': 'logf({0})',
         'aten.rsqrt.default': 'rsqrtf({0})', 'aten.neg.default': '(-({0}))', 'aten.clone.default': '({0})',
         'aten.sqrt.default': 'sqrtf({0})', 'aten.sigmoid.default': '(1.0f / (1.0f + expf(-({0}))))',
         'aten.relu.default': 'fmaxf({0}, 0.0f)'}
BINARY = {'aten.add.Tensor': '({0} + {1})', 'aten.sub.Tensor': '({0} - {1})', 'aten.mul.Tensor': '({0} * {1})',
          'aten.div.Tensor': '({0} / {1})', 'aten.eq.Tensor': '({0} == {1})', 'aten.eq.Scalar': '({0} == {1})',
          'aten.rsub.Scalar': '({1} - {0})', 'aten.maximum.default': 'fmaxf({0}, {1})',
          'aten.minimum.default': 'fminf({0}, {1})', 'aten.lt.Tensor': '({0} < {1})', 'aten.gt.Tensor': '({0} > {1})'}
PRODUCTS = {'aten.mm.default', 'aten.bmm.default'}
REDUCTIONS = {'aten.mean.dim', 'aten.sum.dim_IntList'}


def _ref(arg):
    if isinstance(arg, dict) and set(arg) == {'ref'}:
        return arg['ref']
    name = getattr(arg, 'name', None)
    return name if isinstance(name, str) and type(arg).__name__ == 'Ref' else None


def _dt(value):
    return value.dtype.removeprefix('torch.')


def _numel(shape):
    return math.prod(shape) if shape else 1


@dataclass
class View:
    root: str
    dims: list            # per view dim: root dim index, or None (inserted size-1 dim)
    fixed: dict           # root dim -> constant index (selected away)


@dataclass
class Exchange:
    name: str
    n: int                # values per CTA partial
    kind: str             # 'vector' (reduce-scatter + all-gather), 'scalar' (sum), 'lse' ((max, sumexp) pairs)
    offset: int = 0       # arena words
    reduced_offset: int = 0


@dataclass
class Step:
    name: str
    code: str
    deps: set = field(default_factory=set)       # steps that must precede (same CTA ordering)
    xdeps: set = field(default_factory=set)      # steps that must precede across a grid barrier
    order: int = 0
    warp0: bool = False                           # executed by warp 0 only (small replicated chains)


class _Region:
    """Analysis of one region (init / body / output) under a fixed grid."""

    def __init__(self, graph, label, G, external):
        self.g, self.label, self.G = graph, label, G
        self.values = graph.values
        self.ops = list(graph.operations)
        self.external = external          # input name -> dict(kind=global|carry, ptr=..., ...)
        self.views = {}
        self.op_of = {}
        for op in self.ops:
            for out in op.outputs:
                self.op_of[out] = op
        self.consumers = {}
        for op in self.ops:
            for name in self._input_refs(op):
                self.consumers.setdefault(name, []).append(op)
        for name in graph.inputs:
            self.views[name] = View(name, list(range(len(self.values[name].shape))), {})

    @staticmethod
    def _input_refs(op):
        out = []

        def walk(a):
            r = _ref(a)
            if r is not None:
                out.append(r)
            elif isinstance(a, (tuple, list)):
                for x in a:
                    walk(x)
        walk(op.args)
        return out

    def shape(self, name):
        return tuple(self.values[name].shape)

    def dtype(self, name):
        return _dt(self.values[name])


def _view_of(region, op, src):
    """Compose a view op over its source's view info."""
    v = region.views[src]
    t = op.target
    a = op.args
    dims, fixed = list(v.dims), dict(v.fixed)
    if t == 'aten.select.int':
        dim = a[1] % len(dims)
        rd = dims.pop(dim)
        if rd is not None:
            fixed[rd] = int(a[2]) % region.values[v.root].shape[rd] if region.values[v.root].shape[rd] else 0
    elif t == 'aten.unsqueeze.default':
        dim = a[1] % (len(dims) + 1)
        dims.insert(dim, None)
    elif t == 'aten.t.default':
        if len(dims) == 2:
            dims = [dims[1], dims[0]]
    elif t in ('aten.view.default', 'aten.reshape.default', 'aten.squeeze.dim', 'aten.alias.default'):
        new_shape = tuple(region.shape(op.outputs[0]))
        old_shape = tuple(region.shape(src))
        old_core = [(d, s) for d, s in zip(dims, old_shape) if s != 1]
        new_core = [s for s in new_shape if s != 1]
        if [s for _, s in old_core] != new_core:
            raise LoweringError(f'vector_resident: {t} at {op.name} changes non-unit dimensions')
        dropped = [d for d, s in zip(dims, old_shape) if s == 1 and d is not None]
        for d in dropped:
            fixed[d] = 0
        it = iter(old_core)
        dims = [next(it)[0] if s != 1 else None for s in new_shape]
    else:
        raise LoweringError(f'vector_resident: unsupported view {t}')
    return View(v.root, dims, fixed)


class VresLowering:
    def __init__(self, graph, G, options):
        self.graph, self.G = graph, G
        self.options = options
        md = graph.metadata
        self.np_, self.ns = len(md['parameter_names']), len(md['state_names'])
        self.nc, self.ni, self.npred = len(md['carry_names']), md['n_inputs'], md['prediction_count']
        self.loop_inputs = list(graph.inputs)
        self.loop_outputs = list(graph.outputs)
        if len(self.loop_inputs) != self.np_ + self.ns + self.ni + 1:
            raise LoweringError('vector_resident: loop ABI arity mismatch')
        self.smem = {}          # (region, value root) -> (byte offset, bytes)
        self.smem_top = 0
        self.arena_words = 0
        self.exchanges = []
        self.notes = []

    # ------------------------------------------------------------------ memory
    def alloc_smem(self, key, nbytes):
        if key in self.smem:
            return self.smem[key][0]
        off = (self.smem_top + 15) // 16 * 16
        self.smem[key] = (off, nbytes)
        self.smem_top = off + nbytes
        return off

    def alloc_arena(self, words):
        off = (self.arena_words + 63) // 64 * 64
        self.arena_words = off + words
        return off

    # ------------------------------------------------------------------ analysis
    def analyze(self, region):
        """Views, shardings, placements and product plans for one region."""
        R = region
        G = self.G
        R.shard = {}       # root -> (axis, N) or None
        R.place = {}       # root -> 'global' | 'carry' | 'smem' | 'local' | 'inline' | 'virtual'
        R.plans = {}       # op name -> dict
        for name in R.g.inputs:
            info = R.external[name]
            R.place[name] = info['kind']
            R.shard[name] = None
        owner = {}         # state root -> (axis, N) partition all its accesses must use

        def root_shard_in(view_name, out_rank=None, out_shape=None):
            """Sharding of a view expressed as (view axis, N) or None."""
            v = R.views[view_name]
            s = R.shard.get(v.root)
            if s is None:
                return None
            axis, N = s
            if axis in v.fixed:
                raise LoweringError(f'vector_resident: view {view_name} selects inside a sharded dimension')
            if axis not in v.dims:
                raise LoweringError(f'vector_resident: view {view_name} drops a sharded dimension')
            return v.dims.index(axis), N

        def bcast_axis(view_name, out_shape):
            s = root_shard_in(view_name)
            if s is None:
                return None
            ax, N = s
            vs = R.shape(view_name)
            return ax + (len(out_shape) - len(vs)), N

        for op in R.ops:
            out = op.outputs[0]
            t = op.target
            ins = R._input_refs(op)
            if t in VIEW_TARGETS:
                R.views[out] = _view_of(R, op, ins[0])
                continue
            R.views[out] = View(out, list(range(len(R.shape(out)))), {})
            oshape = R.shape(out)
            if t in PRODUCTS:
                plan = self._plan_product(R, op, owner)
                R.plans[op.name] = plan
                R.shard[out] = plan['out_shard']
                R.place[out] = 'local' if plan['out_shard'] else 'smem'
                continue
            if t == 'aten.index_select.default':
                src, dim, idx = ins[0], op.args[1], ins[1]
                if R.place.get(R.views[src].root) not in ('global',) or R.shard.get(R.views[idx].root):
                    raise LoweringError('vector_resident: index_select needs a global source and replicated index')
                R.shard[out] = None
                R.place[out] = 'smem'
                R.plans[op.name] = {'kind': 'index_select', 'dim': dim % len(R.shape(src))}
                if _numel(oshape) > SMEM_LOCAL_LIMIT:
                    raise LoweringError('vector_resident: index_select result too large to replicate')
                continue
            if t == 'aten.arange.default':
                R.shard[out] = None
                R.place[out] = 'virtual'
                continue
            if t in ('aten.zeros.default', 'aten.full.default', 'aten.ones.default'):
                R.shard[out] = None
                R.place[out] = 'smem'
                if _numel(oshape) > SMEM_LOCAL_LIMIT:
                    raise LoweringError('vector_resident: large constant fill')
                R.plans[op.name] = {'kind': 'fill'}
                continue
            if t in REDUCTIONS:
                src = ins[0]
                rdims = [d % len(R.shape(src)) for d in op.args[1]]
                keep = bool(op.args[2]) if len(op.args) > 2 else bool(op.kwargs.get('keepdim', False))
                s = root_shard_in(src)
                if s is not None and s[0] in rdims:
                    kind = 'reduce_sharded'
                    R.shard[out] = None
                elif s is not None:
                    raise LoweringError('vector_resident: reduction over a non-sharded dimension of a sharded value')
                else:
                    kind = 'reduce_local'
                    R.shard[out] = None
                if rdims != [len(R.shape(src)) - 1]:
                    raise LoweringError('vector_resident: only last-dimension reductions are supported')
                R.place[out] = 'smem'
                R.plans[op.name] = {'kind': kind, 'mean': t == 'aten.mean.dim', 'keep': keep}
                continue
            if t == 'aten._log_softmax.default':
                src = ins[0]
                dim = op.args[1] % len(R.shape(src))
                if dim != len(R.shape(src)) - 1:
                    raise LoweringError('vector_resident: log_softmax over the last dimension only')
                s = root_shard_in(src)
                R.shard[out] = (dim, s[1]) if s is not None else None
                if s is not None and s[0] != dim:
                    raise LoweringError('vector_resident: log_softmax over a non-sharded axis of a sharded value')
                R.place[out] = 'local' if s is not None else 'smem'
                R.plans[op.name] = {'kind': 'log_softmax_sharded' if s is not None else 'log_softmax_local'}
                continue
            if t in UNARY or t in BINARY or t in ('aten.where.self', 'aten._to_copy.default'):
                shards = []
                for name in ins:
                    s = bcast_axis(name, oshape)
                    if s is not None:
                        shards.append(s)
                if shards:
                    if len(set(shards)) != 1:
                        raise LoweringError(f'vector_resident: conflicting input shardings at {op.name}: {shards}')
                    R.shard[out] = shards[0]
                elif _numel(oshape) >= LARGE:
                    axis = max(range(len(oshape)), key=lambda d: oshape[d])
                    R.shard[out] = (axis, oshape[axis])
                else:
                    R.shard[out] = None
                local = self._local_numel(oshape, R.shard[out])
                R.place[out] = 'inline' if local > SMEM_LOCAL_LIMIT else ('local' if R.shard[out] else 'smem')
                R.plans[op.name] = {'kind': 'elementwise'}
                continue
            raise LoweringError(f'vector_resident: unsupported operation {t} at {op.name}')
        # casts feeding a product's matrix operand are evaluated during the load
        for op in R.ops:
            plan = R.plans.get(op.name)
            if plan and plan.get('kind') == 'product':
                for cast in plan['matrix_casts']:
                    R.place[cast] = 'inline'
        self._inline_single_use(R)
        if self.options.get('vres_head_fusion', 'auto') and R.label == 'body':
            self._match_head_fusion(R)
        # state ownership: elementwise reads of explicit state must be owner-partition expressions
        state_roots = {n for n, e in R.external.items() if e.get('role') == 'state'}
        for op in R.ops:
            plan = R.plans.get(op.name, {})
            if plan.get('kind') != 'elementwise':
                continue
            for r in R._input_refs(op):
                root = R.views[r].root
                if root not in state_roots:
                    continue
                own = owner.get(root)
                out = op.outputs[0]
                if own is None or R.shape(out) != R.shape(root) or R.shard.get(out) != own:
                    raise LoweringError(f'vector_resident: {op.name} reads state {root} outside its owner partition')
        R.owner = owner
        return R

    def _expr_roots(self, R, name, seen=None):
        """Materialized/external roots an expression reads (through inline values)."""
        seen = set() if seen is None else seen
        root = R.views[name].root
        if root in seen:
            return seen
        seen.add(root)
        if R.place.get(root) == 'inline':
            for r in R._input_refs(R.op_of[root]):
                self._expr_roots(R, r, seen)
        return seen

    def _onehot_index(self, R, c):
        """c == (arange == scalar index) [cast to float]: return the index view (sum_v c_v M_v is one row)."""
        root = R.views[c].root
        op = R.op_of.get(root)
        if op is not None and op.target == 'aten._to_copy.default' and R.place.get(root) == 'inline':
            root = R.views[R._input_refs(op)[0]].root
            op = R.op_of.get(root)
        if op is None or op.target != 'aten.eq.Tensor' or R.place.get(root) not in ('inline',):
            return None
        a, b = R._input_refs(op)[:2]
        ra, rb = R.views[a].root, R.views[b].root
        for ar, ix in ((ra, b), (rb, a)):
            aop = R.op_of.get(ar)
            if (aop is not None and aop.target == 'aten.arange.default' and len(aop.args) == 1
                    and R.shard.get(R.views[ix].root) is None and _numel(R.shape(ix)) == 1):
                return ix
        return None

    def _match_head_fusion(self, R):
        """logits = M n (dot, split F) -> log_softmax -> g = exp(lsm) - c (FP32, c independent of lsm)
        -> g @ M (axpy, split K over the same rows): one pass over M with online softmax statistics."""
        for op in R.ops:
            plan = R.plans.get(op.name, {})
            if plan.get('kind') != 'product' or plan['split'] != 'K' or plan['form'] != 'axpy':
                continue
            vec = plan['vec']
            vroot = R.views[vec].root
            sub = R.op_of.get(vroot)
            if sub is None or sub.target != 'aten.sub.Tensor' or R.place.get(vroot) != 'inline':
                continue
            if any(R.dtype(x) != 'float32' for x in (vec, vroot)):
                continue
            a, c = R._input_refs(sub)[:2] if len(R._input_refs(sub)) >= 2 else (None, None)
            if a is None or sub.kwargs.get('alpha', 1) != 1:
                continue
            ex = R.op_of.get(R.views[a].root)
            if ex is None or ex.target != 'aten.exp.default' or R.place.get(R.views[a].root) != 'inline':
                continue
            lsm = R.views[R._input_refs(ex)[0]].root
            ls = R.op_of.get(lsm)
            if ls is None or ls.target != 'aten._log_softmax.default':
                continue
            if R.plans[ls.name]['kind'] != 'log_softmax_sharded':
                continue
            logits = R.views[R._input_refs(ls)[0]].root
            lop = R.op_of.get(logits)
            lplan = R.plans.get(lop.name, {}) if lop is not None else {}
            if (lplan.get('kind') != 'product' or lplan['split'] != 'F' or lplan['form'] != 'dot'
                    or lplan['matrix_root'] != plan['matrix_root'] or lplan['F'] != plan['K']
                    or lplan['cast_to'] != plan['cast_to'] or lplan['matrix_dtype'] != plan['matrix_dtype']):
                continue
            if lsm in self._expr_roots(R, c) or logits in self._expr_roots(R, c):
                continue
            vw = {'bfloat16': 8, 'float32': 4}.get(lplan['matrix_dtype'], 1)
            if lplan['K'] % (32 * vw) or lplan['sF'] % vw:
                # the single-pass emitter needs whole 16-byte row chunks per lane; keep the two-pass head
                self.notes.append(f'head fusion skipped for {lop.name}: K={lplan["K"]} is not a multiple of {32 * vw}')
                continue
            lplan['fuse_grad'] = op.name
            plan['fused_from'] = lop.name
            plan['fused_c'] = c
            plan['fused_onehot'] = self._onehot_index(R, c)
            R.plans[ls.name]['pairs_from'] = lop.name
            self.notes.append(f'head fusion: {lop.name} + {ls.name} + {op.name} share one pass over {plan["matrix_root"]}')

    def _consumers(self, R, name):
        out = []
        for op in R.consumers.get(name, []):
            if op.target in VIEW_TARGETS:
                out += self._consumers(R, op.outputs[0])
            else:
                out.append(op)
        return out

    def _inline_single_use(self, R):
        """Evaluate single-use small elementwise values inside their consumer (no broadcast amplification)."""
        changed = True
        while changed:
            changed = False
            for op in R.ops:
                if R.plans.get(op.name, {}).get('kind') != 'elementwise':
                    continue
                out = op.outputs[0]
                if R.place.get(out) not in ('smem', 'local'):
                    continue
                cons = self._consumers(R, out)
                if len(cons) != 1 or out in R.g.outputs:
                    continue
                c = cons[0]
                ckind = R.plans.get(c.name, {}).get('kind')
                if ckind == 'elementwise':
                    cout = c.outputs[0]
                    if (self._local_numel(R.shape(cout), R.shard.get(cout))
                            > self._local_numel(R.shape(out), R.shard.get(out))):
                        continue
                elif ckind not in ('reduce_local', 'reduce_sharded', 'log_softmax_sharded', 'log_softmax_local',
                                   'product', 'index_select'):
                    continue
                R.place[out] = 'inline'
                changed = True

    def _local_numel(self, shape, shard):
        if shard is None:
            return _numel(shape)
        axis, N = shard
        chunk = -(-N // self.G)
        return _numel(shape) // shape[axis] * chunk

    def _matrix_source(self, R, name):
        """Look through elementwise casts to the stored matrix; returns (view name, cast chain)."""
        chain = []
        cur = name
        while True:
            v = R.views[cur]
            op = R.op_of.get(v.root)
            if op is not None and op.target == 'aten._to_copy.default' and len(R.consumers.get(v.root, [])) == 1:
                if v.dims != list(range(len(R.shape(v.root)))) or v.fixed:
                    raise LoweringError('vector_resident: view of a cast matrix operand')
                chain.append(v.root)
                cur = R._input_refs(op)[0]
                continue
            return cur, chain

    def _plan_product(self, R, op, owner):
        G = self.G
        a, b = R._input_refs(op)
        out = op.outputs[0]
        bmm = op.target == 'aten.bmm.default'
        sa, sb = R.shape(a), R.shape(b)
        if bmm:
            if sa[0] != 1 or sb[0] != 1:
                raise LoweringError('vector_resident: batched products require batch 1')
            sa, sb = sa[1:], sb[1:]
        M, K = sa
        K2, N = sb
        if K != K2:
            raise LoweringError('vector_resident: product contraction mismatch')
        a_src, a_cast = self._matrix_source(R, a)
        b_src, b_cast = self._matrix_source(R, b)
        na, nb = M * K, K * N
        if nb >= LARGE and M == 1:
            mat, mat_cast, vec, vec_is_a, F = b_src, b_cast, a, True, N
            mdims = (0, 1)           # (K dim, F dim) in the (squeezed) matrix operand
        elif na >= LARGE and N == 1:
            mat, mat_cast, vec, vec_is_a, F = a_src, a_cast, b, False, M
            mdims = (1, 0)
        else:
            raise LoweringError(f'vector_resident: product {op.name} is not vector x large matrix ({M}x{K}x{N})')
        mv = R.views[mat]
        mroot = mv.root
        if R.place.get(mroot) not in ('global', 'resident'):
            raise LoweringError(f'vector_resident: product {op.name} matrix operand must be a stored tensor')
        rstrides = tuple(R.values[mroot].strides)
        rshape = tuple(R.values[mroot].shape)
        mshape = R.shape(mat)
        off = 1 if bmm else 0
        kdim, fdim = mdims[0] + off, mdims[1] + off
        kroot, froot = mv.dims[kdim], mv.dims[fdim]
        if kroot is None or froot is None:
            raise LoweringError('vector_resident: matrix operand dims must map to stored dims')
        sK, sF = rstrides[kroot], rstrides[froot]
        base_fixed = sum(rstrides[d] * i for d, i in mv.fixed.items())
        if sK == 1:
            form = 'dot'
        elif sF == 1:
            form = 'axpy'
        else:
            raise LoweringError('vector_resident: matrix operand is not contiguous along K or F')
        # vector operand sharding along K?
        vv = R.views[vec]
        vshard = R.shard.get(vv.root)
        vec_k_axis = (len(R.shape(vec)) - 1) if vec_is_a else (len(R.shape(vec)) - 2)
        vsh = None
        if vshard is not None:
            ax = vv.dims.index(vshard[0]) if vshard[0] in vv.dims else None
            if ax != vec_k_axis:
                raise LoweringError(f'vector_resident: product {op.name} vector operand sharded off the contraction')
            vsh = vshard
        if vsh is not None:
            split = 'K'
        elif F >= 4 * G:
            split = 'F'
        else:
            split = 'K'
        part_dim, part_N = (froot, F) if split == 'F' else (kroot, K)
        if R.external.get(mroot, {}).get('role') == 'state':
            prev = owner.setdefault(mroot, (part_dim, part_N))
            if prev != (part_dim, part_N):
                raise LoweringError(f'vector_resident: state {mroot} accessed with conflicting partitions')
        oshape = R.shape(out)
        if split == 'F':
            out_axis = len(oshape) - 1 if vec_is_a else len(oshape) - 2
            out_shard = (out_axis, F)
        else:
            out_shard = None
            ex = Exchange(f'{R.label}_{op.name}', F, 'vector')
            self.exchanges.append(ex)
        mdt = R.dtype(mroot)
        cast_to = R.dtype(mat_cast[0]) if mat_cast else mdt
        return {'kind': 'product', 'form': form, 'split': split, 'matrix_root': mroot, 'matrix_dtype': mdt,
                'cast_to': cast_to, 'matrix_casts': mat_cast, 'sK': sK, 'sF': sF, 'base': base_fixed,
                'K': K, 'F': F, 'vec': vec, 'vec_is_a': vec_is_a, 'vec_shard': vsh, 'out_shard': out_shard,
                'exchange': None if split == 'F' else ex.name, 'rshape': rshape,
                'kroot': kroot, 'froot': froot, 'vec_view_matrix': mat}


def lower_vector_resident(graph, *, vres_grid=132, vres_resident_state='auto', **options):
    """Entry point; see module docstring.

    vres_resident_state: 'auto' keeps owner-partitioned explicit state in shared memory across
    iterations when the plan fits (falls back to in-place global state otherwise); True requires it;
    False keeps state in global memory.
    """
    G = int(vres_grid)
    if not 1 <= G <= 1024:
        raise LoweringError('vres_grid must be in [1, 1024]')
    if type(graph.max_steps) is not int or not 0 <= graph.max_steps <= 1024:
        raise LoweringError('inference loop requires static max_steps in [0,1024]')
    if vres_resident_state not in ('auto', True, False):
        raise LoweringError("vres_resident_state must be 'auto', True or False")
    from .vres_codegen import emit_program
    if vres_resident_state is False:
        return emit_program(VresLowering(graph, G, {**options, 'vres_resident_state': False}))
    try:
        return emit_program(VresLowering(graph, G, {**options, 'vres_resident_state': True}))
    except LoweringError as error:
        if vres_resident_state is True or 'shared memory plan' not in str(error):
            raise
    return emit_program(VresLowering(graph, G, {**options, 'vres_resident_state': False}))
