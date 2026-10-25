# P-OPS2-D adaptation from 04ca670a17bb51ca6123a5c0d9a1002a27fbc83c.
# Adapted from P-OPS-B 286c2371624be08c6c7106c47df546e1e550f861; only the local header path changes.
"""FP32 matrix lowering policy: engine admission, tile/split-K selection and phase packing.

New P-OPS-B code. Selection uses only operation properties (shapes, strides,
alignment, batch) and the target resident grid, never model or layer names.
Arithmetic stays strict FP32 SIMT fmaf (see fp32_matrix.cuh): k ascending inside
each split-K slice and slices summed in fixed slice order. No TF32 or tensor cores.

`MatrixPolicy('frozen')` reproduces the Stage 3 lowering byte for byte; the default
`MatrixPolicy()` selects the cp.async engine where its layout contract holds and
falls back to the pinned/generic bodies otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import math
from pathlib import Path

ENGINE_SOURCE = (Path(__file__).parent / 'kernels' / 'fp32_matrix.cuh').read_text()


L2_PREFETCH_SOURCE = r"""
#ifndef POPS_L2_PREFETCH
#define POPS_L2_PREFETCH
namespace pops {
// Option l2_prefetch (rejected by measurement, off by default): stage an upcoming product's operands
// into L2 with the sm_90 bulk prefetch hint. Data movement only; each thread covers at most one 16 KiB
// chunk per grid-stride step and the tail below 16 bytes is skipped.
__device__ __forceinline__ void l2_prefetch(const float* p, long long bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr long long CH = 16384;
  const long long n = bytes / 16 * 16;
  for (long long off = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * CH; off < n;
       off += (long long)gridDim.x * blockDim.x * CH) {
    const unsigned size = (unsigned)(n - off < CH ? n - off : CH);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(reinterpret_cast<const char*>(p) + off), "r"(size)
                 : "memory");
  }
#endif
}
}  // namespace pops
#endif
"""


def engine_source(policy):
    return ''.join(f'#define {d}\n' for d in policy.defines) + ENGINE_SOURCE
BK, BN, THREADS = 16, 64, 256


@dataclass(frozen=True)
class MatrixPolicy:
    mode: str = 'cpasync'          # 'frozen' | 'cpasync'
    grid: int = 264                # resident CTA slots assumed by the cost model (H100: 132 SMs x 2)
    stages: int = 2
    tiles: tuple = ((64, 64), (32, 64), (128, 64), (64, 128))
    smem_limit: int = 49152        # dynamic shared bytes per CTA (above 48 KiB uses the opt-in attribute)
    splits: tuple = (1, 2, 4, 8)
    min_slice_steps: int = 2       # each split-K slice covers at least this many k steps
    pack_phases: bool = True       # co-schedule independent products in one barrier interval
    # Cost model (microseconds per task on one resident CTA slot) fitted by pops_b/fit_cost.py to
    # 1,090 in-context product phases of complete H100 updates (two CTAs/SM, inlined 2-stage engine,
    # release/acquire split-K arrival; median error 5.3%, p90 15%). A phase costs the maximum slot
    # load under vbase rotation.
    task_us: dict = field(default_factory=lambda: {'32x64': 0.60, '64x64': 1.34, '128x64': 3.71,
                                                    '64x128': 2.41, '128x128': 4.0})
    step_us: dict = field(default_factory=lambda: {'32x64': 0.78, '64x64': 1.08, '128x64': 1.68,
                                                    '64x128': 1.85, '128x128': 3.4})
    split_store_us: float = 0.11    # per split task, per 4096 output elements (partial store + arrival)
    split_reduce_us: float = 5.76   # last arriver, per (KSPLIT-1)/KSPLIT, per 4096 output elements
    defines: tuple = ('POPS_INLINE', 'POPS_RELACQ')  # engine compile options; part of the source identity
    elide_copies: bool = True      # read products through permuted copies whose only consumers are products
    joint_options: int = 8         # candidate configurations kept per product for joint phase selection
    joint: bool = True             # choose packed products' configurations together (modeled makespan)
    stage_depths: tuple = (16,)    # k values per pipeline stage (16 or 32)
    deep_stage_factor: float = 0.9 # modeled cost of a 32-deep stage relative to two 16-deep stages
    min_blocks_per_sm: int = 2     # emits __launch_bounds__(256, n): register cap for n resident CTAs/SM
    local_pointers: bool = True    # declare tensor pointers per phase from opaque args/arena copies
    l2_prefetch: bool = False      # stage the next product phase's cold operands into L2 one phase early
    l2_prefetch_limit: int = 16 << 20  # bytes per product phase (H100 L2: 50 MiB)
    _explicit_residency: bool = False  # frozen mode only: honour min_blocks_per_sm/grid as given

    def __post_init__(self):
        if self.mode not in ('frozen', 'cpasync'):
            raise ValueError('matrix policy mode must be frozen or cpasync')
        if self.mode == 'frozen' and self.min_blocks_per_sm == 2 and self.grid == 264 and not self._explicit_residency:
            object.__setattr__(self, 'min_blocks_per_sm', 1)
            object.__setattr__(self, 'grid', 128)
        if not 2 <= self.stages <= 6 or self.grid <= 0 or not 0 < self.smem_limit <= 227 * 1024:
            raise ValueError('invalid matrix pipeline depth, grid or shared-memory limit')

    @property
    def frozen(self):
        return self.mode == 'frozen'

    def to_dict(self):
        value = asdict(self)
        value.pop('_explicit_residency', None)
        value['step_us'] = dict(self.step_us)
        value['task_us'] = dict(self.task_us)
        return value


def stage_floats(bm, bn, ak, bkc, kb=BK):
    sa = kb + 4 if ak else bm + 4
    sb = kb + 4 if bkc else bn + 4
    return (bm * sa if ak else kb * sa) + (bn * sb if bkc else kb * sb)


def stages_for(policy, bm, bn, ak, bkc, kb=BK):
    """Deepest ring (<= policy.stages) that fits the shared-memory limit, or 0."""
    for st in range(policy.stages, 1, -1):
        if 4 * st * stage_floats(bm, bn, ak, bkc, kb) + 64 <= policy.smem_limit:
            return st
    return 0


def admit(la, lb, m, n, k, batch):
    """Engine fields when the cp.async contract holds for these operand layouts, else None.

    la/lb are plain strided layouts or factored layouts (views through elided copies).
    Each matrix dimension may have two atoms; tiles must not straddle an atom boundary.
    """
    try:
        fa = la if isinstance(la, FLayout) else FLayout.of(la)
        fb = lb if isinstance(lb, FLayout) else FLayout.of(lb)
        rank = len(fa.dims)
        A, B = operand(fa, rank), operand(fb, rank)
    except FactorError:
        return None
    (mia, sam, sam1), (kia, sak, sak1) = A['rows'], A['cols']
    (kib, sbk, sbk1), (nib, sbn, sbn1) = B['rows'], B['cols']
    ak, bkc = sak == 1, sbk == 1
    if not (ak or sam == 1) or not (bkc or sbn == 1):
        return None
    if n % BN or k % BK or batch <= 0 or kia % BK or kib % BK or nib % BN:
        return None
    nb2 = max(A['nb2'], B['nb2'])
    batches = []
    for op in (A, B):
        if op['nb2'] == nb2:
            batches.append((op['bs'], op['bs2']))
        elif op['nb2'] == 1:
            batches.append((op['bs'] * nb2, op['bs']))
        else:
            return None
    (bsa, bsa2), (bsb, bsb2) = batches
    lead = (sam if ak else sak, sam1, sak1, sbn if bkc else sbk, sbn1, sbk1, bsa, bsa2, bsb, bsb2,
            fa.offset, fb.offset)
    if any(v % 4 for v in lead):
        return None
    return {'ak': ak, 'bkc': bkc, 'sam': sam, 'sak': sak, 'sbk': sbk, 'sbn': sbn, 'bsa': bsa, 'bsb': bsb,
            'bsa2': bsa2, 'bsb2': bsb2, 'nb2': nb2, 'mia': mia, 'sam1': sam1, 'kia': kia, 'sak1': sak1,
            'kib': kib, 'sbk1': sbk1, 'nib': nib, 'sbn1': sbn1,
            'a_root': fa.root, 'a_offset': fa.offset, 'b_root': fb.root, 'b_offset': fb.offset}


def step_cost(policy, row):
    base = policy.step_us[f"{row['bm']}x{row['bn']}"]
    return base if row['kb'] == 16 else (row['kb'] // 16) * base * policy.deep_stage_factor


def candidates(policy, m, n, k, batch, m_inner=None, n_inner=None, ak=True, bkc=False, k_inner=None):
    rows = []
    for bm, bn in policy.tiles:
        if m % bm or (m_inner or m) % bm or n % bn or (n_inner or n) % bn:
            continue
        for kb in policy.stage_depths:
            stages = stages_for(policy, bm, bn, ak, bkc, kb)
            if not stages or (k_inner or k) % kb:
                continue
            for ks in policy.splits:
                if k % (kb * ks) or k // (kb * ks) < min(policy.min_slice_steps, k // kb):
                    continue
                if kb > 32 and k // (kb * ks) != 1:
                    continue  # 64-deep stages only for single-stage slices (short-K products)
                tiles = batch * (m // bm) * (n // bn)
                row = {'bm': bm, 'bn': bn, 'kb': kb, 'ksplit': ks, 'tiles': tiles, 'tasks': tiles * ks,
                       'steps': k // (kb * ks), 'stages': stages, 'waves': math.ceil(tiles * ks / policy.grid)}
                row['estimate_us'] = round(makespan(policy, [row]), 4)
                rows.append(row)
    return rows


def select(policy, m, n, k, batch, m_inner=None, n_inner=None, ak=True, bkc=False):
    rows = candidates(policy, m, n, k, batch, m_inner, n_inner, ak, bkc)
    if not rows:
        return None
    # Deterministic: estimated time, then fewer partial bytes, then larger tiles.
    return min(rows, key=lambda r: (r['estimate_us'], r['ksplit'], -r['bm'] * r['bn'], -r['bm']))


def make_spec(policy, a, b, out, la, lb, shape, m, n, k, batch):
    flags = admit(la, lb, m, n, k, batch)
    if flags is None:
        return None
    rows = candidates(policy, m, n, k, batch, flags['mia'], flags['nib'], flags['ak'], flags['bkc'],
                      math.gcd(flags['kia'], flags['kib']))
    if not rows:
        return None
    rows.sort(key=lambda r: (r['estimate_us'], r['ksplit'], -r['bm'] * r['bn'], -r['bm']))
    choice = rows[0]
    return {'a': a, 'b': b, 'out': out, 'm': m, 'n': n, 'k': k, 'batch': batch, **flags, **choice,
            'scm': n, 'scn': 1, 'bsc': m * n, 'options': rows[:policy.joint_options],
            'shared_bytes': 4 * choice['stages'] * stage_floats(choice['bm'], choice['bn'], flags['ak'], flags['bkc'],
                                                                choice['kb'])}


def task_us(policy, row):
    """Modeled time of one task; the last-arriver reduction is amortized over the tile's slices."""
    tile = f"{row['bm']}x{row['bn']}"
    cost = policy.task_us[tile] + row['steps'] * step_cost(policy, row)
    if row['ksplit'] > 1:
        area = row['bm'] * row['bn'] / 4096
        cost += area * (policy.split_store_us + policy.split_reduce_us * (row['ksplit'] - 1) / row['ksplit'])
    return cost


def makespan(policy, rows):
    """Max modeled CTA load when products share one phase with cumulative vbase rotation."""
    load, vbase = [0.0] * policy.grid, 0
    for row in rows:
        tau, tasks = task_us(policy, row), row['tasks']
        full, extra = divmod(tasks, policy.grid)
        if full:
            load = [x + full * tau for x in load]
        for i in range(extra):
            load[(vbase + i) % policy.grid] += tau
        vbase += tasks
    return max(load)


def rebalance(policy, phase):
    """Joint configuration of the products packed into one phase (coordinate descent, deterministic)."""
    specs = phase['matrix']
    choice = [s['options'].index(next(o for o in s['options'] if all(o[f] == s[f] for f in ('bm', 'bn', 'kb', 'ksplit'))))
              for s in specs]
    best = makespan(policy, [s['options'][c] for s, c in zip(specs, choice)])
    for _ in range(4):
        improved = False
        for g, spec in enumerate(specs):
            for c in range(len(spec['options'])):
                if c == choice[g]:
                    continue
                trial = choice[:g] + [c] + choice[g + 1:]
                value = makespan(policy, [s['options'][x] for s, x in zip(specs, trial)])
                if value < best - 1e-9:
                    best, choice, improved = value, trial, True
        if not improved:
            break
    for spec, c in zip(specs, choice):
        row = spec['options'][c]
        spec.update({k: row[k] for k in ('bm', 'bn', 'kb', 'ksplit', 'tiles', 'tasks', 'steps', 'stages', 'waves', 'estimate_us')})
        spec['shared_bytes'] = 4 * row['stages'] * stage_floats(row['bm'], row['bn'], spec['ak'], spec['bkc'], row['kb'])
    phase['tiles'] = sum(s['tasks'] for s in specs)
    phase['shared_bytes'] = max(s['shared_bytes'] for s in specs)
    phase['modeled_makespan_us'] = round(best, 4)
    return phase


def schedule(phases, layouts):
    """Pack each cp.async product into the nearest earlier product phase when legal.

    Moving phase P earlier past phases X..Q is legal when P reads no storage root
    written by any of them (true dependence). P's outputs are fresh SSA roots, so no
    earlier phase reads them, and no phase writes P's inputs except their producers.
    Only matrix products are packed together; other phases are never merged.
    """
    def roots(names):
        return {layouts[name].root for name in names}

    packed = []
    for phase in phases:
        if phase.get('engine') == 'pops_cpasync':
            reads = roots(phase['reads'])
            target = None
            for index in range(len(packed) - 1, -1, -1):
                other = packed[index]
                if reads & roots(other['writes']):
                    break
                if other.get('engine') == 'pops_cpasync':
                    target = index
                    break
            if target is not None:
                host = packed[target]
                host['names'] += phase['names']
                host['targets'] += phase['targets']
                host['reads'] = list(dict.fromkeys(host['reads'] + phase['reads']))
                host['writes'] = host['writes'] + phase['writes']
                host['matrix'] = host['matrix'] + phase['matrix']
                host['tiles'] = host['tiles'] + phase['tiles']
                host['shared_bytes'] = max(host['shared_bytes'], phase['shared_bytes'])
                continue
        packed.append(phase)
    return packed


def scratch_words(phase):
    """Counter and partial words used by one packed phase (a64-aligned pieces)."""
    a64 = lambda words: (words + 63) // 64 * 64
    counters = a64(sum(s['tiles'] for s in phase['matrix'] if s['ksplit'] > 1))
    partials = sum(a64(s['tiles'] * s['ksplit'] * s['bm'] * s['bn']) for s in phase['matrix'] if s['ksplit'] > 1)
    return counters, partials


def layout_struct(s):
    """Compile-time layout of one product: (struct name, struct definition)."""
    fields = {
        'SAM': s['sam'], 'SAK': s['sak'], 'SAM1': s.get('sam1', 0), 'MIA': s.get('mia', s['m']),
        'KIA': s.get('kia', s['k']), 'SAK1': s.get('sak1', 0), 'BSA': s['bsa'], 'BSA2': s.get('bsa2', 0),
        'SBK': s['sbk'], 'SBN': s['sbn'], 'SBK1': s.get('sbk1', 0), 'SBN1': s.get('sbn1', 0),
        'KIB': s.get('kib', s['k']), 'NIB': s.get('nib', s['n']), 'BSB': s['bsb'], 'BSB2': s.get('bsb2', 0),
        'SCM': s['scm'], 'SCN': s['scn'], 'SCM1': s.get('scm1', 0), 'SCN1': s.get('scn1', 0),
        'MIC': s.get('mic', s['m']), 'NIC': s.get('nic', s['n']),
        'BSC': s['bsc'] * s.get('nb2', 1), 'BSC2': s['bsc'],
        'K': s['k'], 'NB2': s.get('nb2', 1), 'TILES_M': s['m'] // s['bm'], 'TILES_N': s['n'] // s['bn'],
        'KSPLIT': s['ksplit'], 'TASKS': s['tasks']}
    ints = {'MIA', 'KIA', 'KIB', 'NIB', 'MIC', 'NIC', 'K', 'NB2', 'TILES_M', 'TILES_N', 'KSPLIT', 'TASKS'}
    body = ' '.join(f'static constexpr {"int" if k in ints else "long long"} {k} = {v}{"" if k in ints else "LL"};'
                    for k, v in fields.items())
    name = 'pops_L_' + hashlib.sha256(body.encode()).hexdigest()[:12]
    guard = name.upper()
    return name, f'#ifndef {guard}\n#define {guard}\nstruct {name} {{ {body} }};\n#endif'


def emit(phase, ident, base, counter_words):
    """CUDA for a packed phase. Matrix scratch at arena word `base`: arrival counters occupy
    [base, base + counter_words) in EVERY phase (they must stay zero between phases), partials follow.

    Returns (phase code, {layout struct name: definition}) for file-scope emission."""
    counters, _ = scratch_words(phase)
    if counters > counter_words:
        raise ValueError('phase counters exceed the program counter region')
    cnt_cursor, part_cursor, vbase = 0, counter_words, 0
    a64 = lambda words: (words + 63) // 64 * 64
    lines, structs = [], {}
    for s in phase['matrix']:
        split = s['ksplit'] > 1
        part = f'arena + {base + part_cursor}LL' if split else 'nullptr'
        cnt = f'reinterpret_cast<unsigned*>(arena + {base + cnt_cursor}LL)' if split else 'nullptr'
        name, definition = layout_struct(s)
        structs[name] = definition
        lines.append(f"pops::gemm_phase<{s['bm']},{s['bn']},{str(s['ak']).lower()},{str(s['bkc']).lower()},{s['stages']},{name},{s['kb']}>("
                     f"{ident(s['a_root'])} + {s['a_offset']}LL, {ident(s['b_root'])} + {s['b_offset']}LL, "
                     f"{ident(s['out'])}, {part}, {cnt}, {vbase}, smem);")
        if split:
            cnt_cursor += s['tiles']
            part_cursor += a64(s['tiles'] * s['ksplit'] * s['bm'] * s['bn'])
        vbase += s['tasks']
    return '\n'.join(lines), structs


# ----------------------------------------------------------------------------------------------
# Factored layouts: a logical dimension may be a product of atoms (size, stride), outer first.
# Used to read matrix operands through permuted copies (`clone` of a permuted view) without
# materializing the copy. Pure layout algebra; no values are read.

class FactorError(ValueError):
    pass


def _canon(atoms):
    atoms = [(int(n), int(s)) for n, s in atoms if n != 1]
    merged = []
    for n, s in atoms:
        if merged and merged[-1][1] == n * s:
            merged[-1] = (merged[-1][0] * n, s)
        else:
            merged.append((n, s))
    return tuple(merged) or ((1, 0),)


@dataclass(frozen=True)
class FLayout:
    dims: tuple          # tuple of atom tuples, one per logical dimension
    root: str
    offset: int = 0

    @classmethod
    def of(cls, layout):
        return cls(tuple(_canon([(d, s)]) for d, s in zip(layout.shape, layout.strides)), layout.root, layout.offset)

    @property
    def shape(self):
        return tuple(math.prod(n for n, _ in d) for d in self.dims)

    def permute(self, order):
        return FLayout(tuple(self.dims[int(i) % len(self.dims)] for i in order), self.root, self.offset)

    def transpose(self, a, b):
        order = list(range(len(self.dims)))
        a, b = a % len(order), b % len(order)
        order[a], order[b] = order[b], order[a]
        return self.permute(order)

    def reshape(self, shape):
        if math.prod(shape) != math.prod(self.shape):
            raise FactorError('reshape changes numel')
        atoms = [a for d in self.dims for a in d if a[0] != 1]
        dims = []
        for size in shape:
            need, parts = int(size), []
            while need > 1:
                if not atoms:
                    raise FactorError('reshape exhausted atoms')
                n, s = atoms.pop(0)
                if n <= need:
                    if need % n:
                        raise FactorError('reshape boundary splits an atom')
                    parts.append((n, s))
                    need //= n
                else:
                    if n % need:
                        raise FactorError('reshape boundary splits an atom')
                    parts.append((need, s * (n // need)))
                    atoms.insert(0, (n // need, s))
                    need = 1
            dims.append(_canon(parts))
        if any(n != 1 for n, _ in atoms):
            raise FactorError('reshape left atoms')
        return FLayout(tuple(dims), self.root, self.offset)

    def unsqueeze(self, dim):
        dims = list(self.dims)
        dims.insert(dim % (len(dims) + 1), ((1, 0),))
        return FLayout(tuple(dims), self.root, self.offset)


def apply_view(fl, base, args, shape):
    """Apply one ATen view operation to a factored layout, or raise FactorError."""
    if base in {'detach', 'alias'}:
        out = fl
    elif base == 't':
        if len(fl.dims) > 2:
            raise FactorError('t requires rank <= 2')
        out = fl.permute(list(reversed(range(len(fl.dims)))))
    elif base == 'transpose':
        out = fl.transpose(int(args[1]), int(args[2]))
    elif base == 'permute':
        out = fl.permute(args[1])
    elif base in {'view', '_unsafe_view', 'reshape'}:
        out = fl.reshape(tuple(shape))
    elif base == 'unsqueeze':
        out = fl.unsqueeze(int(args[1]))
    elif base == 'squeeze':
        out = fl.reshape(tuple(shape))
    else:
        raise FactorError(f'{base} is not a factored view')
    if tuple(out.shape) != tuple(shape):
        raise FactorError('factored view shape mismatch')
    return out


def _two(dim):
    """(inner size, inner stride, outer stride) for a dimension of at most two atoms."""
    atoms = [a for a in dim if a[0] != 1] or [(1, 0)]
    if len(atoms) == 1:
        return atoms[0][0], atoms[0][1], 0
    if len(atoms) == 2:
        return atoms[1][0], atoms[1][1], atoms[0][1]
    raise FactorError('more than two atoms in a matrix dimension')


def operand(fl, rank):
    """Map a factored rank-2/3 matrix operand to engine fields (rows, cols, batch)."""
    if len(fl.dims) != rank or rank not in (2, 3):
        raise FactorError('matrix operand rank')
    rows, cols = _two(fl.dims[-2]), _two(fl.dims[-1])
    if rank == 3:
        bi, bs_in, bs_out = _two(fl.dims[0])
        nb2, bs, bs2 = (bi, bs_out, bs_in) if bs_out or len([a for a in fl.dims[0] if a[0] != 1]) == 2 else (1, bs_in, 0)
    else:
        nb2, bs, bs2 = 1, 0, 0
    return {'rows': rows, 'cols': cols, 'nb2': nb2, 'bs': bs, 'bs2': bs2, 'offset': fl.offset, 'root': fl.root}
