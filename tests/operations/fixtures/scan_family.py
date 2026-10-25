"""S108F declarations for the maintained selective state-space family s99k-gdssm-v1.

Model, parameter inventory and initialisation are those of S99K (s99k_ssm/declaration.py at tag
s103j-scan-performance-v1, research/s99/S99K/SPEC.md), reproduced here without behavioural change:

  per ssm block:  u = rms(x) * g;  z = u . w_in  (columns v | q | c | k | r, widths E, E, E, N, N)
                  dl = softplus(c + dt_bias);  alpha_t = keep_t * exp(-dl_t * exp(a_log))      (per channel e, state n)
                  h_t = alpha_t h_{t-1} + dl_t v_t k_t;  s_t = h_t . r_t + d_skip * v_t;  o = s * silu(q);  x += o . w_out
  per ffn block:  x += silu(rms(x) * g . w1) . w2
  head:           logits = rms(x) * g_f . emb^T (tied);  mean next-byte cross entropy
  update:         global-norm clip, AdamW (decay on emb/w_in/w_out/w1/w2), fp32 masters

This is Mamba-1-*inspired* (selective diagonal recurrence, softplus step, SiLU gate, D skip, B/C shared over channels),
without the short convolution, x_proj/dt_proj low-rank step, or Mamba-2 heads. No Mamba compatibility is claimed.

S108F changes only the admission bounds (a generator resource property, not model semantics): up to 32 blocks and
D <= 2048, T <= 8192, so the ~100M declarations are expressible. Every other refusal of S99K is kept. The numerical
policy is named per execution (see PLAN.md): s108f-pbf16-v1 is the practical BF16 policy of the conventional control.
"""
import hashlib
import json
import math

import numpy as np

FAMILY = 's99k-gdssm-v1'
DECL_SCHEMA = 's108f-decl-v1'
INIT = 's99k-init-v1'
EPS = 1e-5
POLICIES = ('s108f-pbf16-v1',)


class Refused(ValueError):
    def __init__(self, code, detail=''):
        super().__init__(code + (': ' + detail if detail else ''))
        self.code = code


BOUNDS = dict(V=(16, 4096), D=(64, 2048), E=(16, 8192), F=(16, 16384), T=(1, 8192), blocks=(1, 32), BT=(1, 65536),
              N=(4, 8, 16, 32))
MATRIX_ROLES = ('emb', 'w_in', 'w_out', 'w1', 'w2')
OPT_DEFAULT = dict(kind='adamw', lr=1e-3, beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1)
CLIP_DEFAULT = dict(max_norm=1.0)


def canonical(decl):
    return json.dumps(decl, sort_keys=True, separators=(',', ':'))


def digest(decl):
    return hashlib.sha256(canonical(decl).encode()).hexdigest()


def admit(decl):
    """Normalized admitted declaration or Refused(code). Device-independent rules only."""
    if not isinstance(decl, dict) or decl.get('schema') != DECL_SCHEMA:
        raise Refused('SCHEMA', 'schema must be ' + DECL_SCHEMA)
    known = {'schema', 'name', 'vocab', 'd_model', 'batch', 'seq_len', 'blocks', 'norm_eps', 'seed', 'head',
             'precision', 'optimizer', 'clip', 'state', 'execution', 'family', 'init'}
    if decl.get('family', FAMILY) != FAMILY or decl.get('init', INIT) != INIT:
        raise Refused('FAMILY', 'only %s with %s' % (FAMILY, INIT))
    extra = set(decl) - known
    if extra:
        raise Refused('UNKNOWN_FIELD', ','.join(sorted(extra)))
    ints = {}
    for k in ('vocab', 'd_model', 'batch', 'seq_len', 'seed'):
        v = decl.get(k)
        if type(v) is not int:
            raise Refused('INTEGER_REQUIRED', k)
        ints[k] = v
    V, D, B, T = ints['vocab'], ints['d_model'], ints['batch'], ints['seq_len']
    if not (BOUNDS['V'][0] <= V <= BOUNDS['V'][1] and V % 8 == 0):
        raise Refused('VOCAB_BOUNDS', str(V))
    if not (BOUNDS['D'][0] <= D <= BOUNDS['D'][1] and D % 8 == 0):
        raise Refused('WIDTH_BOUNDS', str(D))
    if not BOUNDS['T'][0] <= T <= BOUNDS['T'][1]:
        raise Refused('LENGTH_BOUNDS', str(T))
    if B < 1 or B * T > BOUNDS['BT'][1]:
        raise Refused('ROWS_BOUNDS', 'B*T=%d' % (B * T))
    if decl.get('norm_eps', EPS) != EPS:
        raise Refused('NORM_EPS', 'only 1e-5 is admitted')
    if decl.get('precision', POLICIES[0]) not in POLICIES:
        raise Refused('PRECISION', str(decl.get('precision')))
    if decl.get('head', {'tied': True}) != {'tied': True}:
        raise Refused('UNTIED_HEAD', 'only a tied embedding/head is admitted')
    state = decl.get('state', {'carry_across_updates': False})
    if set(state) - {'carry_across_updates', 'reset'}:
        raise Refused('STATE_FIELD', ','.join(sorted(set(state) - {'carry_across_updates', 'reset'})))
    if state.get('carry_across_updates', False) is not False:
        raise Refused('CROSS_UPDATE_STATE_CARRY', 'every update starts each sequence from h=0')
    if state.get('reset', 'document_start') != 'document_start':
        raise Refused('RESET_POLICY', str(state.get('reset')))
    if decl.get('execution', {'devices': 1}) != {'devices': 1}:
        raise Refused('DISTRIBUTED', 'single-device execution only')
    opt = dict(OPT_DEFAULT, **decl.get('optimizer', {}))
    if opt['kind'] != 'adamw' or set(opt) != set(OPT_DEFAULT):
        raise Refused('OPTIMIZER', 'AdamW with lr/beta1/beta2/eps/weight_decay only')
    if not all(isinstance(opt[k], (int, float)) and math.isfinite(opt[k]) for k in OPT_DEFAULT if k != 'kind'):
        raise Refused('OPTIMIZER', 'finite hyperparameters')
    if not (0 < opt['lr'] and 0 <= opt['beta1'] < 1 and 0 <= opt['beta2'] < 1 and opt['eps'] >= 0 and opt['weight_decay'] >= 0):
        raise Refused('OPTIMIZER', 'invalid hyperparameters')
    clip = dict(CLIP_DEFAULT, **decl.get('clip', {}))
    if set(clip) != {'max_norm'} or not (isinstance(clip['max_norm'], (int, float)) and math.isfinite(clip['max_norm']) and clip['max_norm'] > 0):
        raise Refused('CLIP', 'positive finite max_norm only')
    blocks = decl.get('blocks')
    if not isinstance(blocks, list) or not BOUNDS['blocks'][0] <= len(blocks) <= BOUNDS['blocks'][1]:
        raise Refused('BLOCK_COUNT', 'between %d and %d blocks' % BOUNDS['blocks'])
    out_blocks = []
    for i, b in enumerate(blocks):
        kind = b.get('kind') if isinstance(b, dict) else None
        if kind == 'ssm':
            if set(b) != {'kind', 'd_inner', 'd_state'}:
                raise Refused('SSM_FIELDS', 'block %d' % i)
            E, N = b['d_inner'], b['d_state']
            if type(E) is not int or not (BOUNDS['E'][0] <= E <= BOUNDS['E'][1] and E % 8 == 0):
                raise Refused('INNER_BOUNDS', 'block %d E=%r' % (i, E))
            if type(N) is not int or N not in BOUNDS['N']:
                raise Refused('STATE_BOUNDS', 'block %d N=%r (4, 8, 16 or 32)' % (i, N))
            out_blocks.append(dict(kind='ssm', d_inner=E, d_state=N))
        elif kind == 'ffn':
            if set(b) - {'kind', 'd_hidden', 'activation'} or b.get('activation', 'silu') != 'silu':
                raise Refused('FFN_FIELDS', 'block %d (SiLU FFN only)' % i)
            F = b.get('d_hidden')
            if type(F) is not int or not (BOUNDS['F'][0] <= F <= BOUNDS['F'][1] and F % 8 == 0):
                raise Refused('HIDDEN_BOUNDS', 'block %d F=%r' % (i, F))
            out_blocks.append(dict(kind='ffn', d_hidden=F, activation='silu'))
        elif kind in ('attention', 'moe', 'mamba2', 'hyena', 'conv', 'mamba'):
            raise Refused('UNSUPPORTED_BLOCK', '%s (block %d)' % (kind, i))
        else:
            raise Refused('UNKNOWN_BLOCK', 'block %d' % i)
    if not any(b['kind'] == 'ssm' for b in out_blocks):
        raise Refused('NO_STATE_SPACE_BLOCK', 'at least one ssm block is required')
    return dict(schema=DECL_SCHEMA, family=FAMILY, name=str(decl.get('name', 'unnamed')), vocab=V, d_model=D,
                batch=B, seq_len=T, blocks=out_blocks, norm_eps=EPS, seed=ints['seed'], head={'tied': True},
                precision=decl.get('precision', POLICIES[0]), init=INIT, optimizer=opt, clip=clip,
                state=dict(carry_across_updates=False, reset='document_start'), execution={'devices': 1})


def inventory(d):
    """Ordered unique parameters (S99K SPEC section 1 order): id, role, shape, decay group."""
    V, D = d['vocab'], d['d_model']
    rows = [('emb', 'emb', (V, D))]
    for i, b in enumerate(d['blocks']):
        p = 'blocks.%d.' % i
        if b['kind'] == 'ssm':
            E, N = b['d_inner'], b['d_state']
            rows += [(p + 'norm', 'norm', (D,)), (p + 'w_in', 'w_in', (D, 3 * E + 2 * N)),
                     (p + 'a_log', 'a_log', (E, N)), (p + 'dt_bias', 'dt_bias', (E,)),
                     (p + 'd_skip', 'd_skip', (E,)), (p + 'w_out', 'w_out', (E, D))]
        else:
            F = b['d_hidden']
            rows += [(p + 'norm', 'norm', (D,)), (p + 'w1', 'w1', (D, F)), (p + 'w2', 'w2', (F, D))]
    rows.append(('final_norm', 'norm', (D,)))
    return [dict(id=i, role=r, shape=s, group=0 if r in MATRIX_ROLES else 1) for i, r, s in rows]


def n_params(d):
    return int(sum(np.prod(p['shape']) for p in inventory(d)))


def softplus_inv(y):
    return y + np.log(-np.expm1(-y))


def initialize(d, seed=None):
    """s99k-init-v1: float32 masters in inventory order (byte-identical rule to S99K initialize)."""
    rng = np.random.Generator(np.random.PCG64(d['seed'] if seed is None else seed))
    nb = len(d['blocks'])
    out = {}
    for p in inventory(d):
        shape, role = p['shape'], p['role']
        if role == 'emb':
            w = rng.normal(0.0, 0.02, shape)
        elif role == 'norm':
            w = np.ones(shape)
        elif role in ('w_in', 'w1'):
            w = rng.normal(0.0, 1.0 / math.sqrt(shape[0]), shape)
        elif role == 'a_log':
            w = np.broadcast_to(np.log(np.arange(1, shape[1] + 1, dtype=np.float64)), shape).copy()
        elif role == 'dt_bias':
            dt = np.exp(rng.uniform(math.log(1e-3), math.log(1e-1), shape))
            w = softplus_inv(dt)
        elif role == 'd_skip':
            w = np.ones(shape)
        elif role in ('w_out', 'w2'):
            w = rng.normal(0.0, 1.0 / math.sqrt(2 * nb * shape[0]), shape)
        else:
            raise AssertionError(role)
        out[p['id']] = np.ascontiguousarray(w, dtype=np.float32)
    return out


def make(name, *, D, E, N, F, pairs, L, tokens=16384, V=256, seed=1081, lr=2e-3):
    """Declaration builder: `pairs` x [ssm(E, N), ffn(F)], batch = tokens / L (S107F protocol)."""
    if tokens % L:
        raise Refused('ROWS_BOUNDS', 'tokens %% L')
    blocks = []
    for _ in range(pairs):
        blocks += [dict(kind='ssm', d_inner=E, d_state=N), dict(kind='ffn', d_hidden=F)]
    return admit(dict(schema=DECL_SCHEMA, name=name, vocab=V, d_model=D, batch=tokens // L, seq_len=L, seed=seed,
                      blocks=blocks, optimizer=dict(kind='adamw', lr=lr, beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1),
                      clip=dict(max_norm=1.0)))
