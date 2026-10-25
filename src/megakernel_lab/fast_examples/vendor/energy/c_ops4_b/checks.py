"""Same-state numerical checks shared by the CPU recipe task and the GPU native gates (c_ops4_b/POLICY.json).

e(X) = ||Q_X - Q_R|| / max(||Q_R||, 1e-30) against the FP64 reference R (c_ops4_b/reference.py), each
implementation evaluated from the SAME input bytes (the subject's own chain state). Calibration implementations:
C1 = eager autograd control, C2 = eager hand-written-backward control (c_ops4_b/controls.py), both FP32 under
the recipe's BF16 product policy. PASS iff e(subject) <= max(KAPPA * max(e(C1), e(C2)), FLOOR).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import reference as R
from megakernel_lab.fast_examples.vendor.energy.examples import energy as recipe

KAPPA, FLOOR = 2.0, 1e-6


def rel(a, b):
    a, b = torch.as_tensor(a).double().flatten(), torch.as_tensor(b).double().flatten()
    return float((a - b).norm() / max(float(b.norm()), 1e-30))


def judge(errors, subject='native'):
    calibration = max(errors[k] for k in errors if k != subject)
    bound = max(KAPPA * calibration, FLOOR)
    finite = math.isfinite(errors[subject])
    return {'errors': errors, 'bound': bound, 'passed': bool(finite and errors[subject] <= bound)}


def names_and_decay(cfg):
    inv = recipe.inventory(cfg)
    return [n for n, _, _ in inv], {n: d for n, _, d in inv}


def p64(params):
    return {k: v.detach().double() for k, v in params.items()}


def packed(params):
    return {k: (v.to(torch.bfloat16) if v.dim() == 2 else v) for k, v in params.items()}


def noise(cfg, key, k, t):
    from megakernel_lab.operation_compiler.random import philox_field
    z = recipe.dims(cfg)
    dev = key.device
    return philox_field(key, torch.as_tensor(k, dtype=torch.int64, device=dev).reshape(()),
                        torch.as_tensor(t, dtype=torch.int64, device=dev).reshape(()), [z['B'], z['D']], 'normal')


def step_increments(cfg, params, x_prev, k, t, key, subject_next=None):
    """Increments x_k - x_{k-1} from x_prev for R (FP64, FP64 normals), C1, C2 (FP32 device normals)."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import Control
    z = recipe.dims(cfg)
    seed = cfg['seed']
    _, _, x_r = R.chain_step(p64(params), x_prev.double(), int(k), int(t), cfg, R.Arith('fp64'), z['L'], seed)
    xi = noise(cfg, key, k, t)
    out = {'R': x_r - x_prev.double()}
    for label, variant in (('C1', 'eager'), ('C2', 'manual')):
        ctrl = Control(cfg, variant)
        out[label] = ctrl._chain_step(packed(params), x_prev, xi).double() - x_prev.double()
    if subject_next is not None:
        out['native'] = subject_next.double() - x_prev.double()
    return out


def update_observables(cfg, params, opt, xpos, xneg):
    """loss, raw gradients, new params, m, v from (params, optimizer state, x+, x-) for R, C1, C2."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import Control
    names, decay = names_and_decay(cfg)
    z = recipe.dims(cfg)
    P = p64(params)
    loss, E, G = R.loss_and_grads(P, xpos.double(), xneg.double(), cfg['alpha'], R.Arith('fp64'), z['L'])
    steps = {n: int(opt[f'step:{n}']) for n in names}
    m = {n: opt[f'exp_avg:{n}'] for n in names}
    v = {n: opt[f'exp_avg_sq:{n}'] for n in names}
    new, norm, coef = R.adamw(cfg, names, decay, P, G, m, v, steps)
    obs = {'R': pack(names, loss, G, {n: new[n][0] for n in names}, {n: new[n][1] for n in names},
                     {n: new[n][2] for n in names})}
    for label, variant in (('C1', 'eager'), ('C2', 'manual')):
        ctrl = Control(cfg, variant)
        lc, _, Gc = ctrl._param_grads(params, xpos, xneg)
        out = ctrl._optimizer(params, m, v, {n: opt[f'step:{n}'] for n in names}, Gc, names, decay)
        obs[label] = pack(names, lc, Gc, {n: out[n][0] for n in names}, {n: out[n][1] for n in names},
                          {n: out[n][2] for n in names})
    return obs, {'norm': float(norm), 'coef': float(coef)}


def pack(names, loss, grads, params, m, v):
    flat = lambda d: torch.cat([d[n].detach().double().flatten().cpu() for n in names])
    return {'loss': torch.as_tensor(loss).double().reshape(1).cpu(), 'grads': flat(grads), 'params': flat(params),
            'm': flat(m), 'v': flat(v)}


def compare(obs, subject='native', reference='R', skip=()):
    rows = {}
    for q in obs[reference]:
        if q in skip:
            continue
        errors = {label: rel(obs[label][q], obs[reference][q]) for label in obs if label != reference}
        rows[q] = judge(errors, subject)
    return rows


def optimizer_from_gradients(cfg, params, opt, grads):
    """FP64 clip + AdamW from the subject's own raw gradients (identical-gradient optimizer tier)."""
    names, decay = names_and_decay(cfg)
    steps = {n: int(opt[f'step:{n}']) for n in names}
    new, norm, coef = R.adamw(cfg, names, decay, p64(params), {n: grads[n].double() for n in names},
                              {n: opt[f'exp_avg:{n}'] for n in names}, {n: opt[f'exp_avg_sq:{n}'] for n in names}, steps)
    return new, float(norm), float(coef)


def recipe_vs_reference(cfg, images, rows):
    """CPU: eager user functions (examples/energy.py) versus FP64 R and the p1 emulation, same state."""
    from megakernel_lab.fast_examples.vendor.energy.cops4b import bindings
    from megakernel_lab.operation_compiler.optim import prepare_optimizer
    torch.manual_seed(0)
    params, opt, state, (img, order), _ = bindings(cfg, rows, images=images)
    init, step, loss_fn = recipe.make_functions(cfg)
    z = recipe.dims(cfg)
    seed, t = cfg['seed'], int(state['t'])
    report = {}
    with torch.no_grad():
        x0 = init(params, state, img, order)['x']
    x0_ref, fresh, idx = R.init_negatives(cfg, state['buf'], t, seed)
    report['init_exact'] = bool(torch.equal(x0, x0_ref))
    report['fresh_rows'] = int(fresh.sum())
    x, steps = x0, []
    for k in range(z['K']):
        nxt = step(params, {'x': x}, state, torch.tensor(k), img, order)[0]['x']
        _, _, xr = R.chain_step(p64(params), x.double(), k, t, cfg, R.Arith('fp64'), z['L'], seed)
        _, _, xp = R.chain_step(p64(params), x.double(), k, t, cfg, R.Arith('p1'), z['L'], seed)
        dr = xr - x.double()
        errors = {'recipe': rel(nxt.double() - x.double(), dr), 'p1': rel(xp - x.double(), dr)}
        steps.append({'k': k, **judge(errors, 'recipe')})
        x = nxt
    report['steps'] = steps
    xpos = img[order[t]]
    names, decay = names_and_decay(cfg)
    prepared = prepare_optimizer(recipe.optimizer_config(cfg), params, optimizer_state=opt)
    grads, (value, new_state) = torch.func.grad_and_value(
        lambda p: loss_fn(p, {'x': x}, state, img, order), has_aux=True)(params)
    flat_p = tuple(params[n] for n in names)
    new_p, new_o = prepared.update(flat_p, tuple(grads[n] for n in names), tuple(opt.values()))
    P = p64(params)
    loss_r, _, G_r = R.loss_and_grads(P, xpos.double(), x.double(), cfg['alpha'], R.Arith('fp64'), z['L'])
    loss_p, _, G_p = R.loss_and_grads(P, xpos.double(), x.double(), cfg['alpha'], R.Arith('p1'), z['L'])
    steps0 = {n: 0 for n in names}
    zero = {n: torch.zeros_like(params[n]) for n in names}
    new_r, _, _ = R.adamw(cfg, names, decay, P, G_r, zero, zero, steps0)
    new_pp, _, _ = R.adamw(cfg, names, decay, P, G_p, zero, zero, steps0)
    flat = lambda d: torch.cat([d[n].double().flatten() for n in names])
    obs = {'recipe': {'loss': value.reshape(1), 'grads': flat(grads), 'params': torch.cat([w.flatten() for w in new_p])},
           'p1': {'loss': loss_p.reshape(1), 'grads': flat(G_p), 'params': flat({n: new_pp[n][0] for n in names})},
           'R': {'loss': loss_r.reshape(1), 'grads': flat(G_r), 'params': flat({n: new_r[n][0] for n in names})}}
    report['update'] = compare(obs, subject='recipe')
    slots = (int(state['ptr']) + np.arange(z['B'])) % z['R']
    expected = state['buf'].clone()
    expected[torch.as_tensor(slots)] = x
    report['commit_exact'] = bool(torch.equal(new_state['buf'], expected)) and int(new_state['ptr']) == (
        int(state['ptr']) + z['B']) % z['R'] and int(new_state['t']) == t + 1
    report['passed'] = bool(report['init_exact'] and report['commit_exact']
                            and all(s['passed'] for s in steps) and all(r['passed'] for r in report['update'].values()))
    return report
