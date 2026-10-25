"""C-OPS4-A harness: adaptive-memory training through the common operation compiler.

CPU tasks (Modal CPU, one thread per worker): capture, reference interpretation,
independent FP64 checks, lowering and nvcc builds. GPU tasks (one H100): native
execution checks, timing against conventional controls, and real-text training.
Task names route harness code only; the compiler never sees them.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def err(actual, expected):
    import torch
    a, e = actual.detach().double().cpu(), expected.detach().double().cpu()
    if a.shape != e.shape:
        return {'shape_mismatch': [list(a.shape), list(e.shape)]}
    diff = (a - e).abs()
    scale = e.abs().max().item()
    return {'max_abs': diff.max().item() if diff.numel() else 0.0, 'ref_max': scale,
            'rel_max': diff.max().item() / scale if scale > 0 else diff.max().item(),
            'rel_l2': ((a - e).norm() / e.norm()).item() if e.norm() > 0 else (a - e).norm().item(),
            'finite': bool(torch.isfinite(a).all())}


def worst(errors, key='rel_l2'):
    vals = [(v[key], k) for k, v in errors.items() if key in v]
    return max(vals) if vals else (0.0, None)


# ------------------------------------------------------------------------------------------------ fixtures
def recipe(name):
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import RECIPES
    return RECIPES[name]


def fixture(a, seed=106140, step=8, H=64, lr=3e-3, eta=1.0):
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import batch_inputs, init_params, schedule_rows, synthetic_tokens
    P = init_params(a, seed)
    sc = schedule_rows(H, lr, eta)[step - 1]
    return P, batch_inputs(a, synthetic_tokens(a, seed + step), sc)


def donor_reference(a, P, inputs, dtype, *, mutant=None):
    """Independent loss/gradients: the vendored donor model (explicit masked attention) + autograd."""
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b import model as MD
    x, y, sc = inputs[:3]
    da = a.donor()
    attn = MD.Attn(da, 'ref', 'cpu')
    rope = MD.rope_tables(da, 'cpu', dtype)
    params = {k: v.detach().to(dtype).clone().requires_grad_(True) for k, v in P.items()}
    ce = MD.forward_ce(params, da, x, y, attn, rope, eta=sc[5].to(dtype), mutant=mutant)
    loss = ce.mean()
    grads = torch.autograd.grad(loss, list(params.values()))
    return loss.detach(), dict(zip(params, grads)), ce.detach()


def own_reference(a, P, inputs, dtype, count=None, mutant=None):
    """Our formulation evaluated eagerly with autograd (no compiler), in the given dtype."""
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import reference_loss
    params = {k: v.detach().to(dtype).clone().requires_grad_(True) for k, v in P.items()}
    cast = tuple(t.to(dtype) if t.is_floating_point() else t for t in inputs)
    loss = reference_loss(params, a, *cast, count=count, mutant=mutant)
    if not loss.requires_grad:
        return loss.detach(), {k: torch.zeros_like(v) for k, v in params.items()}
    grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
    return loss.detach(), {k: torch.zeros_like(v) if g is None else g for (k, v), g in zip(params.items(), grads)}


def donor_optimizer(a, P, grads, state, sc):
    """The donor's unchanged Engine._update applied to supplied gradients (FP32)."""
    import torch
    from types import SimpleNamespace
    from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b import train as TR
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import inventory
    inv = inventory(a)
    obj = SimpleNamespace(kind={n: k for n, _, k in inv}, adam=[n for n, _, k in inv if k != 'matrix'])
    groups = {}
    for n, _, k in inv:
        if k == 'matrix':
            groups.setdefault(tuple(P[n].shape), []).append(n)
    obj.groups = groups
    mom = {n: state[f'momentum:{n}'] for n, _, k in inv if k == 'matrix'}
    m = {n: state[f'exp_avg:{n}'] for n in obj.adam}
    v = {n: state[f'exp_avg_sq:{n}'] for n in obj.adam}
    with torch.no_grad():
        new, norm = TR.Engine._update(obj, dict(P), dict(grads), mom, m, v, torch.cat([sc[:5], sc[5:]]))
    params = {n: new[n][0] for n in P}
    out_state = {}
    for n, _, k in inv:
        if k == 'matrix':
            out_state[f'momentum:{n}'] = new[n][1]
        else:
            out_state[f'exp_avg:{n}'], out_state[f'exp_avg_sq:{n}'] = new[n][1], new[n][2]
    return params, out_state, norm


# --------------------------------------------------------------------------------------------------- tasks
def task_probe(spec, out):
    """Capture + reference-interpreter/eager/donor agreement + lowering (and optional nvcc build)."""
    import torch
    torch.set_num_threads(1)
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import capture, init_params
    from megakernel_lab.operation_compiler.muon import prepare_muon
    a = recipe(spec['recipe'])
    P, inputs = fixture(a, seed=spec.get('seed', 106140))
    result = {'recipe': spec['recipe'], 'arch': vars(a), 'parameters': sum(v.numel() for v in P.values())}
    t0 = time.perf_counter()
    cap = capture(a, P, inputs, storage_mode=spec.get('storage_mode', 'save_body_tape'), **spec.get('capture_options', {}))
    result['capture_seconds'] = time.perf_counter() - t0
    result['regions'] = {name: {'operations': len(g.operations),
                                'targets': dict(Counter(op.target for op in g.operations))}
                         for name, g in cap.ir.regions.items()}
    result['body_tape'] = cap.ir.metadata.get('body_tape')
    result['init_tape'] = cap.ir.metadata.get('init_tape')
    result['sequence_specs'] = cap.ir.metadata.get('sequence_specs')
    count = torch.tensor(spec.get('count', a.chunks), dtype=torch.int64)
    flat = (*cap.example_args[:-1], count)
    save(out / 'graph.json', cap.ir.to_dict())
    torch.save({'args': flat}, out / 'inputs.pt')
    if spec.get('interpret', True):
        t0 = time.perf_counter()
        graph = cap.unpack(cap.graph_step(*flat))
        result['graph_step_seconds'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        eager = cap.unpack(cap.step_fn(*flat))
        result['step_fn_seconds'] = time.perf_counter() - t0
        result['graph_vs_eager'] = {
            'loss': err(graph['loss'], eager['loss']),
            'grads_worst': worst({k: err(graph['grads'][k], v) for k, v in eager['grads'].items()}),
            'params_worst': worst({k: err(graph['params'][k], v) for k, v in eager['params'].items()}),
            'state_worst': worst({k: err(graph['optimizer_state'][k], v) for k, v in eager['optimizer_state'].items()})}
        if spec.get('donor', True):
            loss64, grads64, _ = donor_reference(a, P, inputs, torch.float64)
            result['graph_vs_donor_fp64'] = {
                'loss': err(graph['loss'], loss64),
                'grads': {k: err(graph['grads'][k], v) for k, v in grads64.items()}}
            result['graph_vs_donor_fp64']['grads_worst'] = worst(result['graph_vs_donor_fp64']['grads'])
            loss32, grads32, _ = donor_reference(a, P, inputs, torch.float32)
            result['donor_fp32_vs_fp64'] = {'loss': err(loss32, loss64),
                                            'grads_worst': worst({k: err(grads32[k], v) for k, v in grads64.items()})}
            # Optimizer: donor's unchanged update on the graph's gradients.
            state0 = prepare_muon(cap.config, P).state
            dp, dstate, _ = donor_optimizer(a, P, graph['grads'], state0, inputs[2])
            result['optimizer_vs_donor'] = {
                'params': worst({k: err(graph['params'][k], v) for k, v in dp.items()}),
                'state': worst({k: err(graph['optimizer_state'][k], v) for k, v in dstate.items()}),
                'params_max_abs': max(err(graph['params'][k], v)['max_abs'] for k, v in dp.items())}
    if spec.get('lower', True):
        from megakernel_lab.operation_compiler.lower import lower
        t0 = time.perf_counter()
        try:
            program = lower(cap.ir, **spec.get('lower_options', {}))
            result['lower'] = {'ok': True, 'seconds': time.perf_counter() - t0,
                               'arena_bytes': program.manifest['arena_bytes'],
                               'boundary_tape_bytes': program.manifest.get('boundary_tape_bytes'),
                               'body_tape_bytes': program.manifest.get('body_tape_bytes'),
                               'schedule': program.manifest.get('compiler_selection'),
                               'source_bytes': len(program.source)}
            (out / 'program.cu').write_text(program.source)
            save(out / 'manifest.json', program.manifest)
            if spec.get('build'):
                from megakernel_lab.operation_compiler.runtime import build
                t0 = time.perf_counter()
                library = build(program, cache_dir=spec['cache_dir'], arch='sm_90')
                result['build'] = {'seconds': time.perf_counter() - t0, 'library': str(library),
                                   'library_sha256': sha(library), 'source_hash': program.cache_key}
        except Exception as exc:  # noqa: BLE001 -- recorded diagnostic
            result['lower'] = {'ok': False, 'seconds': time.perf_counter() - t0, 'error': repr(exc)[:4000],
                               'traceback': traceback.format_exc()[-6000:]}
    save(out / 'prepare.json', result)
    return result


def task_fd(spec, out):
    """Independent FP64 checks on small cases: donor-model central differences along random directions vs
    autograd of our formulation; donor/own fault mutants show the gradient gate's sensitivity."""
    import torch
    torch.set_num_threads(1)
    a = recipe(spec['recipe'])
    P, inputs = fixture(a, seed=spec.get('seed', 106140))
    loss, grads = own_reference(a, P, inputs, torch.float64)
    dloss, dgrads, _ = donor_reference(a, P, inputs, torch.float64)
    result = {'recipe': spec['recipe'], 'loss_own_vs_donor': abs(loss.item() - dloss.item()) / abs(dloss.item()),
              'grads_own_vs_donor': worst({k: err(grads[k], v) for k, v in dgrads.items()})}
    from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b import model as MD
    da = a.donor()
    attn, rope = MD.Attn(da, 'ref', 'cpu'), MD.rope_tables(da, 'cpu', torch.float64)
    x, y, sc = inputs[:3]

    def donor_loss(params):
        with torch.no_grad():
            return MD.forward_ce(params, da, x, y, attn, rope, eta=sc[5].double()).mean().item()
    P64 = {k: v.double() for k, v in P.items()}
    g = torch.Generator().manual_seed(spec.get('fd_seed', 4242))
    eps = spec.get('eps', 1e-4)
    rows = []
    for r in range(spec.get('directions', 6)):
        v = {k: torch.randn(t.shape, generator=g, dtype=torch.float64) for k, t in P64.items()}
        norm = math.sqrt(sum((t * t).sum().item() for t in v.values()))
        v = {k: t / norm for k, t in v.items()}
        plus = donor_loss({k: P64[k] + eps * v[k] for k in P64})
        minus = donor_loss({k: P64[k] - eps * v[k] for k in P64})
        fd = (plus - minus) / (2 * eps)
        analytic = sum((grads[k] * v[k]).sum().item() for k in P64)
        rows.append({'fd': fd, 'autograd_own': analytic, 'rel': abs(fd - analytic) / max(abs(fd), 1e-30)})
    result['directional'] = rows
    result['directional_worst'] = max(r['rel'] for r in rows)
    faults = {}
    for name, fn in (('detach_write', lambda: own_reference(a, P, inputs, torch.float64, mutant='detach_write')),
                     ('read_after_write', lambda: donor_reference(a, P, inputs, torch.float64, mutant='read_after_write')[:2]),
                     ('future_label', lambda: donor_reference(a, P, inputs, torch.float64, mutant='future_label')[:2])):
        floss, fgrads = fn()
        faults[name] = {'loss_rel': abs(floss.item() - loss.item()) / abs(loss.item()),
                        'grads_worst_rel_l2': worst({k: err(fgrads[k], v) for k, v in grads.items()})}
    result['faults'] = faults
    tol = spec.get('fd_tolerance', 1e-6)
    result['passed'] = bool(result['directional_worst'] <= tol and result['grads_own_vs_donor'][0] <= 1e-8
                            and result['loss_own_vs_donor'] <= 1e-12
                            and all(f['grads_worst_rel_l2'][0] > 1e-3 for f in faults.values()))
    return result




TASKS = {'probe': task_probe, 'fd': task_fd, }


# ------------------------------------------------------------------------------------------------- GPU side
def load_native(folder, *, blocks=None):
    import torch
    from megakernel_lab.operation_compiler.lower import GeneratedProgram
    from megakernel_lab.operation_compiler.runtime import Runtime
    folder = Path(folder)
    receipt = json.loads((folder / 'prepare.json').read_text())
    library = Path(receipt['build']['library'])
    if sha(library) != receipt['build']['library_sha256']:
        raise ValueError('CPU-built library hash differs from its receipt')
    program = GeneratedProgram.load(library.parent)
    if program.cache_key != receipt['build']['source_hash']:
        raise ValueError('CPU-built program identity differs from its receipt')
    graph = json.loads((folder / 'graph.json').read_text())
    flat = tuple(t.cuda() for t in torch.load(folder / 'inputs.pt', weights_only=True, map_location='cpu')['args'])
    t0 = time.perf_counter()
    runtime = Runtime(program, flat, library=library, blocks=blocks)
    torch.cuda.synchronize()
    meta = graph['metadata']
    layout = {'names': meta['parameter_names'], 'state_names': meta['optimizer_state_names'],
              'n_inputs': meta['n_inputs'], 'runtime_setup_s': time.perf_counter() - t0}
    return runtime, flat, receipt, layout


class Native:
    """Named access to a bound native runtime: params/state commit in place on step()."""

    def __init__(self, folder, blocks=None):
        self.runtime, self.flat, self.receipt, self.layout = load_native(folder, blocks=blocks)
        n, no = len(self.layout['names']), len(self.layout['state_names'])
        self.params = dict(zip(self.layout['names'], self.flat[:n]))
        self.state = dict(zip(self.layout['state_names'], self.flat[n:n + no]))
        self.x, self.y, self.sc, self.cos, self.sin, self.mask = self.flat[n + no:n + no + 6]
        self.count = self.flat[-1]

    def inputs(self):
        return (self.x, self.y, self.sc, self.cos, self.sin, self.mask)

    def snapshot(self):
        return ({k: v.clone() for k, v in self.params.items()}, {k: v.clone() for k, v in self.state.items()})

    def restore(self, snap):
        import torch
        with torch.no_grad():
            for k, v in snap[0].items():
                self.params[k].copy_(v)
            for k, v in snap[1].items():
                self.state[k].copy_(v)

    def step(self):
        import torch
        self.runtime.step()
        torch.cuda.synchronize()
        return (self.runtime.loss.clone(), {k: v.clone() for k, v in self.runtime.grads.items()},
                {k: v.clone() for k, v in self.params.items()}, {k: v.clone() for k, v in self.state.items()})


def gates_policy():
    return json.loads((Path(__file__).parent / 'cops4_a' / 'POLICY.json').read_text())['native_gates']


def check_state(native, a, label, *, counts=None, causality=False, repeat=True,
                defer_cpu_optimizer=False, cpu_audit_root=None):
    """Native complete update at the bound state vs FP64/FP32 eager autograd and the donor optimizer."""
    import torch
    policy = gates_policy()
    snap = native.snapshot()
    P0, S0 = snap
    inputs = native.inputs()
    nC = a.chunks
    result = {'label': label, 'counts': {}}
    for count in (counts or [nC]):
        native.count.fill_(count)
        ref64 = own_reference(a, P0, inputs, torch.float64, count=count)
        ref32 = own_reference(a, P0, inputs, torch.float32, count=count)
        loss, grads, params, state = native.step()
        row = {'loss_native': loss.item(), 'loss_fp64': ref64[0].item(),
               'loss_err': abs(loss.double().item() - ref64[0].item()) / max(abs(ref64[0].item()), 1e-30),
               'loss_err_eager32': abs(ref32[0].double().item() - ref64[0].item()) / max(abs(ref64[0].item()), 1e-30)}
        grad_rows = {}
        for k in P0:
            e_n, e_32 = err(grads[k], ref64[1][k]), err(ref32[1][k], ref64[1][k])
            bound = max(policy['gradient_ratio'] * e_32['rel_l2'], policy['gradient_floor'])
            grad_rows[k] = {'native': e_n['rel_l2'], 'eager32': e_32['rel_l2'], 'bound': bound,
                            'ref_norm': ref64[1][k].norm().item(), 'passed': e_n['rel_l2'] <= bound or (
                                ref64[1][k].norm().item() == 0 and grads[k].abs().max().item() == 0)}
        row['loss_bound'] = max(policy['loss_ratio'] * row['loss_err_eager32'], policy['loss_floor'])
        row['loss_passed'] = row['loss_err'] <= row['loss_bound'] or (loss.item() == 0 and ref64[0].item() == 0)
        row['grads'] = grad_rows
        row['grads_passed'] = all(g['passed'] for g in grad_rows.values())
        row['worst_grad'] = max(((g['native'] / g['bound'], k) for k, g in grad_rows.items()))
        if count == nC:
            dp, dstate, _ = donor_optimizer(a, P0, grads, S0, native.sc)
            opt = {}
            for k in P0:
                delta_n, delta_d = params[k] - P0[k], dp[k] - P0[k]
                muon = any(name == f'momentum:{k}' for name in S0)
                bound = policy['muon_delta'] if muon else policy['adamw_delta']
                e = err(delta_n, delta_d)
                opt[k] = {'delta_rel_l2': e['rel_l2'], 'bound': bound, 'passed': e['rel_l2'] <= bound}
            st = {k: err(state[k], v)['rel_l2'] for k, v in dstate.items()}
            # Diagnostic (not a gate): the same donor update evaluated on CPU, i.e. the spread between two
            # correct implementations of the BF16 Newton-Schulz step on identical inputs.
            cpu_audit = None
            if defer_cpu_optimizer:
                if cpu_audit_root is None or Path(label).name != label:
                    raise ValueError('deferred CPU optimizer diagnostics require an explicit output and safe label')
                destination = Path(cpu_audit_root) / f'{label}-count{count}.pt'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError('deferred CPU optimizer diagnostic must not overwrite prior evidence')
                torch.save({'parameters': {k: v.detach().cpu() for k, v in P0.items()},
                            'optimizer_state': {k: v.detach().cpu() for k, v in S0.items()},
                            'native_gradients': {k: v.detach().cpu() for k, v in grads.items()},
                            'schedule': native.sc.detach().cpu(),
                            'gpu_donor_parameters': {k: v.detach().cpu() for k, v in dp.items()}}, destination)
                cpu_audit = {'pending': True, 'path': str(destination), 'sha256': sha(destination),
                             'bytes': destination.stat().st_size,
                             'scope': 'same-input donor optimizer CPU-versus-GPU diagnostic; GPU numerical gates unchanged'}
                spread = None
            else:
                cpu_p, _, _ = donor_optimizer(a, {k: v.cpu() for k, v in P0.items()}, {k: v.cpu() for k, v in grads.items()},
                                              {k: v.cpu() for k, v in S0.items()}, native.sc.cpu())
                spread = {k: err(cpu_p[k] - P0[k].cpu(), dp[k].cpu() - P0[k].cpu())['rel_l2'] for k in P0}
            delta_norm = {k: (params[k] - P0[k]).norm().item() for k in P0}
            row['optimizer'] = {'params': opt, 'state_worst': max(st.values()),
                                'donor_cpu_vs_gpu_delta_rel_l2': spread, 'native_delta_norm': delta_norm,
                                'passed': all(v['passed'] for v in opt.values())
                                and max(st.values()) <= policy['state_rel']}
            if cpu_audit is not None:
                row['optimizer']['cpu_optimizer_audit'] = cpu_audit
            if repeat:
                native.restore(snap)
                again = native.step()
                row['repeat_bitwise'] = bool(torch.equal(again[0], loss) and all(
                    torch.equal(again[1][k], grads[k]) for k in grads) and all(
                    torch.equal(again[2][k], params[k]) for k in params))
        result['counts'][str(count)] = row
        native.restore(snap)
    if causality:
        result['causality'] = causality_check(native, a)
    native.count.fill_(nC)
    native.restore(snap)
    result['passed'] = all(r['loss_passed'] and r['grads_passed'] and r.get('optimizer', {}).get('passed', True)
                           and r.get('repeat_bitwise', True) for r in result['counts'].values()) and (
        not causality or result['causality']['passed'])
    return result


def causality_check(native, a):
    """Prefix losses (count = k) are bitwise unchanged when bytes from chunk j on are edited; later ones change."""
    import torch
    snap = native.snapshot()
    x0, y0 = native.x.clone(), native.y.clone()
    nC, C = a.chunks, a.chunk
    j = nC // 2
    ks = sorted({1, max(1, j - 1), j, j + 1, nC})

    def prefix_losses():
        out = {}
        for k in ks:
            native.count.fill_(k)
            out[k] = native.step()[0]
            native.restore(snap)
        return out
    before = prefix_losses()
    with torch.no_grad():
        g = torch.Generator(device='cuda').manual_seed(7)
        native.x[:, j * C:] = torch.randint(0, a.vocab, native.x[:, j * C:].shape, device='cuda', generator=g)
        native.y[:, j * C - 1:] = torch.randint(0, a.vocab, native.y[:, j * C - 1:].shape, device='cuda', generator=g)
    after = prefix_losses()
    with torch.no_grad():
        native.x.copy_(x0)
        native.y.copy_(y0)
    rows = {str(k): {'before': before[k].item(), 'after': after[k].item(),
                     'bitwise_equal': bool(torch.equal(before[k], after[k]))} for k in ks}
    # Label y[j*C-1] belongs to chunk j-1, so prefixes of at most j-1 chunks must be bitwise unchanged.
    unaffected = [k for k in ks if k <= j - 1]
    affected = [k for k in ks if k >= j]
    passed = all(rows[str(k)]['bitwise_equal'] for k in unaffected) and all(
        not rows[str(k)]['bitwise_equal'] for k in affected)
    return {'edit_chunk': j, 'edited': 'x from position j*C, y from j*C-1 (chunk j-1 last label onward)',
            'rows': rows, 'unaffected_prefixes': unaffected, 'affected_prefixes': affected, 'passed': passed}


def task_native_check(spec, out):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import schedule_rows, synthetic_tokens
    a = recipe(spec['recipe'])
    native = Native(spec['folder'])
    nC = a.chunks
    result = {'recipe': spec['recipe'], 'runtime_setup_s': native.layout['runtime_setup_s'],
              'blocks': native.runtime.blocks, 'launch_policy': native.runtime.launch_policy,
              'arena_bytes': native.runtime.allocated_bytes, 'states': []}
    counts = spec.get('counts', [0, 1, nC // 2, nC])
    deferred = {'defer_cpu_optimizer': bool(spec.get('defer_cpu_optimizer', False)),
                'cpu_audit_root': Path(out) / 'cpu-optimizer-audit'}
    result['states'].append(check_state(native, a, 'fresh', counts=counts,
                                        causality=spec.get('causality', True), **deferred))
    save(out / 'native-check.json', result)
    train = spec.get('train_updates', 0)
    if train:
        rows = schedule_rows(train, 3e-3, 1.0).cuda()
        losses = []
        for i in range(train):
            tokens = synthetic_tokens(a, 5000 + i).cuda()
            with torch.no_grad():
                native.x.copy_(tokens[:, :-1])
                native.y.copy_(tokens[:, 1:])
                native.sc.copy_(rows[i])
            native.runtime.step()
            losses.append(native.runtime.loss.item())
        result['train_losses'] = losses
        tokens = synthetic_tokens(a, 99)
        with torch.no_grad():
            native.x.copy_(tokens[:, :-1].cuda())
            native.y.copy_(tokens[:, 1:].cuda())
        result['states'].append(check_state(native, a, f'after_{train}', counts=[nC], causality=False, **deferred))
    result['passed'] = all(s['passed'] for s in result['states'])
    result['cpu_optimizer_audit_pending'] = bool(spec.get('defer_cpu_optimizer', False) and any(
        row.get('optimizer', {}).get('cpu_optimizer_audit', {}).get('pending')
        for state in result['states'] for row in state['counts'].values()))
    save(out / 'native-check.json', result)
    return result


def run_arm(spec, out, label):
    """One arm in a fresh process (clean peak memory, compile and setup accounting)."""
    import subprocess
    import sys
    target = Path(out) / f'{label}.json'
    spec = {**spec, 'out': str(target)}
    lane = Path(__file__).parent
    env = dict(os.environ)
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, '-I', '-m', 'megakernel_lab.fast_examples.vendor.adaptive.c4a_gpu', json.dumps(spec)], cwd=str(lane), env=env,
                          capture_output=True, text=True, timeout=spec.get('timeout', 3600))
    (Path(out) / f'{label}.log').write_text(proc.stdout[-20000:] + '\n----\n' + proc.stderr[-20000:])
    result = json.loads(target.read_text()) if target.exists() else {'passed': False, 'returncode': proc.returncode}
    result['wall_s'] = time.perf_counter() - started
    return result


def task_timing(spec, out):
    """Complete-update time / memory / setup: native and conventional arms, each in a fresh process, R rounds."""
    rows = {}
    for r in range(spec.get('rounds', 2)):
        for arm in spec['arms']:
            label = f'{arm}-r{r}'
            if arm.startswith('native'):
                arm_spec = {'kind': 'native', 'recipe': spec['recipe'], 'folder': spec['folders'][arm],
                            'iters': spec.get('iters', 30), 'warmup': spec.get('warmup', 5),
                            'profile': spec.get('profile', False) and r == 0}
            else:
                arm_spec = {'kind': 'conv', 'recipe': spec['recipe'], 'arm': arm, 'iters': spec.get('iters', 30),
                            'warmup': spec.get('warmup', 5), 'backend': spec.get('backend', 'flex')}
            rows[label] = run_arm(arm_spec, out, label)
            save(out / 'timing-partial.json', rows)
    summary = {}
    for arm in spec['arms']:
        med = [rows[f'{arm}-r{r}'].get('timing', {}).get('median_ms') for r in range(spec.get('rounds', 2))]
        summary[arm] = {'median_ms_by_round': med, 'passed': all(rows[f'{arm}-r{r}'].get('passed') for r in range(spec.get('rounds', 2)))}
    return {'recipe': spec['recipe'], 'rows': rows, 'summary': summary,
            'passed': all(v['passed'] for v in summary.values())}


def task_train(spec, out):
    """Bounded real-text comparison: native vs unchanged conventional recipe, same init/data/schedule."""
    common = {'recipe': spec['recipe'], 'updates': spec['updates'], 'seed': spec.get('seed', 106140),
              'lr': spec.get('lr', 3e-3), 'eta': spec.get('eta', 1.0)}
    rows, params = {}, {}
    for arm in spec['arms']:
        path = str(Path(out) / f'{arm}-final.pt')
        if arm.startswith('native'):
            arm_spec = {**common, 'kind': 'train_native', 'folder': spec['folders'][arm], 'params_out': path}
        else:
            arm_spec = {**common, 'kind': 'train_conv', 'arm': arm, 'params_out': path}
        rows[arm] = run_arm(arm_spec, out, f'train-{arm}')
        if rows[arm].get('passed'):
            params[arm] = path
        save(out / 'train-partial.json', rows)
    rows['evaluation'] = run_arm({'kind': 'evaluate', 'recipe': spec['recipe'], 'params': params,
                                  'eta': common['eta'], 'eval_units': spec.get('eval_units', 64)}, out, 'evaluate')
    return {'rows': rows, 'passed': all(r.get('passed') for r in rows.values())}


def level_profile(runtime, runs=5):
    """Per-barrier-level CTA0 stamps (profile mode; diagnostic only), attributed to region stage phases.
    Parsing adapted from P6-C p6c.py (6f63852b4)."""
    import statistics
    import torch
    manifest = runtime.manifest
    stamps = []
    for _ in range(runs):
        runtime.step(profile=True)
        torch.cuda.synchronize()
        stamps.append(runtime._ticks.cpu().tolist())
    regions = manifest['regions']
    region_of = {'F': manifest['splits'].get('forward_region', 'body_saved'), 'R': 'body_vjp_saved'}

    def names(member):
        region, stage = member
        key = region_of.get(region[0]) if region[:2] in ('F(', 'R(') else region
        key = {'init': next((k for k in regions if k.startswith('init') and 'vjp' not in k), 'init'),
               'init_vjp': next((k for k in regions if k.startswith('init_vjp')), 'init_vjp'),
               'terminal': 'terminal_vjp', 'optimizer': 'optimizer'}.get(key, key)
        if key not in regions or 'stages' not in regions[key] or stage >= len(regions[key]['stages']):
            return [f'{region}:{stage}']
        phases = regions[key]['phases']
        return [f"{phases[p]['kind']}:{'/'.join(phases[p]['names'][:2])}" for p in regions[key]['stages'][stage]['phases']]
    out = []
    for block in manifest.get('tick_blocks', []):
        rows = []
        for it in range(min(block['iterations'], 2)):
            base = block['base'] + it * (block['levels'] + 1)
            for level in range(block['levels']):
                vals = [r[base + level + 1] - r[base + level] for r in stamps]
                if any(v < 0 or v > 10**10 for v in vals):
                    continue
                rows.append({'it': it, 'level': level, 'ns': statistics.median(vals),
                             'members': [n for m in block['members'][level] for n in names(m)]})
        out.append({'block': block['block'], 'levels': block['levels'], 'iterations': block['iterations'],
                    'total_ns_first_iters': sum(r['ns'] for r in rows), 'rows': rows})
    return out


def task_profile(spec, out):
    native = Native(spec['folder'])
    native.runtime.step()
    levels = level_profile(native.runtime, spec.get('runs', 5))
    kinds = Counter()
    for block in levels:
        for row in block['rows']:
            share = row['ns'] / max(1, len(row['members']))
            for m in row['members']:
                kinds[(block['block'], m.split(':')[0])] += share
    save(out / 'levels.json', levels)
    top = sorted(((row['ns'], b['block'], row['level'], row['members'][:4]) for b in levels for row in b['rows']), reverse=True)[:60]
    return {'top_levels': top, 'kind_ns': {f'{b}:{k}': v for (b, k), v in kinds.most_common(40)}, 'passed': True}


def task_update_check(spec, out):
    rows = {}
    for arm in spec['arms']:
        arm_spec = {'kind': 'update_check', 'recipe': spec['recipe'], 'arm': arm,
                    'folder': spec.get('folders', {}).get(arm)}
        rows[arm] = run_arm(arm_spec, out, f'update-{arm}')
    return {'rows': rows, 'passed': all(r.get('passed') for r in rows.values())}


GPU_TASKS = {'native_check': task_native_check, 'timing': task_timing, 'train': task_train, 'profile': task_profile,
             'update_check': task_update_check}






