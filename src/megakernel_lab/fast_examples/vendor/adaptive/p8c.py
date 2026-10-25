"""P8-C harness: whole-update performance of the adaptive-memory learner through the common compiler.

Reuses the frozen C-OPS4-A harness (c4a.py, c4a_gpu.py, c4a_conv.py; snapshot hashes in p8_c/SNAPSHOT.json)
for fixtures, the donor optimizer, the native binding and the conventional arms. Adds: capture/lowering
summaries of compiler decisions, the declared mixed-precision gates of p8_c/POLICY.json (native vs FP64,
calibrated by the eager mixed-policy arithmetic reference on the same state), and task routing for the
owned P8-C cloud adapter. Task names route harness code only; the compiler never sees them.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback

from megakernel_lab.fast_examples.vendor.adaptive import c4a
from megakernel_lab.fast_examples.vendor.adaptive.c4a import err, save, sha, worst

HERE = Path(__file__).resolve().parent


def _register_fixtures():
    """P8-C harness fixtures (no compiler effect): a small head-width-32 recipe exercising the window rule."""
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import RECIPES, Arch
    RECIPES.setdefault('w32', Arch(ctx=512, window=128, chunk=64, d=64, heads=2, layers=2, hidden=128, vocab=64,
                                   m_static=64, m_fast=64, batch=2))


_register_fixtures()


def policy():
    return json.loads((HERE / 'p8_c' / 'POLICY.json').read_text())


# ----------------------------------------------------------------------------------------------- references
def own_reference(a, P, inputs, dtype, count=None, precision=None):
    """Eager autograd of our formulation; ``precision`` applies a declared ops.precision policy (FP32 only)."""
    import torch
    from contextlib import nullcontext
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import reference_loss
    from megakernel_lab.operation_compiler.precision import precision_policy
    if precision is not None and dtype != torch.float32:
        raise ValueError('a precision policy is an FP32 program')
    params = {k: v.detach().to(dtype).clone().requires_grad_(True) for k, v in P.items()}
    cast = tuple(t.to(dtype) if t.is_floating_point() else t for t in inputs)
    with (precision_policy(precision) if precision is not None else nullcontext()):
        loss = reference_loss(params, a, *cast, count=count)
    if not loss.requires_grad:
        return loss.detach(), {k: torch.zeros_like(v) for k, v in params.items()}
    grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
    return loss.detach(), {k: torch.zeros_like(v) if g is None else g for (k, v), g in zip(params.items(), grads)}


def check_state(native, a, label, *, precision=None, counts=None, causality=False, repeat=True,
                defer_cpu_optimizer=False, cpu_audit_root=None):
    """Native complete update at the bound state. FP32 programs: unchanged C-OPS4-A gates (c4a.check_state).
    Declared mixed programs: same structure, calibrated by the eager mixed-policy arithmetic reference."""
    import torch
    if precision is None:
        return c4a.check_state(native, a, label, counts=counts, causality=causality, repeat=repeat,
                               defer_cpu_optimizer=defer_cpu_optimizer, cpu_audit_root=cpu_audit_root)
    gates = c4a.gates_policy()
    snap = native.snapshot()
    P0, S0 = snap
    inputs = native.inputs()
    nC = a.chunks
    result = {'label': label, 'precision': precision, 'counts': {}}
    for count in (counts or [nC]):
        native.count.fill_(count)
        ref64 = own_reference(a, P0, inputs, torch.float64, count=count)
        refm = own_reference(a, P0, inputs, torch.float32, count=count, precision=precision)
        loss, grads, params, state = native.step()
        l64 = ref64[0].item()
        row = {'loss_native': loss.item(), 'loss_fp64': l64, 'loss_mixed_eager': refm[0].item(),
               'loss_err': abs(loss.double().item() - l64) / max(abs(l64), 1e-30),
               'loss_err_mixed_eager': abs(refm[0].double().item() - l64) / max(abs(l64), 1e-30)}
        grad_rows = {}
        for k in P0:
            e_n, e_m = err(grads[k], ref64[1][k]), err(refm[1][k], ref64[1][k])
            bound = max(gates['gradient_ratio'] * e_m['rel_l2'], gates['gradient_floor'])
            grad_rows[k] = {'native': e_n['rel_l2'], 'mixed_eager': e_m['rel_l2'], 'bound': bound,
                            'native_vs_mixed_eager': err(grads[k], refm[1][k])['rel_l2'],
                            'ref_norm': ref64[1][k].norm().item(), 'passed': e_n['rel_l2'] <= bound or (
                                ref64[1][k].norm().item() == 0 and grads[k].abs().max().item() == 0)}
        row['loss_bound'] = max(gates['loss_ratio'] * row['loss_err_mixed_eager'], gates['loss_floor'])
        row['loss_passed'] = row['loss_err'] <= row['loss_bound'] or (loss.item() == 0 and l64 == 0)
        row['grads'] = grad_rows
        row['grads_passed'] = all(g['passed'] for g in grad_rows.values())
        row['worst_grad'] = max(((g['native'] / g['bound'], k) for k, g in grad_rows.items()))
        if count == nC:
            dp, dstate, _ = c4a.donor_optimizer(a, P0, grads, S0, native.sc)
            opt = {}
            for k in P0:
                muon = any(name == f'momentum:{k}' for name in S0)
                bound = gates['muon_delta'] if muon else gates['adamw_delta']
                e = err(params[k] - P0[k], dp[k] - P0[k])
                opt[k] = {'delta_rel_l2': e['rel_l2'], 'bound': bound, 'passed': e['rel_l2'] <= bound}
            st = {k: err(state[k], v)['rel_l2'] for k, v in dstate.items()}
            row['optimizer'] = {'params': opt, 'state_worst': max(st.values()),
                                'passed': all(v['passed'] for v in opt.values()) and max(st.values()) <= gates['state_rel']}
            if defer_cpu_optimizer:
                if cpu_audit_root is None or Path(label).name != label:
                    raise ValueError('an explicit diagnostic directory and safe label are required')
                destination = Path(cpu_audit_root) / f'{label}-count{count}.pt'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                torch.save({'parameters': {k: v.detach().cpu() for k, v in P0.items()},
                            'optimizer_state': {k: v.detach().cpu() for k, v in S0.items()},
                            'native_gradients': {k: v.detach().cpu() for k, v in grads.items()},
                            'schedule': native.sc.detach().cpu(),
                            'gpu_donor_parameters': {k: v.detach().cpu() for k, v in dp.items()}}, destination)
                row['optimizer']['cpu_optimizer_audit'] = dict(pending=True, path=str(destination),
                                                               sha256=c4a.sha(destination))
            if repeat:
                native.restore(snap)
                again = native.step()
                row['repeat_bitwise'] = bool(torch.equal(again[0], loss) and all(
                    torch.equal(again[1][k], grads[k]) for k in grads) and all(
                    torch.equal(again[2][k], params[k]) for k in params))
        result['counts'][str(count)] = row
        native.restore(snap)
    if causality:
        result['causality'] = c4a.causality_check(native, a)
    native.count.fill_(nC)
    native.restore(snap)
    result['passed'] = all(r['loss_passed'] and r['grads_passed'] and r.get('optimizer', {}).get('passed', True)
                           and r.get('repeat_bitwise', True) for r in result['counts'].values()) and (
        not causality or result['causality']['passed'])
    return result


# ---------------------------------------------------------------------------------------------- summaries
def program_summary(manifest):
    """Compact compiler-decision summary of a lowered loop program (diagnostic receipt)."""
    regions = manifest.get('regions', {})
    out = {'arena_bytes': manifest.get('arena_bytes'), 'regions': {}}
    for name, region in regions.items():
        phases = region.get('phases', [])
        kinds = Counter(p.get('kind') for p in phases)
        decisions = region.get('matrix_decisions', [])
        engines = Counter((d.get('selected'), (d.get('reason') or '')[:60]) for d in decisions)
        out['regions'][name] = {
            'phases': len(phases), 'barriers': region.get('barrier_count'), 'kinds': dict(kinds.most_common(30)),
            'stages': len(region.get('stages', [])),
            'matrix': [[sel, reason, n] for (sel, reason), n in engines.most_common(12)],
            'attention_groups': len(region.get('attention_groups', [])),
            'attention_refusals': region.get('attention_refusals', [])[:12],
            'group_rules': sorted(region.get('group_rules', {}))}
    for key in ('loop_schedule_policy', 'compiler_selection', 'body_tape_bytes', 'boundary_tape_bytes'):
        if key in manifest:
            out[key] = manifest[key]
    return out


def task_probe(spec, out):
    """c4a.task_probe (capture, optional interpretation/donor checks, lowering, build) + decision summaries."""
    for key, value in spec.get('env', {}).items():   # diagnostic compiler switches only
        os.environ[key] = value
    result = c4a.task_probe(spec, out)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        result['summary'] = program_summary(manifest)
        save(out / 'summary.json', result['summary'])
    graph = json.loads((out / 'graph.json').read_text()) if (out / 'graph.json').exists() else {}
    meta = graph.get('metadata', {})
    for key in ('precision_policy', 'attention_tape', 'init_attention_tape', 'cast_reuse', 'tape_selection'):
        if key in meta:
            result[key] = meta[key]
    result['passed'] = bool(result.get('lower', {}).get('ok', not spec.get('lower', True)))
    return result


def task_dump_joint(spec, out):
    """Diagnostic: write the generated code of every joint graph handed to the tape partitioner (no lowering)."""
    import torch
    torch.set_num_threads(1)
    from megakernel_lab.operation_compiler import loop as L
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import capture
    a = c4a.recipe(spec['recipe'])
    P, inputs = c4a.fixture(a)
    original = L._partition_body_tape
    dumped = []

    def spy(joint, module, **kwargs):
        path = out / f'joint-{len(dumped)}.py'
        path.write_text(module.code)
        lines = []
        for node in module.graph.nodes:
            val = node.meta.get('val')
            desc = (tuple(val.shape), str(val.dtype), tuple(val.stride())) if isinstance(val, torch.Tensor) else None
            lines.append(f"{node.name} {node.op} {node.target} {[getattr(x, 'name', x) for x in node.args]} {desc}")
        (out / f'joint-{len(dumped)}-nodes.txt').write_text('\n'.join(lines))
        dumped.append({'file': path.name, 'nodes': len(lines), 'kwargs': {k: str(v)[:200] for k, v in kwargs.items()}})
        return original(joint, module, **kwargs)
    L._partition_body_tape = spy
    try:
        capture(a, P, inputs, **spec.get('capture_options', {}))
    finally:
        L._partition_body_tape = original
    return {'dumped': dumped, 'passed': True}


TASKS = {'probe': task_probe, 'fd': c4a.task_fd, 'dump_joint': task_dump_joint}


# ----------------------------------------------------------------------------------------------- GPU tasks
def task_native_check(spec, out):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import schedule_rows, synthetic_tokens
    a = c4a.recipe(spec['recipe'])
    precision = spec.get('precision')
    native = c4a.Native(spec['folder'])
    nC = a.chunks
    result = {'recipe': spec['recipe'], 'precision': precision, 'runtime_setup_s': native.layout['runtime_setup_s'],
              'blocks': native.runtime.blocks, 'arena_bytes': native.runtime.allocated_bytes, 'states': []}
    counts = spec.get('counts', [0, 1, nC // 2, nC])
    deferred = dict(defer_cpu_optimizer=spec.get('defer_cpu_optimizer', False),
                    cpu_audit_root=out / 'cpu-optimizer-audit')
    result['states'].append(check_state(native, a, 'fresh', precision=precision, counts=counts,
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
        result['states'].append(check_state(native, a, f'after_{train}', precision=precision, counts=[nC], **deferred))
    result['passed'] = all(s['passed'] for s in result['states'])
    save(out / 'native-check.json', result)
    return result




def task_compare(spec, out):
    """Two native builds of the same captured program class, one update each from the identical bound state and
    inputs: bitwise equality of loss, gradients, new parameters and optimizer state (for pure lowering changes)."""
    import torch
    a = c4a.recipe(spec['recipe'])
    rows = {}
    results = []
    for folder in spec['folders']:
        native = c4a.Native(folder)
        results.append(native.step())
        del native
        torch.cuda.synchronize()
    (l0, g0, p0, s0), (l1, g1, p1, s1) = results
    rows['loss_equal'] = bool(torch.equal(l0, l1))
    rows['loss'] = [l0.item(), l1.item()]
    rows['grads_unequal'] = [k for k in g0 if not torch.equal(g0[k], g1[k])]
    rows['grads_rel_l2'] = {k: c4a.err(g1[k], g0[k])['rel_l2'] for k in rows['grads_unequal']}
    rows['params_unequal'] = [k for k in p0 if not torch.equal(p0[k], p1[k])]
    rows['state_unequal'] = [k for k in s0 if not torch.equal(s0[k], s1[k])]
    rows['passed'] = rows['loss_equal'] and not (rows['grads_unequal'] or rows['params_unequal'] or rows['state_unequal'])
    return rows


def task_profile_members(spec, out):
    """Diagnostic (member_ticks builds): per level, each member's latest-CTA finish minus the level start (ns),
    attributed to the member's region-stage phases; never a timing metric."""
    import statistics
    import torch
    native = c4a.Native(spec['folder'])
    native.runtime.step()
    torch.cuda.synchronize()
    manifest = native.runtime.manifest
    runs = []
    for _ in range(spec.get('runs', 3)):
        native.runtime._ticks.zero_()
        native.runtime.step(profile=True)
        torch.cuda.synchronize()
        runs.append(native.runtime._ticks.cpu().tolist())
    regions = manifest['regions']
    rows = []
    for block in manifest.get('tick_blocks', []):
        if 'member_base' not in block or block['iterations'] != 1:
            continue
        width = block['member_width']
        for level in range(block['levels']):
            start = [r[block['base'] + level] for r in runs]
            for k, (region, stage) in enumerate(block['members'][level]):
                ends = [r[block['member_base'] + level * width + k] for r in runs]
                ns = statistics.median(e - s0 for e, s0 in zip(ends, start))
                key = {'init': next((n for n in regions if n.startswith('init') and 'vjp' not in n), 'init'),
                       'init_vjp': next((n for n in regions if n.startswith('init_vjp')), 'init_vjp')}.get(region, region)
                names = []
                if key in regions and stage < len(regions[key].get('stages', [])):
                    phases = regions[key]['phases']
                    names = [f"{phases[q]['kind']}:{'/'.join(phases[q]['names'][:2])}"
                             for q in regions[key]['stages'][stage]['phases']]
                rows.append({'block': block['block'], 'level': level, 'member': k, 'ns': ns, 'names': names})
    save(out / 'members.json', rows)
    top = sorted(rows, key=lambda r: -r['ns'])[:80]
    return {'passed': True, 'top': [[round(r['ns'] / 1e3, 1), r['block'], r['level'], r['names'][:3]] for r in top]}


def task_conv_profile(spec, out):
    """Diagnostic: torch.profiler kernel timeline of the conventional arm's complete update (not a metric)."""
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive import c4a_conv
    from megakernel_lab.fast_examples.vendor.adaptive import c4a_gpu
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import schedule_rows
    a = c4a.recipe(spec['recipe'])
    engine, info = c4a_conv.make_engine(a, spec.get('arm', 'bf16'), backend=spec.get('backend', 'flex'))
    sc = schedule_rows(64, 3e-3, 1.0)[7].cuda()
    batches = c4a_gpu.synthetic_batches(a)
    for i in range(6):
        c4a_conv.stage(engine, batches[i % 4][:, :-1], batches[i % 4][:, 1:], sc)
        engine.step()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(spec.get('steps', 3)):
            c4a_conv.stage(engine, batches[i % 4][:, :-1], batches[i % 4][:, 1:], sc)
            engine.step()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    events.sort(key=lambda e: e.time_range.start)
    timeline = [[e.name[:160], e.time_range.start, e.time_range.end - e.time_range.start] for e in events]
    save(out / 'timeline.json', timeline)
    by = Counter()
    for name, _, dur in timeline:
        by[name] += dur
    steps = spec.get('steps', 3)
    return {'passed': True, 'kernels_per_step': len(timeline) / steps,
            'busy_us_per_step': sum(d for _, _, d in timeline) / steps,
            'top': [[n, round(v / steps, 1)] for n, v in by.most_common(60)]}


GPU_TASKS = {'conv_profile': task_conv_profile, 'profile_members': task_profile_members, 'compare': task_compare, 'native_check': task_native_check, 'timing': c4a.task_timing, 'train': c4a.task_train,
             'profile': c4a.task_profile, 'update_check': c4a.task_update_check}










