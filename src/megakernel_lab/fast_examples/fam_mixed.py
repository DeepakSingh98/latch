"""Step-14 mixed-operation programs (explicit BF16 products, pairwise-bmm auxiliary, shared block) via capture_module.

User model: recovered_release/step14_workloads.py, copied unchanged from LATCH-STEP-14-SPEED
(research/latch_release/workloads.py at step-14-speed); fresh_pairwise is the integration.3 untuned
program (integration/harness/mixed/workloads.py @5e592f9309). SGD lr .01, momentum .9, no decay.
Historical: r14_mixed_b 0.2970 vs 0.3010 (0.987; i3 1.001), r13_mixed 0.953, fresh_pairwise i3 0.794;
denominator compiled_default_graph (Inductor default mode + CUDA graph beat max-autotune here).
"""
import torch
from dataclasses import replace
from megakernel_lab.fast_examples.engines import Conventional, Native
from megakernel_lab.fast_examples.recovered_release import step14_workloads as W

W.CASES['fresh_pairwise'] = W.Case('fresh_pairwise', batch=576, inputs=344, width=704, outputs=104, depth=2,
                                   auxiliary=.15, branch='pairwise', chunk=32, shared=True, seed=28101)
CASES = {n: dict(case=W.CASES[n]) for n in ('r14_mixed_b', 'r13_mixed', 'fresh_pairwise')}
LR, MOMENTUM, DECAY = W.LR, W.MOMENTUM, 0.0
REFERENCE_ARM = 'eager'
PREPARE_TIMEOUT, RUN_TIMEOUT, AUDIT_TIMEOUT = 3600, 1800, 1800
NATIVE = ('latch', 'latch-legacy')
CONVENTIONAL = ('eager', 'eager-graph', 'compiled', 'compiled-default')


def native_variants(c):
    return {'latch': dict(), 'latch-legacy': dict(execution='legacy')}


def capture(c, precision=None):
    from megakernel_lab.operation_compiler import capture_module, SGDConfig
    torch.set_num_threads(1)
    m = W.ComposedMLP(c['case'])
    inputs, targets = W.fixture(c['case'])
    cap = capture_module(m, W.loss_fn(c['case']), inputs, targets,
                         optimizer=SGDConfig(lr=LR, momentum=MOMENTUM, weight_decay=DECAY))
    params = {n: p.detach().clone() for n, p in m.named_parameters()}
    return cap, dict(parameters=params, optimizer={n: torch.zeros_like(p) for n, p in params.items()}), \
        dict(parameters=sum(p.numel() for p in params.values()))


def arms(c, prep):
    return list(CONVENTIONAL) + [v for v in NATIVE if prep.get('variants', {}).get(v, {}).get('ok')]


def audited_arms(c):
    return list(CONVENTIONAL) + list(NATIVE)


def arm_label(c, arm):
    return f'{arm} (explicit BF16 products)'


def make_model(c, arm):
    return W.ComposedMLP(c['case'])


def objective(model, batch, c):
    return W.loss_fn(c['case'])(model(batch[0]), *batch[1:])


def objective_functional(call, batch, c):
    return W.loss_fn(c['case'])(call(batch[0]), *batch[1:])


def initial_state(c, params):
    return {n: torch.zeros_like(p) for n, p in params.items()}


def make_optimizer(c, params, arm):
    return torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=DECAY, foreach=True)


def bind_state(c, opt, params, state):
    for n, p in params.items():
        opt.state[p]['momentum_buffer'] = state[n]


def functional_update(c, params, grads, state):
    for n, p in params.items():
        v = MOMENTUM * state[n] + grads[n] + DECAY * p
        state[n].copy_(v)
        p.copy_(p - LR * v)


def pool(c, device, engine=None):
    result = []
    for i in range(8):
        inputs, targets = W.fixture(c['case'], seed_offset=1 + i)
        result.append(tuple(t.to(device) for t in (*inputs, *targets)))
    return result


def engine(c, arm, d):
    if arm in NATIVE:
        return Native(d / arm, label=arm)
    return Conventional(__import__(__name__, fromlist=['*']), c, arm, pool(c, 'cuda')[0])


def reference(c, s):
    m = W.ComposedMLP(c['case'])
    m.load_state_dict(s['parameters'])
    params = dict(m.named_parameters())
    opt = torch.optim.SGD(list(params.values()), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    for n, p in params.items():
        opt.state[p]['momentum_buffer'] = s['optimizer'][n].clone()
    batch = pool(c, 'cpu')[1]
    loss = objective(m, batch, c)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in params.items()}
    opt.step()
    return dict(loss=loss.detach(), gradients=grads, before_parameters=s['parameters'],
                after_parameters={n: p.detach().clone() for n, p in params.items()})
