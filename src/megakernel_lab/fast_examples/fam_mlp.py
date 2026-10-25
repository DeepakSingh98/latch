"""Historical mixed-operation MLP suite (latch-comparisons-suite-v1) through the public capture_module API.

User model only (mlp_models.py, copied unchanged from the independent test's historical_models.py, which wraps
the recovered research/latch_ops/examples/models.py equations). Momentum SGD .01/.9, coupled decay .001.
FP32 cases: strict lowering; BF16 cases: explicit BF16 products in the user model + precision_policy='practical'.
"""
import torch
from megakernel_lab.fast_examples.engines import Conventional, Native
from megakernel_lab.fast_examples import mlp_models as M

CASES = M.CASES
REFERENCE_ARM = 'eager'
PREPARE_TIMEOUT, RUN_TIMEOUT, AUDIT_TIMEOUT = 3600, 1800, 1800
NATIVE = ('latch', 'latch-legacy', 'latch-bf16proj', 'latch-common')


def native_variants(c):
    policy = 'practical' if c['native_bf16'] else 'strict'
    variants = {'latch': dict(precision_policy=policy),          # normal automatic API
                'latch-legacy': dict(execution='legacy', precision_policy=policy)}                        # explicit retained-path diagnostic
    variants['latch-common'] = dict(execution='compose', compose_selection='common', precision_policy=policy)
    if c['native_bf16']:
        # Historical LATCH-COMPARISONS native BF16 recipe: FP32 user model captured with the named
        # precision='bf16_projections' recipe, lowered practical (separate arithmetic declaration).
        variants['latch-bf16proj'] = dict(execution='compose', precision_policy='practical',
                                          _capture_precision='bf16_projections')
    return variants


def capture(c, precision=None):
    from megakernel_lab.operation_compiler import capture_module, SGDConfig
    torch.set_num_threads(1)
    m = M.model(c if precision is None else dict(c, native_bf16=False))
    x, y = M.batch(c, 0)
    cap = capture_module(m, M.loss_fn(c), (x,), (y,), precision=precision,
                         optimizer=SGDConfig(lr=c['lr'], momentum=c['momentum'], weight_decay=c['decay']))
    params = {n: p.detach().clone() for n, p in m.named_parameters()}
    initial = dict(parameters=params, optimizer={n: torch.zeros_like(p) for n, p in params.items()})
    return cap, initial, dict(parameters=sum(p.numel() for p in params.values()))


def conventional_arms(c):
    arms = ['eager', 'eager-graph', 'compiled']
    if not c['native_bf16']:
        arms.append('compiled-bf16')       # practical control: explicit BF16 products, FP32 elsewhere
    return arms


def arms(c, prep):
    native = [v for v in NATIVE if prep.get('variants', {}).get(v, {}).get('ok')]
    return conventional_arms(c) + native + (['latch-direct'] if 'latch' in native else [])


def audited_arms(c):
    return conventional_arms(c) + list(NATIVE)


def arm_label(c, arm):
    bf = c['native_bf16'] or arm.endswith('-bf16')
    return f"{arm} ({'explicit BF16 products' if bf else 'FP32'})"


def make_model(c, arm):
    return M.model(dict(c, bf16=arm.endswith('-bf16')))


def objective(model, batch, c):
    return M.loss_fn(c)(model(batch[0]), batch[1])


def objective_functional(call, batch, c):
    return M.loss_fn(c)(call(batch[0]), batch[1])


def initial_state(c, params):
    return {n: torch.zeros_like(p) for n, p in params.items()}


def make_optimizer(c, params, arm):
    return torch.optim.SGD(params, lr=c['lr'], momentum=c['momentum'], weight_decay=c['decay'], foreach=True)


def bind_state(c, opt, params, state):
    for n, p in params.items():
        opt.state[p]['momentum_buffer'] = state[n]


def functional_update(c, params, grads, state):
    for n, p in params.items():
        v = c['momentum'] * state[n] + grads[n] + c['decay'] * p
        state[n].copy_(v)
        p.copy_(p - c['lr'] * v)


def pool(c, device, engine=None):
    return [tuple(M.batch(c, i, device)) for i in range(8)]


def engine(c, arm, d):
    if arm == 'latch-direct':
        return Native(d / 'latch', graph=False, label='latch-direct (automatic, no CUDA graph)')
    if arm in NATIVE:
        return Native(d / arm, label=arm)
    return Conventional(__import__(__name__, fromlist=['*']), c, arm, pool(c, 'cuda')[0])


def reference(c, s):
    """Independent CPU autograd + torch.optim.SGD at the saved state on pool batch 1 (same model arithmetic)."""
    m = M.model(c)
    m.load_state_dict(s['parameters'])
    params = dict(m.named_parameters())
    opt = torch.optim.SGD(list(params.values()), lr=c['lr'], momentum=c['momentum'], weight_decay=c['decay'])
    for n, p in params.items():
        opt.state[p]['momentum_buffer'] = s['optimizer'][n].clone()
    x, y = M.batch(c, 1)
    loss = M.loss_fn(c)(m(x), y)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in params.items()}
    opt.step()
    return dict(loss=loss.detach(), gradients=grads, before_parameters=s['parameters'],
                after_parameters={n: p.detach().clone() for n, p in params.items()})
