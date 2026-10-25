"""STEP19-SPEED dynamic-batch matrix programs through public capture_module(dynamic_batch=DynamicBatch(32)).

User model: TwoLayer from research/latch_release/integration2/harness/dynamic/step19_workloads.py @5e592f9309
(reproduced below, unchanged equations/initialization): x -> SiLU(x W1 + b1) W2 + b2, MSE, explicit BF16 products
(FP32 accumulate, BF16-rounded results), FP32 masters; SGD lr .01, momentum .9, coupled decay .001.
One native program (capacity 32, default precision policy, as historically built) is
executed at the stated active batch. Controls are static per-batch programs (historical STEP19 control set:
eager+graph and compiled+graph, BF16 recipe), which can specialize to the exact batch.
Historical: matrix_bf16 B32 27.51 vs 30.21 us (0.911, STEP19-SPEED); i3 0.901; B1 ~1.2-1.27 (slower);
reserved_bf16 768-1536-384 B32 1.03.
"""
import math
import torch
from megakernel_lab.fast_examples.engines import Conventional, Native

POLICY_CASES = {
    'matrix_bf16': dict(inputs=512, hidden=1024, outputs=256, seed=19107),
    'reserved_bf16': dict(inputs=768, hidden=1536, outputs=384, seed=19131),
}
LR, MOMENTUM, DECAY = 0.01, 0.9, 0.001
CASES = {f'{n}-b{b}': dict(program=n, batch=b, **POLICY_CASES[n])
         for n, batches in (('matrix_bf16', (32, 16, 1)), ('reserved_bf16', (32,))) for b in batches}
REFERENCE_ARM = 'eager'
PREPARE_TIMEOUT, RUN_TIMEOUT, AUDIT_TIMEOUT = 3600, 1800, 1800
NATIVE = ('latch',)
CONVENTIONAL = ('eager', 'eager-graph', 'compiled')


def product(x, w):
    return (x.to(torch.bfloat16) @ w.to(torch.bfloat16)).float()


class TwoLayer(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        g = torch.Generator().manual_seed(c['seed'])
        for name, rows, cols in [('first', c['inputs'], c['hidden']), ('head', c['hidden'], c['outputs'])]:
            setattr(self, name, torch.nn.Parameter(torch.randn(rows, cols, generator=g) * (0.35 / math.sqrt(rows))))
        self.first_bias = torch.nn.Parameter(torch.linspace(-0.03, 0.03, c['hidden']))
        self.head_bias = torch.nn.Parameter(torch.linspace(-0.01, 0.01, c['outputs']))

    def forward(self, x):
        return product(torch.nn.functional.silu(product(x, self.first) + self.first_bias), self.head) + self.head_bias


def loss_fn(prediction, target):
    return (prediction - target).square().mean()


def fixture(c, batch, step):
    g = torch.Generator().manual_seed(c['seed'] + 1000 + step * 37 + batch)
    return torch.randn(batch, c['inputs'], generator=g) * 0.4, torch.randn(batch, c['outputs'], generator=g) * 0.1


def native_variants(c):
    return {'latch': dict()}


def capture(c, precision=None):
    from megakernel_lab.operation_compiler import capture_module, SGDConfig, DynamicBatch
    torch.set_num_threads(1)
    m = TwoLayer(c)
    x, y = fixture(c, 32, 0)   # historical capture example: capacity batch
    cap = capture_module(m, loss_fn, (x,), (y,), dynamic_batch=DynamicBatch(maximum=32, axis=0),
                         optimizer=SGDConfig(lr=LR, momentum=MOMENTUM, weight_decay=DECAY))
    params = {n: p.detach().clone() for n, p in m.named_parameters()}
    return cap, dict(parameters=params, optimizer={n: torch.zeros_like(p) for n, p in params.items()}), \
        dict(parameters=sum(p.numel() for p in params.values()), capacity=32)


def arms(c, prep):
    return list(CONVENTIONAL) + [v for v in NATIVE if prep.get('variants', {}).get(v, {}).get('ok')]


def audited_arms(c):
    return list(CONVENTIONAL) + list(NATIVE)


def arm_label(c, arm):
    return f'{arm} (static batch {c["batch"]}, explicit BF16 products)'


def make_model(c, arm):
    return TwoLayer(c)


def objective(model, batch, c):
    return loss_fn(model(batch[0]), batch[1])


def objective_functional(call, batch, c):
    return loss_fn(call(batch[0]), batch[1])


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
    return [tuple(t.to(device) for t in fixture(c, c['batch'], i)) for i in range(8)]


def engine(c, arm, d):
    if arm in NATIVE:
        return Native(d / arm, label=f'{arm} (DynamicBatch capacity 32, active {c["batch"]})')
    return Conventional(__import__(__name__, fromlist=['*']), c, arm, pool(c, 'cuda')[0])


def reference(c, s):
    m = TwoLayer(c)
    m.load_state_dict(s['parameters'])
    params = dict(m.named_parameters())
    opt = torch.optim.SGD(list(params.values()), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    for n, p in params.items():
        opt.state[p]['momentum_buffer'] = s['optimizer'][n].clone()
    x, y = pool(c, 'cpu')[1]
    loss = loss_fn(m(x), y)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in params.items()}
    opt.step()
    return dict(loss=loss.detach(), gradients=grads, before_parameters=s['parameters'],
                after_parameters={n: p.detach().clone() for n, p in params.items()})
