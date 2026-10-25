"""LATCH-FFN-RETAIN extracted FFN sublayer programs through public capture_module(...).build().

User model: recovered_release/ffn/{extract,model,reference}.py from research/latch_release/ffn @5e592f9309 (only the
`research.latch_release.ffn` import prefix remapped; original hashes in recovered_release/ffn/ORIGINAL.sha256).
x + down(act(up(LayerNorm(x)))), explicit BF16 projection operands/bias with FP32 promotion, FP32 LayerNorm/residual,
MSE to a fixed target; AdamW lr 1e-3, betas .9/.999, eps 1e-8, wd .01 after global-norm clip 1.0.
Programs: ffn (B32 T512 D640 ff2560 ReLU), ffn-d768 (B16 D768 ff3072), ffn-gelu (B32 D512 ff2048 GELU-tanh).
Historical (DECLARED.json score: fastest of eager / compiled max-autotune + graph / same under BF16 autocast):
ffn 0.993 vs 1.070 ms (0.928); i4 0.900 fresh; d768 0.974 (i4 0.951); gelu 1.122 (i4 1.033).
"""
import math
import torch
import torch.nn.functional as Fn
from megakernel_lab.fast_examples.engines import Conventional, Native
from megakernel_lab.fast_examples.recovered_release.ffn import extract as X

CASES = {p: dict(program=p) for p in ('ffn', 'ffn-d768', 'ffn-gelu')}
LR, BETAS, EPS, DECAY, CLIP = 1e-3, (0.9, 0.999), 1e-8, 0.01, 1.0
REFERENCE_ARM = 'eager'
PREPARE_TIMEOUT, RUN_TIMEOUT, AUDIT_TIMEOUT = 3600, 3600, 3600
NATIVE = ('latch', 'latch-legacy')
CONVENTIONAL = ('eager', 'compiled', 'compiled-amp')


def native_variants(c):
    # Explicit legacy is a diagnostic; the normal API selects the compatible retained schedule.
    return {'latch': dict(), 'latch-legacy': dict(execution='legacy')}


def capture(c, precision=None):
    from megakernel_lab.operation_compiler import capture_module
    torch.set_num_threads(1)
    m = X.Program(c['program'])
    x, target = X.inputs(program=c['program'])
    cap = capture_module(m, X.mse, (x,), (target,), optimizer=X.adamw(c['program']))
    params = {n: p.detach().clone() for n, p in m.named_parameters()}
    return cap, dict(parameters=params, optimizer=initial_state(c, params)), \
        dict(parameters=sum(p.numel() for p in params.values()))


def arms(c, prep):
    return list(CONVENTIONAL) + [v for v in NATIVE if prep.get('variants', {}).get(v, {}).get('ok')]


def audited_arms(c):
    return list(CONVENTIONAL) + list(NATIVE)


def arm_label(c, arm):
    return {'compiled-amp': 'compiled whole update under torch.autocast(bf16), FP32 masters',
            'compiled': 'compiled explicit-BF16-projection recipe'}.get(arm, arm)


class AMP(torch.nn.Module):
    """Same parameters; plain F.linear under BF16 autocast (historical practical AMP control)."""
    def __init__(self, program):
        super().__init__()
        self.inner = X.Program(program)

    def forward(self, x):
        b = self.inner.block
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = Fn.layer_norm(x, b.ff_norm.normalized_shape, b.ff_norm.weight, b.ff_norm.bias, b.ff_norm.eps)
            u = Fn.linear(h, b.ff.up.weight, b.ff.up.bias)
            u = Fn.gelu(u, approximate='tanh') if self.inner.activation == 'gelu_tanh' else Fn.relu(u)
            y = Fn.linear(u, b.ff.down.weight, b.ff.down.bias)
        return x + y.float()


def make_model(c, arm):
    return AMP(c['program']) if arm == 'compiled-amp' else X.Program(c['program'])


def _canon(name):
    return name[len('inner.'):] if name.startswith('inner.') else name


def objective(model, batch, c):
    return X.mse(model(batch[0]), batch[1])


def objective_functional(call, batch, c):
    return X.mse(call(batch[0]), batch[1])


def initial_state(c, params):
    s = {}
    for n, p in params.items():
        n = _canon(n)
        s[f'exp_avg:{n}'] = torch.zeros_like(p)
        s[f'exp_avg_sq:{n}'] = torch.zeros_like(p)
        s[f'step:{n}'] = torch.zeros((), dtype=torch.float32, device=p.device)
    return s


def make_optimizer(c, params, arm):
    return torch.optim.AdamW(params, lr=LR, betas=BETAS, eps=EPS, weight_decay=DECAY, fused=True)


def bind_state(c, opt, params, state):
    for n, p in params.items():
        n = _canon(n)
        opt.state[p]['exp_avg'] = state[f'exp_avg:{n}']
        opt.state[p]['exp_avg_sq'] = state[f'exp_avg_sq:{n}']
        opt.state[p]['step'] = state[f'step:{n}']


def pre_step(c, params):
    torch.nn.utils.clip_grad_norm_(params, CLIP)


def functional_update(c, params, grads, state):
    total = torch.sqrt(sum((g.float() ** 2).sum() for g in grads.values()))
    scale = torch.clamp(CLIP / (total + 1e-6), max=1.0)
    for n, p in params.items():
        k = _canon(n)
        g = grads[n] * scale
        step = state[f'step:{k}'] + 1
        m = BETAS[0] * state[f'exp_avg:{k}'] + (1 - BETAS[0]) * g
        v = BETAS[1] * state[f'exp_avg_sq:{k}'] + (1 - BETAS[1]) * g * g
        state[f'step:{k}'].copy_(step)
        state[f'exp_avg:{k}'].copy_(m)
        state[f'exp_avg_sq:{k}'].copy_(v)
        denom = (v / (1 - BETAS[1] ** step)).sqrt() + EPS
        p.copy_(p * (1 - LR * DECAY) - LR * (m / (1 - BETAS[0] ** step)) / denom)


def pool(c, device, engine=None):
    s = X.shape(c['program'])
    result = []
    for i in range(8):
        g = torch.Generator().manual_seed(5 + i)          # seed 5 = the historical capture example
        x = torch.randn(s['batch'], s['seq'], s['width'], generator=g)
        t = torch.randn(s['batch'], s['seq'], s['width'], generator=g)
        result.append((x.to(device), t.to(device)))
    return result


def engine(c, arm, d):
    if arm in NATIVE:
        return Native(d / arm, label=arm)
    e = Conventional(__import__(__name__, fromlist=['*']), c, arm, pool(c, 'cuda')[0])
    if arm == 'compiled-amp':      # state/grad names follow the plain program's parameter names
        e.params = {_canon(n): p for n, p in e.params.items()}
        e.grads = {_canon(n): g for n, g in e.grads.items()}
    return e


def reference(c, s):
    m = X.Program(c['program'])
    m.load_state_dict(s['parameters'], strict=False)
    params = dict(m.named_parameters())
    opt = torch.optim.AdamW(list(params.values()), lr=LR, betas=BETAS, eps=EPS, weight_decay=DECAY)
    for n, p in params.items():
        opt.state[p]['exp_avg'] = s['optimizer'][f'exp_avg:{n}'].clone()
        opt.state[p]['exp_avg_sq'] = s['optimizer'][f'exp_avg_sq:{n}'].clone()
        opt.state[p]['step'] = s['optimizer'][f'step:{n}'].float().clone()
    x, t = pool(c, 'cpu')[1]
    loss = X.mse(m(x), t)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in params.items()}
    torch.nn.utils.clip_grad_norm_(list(params.values()), CLIP)
    opt.step()
    return dict(loss=loss.detach(), gradients=grads, before_parameters=s['parameters'],
                after_parameters={n: p.detach().clone() for n, p in params.items()})
