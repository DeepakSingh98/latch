"""Historical ordinary byte transformer (P6-D/P9-A 'o-l-bf16', ~101.7M) through the released compiler.

User model: recovered research/latch_ops/examples/transformer.py (sha ef4763a3 original; import line mapped to
the installed package). Explicit BF16 projection operands/results, FP32 embeddings/residual/RMSNorm/attention
bmm+softmax/loss; momentum SGD lr .01, momentum .9, coupled decay 1e-4 on every parameter; seed 20261002.
Native: public capture_training + SGDConfig, then lower(ir, policy='auto', precision_policy=...) and runtime.build
(the lower/build/Runtime path of the release's shipped runtime-loop example). The normal API now automatically selects a compatible implementation; explicit legacy is diagnostic.
Data: deterministic random byte batches (historical timing used WikiText-2 bytes; this dense computation has no
data-dependent control flow, so timing is data independent; numerical states differ from history).
Controls (historical P6-D/P8-A/P9-A definitions, reimplemented from vendored scale_amp.py/p8a_controls.py):
  compiled       matched arithmetic: same BF16-projection loss, torch.compile(fullgraph, max_autotune +
                 coordinate_descent_tuning, no inductor cudagraphs) + compiled optimizer.step, whole-step CUDA graph
  compiled-sdpa32 same but attention through F.scaled_dot_product_attention(is_causal) in FP32 (same mathematics)
  amp-ma-cdt     historical practical denominator: FP32 model under torch.autocast(bf16) with causal SDPA, same
                 compile options and SGD; historically fails same-state Q/K gates, admitted by a 1200-update
                 learning check
"""
from dataclasses import replace
import math
import torch
import torch.nn.functional as Fn
from megakernel_lab.fast_examples.engines import Engine, NativeRuntime, write_program, _cp
from megakernel_lab.fast_examples.recovered import transformer as T

PRIMARY = T.TransformerConfig(batch=4, sequence=512, width=1024, heads=16, layers=8, matrix_dtype='bfloat16')
CASES = {
    'o-l-bf16': dict(config=PRIMARY, historical='4.793078 native practical vs 4.946915 amp_ma_cdt (0.969x), '
                     'IP9 AB_RESULTS.md; P9-A 4.808 vs 4.961'),
    'o-m-bf16': dict(config=replace(PRIMARY, width=512, heads=8, layers=9),
                     historical='nearby transfer: 3.224 native vs 4.455 matched / 2.520 AMP (AMP failed gates)'),
}
LR, MOMENTUM, DECAY = 0.01, 0.9, 1e-4
REFERENCE_ARM = 'eager'
PREPARE_TIMEOUT, RUN_TIMEOUT, AUDIT_TIMEOUT = 5400, 3600, 5400
NATIVE = ('latch-legacy', 'latch', 'latch-strict', 'latch-consumed')
CONVENTIONAL = ('eager', 'compiled', 'compiled-sdpa32', 'amp-ma-cdt')
OPTIONS = {'triton.cudagraphs': False, 'compile_threads': 1, 'max_autotune': True, 'coordinate_descent_tuning': True}


def native_variants(c):
    return {'latch-legacy': dict(policy='auto', precision_policy='practical', execution='legacy'),
            'latch': dict(policy='auto', precision_policy='practical'),
            'latch-strict': dict(policy='auto', precision_policy='strict'),
            # LATCH-STEP-15 selectable path: optimizer consumes gradients, no public gradient outputs
            'latch-consumed': dict(policy='auto', precision_policy='practical', gradient_outputs='consumed')}


def capture(c, precision=None):
    from megakernel_lab.operation_compiler import capture_training, SGDConfig
    torch.set_num_threads(1)
    cfg = c['config']
    params = T.parameters(cfg)
    momentum = {k: torch.zeros_like(v) for k, v in params.items()}
    captured = capture_training(T.make_loss(cfg), params, momentum, pool(c, 'cpu')[0],
                                SGDConfig(lr=LR, momentum=MOMENTUM, weight_decay=DECAY), precision=precision)
    initial = dict(parameters={k: v.clone() for k, v in params.items()},
                   optimizer={k: v.clone() for k, v in momentum.items()})
    return captured, initial, dict(parameters=sum(v.numel() for v in params.values()),
                                   operations=len(captured.ir.operations))


def build(captured, target, cache, kwargs):
    import time
    from megakernel_lab.operation_compiler.lower import lower
    t = time.perf_counter()
    program = lower(captured.ir, **kwargs)
    return write_program(program, captured, target, cache, extra=dict(lower_kwargs=kwargs,
                                                                       lower_seconds=time.perf_counter() - t))


def arms(c, prep):
    return list(CONVENTIONAL) + [v for v in NATIVE if prep.get('variants', {}).get(v, {}).get('ok')]


def audited_arms(c):
    return list(CONVENTIONAL) + list(NATIVE)


def pool(c, device, engine=None):
    cfg = c['config']
    result = []
    for i in range(8):
        g = torch.Generator().manual_seed(cfg.seed + 101 + i)
        tokens = torch.randint(0, cfg.vocab, (cfg.batch, cfg.sequence), generator=g)
        targets = torch.randint(0, cfg.vocab, (cfg.batch, cfg.sequence), generator=g)
        positions, mask = T.static_inputs(cfg)
        result.append(tuple(v.to(device) for v in (tokens, targets, positions, mask)))
    return result


def sdpa_block(params, hidden, cfg, layer, autocast):
    """Vendored scale_amp.sdpa_block; with autocast=False projections use the explicit BF16 linear()."""
    b, t, d, h = cfg.batch, cfg.sequence, cfg.width, cfg.heads
    hd = d // h
    prefix = f'block{layer}.'
    normalized = T.rms_norm(hidden, params[prefix + 'attention_norm'], cfg.epsilon).reshape(b * t, d)

    def mm(x, w):
        return x @ w if autocast else T.linear(x, w, cfg)

    def project(name):
        return mm(normalized, params[prefix + name]).reshape(b, t, h, hd).transpose(1, 2)

    attended = Fn.scaled_dot_product_attention(project('query'), project('key'), project('value'), is_causal=True)
    attended = attended.transpose(1, 2).reshape(b * t, d)
    hidden = hidden + mm(attended, params[prefix + 'attention_out']).reshape(b, t, d)
    normalized = T.rms_norm(hidden, params[prefix + 'feedforward_norm'], cfg.epsilon).reshape(b * t, d)
    wide = Fn.silu(mm(normalized, params[prefix + 'up']))
    return hidden + mm(wide, params[prefix + 'down']).reshape(b, t, d)


def loss_for(arm, cfg):
    if arm in ('eager', 'compiled'):
        return T.make_loss(cfg)
    autocast = arm == 'amp-ma-cdt'
    model = replace(cfg, matrix_dtype='float32') if autocast else cfg

    def loss(params, tokens, targets, positions, mask):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            hidden = T.embed(params, tokens, positions)
            for layer in range(model.layers):
                hidden = sdpa_block(params, hidden, model, layer, autocast)
            return T.cross_entropy(params, hidden, targets, model).float()
    return loss


class TorchSGD(Engine):
    def __init__(self, c, arm):
        cfg = c['config']
        self.label = arm
        initial = T.parameters(cfg, device='cuda')
        self.params = {k: v.requires_grad_(True) for k, v in initial.items()}
        self.opt_state = {k: torch.zeros_like(v) for k, v in self.params.items()}
        for p in self.params.values():
            p.grad = torch.zeros_like(p)
        self.opt = torch.optim.SGD(list(self.params.values()), lr=LR, momentum=MOMENTUM, weight_decay=DECAY,
                                   foreach=True)
        for k, p in self.params.items():
            self.opt.state[p]['momentum_buffer'] = self.opt_state[k]
        self.batch = tuple(v.clone() for v in pool(c, 'cuda')[0])
        loss_fn = loss_for(arm, cfg)
        self.grads = {k: p.grad for k, p in self.params.items()}
        if arm == 'eager':
            self.loss_fn, self.opt_step = loss_fn, self.opt.step
            self.loss = torch.zeros((), device='cuda')
            return
        self.loss_fn = torch.compile(loss_fn, backend='inductor', fullgraph=True, options=OPTIONS)
        self.opt_step = torch.compile(self.opt.step, backend='inductor', fullgraph=False, options=OPTIONS)
        self.compiled = True
        snapshot = self.state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.loss = self._step()
        torch.cuda.synchronize()
        self.load(snapshot)

    def _step(self):
        self.opt.zero_grad(set_to_none=False)
        loss = self.loss_fn(self.params, *self.batch)
        loss.backward()
        self.opt_step()
        return loss.detach()

    def stage(self, batch):
        for dst, src in zip(self.batch, batch):
            dst.copy_(src)

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.loss = self._step()

    def state(self):
        return dict(parameters=_cp(self.params), optimizer=_cp(self.opt_state))

    def load(self, s):
        with torch.no_grad():
            for k, p in self.params.items():
                p.copy_(s['parameters'][k])
            for k, v in self.opt_state.items():
                v.copy_(s['optimizer'][k])


def engine(c, arm, d):
    if arm in NATIVE:
        return NativeRuntime(d / arm, label=arm, n_inputs=4)
    return TorchSGD(c, arm)


def reference(c, s):
    """Independent CPU eager autograd + torch.optim.SGD at the saved state on pool batch 1 (same BF16 recipe)."""
    cfg = c['config']
    params = {k: v.clone().requires_grad_(True) for k, v in s['parameters'].items()}
    opt = torch.optim.SGD(list(params.values()), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    for k, p in params.items():
        opt.state[p]['momentum_buffer'] = s['optimizer'][k].clone()
    loss = T.make_loss(cfg)(params, *pool(c, 'cpu')[1])
    loss.backward()
    grads = {k: p.grad.detach().clone() for k, p in params.items()}
    opt.step()
    return dict(loss=loss.detach(), gradients=grads, before_parameters=s['parameters'],
                after_parameters={k: p.detach().clone() for k, p in params.items()})
