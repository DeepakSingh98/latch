"""P-OPS4-D fixed long-context attention region (adapted from P-OPS3-B pops3b_region.py @ e68b4986b).

The region is the ordinary model code of examples/transformer.py `block` between the BF16 query/key/value
projection products and the BF16 cast that feeds the output projection (strict-BF16 matrix recipe), plus the
reverse-mode derivative PyTorch derives for it:

    inputs   qb, kb, vb : BF16 [B*T, D]   (the projection products)
             mask       : FP32 [T, T]     (the model's explicit additive causal mask input)
             g          : BF16 [B*T, D]   (derivative of the BF16 attended activation)
    outputs  out        : BF16 [B*T, D]   (attended.to(bf16), the output projection's operand)
             dqb,dkb,dvb: BF16 [B*T, D]   (derivatives of the projection products)

BF16 autocast produces the same boundary values in the practical control. Every layout copy, dtype conversion,
saved forward state and scratch between those boundaries is part of the region. See pops4_d/SELECTION.json.
"""
from __future__ import annotations

import math

import torch

SHAPES = {
    # name: (batch, sequence, heads, head width)
    'c2048': (4, 2048, 4, 64),        # frozen headline (integration_context2048 attention)
    'c1024': (4, 1024, 4, 64),        # frozen check: shorter context
    'c4096': (2, 4096, 4, 64),        # frozen check: longer context, same tokens
}


def config(name):
    b, t, h, hd = SHAPES[name]
    return dict(batch=b, sequence=t, heads=h, head_width=hd, width=h * hd)


def causal_mask(t, dtype=torch.float32, device='cpu'):
    # Identical to examples.transformer.static_inputs.
    return torch.triu(torch.full((t, t), -10000.0, dtype=dtype, device=device), diagonal=1)


def attention_region(qb, kb, vb, mask, cfg):
    """Model code (examples/transformer.py block, strict-BF16 recipe) from projection products to the output cast."""
    b, t, h, hd, d = cfg['batch'], cfg['sequence'], cfg['heads'], cfg['head_width'], cfg['width']

    def heads(x):
        # linear() returns (...).float(); block's project() then reshapes/permutes to [B*H, T, HD].
        return x.float().reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)

    query, key, value = heads(qb), heads(kb), heads(vb)
    scores = torch.bmm(query, key.transpose(1, 2)) * (1.0 / math.sqrt(hd)) + mask
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.bmm(probabilities, value).reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d)
    # linear(attended, W_out) casts its activation operand to BF16.
    return attended.to(torch.bfloat16)


def autocast_region(qb, kb, vb, mask, cfg):
    """The same model code as the practical BF16 autocast control executes it (projection outputs already BF16)."""
    b, t, h, hd, d = cfg['batch'], cfg['sequence'], cfg['heads'], cfg['head_width'], cfg['width']
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, cache_enabled=False):
        def heads(x):
            return x.reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)
        query, key, value = heads(qb), heads(kb), heads(vb)
        scores = torch.bmm(query, key.transpose(1, 2)) * (1.0 / math.sqrt(hd)) + mask
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.bmm(probabilities, value).reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d)
    return attended.to(torch.bfloat16)


def region_step(cfg):
    def step(qb, kb, vb, mask, g):
        out, pullback = torch.func.vjp(lambda a, b, c: attention_region(a, b, c, mask, cfg), qb, kb, vb)
        dq, dk, dv = pullback(g)
        return out, dq, dk, dv
    return step


def example_inputs(cfg, *, seed=0, device='cpu', scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    n, d = cfg['batch'] * cfg['sequence'], cfg['width']
    q, k, v = ((torch.randn(n, d, generator=gen) * scale).to(torch.bfloat16) for _ in range(3))
    g = (torch.randn(n, d, generator=gen) * 1e-3).to(torch.bfloat16)
    return tuple(x.to(device) for x in (q, k, v, causal_mask(cfg['sequence']), g))


def capture_graph(cfg):
    """make_fx graph of the region (same decompositions as capture_training) as a lower()-ready dict."""
    from torch.fx.experimental.proxy_tensor import make_fx
    from megakernel_lab.operation_compiler.capture import capture_decompositions
    args = example_inputs(cfg)
    gm = make_fx(region_step(cfg), decomposition_table=capture_decompositions(), tracing_mode='fake',
                 _allow_non_fake_inputs=True)(*args)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return fx_to_graph(gm), gm


def fx_to_graph(gm):
    values, operations, inputs, outputs = {}, [], [], []

    def encode(obj):
        if isinstance(obj, torch.fx.Node):
            return {'ref': obj.name}
        if isinstance(obj, (tuple, list)):
            return [encode(x) for x in obj]
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return str(obj).removeprefix('torch.')
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        raise TypeError(f'unsupported literal {type(obj).__name__}')

    for node in gm.graph.nodes:
        if node.op == 'output':
            outputs = [x.name for x in node.args[0]]
            continue
        val = node.meta['val']
        values[node.name] = {'shape': [int(x) for x in val.shape], 'strides': [int(x) for x in val.stride()],
                             'dtype': str(val.dtype).removeprefix('torch.'),
                             'role': 'input' if node.op == 'placeholder' else 'temporary'}
        if node.op == 'placeholder':
            inputs.append(node.name)
        elif node.op == 'call_function':
            operations.append({'name': node.name, 'target': str(node.target),
                               'args': encode(node.args), 'kwargs': encode(node.kwargs)})
        else:
            raise ValueError('unsupported node ' + node.op)
    return {'inputs': inputs, 'outputs': outputs, 'values': values, 'operations': operations}


# ------------------------------------------------------------------------------------------- conventional arms
def _sdpa_region(backend, causal, mask_dtype=None, contiguous=False):
    """BF16 SDPA attention with the region's boundaries (views of the BF16 projection products, BF16 output)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = {'flash': SDPBackend.FLASH_ATTENTION, 'efficient': SDPBackend.EFFICIENT_ATTENTION,
                'cudnn': SDPBackend.CUDNN_ATTENTION, 'math': SDPBackend.MATH}
    chosen = None if backend == 'auto' else [backends[backend]]
    converted = {}

    def mask_operand(mask):
        # The model's mask is a constant input: eager arms convert it once (first use), as a user would keep a
        # BF16 copy; compiled arms convert inside their graph.
        if mask_dtype is None or mask.dtype == mask_dtype:
            return mask
        if torch.compiler.is_compiling():
            return mask.to(mask_dtype)
        key = (mask.data_ptr(), mask._version, tuple(mask.shape))
        if key not in converted:
            converted.clear()
            converted[key] = mask.to(mask_dtype)
        return converted[key]

    def fn(qb, kb, vb, mask, cfg):
        b, t, h, hd, d = cfg['batch'], cfg['sequence'], cfg['heads'], cfg['head_width'], cfg['width']
        q, k, v = (x.view(b, t, h, hd).transpose(1, 2) for x in (qb, kb, vb))
        if contiguous:
            q, k, v = (x.contiguous() for x in (q, k, v))
        kwargs = dict(scale=1.0 / math.sqrt(hd))
        if causal:
            kwargs['is_causal'] = True
        else:
            kwargs['attn_mask'] = mask_operand(mask)
        if chosen is None:
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)
        else:
            with sdpa_kernel(chosen):
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)
        return o.transpose(1, 2).reshape(b * t, d)
    return fn


def _flex_region(dtype):
    """FlexAttention as normally used: block mask built once (eager), flex_attention compiled by Inductor."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    compiled = torch.compile(flex_attention, fullgraph=True)
    cache = {}

    def fn(qb, kb, vb, mask, cfg):
        b, t, h, hd, d = cfg['batch'], cfg['sequence'], cfg['heads'], cfg['head_width'], cfg['width']
        q, k, v = (x.to(dtype).view(b, t, h, hd).transpose(1, 2) for x in (qb, kb, vb))
        if t not in cache:
            cache[t] = create_block_mask(lambda bb, hh, i, j: i >= j, None, None, t, t, device=qb.device)
        o = compiled(q, k, v, block_mask=cache[t], scale=1.0 / math.sqrt(hd))
        return o.transpose(1, 2).reshape(b * t, d).to(torch.bfloat16)
    return fn


MA = {'max_autotune': True}
MA_CDT_COMBO = {'max_autotune': True, 'coordinate_descent_tuning': True, 'combo_kernels': True}
ARMS = {
    # name: (attention callable factory, compile options or None, numerical class)
    'eager_fp32': (lambda: attention_region, None, 'strict FP32 (model code)'),
    'inductor_fp32_ma': (lambda: attention_region, MA, 'strict FP32 compiled'),
    'sdpa_flash_causal': (lambda: _sdpa_region('flash', True), None, 'BF16 SDPA flash, is_causal'),
    'sdpa_efficient_causal': (lambda: _sdpa_region('efficient', True), None, 'BF16 SDPA mem-efficient, is_causal'),
    'sdpa_cudnn_causal': (lambda: _sdpa_region('cudnn', True), None, 'BF16 SDPA cuDNN, is_causal'),
    'sdpa_cudnn_causal_contig': (lambda: _sdpa_region('cudnn', True, contiguous=True), None,
                                 'BF16 SDPA cuDNN, is_causal, head-major contiguous copies'),
    'sdpa_efficient_mask': (lambda: _sdpa_region('efficient', False, torch.bfloat16), None,
                            'BF16 SDPA mem-efficient, additive BF16 mask'),
    'sdpa_cudnn_mask': (lambda: _sdpa_region('cudnn', False, torch.bfloat16), None, 'BF16 SDPA cuDNN, additive BF16 mask'),
    'inductor_sdpa_causal_ma': (lambda: _sdpa_region('auto', True), MA, 'BF16 SDPA (auto backend) compiled, is_causal'),
    'inductor_sdpa_mask_ma': (lambda: _sdpa_region('auto', False, torch.bfloat16), MA,
                              'BF16 SDPA (auto backend) compiled, additive mask'),
    'autocast_eager': (lambda: autocast_region, None, 'BF16 autocast model code (practical eager control)'),
    'autocast_inductor_ma': (lambda: autocast_region, MA, 'BF16 autocast model code, Inductor max-autotune'),
    'autocast_inductor_ma_cdt_combo': (lambda: autocast_region, MA_CDT_COMBO,
                                       'BF16 autocast model code, practical winning compile options'),
    'flex_bf16': (lambda: _flex_region(torch.bfloat16), None, 'BF16 FlexAttention, causal block mask'),
}
CAUSAL_ARMS = {'sdpa_flash_causal', 'sdpa_efficient_causal', 'sdpa_cudnn_causal', 'sdpa_cudnn_causal_contig',
               'inductor_sdpa_causal_ma', 'flex_bf16'}


def _compile(fn, options):
    from torch._inductor import config as inductor_config
    opts = {'triton.cudagraphs': False, 'compile_threads': 1, 'emulate_precision_casts': True}
    opts.update({k: v for k, v in options.items() if k != 'coordinate_descent_tuning'})
    if options.get('coordinate_descent_tuning'):
        inductor_config.coordinate_descent_tuning = True
    return torch.compile(fn, backend='inductor', fullgraph=True, options=opts)


def make_arm(name, cfg):
    """Returns step(qb, kb, vb, mask, g) -> (out, dq, dk, dv) using ordinary autograd (compiled where listed)."""
    factory, options, _ = ARMS[name]
    fn = factory()
    if options is not None:
        fn = _compile(fn, options)

    def step(qb, kb, vb, mask, g):
        out = fn(qb, kb, vb, mask, cfg)
        dq, dk, dv = torch.autograd.grad(out, (qb, kb, vb), g)
        return out, dq, dk, dv
    return step


def attention_callable(name, *, compiled=True):
    """The arm's attention arithmetic as fn(qb, kb, vb, mask, cfg) -> BF16 [B*T, D]."""
    factory, options, _ = ARMS[name]
    fn = factory()
    if compiled and options is not None:
        fn = _compile(fn, options)
    return fn


def reference_fp64(qb, kb, vb, mask, g, cfg):
    """FP64 evaluation of the region semantics at the same BF16-exact inputs (no output rounding)."""
    b, t, h, hd, d = cfg['batch'], cfg['sequence'], cfg['heads'], cfg['head_width'], cfg['width']

    def heads(x):
        return x.double().reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)

    q, k, v = (heads(x).requires_grad_(True) for x in (qb, kb, vb))
    scores = torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(hd)) + mask.double()
    p = torch.softmax(scores, dim=-1)
    o = torch.bmm(p, v)
    go = g.double().reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)
    dq, dk, dv = torch.autograd.grad(o, (q, k, v), go)

    def flat(x):
        return x.reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d).detach()
    return flat(o), flat(dq), flat(dk), flat(dv)


def error_report(outputs, reference):
    """Per-output max abs, max relative (to output max), RMS error vs FP64; elementwise BF16-ulp counts."""
    rows = {}
    for label, actual, expected in zip(('out', 'dq', 'dk', 'dv'), outputs, reference):
        a, e = actual.double(), expected.double()
        err = (a - e).abs()
        scale = e.abs().max().item() or 1.0
        rounded = expected.to(torch.bfloat16).double()
        rows[label] = {'max_abs': err.max().item(), 'rms': err.pow(2).mean().sqrt().item(),
                       'max_rel_to_max': err.max().item() / scale, 'ref_max_abs': scale,
                       'ref_rms': e.pow(2).mean().sqrt().item(),
                       'rel_rms': err.pow(2).mean().sqrt().item() / (e.pow(2).mean().sqrt().item() or 1.0),
                       'bf16_mismatch_vs_rounded_ref': int((a != rounded).sum().item()),
                       'elements': a.numel(), 'finite': bool(torch.isfinite(a).all())}
    return rows


# ------------------------------------------------------------------------------- model substitution (validity)
class patched_model:
    """Temporarily replace examples.transformer.block by the same block with substituted attention arithmetic.

    Everything outside the region is the ordinary strict-BF16-recipe block code; with attention_region it is the
    identical model. recorder(layer, qb, kb, vb, out) observes the region boundary (diagnostics only).
    """

    def __init__(self, attention, recorder=None):
        self.attention, self.recorder = attention, recorder

    def __enter__(self):
        import torch.nn.functional as F
        import megakernel_lab.fast_examples.recovered.transformer as T
        attention, recorder = self.attention, self.recorder

        def block(params, hidden, mask, config, layer):
            if config.matrix_dtype != 'bfloat16':
                raise ValueError('attention substitution is defined for the strict-BF16 recipe only')
            b, t, d, h = config.batch, config.sequence, config.width, config.heads
            hd = d // h
            prefix = f'block{layer}.'
            normalized = T.rms_norm(hidden, params[prefix + 'attention_norm'], config.epsilon)

            def product(name):
                # linear() without its trailing .float(): the BF16 projection product (region input).
                return normalized.reshape(b * t, d).to(torch.bfloat16) @ params[prefix + name].to(torch.bfloat16)

            qb, kb, vb = product('query'), product('key'), product('value')
            cfg = dict(batch=b, sequence=t, heads=h, head_width=hd, width=d)
            attended = attention(qb, kb, vb, mask, cfg)
            if recorder is not None:
                recorder(layer, qb, kb, vb, attended)
            # linear(attended_fp32, W) == (attended_fp32.to(bf16) @ W.to(bf16)).float(); the cast is in the region.
            hidden = hidden + (attended @ params[prefix + 'attention_out'].to(torch.bfloat16)).float().reshape(b, t, d)
            normalized = T.rms_norm(hidden, params[prefix + 'feedforward_norm'], config.epsilon)
            wide = F.silu(T.linear(normalized.reshape(b * t, d), params[prefix + 'up'], config))
            return hidden + T.linear(wide, params[prefix + 'down'], config).reshape(b, t, d)

        self.module, self.original = T, T.block
        T.block = block
        return self

    def __exit__(self, *exc):
        self.module.block = self.original
        return False
