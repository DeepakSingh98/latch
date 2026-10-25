"""CPU reference checks (job script; writes a JSON receipt).

1. Our GPT-2 re-implementation vs Hugging Face transformers GPT2LMHeadModel (FP32 logits).
2. Theorem 1 on the real GPT-2 final MLP and real text (FP64): serial autograd learner vs the
   paper's parallel forward/transpose reconstruction fed the serial costates, plus the paper's
   slow-transpose control; and our GradLev *training* forward (quadratic causal scans) fed the
   same costates.
3. Deployment step formulas: all-FP64 step_math vs the FP64 autograd learner (formula check),
   and policy-P1 step_math (FP32 loop, BF16 head operands) vs the FP64 learner (policy deviation).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import sys

# LATCH-V0-NEXT: the GradLev helpers are the self-contained example package (examples/gradlev/gradlev_example,
# copied from research/latch_gradlev/gl at df184a5f with provenance headers), not the donor research tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples' / 'gradlev'))
from gradlev_example import model as gm  # noqa: E402
from gradlev_example import deploy as gd  # noqa: E402
from gradlev_example.assets import ensure  # noqa: E402
from gradlev_example.data import eval_blocks  # noqa: E402


def mx(a, b):
    return float((a.double() - b.double()).abs().max())


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--T', type=int, default=48)
    ap.add_argument('--T-policy', type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    r = {'torch': torch.__version__}
    t0 = time.time()
    p = gm.gpt2_params(ensure('gpt2.safetensors'))
    inputs, targets = eval_blocks('dev')
    tok, tgt = inputs[:2], targets[:2]

    # 1. HF parity
    from transformers import GPT2Config, GPT2LMHeadModel
    hf = GPT2LMHeadModel(GPT2Config()).eval()
    sd = gm.read_safetensors(ensure('gpt2.safetensors'))
    missing, unexpected = hf.transformer.load_state_dict(sd, strict=False)
    hf.tie_weights()
    with torch.no_grad():
        ref = hf(tok[:1, :256]).logits
        u, x = gm.trunk(p, tok[:1, :256])
        ours = gm.static_logits(p, u, x)
    r['hf_parity'] = {'max_abs_logit_diff': mx(ours, ref), 'logit_scale': float(ref.abs().max()),
                      'missing': [k for k in missing if not k.endswith('attn.bias') and 'masked_bias' not in k],
                      'unexpected': list(unexpected)}

    # 2. Theorem 1 (FP64, exact head values, random learned-gate parameters so eta varies per token)
    T = args.T
    p64 = {k: v.double() for k, v in p.items()}
    g = gm.gate_params()
    gen = torch.Generator().manual_seed(1729)
    g['gate_w'] = torch.randn(gm.D, generator=gen) * (0.5 / gm.D ** 0.5)
    with torch.no_grad():
        u64, x64 = gm.trunk(p64, tok[:1, :T])
    eta1, eta2 = gm.gates(g, x64)
    nll_s, finals, cs = gd.serial_reference(p64, g, u64, x64, eta1, eta2, tgt[:1, :T], head_bf16_values=False)
    gy_s = torch.stack([c[0] for c in cs[0]])
    gz_s = torch.stack([c[1] for c in cs[0]])
    nll_p, gy_p, gz_p, _ = gd.parallel_reconstruction(p64, u64[0], x64[0], eta1[0], eta2[0], tgt[0, :T], gy_s, gz_s,
                                                      head_bf16_values=False)
    _, gy_c, gz_c, _ = gd.parallel_reconstruction(p64, u64[0], x64[0], eta1[0], eta2[0], tgt[0, :T], gy_s, gz_s,
                                                  head_bf16_values=False, slow_transpose=True)
    q = gm.q_params()
    g64 = {k: v.double() for k, v in g.items()}
    q64 = {k: v.double() for k, v in q.items()}
    tot, ce, cons, diag = gm.gradlev_loss(p64, g64, q64, tok[:1, :T], tgt[:1, :T], proposals=(gz_s[None], gy_s[None]))
    r['theorem1_fp64'] = {
        'T': T, 'eta1_range': [float(eta1.min()), float(eta1.max())],
        'serial_nll_mean': float(nll_s.mean()),
        'outputs_max_abs': mx(nll_p, nll_s[0]), 'gy_targets_max_abs': mx(gy_p, gy_s),
        'gz_targets_max_abs': mx(gz_p, gz_s), 'gz_scale': float(gz_s.abs().max()),
        'slow_transpose_control': {'outputs_max_abs': mx(nll_p, nll_s[0]), 'gz_targets_max_abs': mx(gz_c, gz_s),
                                   'gz_rel': rel(gz_c, gz_s)},
        'training_forward_scan': {'ce_max_abs': mx(diag['ce_tokens'][0], nll_s[0]),
                                  'gy_max_abs': mx(diag['gy_t'][0], gy_s), 'gz_max_abs': mx(diag['gz_t'][0], gz_s),
                                  'consistency_at_exact_proposals': float(cons)}}

    # 3a. explicit deployment formulas, all FP64 (E = exact FP64 values)
    c64 = {'b1': p64[f'{gm.L-1}.fc_b'], 'b2': p64[f'{gm.L-1}.pr_b'], 'lnf_w': p64['lnf_w'], 'lnf_b': p64['lnf_b'],
           'E': p64['wte']}
    st = {'W1': p64[f'{gm.L-1}.fc_w'][None].clone(), 'W2': p64[f'{gm.L-1}.pr_w'][None].clone()}
    nll_f = gd.adapt_eager(st, c64, u64, x64, eta1.double(), eta2.double(), tgt[:1, :T])
    r['step_formulas_fp64'] = {'nll_max_abs': mx(nll_f[0], nll_s[0]), 'W1_rel': rel(st['W1'][0], finals[0][0]),
                               'W2_rel': rel(st['W2'][0], finals[0][1]),
                               'W1_change_rel': rel(finals[0][0], p64[f'{gm.L-1}.fc_w'])}

    # 3b. head-precision policies vs the FP64 learner on a longer real prefix (FP32 trunk outputs as inputs)
    Tp = args.T_policy
    with torch.no_grad():
        u32, x32 = gm.trunk(p, tok[:1, :Tp])
    g0 = gm.gate_params()
    e1, e2 = gm.gates(g0, x32)
    refs = {}
    for bf in (True, False):
        nll_ref, fin_ref, _ = gd.serial_reference(p, g0, u32, x32, e1, e2, tgt[:1, :Tp], head_bf16_values=bf)
        refs[bf] = (nll_ref, fin_ref)
    r['mm_out_dtype_overload'] = hasattr(torch.ops.aten.mm, 'dtype')
    pol = {}
    for name, mode, e_bf16 in (('bf16_out', 'bf16', True), ('mixed_bf16_operands_fp32_out', 'mixed', True),
                               ('P1b_mixed_logits_fp32_gl', 'mixed_gl32', True),
                               ('fp32_act_bf16_E', 'fp32', True), ('fp32_exact_E', 'fp32', False)):
        c32 = gd.consts_from(p, 'cpu')
        if not e_bf16:
            c32['E'] = p['wte'].float()
        st32 = gd.initial_state(p, 'cpu')
        nll_32 = gd.adapt_eager(st32, c32, u32, x32, e1, e2, tgt[:1, :Tp], head=mode)
        nll_ref, fin_ref = refs[e_bf16]
        with torch.no_grad():
            W1, W2 = p[f'{gm.L-1}.fc_w'], p[f'{gm.L-1}.pr_w']
            y = gm.gelu_tanh(x32 @ W1 + c32['b1']) @ W2 + c32['b2']
            n, _, _ = gd.layer_norm_parts(u32 + y, c32['lnf_w'], c32['lnf_b'])
            lg = gd.head_logits(n[0], c32['E'], mode)
            frozen = torch.logsumexp(lg, -1) - lg.gather(-1, tgt[0, :Tp, None])[..., 0]
        pol[name] = {'nll_max_abs': mx(nll_32[0], nll_ref[0]),
                     'nll_mean_abs': float((nll_32[0].double() - nll_ref[0]).abs().mean()),
                     'mean_nll': float(nll_32.mean()), 'mean_nll_fp64_ref': float(nll_ref.mean()),
                     'mean_nll_frozen': float(frozen.mean()),
                     'W1_rel': rel(st32['W1'][0], fin_ref[0][0]), 'W2_rel': rel(st32['W2'][0], fin_ref[0][1]),
                     'W1_change_rel_ref': rel(fin_ref[0][0], p[f'{gm.L-1}.fc_w'].double())}
    r['head_policies_vs_fp64'] = {'T': Tp, 'policies': pol}
    r['seconds'] = time.time() - t0
    (out / 'reference.json').write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
