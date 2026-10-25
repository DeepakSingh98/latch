# FGM-A source contract and task adaptation

Inspected on 2026-10-06. This is an adaptation to our text-training task, **not a
reproduction** of [Scaling Laws for Fine-Grained Mixture of Experts,
arXiv:2402.07871v1](https://arxiv.org/html/2402.07871v1).

Source repository: [llm-random/llm-random, pinned commit
810e6451a402f9a37c2f6f43c1bb75bc00f37140](https://github.com/llm-random/llm-random/tree/810e6451a402f9a37c2f6f43c1bb75bc00f37140).
The implementation and initialization in `model.py` are modified adaptations,
with attribution to the llm-random contributors. Upstream is Apache-2.0; its exact
license is preserved in `UPSTREAM_LICENSE.txt`, SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
The pinned recursive repository tree contains no root `NOTICE` file. No upstream
code is imported at runtime, and no live research-city path is used.

## Source paths inspected

All links below resolve the same immutable commit:

- [expert_choice.py](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/research/conditional/moe_layers/expert_choice.py): extraction, weighted merge, final LayerNorm, group reshape.
- [moe_gating.py](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/research/conditional/moe_layers/moe_gating.py): router logits, expert softmax, token top-k and index conversion.
- [expert_types.py](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/research/conditional/moe_layers/expert_types.py): biasless two-projection ReLU experts and initialization fan-ins.
- [model_utils.py](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/research/conditional/utils/model_utils.py): `determine_moe_args`, `get_expert_choice_args`, `get_inner_expert`, and the expert-choice-specific top-k initialization default.
- [argparse.py](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/research/conditional/utils/argparse.py): ReLU, ordinary attention, pre-norm LayerNorm, expert FFN and initialization defaults.
- [initialization.py](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/lizrd/core/initialization.py): the precise truncated-normal definition.
- [llm.py](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/lizrd/core/llm.py): dense ReLU FFN, causal attention, learned position embeddings and pre-norm residual blocks.
- [expert_choice/common.yaml](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/configs/baselines/gpt/expert_choice/common.yaml), [G16/base.yaml](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/configs/baselines/gpt/expert_choice/granularity/16/base.yaml), and [dense/base.yaml](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/configs/baselines/gpt/expert_choice/dense/base.yaml).

## Exact MoE semantics retained

Let `D` be model width, `F` ordinary FFN width, `G` granularity, `E` expansion,
`R` simultaneous routing-group sequences, and `T` sequence length. Then:

`N = E*G` experts, `H = F/G` hidden units per expert,
`K = R/E` selected sequences per expert per position.

We require integer positive settings, `F % G == 0`, `R % E == 0`, and an actual
forward batch divisible by `R`. The upstream `round(topk_fraction*group_size)` is
replaced with an exact validated quotient. This is a deliberate rejection of
configurations that would change active expert work through rounding.

For one group, `x` has shape `[R,T,D]`; the router weight is `[D,N]`. Compute router
logits and **softmax across N experts for each token**. Each expert then takes
**top K across R sequences independently for each position**. Softmax and top-k
operate on different axes. The token output before output normalization is:

`sum_e selected(e,b,t) * softmax(logits[b,t,:])[e] * ReLU(x[b,t] W1[e]) W2[e]`.

There is no selected-weight renormalization, no auxiliary load-balancing loss,
no z-loss, no shared expert, no SwiGLU, and no token-choice capacity dropping.
Expert outputs are summed before an affine `LayerNorm(D, eps=1e-5)`. Its trainable
bias remains meaningful even when no expert selects a token. The model receives
the ordinary residual path as well. Every expert processes exactly `K*T` vectors
per group, but individual tokens can receive zero, one or many experts. Mean
selection count is exactly `G`; identical per-token counts are not guaranteed.
Router gradients flow through selected probabilities; the top-k membership is
discrete. Nonselected router logits still affect the softmax denominator.

Multiple groups preserve upstream reshape grouping. If batch `B=R*M`, sequence
indices `m, M+m, ..., (R-1)*M+m` form group `m`. Time positions are never mixed.
Gradient accumulation steps and data-parallel ranks do not participate in one
another's groups. Training and held-out evaluation must keep `R`, ordering and
group composition explicit. Expert-choice predictions depend on other sequences'
same-position contexts; changing the evaluation batch is not a benign loader
optimization. Changing future tokens in any sequence must leave every earlier
prediction unchanged, apart from floating-point nondeterminism tolerance.

## Initialization retained, including non-obvious upstream details

All weights use upstream `init_type=truncated_normal, init_scale=0.1` by default:
standard deviation before truncation is `sqrt(scale/fan_in)`, and absolute bounds
are `[-2*scale,+2*scale]`. These are **not** two-standard-deviation bounds and this
is **not** upstream `truncated_normal_fixed`. Router and first expert projection
have fan-in `D`.

The second expert projection has fan-in `G*H=F`, **not** `N*H=E*F`. Although
`ExpertFF`'s standalone default uses `N*H`, `get_expert_choice_args` explicitly
defaults `use_topk_initialization=True`; using the standalone class default would
silently change the actual expert-choice recipe. Dense FFN fan-ins are `D` and
`F`. Learned embedding fan-in is 1. Linear biases begin at zero; LayerNorm weights
begin at one and biases at zero.

Parameter-specific seed streams are a declared adaptation. Each stream is SHA-256
derived from the run seed and stable parameter role, so routing/FFN shape changes
cannot advance the attention, embedding, head or norm initialization streams.
Same-seed common backbone tensors match exactly between dense and all G settings.
This preserves each distribution, not bit-identical upstream RNG trajectories.

## Task and execution adaptations

The initial meaningful backbone matches OMNI-L1's scale: `D=640`, 7 blocks,
10 attention heads, sequence length 512, GPT-2 vocabulary 50,257 and untied output
projection. We reuse the corpus/tokenizer and corrected evaluation contract, not
old ordinary losses. Learned positional embeddings, pre-norm LayerNorm residuals,
causal attention and ReLU FFNs follow the original expert-choice family. The
final LayerNorm and biasless untied output follow our text-backbone convention;
upstream's generic prediction-head construction differs. All FGM comparisons use
this same backbone.

The ordinary control uses an `F=2560` two-linear ReLU FFN with both biases and no
extra FFN output norm, matching upstream `FeedForward`. MoE experts have no biases
and retain their extra output norm. Thus MoE-vs-dense tests are architectural
comparisons; the clean granularity comparison is G1/G4/G16 at fixed E and backbone.
An ordinary GELU historical run cannot substitute for this control.

Initial `E=4` gives roughly 168M total parameters with a roughly 99M ordinary
control and unchanged average expert-matrix work. It is large enough to expose
sparse capacity while keeping optimizer state and pilots practical on one GPU.
`R=32`, `K=8` is an explicit task adaptation from the paper's group size 256; the
smaller competition pool may change learning. Accumulating eight forwards does
not restore the paper's routing competition.

| Architecture | Expert count/layer | Expert width | Mean selections/token | Total parameters |
|---|---:|---:|---:|---:|
| Ordinary | — | 2560 | — | 99,104,640 |
| E4, G1 | 4 | 2560 | 1 | 167,921,920 |
| E4, G4 | 16 | 640 | 4 | 167,975,680 |
| E4, G16 | 64 | 160 | 16 | 168,190,720 |

The table is architectural accounting, not a measured learning result. The code
verifies its analytical total against instantiated parameters. All MoE settings
have 91,750,400 expert weights total and 22,937,600 average active expert weights
across seven layers. Router parameters and router projection FLOPs grow with G;
dispatch/scatter vector traffic does too. `parameter_accounting()` includes head,
embedding, biases, norms, active-average definitions and separately scoped FLOP
estimates. It does not mislabel top-k or memory traffic as zero cost: these are
excluded from the matmul estimate and included in measured wall time.

Production router logits, softmax, selected weights and merge accumulation use
FP32. Expert and attention projection matmuls use BF16 autocast. Default attention
uses Torch SDPA with BF16 inputs to permit FlashAttention; this differs from the
paper's high-precision attention mechanism and must be declared in comparisons.
`attention_precision=fp32` keeps the attention operation in FP32 when requested.
FP64 modules are supported solely for small independent numerical audits. Normal
training does not force parameter tensors to BF16. FP32 merge accumulation is a
declared numerical execution choice rather than a claim of bitwise equivalence
to the upstream autocast one-hot/bmm merge.

Indexed gather, batched expert matmul and indexed sum replace upstream one-hot
dispatch. They implement the same mathematical mapping while avoiding dense
one-hot dispatch work. No differentiable surrogate is substituted for top-k.
GPU indexed sums can differ at roundoff level with accumulation order. Tied
routing scores use PyTorch top-k tie behavior and are not assigned artificial
expert identities. Correctness tests should avoid accidental ties for derivative
checks and treat explicit tie fixtures separately.

## Recipes intentionally not mixed

Paper Appendix A reports C4/GPT-2, group size 256, mainly `E=64`, maximum LR
`2e-4`, mixed precision with high-precision router/attention, and training-loss
scaling analysis. The inspected baseline config instead specifies `E=32`, LR
`1e-4`, BF16 and the same expert-choice semantics. Neither is silently presented
as our exact configuration.

The newer [constrained_scaling_laws/common.yaml](https://github.com/llm-random/llm-random/blob/810e6451a402f9a37c2f6f43c1bb75bc00f37140/configs/experiments/constrained_scaling_laws/common.yaml)
uses token-choice, capacity factor 1, SiLU gated experts, effective width 3D,
RoPE/no learned position embedding, FineWeb-Edu, z-loss and load-balancing loss,
and a different schedule. FGM-A does not inherit those model or routing choices.
