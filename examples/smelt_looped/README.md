# Looped TinyMoE training

This example trains a shared-weight recurrent TinyMoE with Latch and a
conventional compiled PyTorch/CUDA-graph control. It adapts
[SMELT](https://arxiv.org/abs/2609.01343) and
[Loop Scaling Laws](https://arxiv.org/abs/2609.40316) to TinyMoE's architecture;
it is not an implementation of either paper's full model or experiment.

Follow [setup](../../docs/SETUP.md), then run from the extracted source root.
Install the same extra on the CPU and GPU hosts:

```sh
python -m pip install '../megakernel_lab-0.0.0-py3-none-any.whl[tinymoe]'
export SMELT_ROOT="$PWD/smelt-output"
bash examples/smelt_looped/quickstart.sh prepare          # CPU, internet
CUDA_VISIBLE_DEVICES= bash examples/smelt_looped/quickstart.sh build
bash examples/smelt_looped/quickstart.sh train            # H100: 32 updates
bash examples/smelt_looped/quickstart.sh conventional     # H100: same model and recipe
bash examples/smelt_looped/quickstart.sh restart          # H100: update 15 to 32
CUDA_VISIBLE_DEVICES= bash examples/smelt_looped/quickstart.sh compare-restart
```

Use a new absolute root and retain it on both hosts. CPU compilation needs the
CUDA toolkit and runs eight concurrent device-unit jobs. Training needs one
H100 80 GB. `prepare` downloads the pinned
[TinyMoE model](https://huggingface.co/FlameF0X/TinyMoE-200m-2x16) and a small
[WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) fixture;
download identities are recorded in the model/data files.

## Model and recipe

`SMELT_LOOP=3,6,2` is the default: zero-based layers 3–8 run twice with shared
weights, giving 18 decoder visits. Every residual update in this span is scaled
by 1/2. Each visit recomputes attention from its updated input; there is no KV
reuse. Full backpropagation sums gradients from both visits. The objective is
mean cross-entropy plus 0.02 × router auxiliary loss across all visits.

Each run performs 32 updates at B8 × S512. LR warms up over eight updates to
3e-4, then stays constant. Two execution warmup updates are undone before the
measured run. The optimizer uses global norm clipping at 1.0 and participating
AdamW with FP32 masters/moments; unused experts remain unchanged. Projections,
attention and expert GEMMs use BF16; cross-entropy uses FP32.

The recipe derives beta2 and matrix-only weight decay from a 1,024-token
sequence even though this example uses S512. It uses beta1 0.9, epsilon 1e-6,
beta2 `0.999^(8/128)` and weight decay 0.00625 at LR 3e-4. This behavior is
preserved in both executors.

`SMELT_LOOP` accepts `FIRST,SPAN,PASSES`; `0,1,1` gives an unlooped pass.
Use a separate root for every loop setting. `SMELT_MODEL`, `SMELT_DATA` and
`SMELT_ARTIFACT` override their default subdirectories; `PYTHON` chooses the
interpreter. Unsupported model changes may be refused by the compiler.

## Outputs and checks

`artifact/` contains the generated program and build descriptions. `latch/`
and `conventional/` each contain `result.json` and `checkpoint-000032.pt`.
The restart stage writes update 15 under `first/` and update 32 under
`resumed/`. `compare-restart` exits successfully only when the native straight
and resumed checkpoints are byte-equal.

Each `result.json` records losses, gradient norms, update timings and dev NLL.
Dev NLL excludes router loss and uses the available validation batches (8,192
targets with the default fixture). Timings cover synchronized updates;
compilation, graph capture and evaluation are separate.

This short example checks execution and restart, not a benefit from adding
recurrence. Width and compute are not matched to the unlooped model, and a
successful run alone does not establish numerical agreement with a reference.
See [release notes and limitations](../../docs/NOTES.md).
