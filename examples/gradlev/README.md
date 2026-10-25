# GradLev: causal adaptation of GPT-2

This example adapts the two weight matrices of GPT-2 small's final MLP after
scoring each token. The rest of the model stays frozen. A block contains 1,024
dependent token steps, lowered through `vector_resident` into one cooperative
kernel. This is a paper-based adaptation of Bo Liu and Qiang Liu's
[GradLev](https://arxiv.org/abs/2609.34174), with the paper's initial gate;
helper training is not included.

Follow [setup](../../docs/SETUP.md), then run from the extracted source root.
Install the same extra on the CPU and GPU hosts:

```sh
python -m pip install '../megakernel_lab-0.0.0-py3-none-any.whl[gradlev]'
export GRADLEV_ROOT="$PWD/gradlev-output"
```

Use a new absolute root and retain that path when transferring build outputs
between hosts. `prepare` needs CPU and internet; `build` needs the CUDA 12.8
toolkit. Execution needs an H100 80 GB with the default 132-CTA grid.

## Run

```sh
bash examples/gradlev/quickstart.sh prepare        # CPU: weights, tokenizer, held-out text
CUDA_VISIBLE_DEVICES= bash examples/gradlev/quickstart.sh build
bash examples/gradlev/quickstart.sh run            # H100: first 16 dev blocks
bash examples/gradlev/quickstart.sh conventional   # H100: compiled CUDA-graph control
bash examples/gradlev/quickstart.sh restart        # H100: carry, restore, reset
bash examples/gradlev/quickstart.sh benchmark      # Otherwise idle H100
```

`prepare` downloads pinned [GPT-2 weights](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)
and tokenizer, then reconstructs held-out windows from
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9).
It reads the needed parquet row groups through HTTP range requests. The default
128 validation windows are a prefix reconstructed by the split rule, not a
whole-file hash-verified dataset. `data.json` records the downloads and split.
Setting `GRADLEV_DEV_WINDOWS=2048` or `GRADLEV_TEST_WINDOWS=8192` requests the
corresponding full split and checks its expected hash.

## Options and outputs

| Variable | Default | Effect |
|---|---|---|
| `GRADLEV_BLOCKS` | `16` | Dev blocks to score |
| `GRADLEV_POLICY` | `P1` | `P1b` keeps softmax-gradient weights in FP32 and fuses the head; rebuild for each policy |
| `GRADLEV_CONVENTIONAL` | `graph_hybrid` for P1 | Override the conventional backend; P1b defaults to `graph_gl32red` |
| `GRADLEV_ROUNDS` | `5` | Alternating benchmark rounds |
| `GRADLEV_ARMS` | Policy-specific | Semicolon-separated benchmark arms |
| `PYTHON` | `python` | Interpreter |

Evaluation writes frozen/adapted NLL, block timing and per-token NLLs under
`run-latch/` and `conventional-graph_hybrid/` (names change with policy/backend).
`benchmark-P1/bench.json` contains timings, memory, kernel counts and agreement
with the eager loop. `restart/state_checks.json` contains state-equality flags;
`build_latch/build.json` records build timings and generated source identity.

The benchmark times a complete block: frozen-trunk prefill, staging, reset,
adaptation and synchronized NLL read-back. It excludes compilation, graph
capture and the first warm block. Use the eager loop for numerical agreement;
the hybrid control has different rounding and is not the reference.

## Numerical behavior and limits

Each token is scored **before** its target updates the fast weights. Fast state
and trunk context reset for each block. P1 uses a BF16 frozen trunk with FP32
masters; the adapted MLP, fast state, normalization, costates and writes are
FP32 with TF32 off. The head uses BF16 operands and FP32 accumulation/results.
P1b additionally retains the softmax-gradient weights in FP32.

The loop supports at most 1,024 steps, small shared-memory carries and batch-1
vector products/reductions. Unsupported accesses and operations are refused;
changing the grid from the compiled value traps. The compiled conventional
control's new-process restart is not guaranteed bitwise. Inspect the saved
checks before claiming state equality or numerical agreement. See
[release notes and limitations](../../docs/NOTES.md).
