# Complete-update comparisons

The installed package includes MLP, transformer, FFN, mixed, dynamic-batch,
adaptive-memory and energy-model examples. Each compares generated execution
with conventional controls and can audit saved updates on CPU.

Follow [setup](SETUP.md) and install the downloaded wheel with its `gpu` extra
on both hosts. From the extracted release root:

```sh
python -m pip install '../megakernel_lab-0.0.0-py3-none-any.whl[gpu]'
python -I -m megakernel_lab.fast_examples list
export LATCH_FAST_ROOT="$PWD/fast-output"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
```

Use a new absolute root for each independent run. Keep the prepared tree and
observations at that same path on the CPU and GPU hosts. `--root` selects this
directory explicitly; `LATCH_FAST_ROOT` supplies the same default when omitted.
`python -I` exercises the installed package without source-tree imports.

## Prepare, compare, audit

```sh
# CPU with CUDA toolkit: capture and compile.
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" prepare mlp shared-64-bf16
# Otherwise idle H100: compare changing updates.
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" compare mlp shared-64-bf16 --rounds 3
# CPU: independent audit, then read the combined result.
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" verify mlp shared-64-bf16
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" report mlp shared-64-bf16
```

Use the same actions with these families and cases:

| Family | Example cases | Computation |
|---|---|---|
| `mlp` | `shared-64-bf16`, `aux-512-fp32`, `heldout-128-bf16` | Residual/shared MLP; SGD or momentum |
| `ordinary` | `o-l-bf16`, `o-m-bf16` | Byte transformer with FP32 masters and BF16 projections |
| `ffn` | `ffn`, `ffn-d768`, `ffn-gelu` | Normalization, up/activation/down, residual, clipped AdamW |
| `mixed` | `r14_mixed_b`, `r13_mixed`, `fresh_pairwise` | BF16 MLP and FP32 pairwise auxiliary branch |
| `dynamic` | `matrix_bf16-b32`, `matrix_bf16-b16`, `matrix_bf16-b1`, `reserved_bf16-b32` | MLP with one binary for batch capacity 32 |

Omitting cases runs the whole family. CPU actions accept `--workers N`, with
one compute thread per worker; provision RAM for each case. GPU arms run
serially and rotate order between rounds. `--arms eager,latch,compiled` selects
a subset; use the same filter for verification and include eager in round zero
to create the shared trained state. `--json PATH` saves a report.

The comparison clock covers a complete update on resident GPU data; it excludes
host input staging and model compilation. Setup, memory and kernel counts are
recorded separately. These short checks do not measure time to a quality target.

## Adaptive memory

These commands use the supplied synthetic byte fixtures with the `m2k`, `m8k`
and `m8k_c64` models and their Muon/AdamW updates:

```sh
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" prepare adaptive --workers 3
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" compare adaptive --rounds 3
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" verify adaptive
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" verify adaptive --arms cpu --workers 3
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" report adaptive
```

The first verification runs on H100 and captures observations plus matched CUDA
controls. Transfer those files back to CPU before the second verification,
which computes independent references. GPU observation capture alone does not
complete the numerical check.

## Energy training

Preparation downloads and verifies public Fashion-MNIST files and stages CPU
reference fixtures. The H100 commands compare updates and run the 400-update
learning checks for the supplied SD and WB models:

```sh
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" prepare energy --workers 2
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" compare energy --rounds 20
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" verify energy
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" train energy
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" verify energy --arms cpu --workers 2
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" report energy
```

Transfer the complete prepared tree and observations back to CPU before its
verification. Retain the separate optimizer-tier results; a passing tier does
not erase failures in another tier.

## Interpret the report

Qualification needs completed timing rounds and passing fresh/trained CPU
audits. The report separates fastest measured, matched-arithmetic and eligible
controls. A run can finish successfully while an arm fails its numerical gate.
For example, the BF16 MLP's relative parameter gate can reject tiny updates to
zero-initialized biases; some compiled controls also fail their audits.

Preserve each family's observations, completion journal and `audit.json`.
Do not reuse a root for overlapping runs: old audits and new observations may
refer to different states. To append disjoint rounds to an existing run, set
`LATCH_FAST_ROUND_START` and keep its original files.

Atomic indexed accumulation can vary in low bits; choose a supported policy
such as `segmented_v1` when deterministic indexed gradients are required. The
harness's in-process restart check alone does not prove cross-process equality.
See [limitations](NOTES.md) and [scoped results](RESULTS.md).
