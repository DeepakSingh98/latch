# TinyMoE selective updates

This example trains every TinyMoE-200M parameter. Shared parameters update on
each 2 × 1,024-token microbatch; experts accumulate raw FP32 gradients and
update once per 16 microbatches. Two generated programs share persistent state:
`accumulate` runs 15 times, then `apply` runs once.

Follow [setup](../../docs/SETUP.md), then run from the extracted source root.
Install the same extra on the CPU and GPU hosts:

```sh
python -m pip install '../megakernel_lab-0.0.0-py3-none-any.whl[tinymoe]'
export SELECTIVE_ROOT="$PWD/selective-output"
bash examples/selective_update/quickstart.sh prepare          # CPU, internet
CUDA_VISIBLE_DEVICES= bash examples/selective_update/quickstart.sh build
bash examples/selective_update/quickstart.sh train            # H100: 64 microbatches
bash examples/selective_update/quickstart.sh restart          # H100: resume from microbatch 24
CUDA_VISIBLE_DEVICES= bash examples/selective_update/quickstart.sh compare-restart
bash examples/selective_update/quickstart.sh conventional     # H100: compiled-grouped control
```

Use a new absolute root and retain it on both hosts. The CPU build needs the
CUDA toolkit and runs 16 compiler jobs for each of two programs. Allow several
GB for artifacts, full-state checkpoints and final parameters. `prepare`
downloads the pinned [TinyMoE model](https://huggingface.co/FlameF0X/TinyMoE-200m-2x16)
and a small [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext)
fixture. This is a short functional example, not a learning comparison.

## Outputs and options

`train/` contains per-microbatch losses, norms and timings, `result.json`,
`checkpoint.pt`, `final-parameters.pt`, evaluation rows and tensor checksums.
`restart` writes to `resumed/`. `compare-restart` checks all loss/norm rows,
evaluations and persistent tensor checksums, including accumulated gradients,
participation flags, moments and counters. The conventional stage writes
`conventional.json` and `rows.npz` plus an agreement report when native final
parameters exist.

| Variable | Default | Effect |
|---|---|---|
| `SELECTIVE_INIT` | `pretrained` | `fresh` prepares a random initial state |
| `SELECTIVE_SEED` | `1743` | Fresh-state initialization seed |
| `SELECTIVE_START_UPDATE` | `32768` pretrained / `0` fresh | Schedule position, a multiple of 16 |
| `SELECTIVE_ORDER_SEED` | `1729` | Training-window order |
| `SELECTIVE_LR_SCALE` | `1.0` | Shared-group LR multiplier and inverse weight-decay multiplier |
| `SELECTIVE_KERNELS` | `0` | `1` adds a separate profiler pass |

The default starts with public pretrained weights and **zero optimizer state**.
The schedule position does not load a trained optimizer checkpoint.

## Update rule and numerical limits

Both groups use their own norm clip at 1.0. At an expert apply, the accumulated
gradient is divided by 16. An expert unused throughout the window retains its
parameters, moments and counter. Expert learning rate is four times the mean
scheduled LR over the window, and the window's decay is applied once.

The expert beta1 bias correction is intentionally **nonstandard**: its counter
`t` advances by 16 per apply, and the rule uses `1 − 0.9^t`. Standard AdamW over
window updates uses `1 − 0.9^(t/16)`. From zero optimizer state the first expert
updates differ substantially. The conventional counterpart uses standard AdamW,
so the two trajectories are not an equal-rule learning comparison. BF16
implementations can also differ in rounding. See
[release notes and limitations](../../docs/NOTES.md).

The persistent BF16 expert casts belong to the runtime's derived state, not
checkpoints. Ordinary PyTorch parameter writes are detected and recast on the
next launch. Writes that bypass PyTorch version counters, such as raw-pointer
writes from another extension, require `runtime.refresh_derived()` before launch.
