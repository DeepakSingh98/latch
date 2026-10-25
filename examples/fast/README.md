# Fast examples

Follow [setup](../../docs/SETUP.md) with the `gpu` extra. The installed package
contains the models, fixtures, controls and audit commands. For a first run:

```sh
export LATCH_FAST_ROOT="$PWD/fast-output"
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" prepare mlp shared-64-bf16
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" compare mlp shared-64-bf16 --rounds 3
CUDA_VISIBLE_DEVICES= python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" verify mlp shared-64-bf16
python -I -m megakernel_lab.fast_examples --root "$LATCH_FAST_ROOT" report mlp shared-64-bf16
```

Prepare and audit on CPU; compare on H100. Use a new absolute output root for
each independent run and preserve it between hosts. A completed comparison
needs the CPU audit before its results are qualified. See the
[full command guide](../../docs/FAST_EXAMPLES.md) and
[numerical limitations](../../docs/NOTES.md).
