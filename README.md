# Latch

Latch compiles supported PyTorch training code into a single GPU program for the
forward pass, loss, backward pass and optimizer update. This is a research preview
for NVIDIA H100 GPUs.

[Project page](https://deepaksingh98.github.io/latch/site/) · [v0 release](https://github.com/DeepakSingh98/latch/releases/tag/v0)

## Install

Use Linux, Python 3.12, CUDA Toolkit 12.8 and an H100 80 GB. Compilation needs a
C++ compiler and Python development headers. See [environment setup](docs/SETUP.md)
if you need to install them.

Download the [source archive, wheel and checksums](https://github.com/DeepakSingh98/latch/releases/tag/v0) into one directory:

```sh
sha256sum -c SHA256SUMS
tar -xzf latch-v0.tar.gz
cd latch-v0
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install '../megakernel_lab-0.0.0-py3-none-any.whl[gpu]'
```

If you cloned the source instead, start at the `python3.12 -m venv` line from its
root and replace the wheel install with `python -m pip install '.[gpu]'`.

## Compile and train

The first example trains a small MLP on synthetic data. No model download or
credentials are needed.

```sh
export LATCH_ROOT="$PWD/latch-output"
CUDA_VISIBLE_DEVICES= python examples/quickstart.py prepare \
  "$LATCH_ROOT/mlp" --cache-dir "$LATCH_ROOT/cache"
python examples/quickstart.py run "$LATCH_ROOT/mlp" --steps 10
```

`prepare` compiles on CPU. `run` trains on the H100 and prints the loss. To try a
wider model with a different learning rate:

```sh
CUDA_VISIBLE_DEVICES= python examples/quickstart.py prepare \
  "$LATCH_ROOT/mlp-wide" --cache-dir "$LATCH_ROOT/cache" \
  --hidden 128 --lr 0.0005
python examples/quickstart.py run "$LATCH_ROOT/mlp-wide" --steps 10
```

Edit [the example](examples/quickstart.py) to change the model. Compile into a new
output directory after each change. If you build and run on separate machines,
use the same absolute paths and Python environment on both.

## Use an agent

Give your coding agent the release folder and access to your H100, then paste:

```text
Read README.md and help me use Latch. Check the environment, install it in a
virtual environment, and run the MLP quickstart. Change its hidden width to 128
and learning rate to 0.0005, then compile and train again. Use the included
PyTorch comparisons and numerical checks where available. Show me the losses
and the commands that worked. Use my existing compute; ask before renting more.

Then help me try this experiment: [describe your model or learning idea].
Start from the closest included example. If an operation is unsupported,
explain what needs to be added rather than silently changing the experiment.
```

## More examples

- [TinyMoE training](examples/tinymoe/README.md)
- [Token-by-token adaptation](examples/gradlev/README.md)
- [Selective weight updates](examples/selective_update/README.md)
- [Repeated layers](examples/smelt_looped/README.md)
- [Smaller models and learning rules](examples/fast/README.md)

[Results](https://deepaksingh98.github.io/latch/site/methodology.html) · [Known limitations](docs/NOTES.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md) · [MIT license](LICENSE)
