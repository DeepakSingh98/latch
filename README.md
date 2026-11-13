# Latch

Latch compiles a PyTorch training step into one GPU kernel: the forward pass,
the loss, the backward pass and the optimizer update run in a single launch.

This is the bare-bones compiler: about 1,300 lines of Python and 200 of CUDA.
The research preview behind the published results is the
[v0 release](https://github.com/DeepakSingh98/latch/releases/tag/v0);
see the [project page](https://deepaksingh98.github.io/latch/site/).

```python
import latch

captured = latch.capture_module(model, loss_fn, (x,), (y,), optimizer=latch.AdamWConfig(lr=1e-3))
captured.build("out")                     # needs nvcc, not a GPU

update = latch.ModuleUpdate.load("out")   # on the GPU
update.stage((x,), (y,))
loss = update.step()                      # one kernel launch
```

## Install

Python 3.11 or later and PyTorch 2.9. Compiling for a GPU needs `nvcc` from the
CUDA Toolkit (tested with 12.8 on an H100). From a clone:

```sh
python -m pip install -e '.[test]'
```

## Try it

```sh
python examples/quickstart.py prepare out    # capture and compile a small MLP
python examples/quickstart.py run out        # train it
python examples/gpt.py --compare             # a character-level GPT, timed against PyTorch
python examples/bench.py                     # the table below
```

Without a GPU, add `--target cpu` to `prepare` (or to `gpt.py`, with
`--steps 5 --batch 2 --context 8`). The same generated source is then compiled
for the host, with an OS thread standing in for each GPU thread. It is slow and
exists for testing.

Read `out/step.cu` after `prepare`: it is the whole training step.

## How it works

| Step | File | What happens |
|---|---|---|
| Capture | `capture.py`, `optim.py` | `torch.func.grad_and_value` and the optimizer run under `make_fx`. PyTorch differentiates; the result is one graph of ATen operations. |
| Lower | `lower.py`, `layout.py`, `ops.py` | Views become strides. Pointwise operations fuse into loops. Every item gets a phase. New weights are written over old ones. Temporaries share one arena. |
| Generate | `codegen.py`, `latch.cuh` | One cooperative kernel: phases separated by grid-wide barriers. |
| Run | `runtime.py` | Compile with `nvcc`, bind tensors once, launch. |

The kernel for the quickstart MLP has eight phases. This is its last one, the
AdamW update of the first layer, writing weights and both moments in place:

```c++
// loop over [64, 64]: add_1 add_2 sub_3
for (int i = tid; i < 4096; i += threads) {
  const float x_mul_7 = (b_arg3_1[i] * 0.9f);
  const float x_mul_8 = (b_mm_4[i] * 0.09999999999999998f);
  const float x_add_1 = (x_mul_7 + x_mul_8);
  ...
  b_add_1[i] = x_add_1;   // first moment
  b_add_2[i] = x_add_2;   // second moment
  b_sub_3[i] = x_sub_3;   // weights
}
```

## What is supported

- **Models:** linear layers, the usual activations, LayerNorm, embeddings,
  attention written with matrix products and softmax, MSE and cross-entropy.
  The tests train MLPs and a small GPT.
- **Optimizers:** `SGDConfig` (momentum, weight decay) and `AdamWConfig`. Any
  object with `init(params)` and `update(params, grads, state)` written in
  tensor operations works the same way.
- **Types:** float32, with int64 and bool for indices and masks.

An operation without a rule in `ops.py` stops compilation with its name.

## Measured

Time per training step on one H100 with PyTorch 2.9.1 and AdamW, from one run of
`examples/bench.py`. "PyTorch" is the same model trained eagerly on the same GPU
with `torch.optim.AdamW(fused=True)`.

| Model | Latch | PyTorch | Speedup |
|---|---|---|---|
| MLP, 3 layers of width 64, batch 64 | 37 µs | 658 µs | 17.7× |
| MLP, 3 layers of width 512, batch 512 | 515 µs | 697 µs | 1.35× |
| MLP, 3 layers of width 2,048, batch 2,048 | 17.6 ms | 3.0 ms | 0.17× |
| GPT, 2 layers, width 64, context 32, batch 8 | 562 µs | 3.43 ms | 6.1× |
| GPT, 4 layers, width 128, context 64, batch 16 | 2.90 ms | 5.54 ms | 1.9× |
| GPT, 6 layers, width 384, context 256, batch 32 | 96.9 ms | 20.4 ms | 0.21× |

Latch wins where launch overhead dominates and loses where matrix products do:
its matrix product is a plain tiled float32 loop, not a tuned one.

## Limits

- Shapes are fixed at capture. Python control flow is specialized to the
  captured call.
- No dropout or other randomness, no convolutions, no mixed precision.
- One GPU. One step per launch.
- Embedding gradients accumulate with atomic adds, so two runs can differ in the
  last bits.
- Compile time grows with the model: about a minute for the 6-layer GPT above.

## Tests

```sh
python -m pytest                          # host emulation
LATCH_TEST_TARGET=cuda python -m pytest   # on a GPU
```

Each training test runs a few steps and compares losses, weights and optimizer
state with the same steps in ordinary PyTorch.

[MIT license](LICENSE)
