# TinyMoE quickstart

Train the pinned TinyMoE-200M model for 32 updates on a small WikiText-103
fixture, compare a conventional control, and check checkpoint restart.
Follow [setup](../../docs/SETUP.md), then run from the extracted source root.
Install the same extra on the CPU and GPU hosts:

```sh
python -m pip install '../megakernel_lab-0.0.0-py3-none-any.whl[tinymoe]'
export TINYMOE_ROOT="$PWD/tinymoe-output"
bash examples/tinymoe/quickstart.sh prepare          # CPU, internet
CUDA_VISIBLE_DEVICES= bash examples/tinymoe/quickstart.sh build
bash examples/tinymoe/quickstart.sh train            # H100: 32 updates
bash examples/tinymoe/quickstart.sh conventional     # H100: same short recipe
bash examples/tinymoe/quickstart.sh evaluate         # H100: test NLL
bash examples/tinymoe/quickstart.sh restart          # H100: update 15 to 32
CUDA_VISIBLE_DEVICES= bash examples/tinymoe/quickstart.sh compare-restart
```

Use a new absolute root and retain it on both hosts, including the generated
artifacts and cache. The CPU build needs the CUDA 12.8 toolkit and runs eight
concurrent compiler jobs; execution needs an H100 80 GB. Exact native restart
binds binaries, runtime, paths, model/data, recipe and optimizer horizon.

`prepare` downloads [TinyMoE weights and tokenizer](https://huggingface.co/FlameF0X/TinyMoE-200m-2x16/tree/9b209f7439aadfac282e0bf44f2fda19771aba90)
and [WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
recording hashes and split identities. The fixture contains 65,537 training
tokens and 8,193 tokens per held-out split, including lookahead. Model weights
and corpus text are not bundled.

The recipe uses B2 × S512 and AdamW with horizon 400, stopping after 32 updates.
Evaluation reports final-readout NLL without router loss. This short continued
training run does not qualify long training: the B2 × S512 recipe has a known
BF16 audit failure at longer-trained states. See [limitations](../../docs/NOTES.md).

## Compare execution

On an otherwise idle H100:

```sh
bash examples/tinymoe/quickstart.sh screen
export TINYMOE_CONVENTIONAL=compiled-grouped
bash examples/tinymoe/quickstart.sh benchmark
```

`screen` compares the available conventional backends. Set
`TINYMOE_CONVENTIONAL` to the strongest numerically eligible backend from your
screen; it controls both the training control and paired benchmark. The clock
includes staging through synchronized update completion, excluding compilation,
restoration, warmup, common batch loading, evaluation and checkpointing.
Timing alone does not establish numerical correctness.

## Audit saved updates

Set `TINYMOE_OBSERVE=1` **before** `train` or `conventional` to save observations
for updates 1 and 32. Allow several additional GB and run observation capture
separately from timing. On CPU, audit each native observation:

```sh
python -m megakernel_lab.tinymoe.audit \
  --model-dir "$TINYMOE_ROOT/model" \
  --observation "$TINYMOE_ROOT/latch/observation-000001.pt" \
  --out "$TINYMOE_ROOT/audit-latch-1"
python -m megakernel_lab.tinymoe.audit \
  --model-dir "$TINYMOE_ROOT/model" \
  --observation "$TINYMOE_ROOT/latch/observation-000032.pt" \
  --out "$TINYMOE_ROOT/audit-latch-32"
```

Use the conventional output directory to audit that arm. Transfer each `.json`
sidecar together with its `.pt` observation. The independent FP64 reference uses
the packaged `tinymoe/POLICY.json` gates. Passing these short-run checks does
not remove the longer-trained limitation. A separate
[reproduction guide](reproduction/README.md) retains the full learning protocols.
