# Edit a learning program and compile it

`model.py` defines a small byte transformer, its loss and AdamW settings in plain
PyTorch. `workflow.py` compiles it and runs the same computation with Latch,
eager PyTorch or a compiled PyTorch control. Defaults use synthetic bytes; these
are API demonstrations, not learning or speed claims.

After installing the release with `[gpu]`, run from the source root:

```sh
export RESEARCHER_ROOT="$PWD/researcher-output"
CUDA_VISIBLE_DEVICES= python examples/researcher/workflow.py prepare "$RESEARCHER_ROOT/ordinary" \
  --cache-dir "$RESEARCHER_ROOT/cache" --config examples/researcher/configs/ordinary.json
python examples/researcher/workflow.py describe "$RESEARCHER_ROOT/ordinary"
python examples/researcher/workflow.py run "$RESEARCHER_ROOT/ordinary" --arm latch --steps 16
python examples/researcher/workflow.py run "$RESEARCHER_ROOT/ordinary" --arm compiled --steps 16
```

Prepare on CPU with the CUDA toolkit, then run on H100. Preserve the absolute
artifact and cache paths. The compiled control can require substantial CPU
compilation on its first GPU-host run.

Change the model or learning configuration and compile a new program:

```sh
CUDA_VISIBLE_DEVICES= python examples/researcher/workflow.py prepare "$RESEARCHER_ROOT/recurrent-exit2" \
  --cache-dir "$RESEARCHER_ROOT/cache" --config examples/researcher/configs/recurrent.json \
  --set exit_after=2 --set aux_weight=0.5
python examples/researcher/workflow.py run "$RESEARCHER_ROOT/recurrent-exit2" --arm latch --steps 16
```

The supplied configs cover ordinary training, intermediate supervision and
shared-weight recurrence. `--set lr=...` changes the static learning rate and
requires recompilation. For a program with an intermediate exit, `run
--aux-weight 0.25` changes its loss weight without recompilation. The default
`segmented_v1` indexed-gradient policy is deterministic; it does not change the
compiler's automatic policy. Run `workflow.py --help` for checkpoint and
comparison commands.

For real text, the optional [data helper](../families/prepare_data.py) writes the
format accepted by `run --data DIRECTORY`. Install
`python -m pip install -r examples/requirements-data.txt`, then run
`python examples/families/prepare_data.py text --out "$RESEARCHER_ROOT/data"`
on CPU. This downloads public FineWeb-Edu row groups and prepares about 64 MiB
of text; it is separate from the no-download example above.
