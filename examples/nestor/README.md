# Nestor training programs

This example builds the complete training update of
[Nestor](https://github.com/Ma-Lab-Berkeley/Nestor) `nestor_100m` with Latch:
forward, backward, global clipping, gradient accumulation and the MuonAdamW
step, generated from the model code as one persistent cooperative program per
launch. It is a build example. It writes the two programs of the measured
B8 × 4096 update and records how they were made. The training harness that
timed them is not included.

The upstream repository has no license, so none of its code is copied here. You clone
it yourself. It is used only to export the step-0 state and the optimizer
groups. `nestor_model.py` is a Latch reimplementation of the training forward
that keeps the upstream parameter names, so upstream state dicts load
unchanged.

## The model

Nestor replaces the residual stream with depth reads. Each layer reads a
softmax mixture of earlier block outputs, then applies RMSNorm. Each block has
a column router that sends every token to k of 8 columns, using a gelu score
MLP, injected noise and a loss-free choice bias. Each column's heads attend
causally over that column's member tokens only. In the published model the
heads are 30 channels wide, padded to 32. The expert layer is a hybrid pool:
public top-2 experts plus per-column private experts, mixed by a path gate.
Block exits export per-column deviations, and the head reads all of them.

`nestor_100m` has width 480 and 7 blocks with (1, 1, 2, 2, 2, 3, 2) layers,
k = (3, 4, 5, 6, 6, 7, 8), (5, 4, 3, 3, 3, 2, 2) heads, 8 public and 8
private experts, a 100,263-token vocabulary and context 4,096.

In `nestor_model.CapturedNestor`, the selected token/column rows of a block
are packed once into ragged expert products and native segmented attention.
Nothing is padded to the longest segment. Q/K/V, the output projections and
the expert products run only on packed rows. `nestor_optimizer.make_rule` is
the upstream MuonAdamW with global norm clipping at 1.0 and gradient
accumulation, written as a Latch `tensor_update`. Muon's Newton-Schulz slices
are zero-padded to a multiple of 8 and iterated in their own orientation. Both
choices give the same per-entry arithmetic and admit tensor-core bodies
(`ns_pad=8, ns_tall=True`).

## Requirements

- **`inventory`:** a clone of `https://github.com/Ma-Lab-Berkeley/Nestor` at
  commit `9c25256fd358f4f3b0b92da175158db20fc035b3`, in an environment that
  can import it. `nestor_inventory.py` refuses any other commit. CPU only.
- **`build`:** a Linux CPU host with the CUDA 12.8 toolkit and no visible GPU.
  It needs `megakernel-lab` installed with the `gpu` extra (torch 2.9.1).
  Tensor work uses one thread, and the build runs 28 concurrent device-unit
  compiler jobs. The programs target `sm_90`.
- **Execution:** one H100 80 GB. Running the programs needs a training harness
  that stages tokens, routing noise, labels and the learning-rate schedule
  (`nestor_optimizer.schedule`). This example does not include one.

## Commands

```sh
export NESTOR_ROOT=/absolute/new/output/dir
export NESTOR_SRC=/path/to/Nestor                   # git checkout at 9c25256f
export UPSTREAM_PYTHON=/path/to/nestor-env/bin/python
bash examples/nestor/quickstart.sh inventory        # upstream env: init-full.pt, inventory-full.json
CUDA_VISIBLE_DEVICES= bash examples/nestor/quickstart.sh build
bash examples/nestor/quickstart.sh smoke            # optional: reduced r1 fixture, capture only
bash examples/nestor/quickstart.sh ir-hash          # optional: captured-graph sha256 of both programs (see Program identity)
bash examples/nestor/quickstart.sh options          # the complete build keyword set
```

The script sets `PYTHONPATH` to this directory only, so `megakernel_lab`
comes from the installed package. `build` writes
`$NESTOR_ROOT/full-{accumulate,final}-b8/` with `artifact/` (the generated
program), `cache/` and `build.json`. `build.json` records the flags, the
operation census, the captured-graph sha256, the input hashes, the hashes of
this directory's sources and the build receipt. Any refusal is recorded there
verbatim.

One update is 8 microbatches of B8 × 4096 (262,144 tokens). The
`accumulate` program runs for the first seven and adds each gradient to an
FP32 accumulator. The `final` program runs for the eighth: it clips, takes the
optimizer step and clears the accumulator. Capture uses zero inputs of the
static shape. Tokens, noise and labels are runtime inputs.

## Options

`nestor_build.py` passes this keyword set to `CapturedModule.build`, inside
`indexed_reduction.policy("segmented_v1")` and after capturing under
`capture_bf16_sdpa()`:

```python
BASE = dict(arch="sm_90", execution="compose", precision_policy="practical", bf16_weight_chunk_k=8192,
            square_sum_association="balanced", gradient_outputs="returned", native_code_only=True,
            device_units=16, device_unit_inline="isolate_all", device_unit_split=150000,
            device_unit_shared_pool=False, static_shared_reserve_bytes=8192,
            compact_codegen="elementwise_full_sum_resource_pure_intervals", device_unit_max_registers=255)
RULES = dict(update_schedule="streamed_prep", attention_rules="streamed",
             segmented_products="flat+tma+rotate+claim+sepi+ktail+fit+bsearch+scatter+route+packsum+kcat",
             matrix_bodies="exact+skinny+skinnybatch+shortk+splitctr",
             accumulation_rules="update+fsload+fsspread+rilp+fsdeepx+rinl+rleaf+rhoist2+rsum1"
                                "+rwarp+rstage+ridx+rearly+rremat+rslot+rnarrow+rmid",
             segmented_attention="folded_forward+fused_backward+claim+rotary",
             tile_readiness="attention", vocab_rules="stream+views+rows",
             elementwise_rules="lanecse+bf16vec+rolling+divc+indexed+permrows+embunroll+rowcopy+acopy",
             graph_rules="cse+depth_reads+castshare")
```

`update_schedule`, `attention_rules`, the `flat+tma+rotate+claim+sepi`
products, `matrix_bodies="exact"` and the `update`, `fsload`, `fsspread` and
`fsdeepx` accumulation tokens are earlier Latch rules. Their descriptions are
in the `CapturedModule.build` docstring, in the option modules under
`megakernel_lab.operation_compiler` (`accumulation_rules`,
`ragged_experts_lower`, `matrix_bodies`). The table below lists the tokens
this example adds. Every new token is opt-in.

With none of the new tokens selected, generated `program.cu` is unchanged for
programs that write each mutable root once and have no internal mutations.
Three changes apply to every program:

- **Effect-order fix.** Group scheduling and row fusion now order
  write-after-read and write-after-write effects on roots written more than
  once and on internal mutations. Earlier lowering could schedule such
  programs incorrectly. These programs may now be ordered differently, and row
  fusion may refuse to combine their mutable writers.
- **Header-closure staging.** When a device unit uses a transformed copy of
  `attention_long_flat.cuh`, the headers that include it are copied beside the
  unit byte for byte, so nested includes resolve the transformed copy. Native
  device-unit cache keys change for every program that uses device units.
- **Header parameters.** `ragged_experts.cuh`, `ragged_tma.cuh` and
  `ragged_claim.cuh` gained optional work-pointer, K-tail and consumer
  parameters. Default callers do not pass them, but the header bytes changed,
  so header hashes, and therefore `source_hash` and cache keys, change for
  programs that include these headers. A few default device functions may also
  compile to different SASS.

"Bitwise" means the token changes the program but not a single result bit.
"Practical" means the token reassociates floating-point work and is accepted
only with `precision_policy="practical"`.

| option | token | what it does | values |
|---|---|---|---|
| `segmented_products` | `ktail` | Admits N-major right products with K % 64 ≠ 0 to the TMA bodies. Shared-memory k rows past K are zeroed before any MMA. Requires `tma`. | bitwise |
| | `fit` | Chooses claimed-product tiles by fewest padded columns or area (including m64n160 and a 128×64 outer tile for small task counts), and counts staged-epilogue shared memory exactly. Requires `claim`, `tma`. | bitwise |
| | `bsearch` | Claimed products map a ticket to its segment by a one-warp prefix scan and a binary search. N-major tiles with BN % 64 ≠ 0 are staged as whole 64-column chunks. Requires `claim`, `tma`. | bitwise |
| | `scatter` | A product read only by a bijective row unpack stores each row at its unpacked position, with the bijection check. Otherwise it falls back to a checked row scatter. Requires `claim`, `tma`. | bitwise |
| | `route` | Routing counts and the stable order run on every CTA. | identical integers |
| | `packsum` | An FP32 sum whose only reader is a routing-pack fixed-k sum is evaluated in that gathered load and never stored. | bitwise |
| | `kcat` | Sibling claimed products that are summed (expert up/gate, attention q/k/v input gradients) run as one K-concatenated product with one BF16 rounding. Requires `bsearch`. | practical |
| `matrix_bodies` | `skinny` | Skinny explicit-BF16 products: split-K on an 8-stage cp.async ring, and K ≤ 16 products with 8 outputs per item and 16-byte stores. | bitwise |
| | `skinnybatch` | Same-geometry skinny splits of one interval run up to 8 per body. Small batches carry the interval's K ≤ 16 products on spare warps. Requires `skinny`. | bitwise |
| | `shortk` | A K ≤ 2 BF16 product with one identity reader is evaluated at that reader and never stored. Requires `skinny`. | bitwise |
| | `splitctr` | Split-K arrival counters in dependency-scheduled compose programs use entry-zeroed per-launch words instead of a zero phase. | bitwise |
| `accumulation_rules` | `rilp` | Exact thread-per-result reductions of 1 to 16 leaves evaluate 2 to 8 results per thread per round, issuing every leaf load before any store. | bitwise |
| | `rinl` | An exact reduction of at most 16 leaves whose FP32 summand is a single-reader elementwise value evaluates that value's expression at each leaf instead of storing it. | bitwise |
| | `rleaf` | A chained row stage recomputes a per-element leaf from one kept register operand and at most two loads, instead of keeping each leaf in its own register array. | bitwise |
| | `rhoist2` | Warp-per-row groups hoist external reads up to 128 registers per thread instead of 96. Takes effect only where the execution policy's group features include `rhoist`. | bitwise |
| | `rsum1` | A one-leaf exact reduction read only at its own element index is evaluated in its readers and never stored. | bitwise |
| | `rwarp` | Large exact last-axis row reductions run warp-per-row and replay the same tree. | bitwise |
| | `rstage` | Row-chain groups store after the last stage. The leaf loops of up to 4 stages are issued before their trees. | bitwise |
| | `ridx` | Index algebra in warp-per-row bodies becomes per-row constants. | bitwise |
| | `rearly` | Readers of a row group's prologue values become epilogue candidates of that group. | bitwise |
| | `rremat` | A row-scaled FP32 group value read only at identity is recomputed by its readers. | bitwise |
| | `rslot` | Slot-axis token-row groups over [tokens, slots, width] run as one warp-per-token-row phase that replays the slot and row trees. | bitwise |
| | `rnarrow` | Rows narrower than a warp are packed several per warp (softmax, log-softmax and their gradients, argmax). | bitwise |
| | `rmid` | Softmax rows of width 33 to 256 that are not a multiple of 32 use the warp-per-row form. | bitwise |
| `segmented_attention` | `folded_forward` | The width-32 forward uses 128-row items, folds the softmax scale into ex2 and reads Q by wgmma from shared memory. | practical |
| | `fused_backward` | Single-pass width-32 backward: the dK/dV items also form dS and add dQ tiles by bulk reductions. The dQ summation order is nondeterministic. | practical |
| | `claim` | Attention items are claimed from an entry-zeroed ticket counter, one ticket ahead. | bitwise |
| | `rotary` | Rotary pack/unpack around segmented attention becomes one row pass, one thread per (row, head). | bitwise |
| `tile_readiness` | `attention` | Attention items publish per-128-row-block counts. Grid barriers whose crossing reads are row-local become waits in the reading products and loops. | bitwise |
| `vocab_rules` | `stream` | The logits product streams the vocabulary weights through a 4-stage ring with resident X rows and a TMA-store epilogue. | bitwise |
| | `views` | Vocabulary-head products address offset views inline. | bitwise |
| | `rows` | The statistics and logits-gradient phases fuse into one staged pass per row (reassociated FP32 exp-sum). Applies to vocabularies of at most 112,000 padded columns; `stream` applies to K ≤ 512. | practical |
| `elementwise_rules` | `lanecse` | float4 lanes share common row-index divisions. | bitwise |
| | `bf16vec` | Contiguous BF16 operands of float4 bodies are loaded as one 8-byte vector. | bitwise |
| | `rolling` | Unroll-1 float4 loops reload each operand for the next index right after its last use. Unrolling is capped by operand registers. | bitwise |
| | `divc` | IEEE division by an exhaustively verified literal becomes a branch-free inline sequence. The shipped table holds the literals verified so far (3, 5, 6, 7, 480, √480, 16380); every other literal keeps its IEEE division. | bitwise |
| | `indexed` | `select_backward`/`slice_backward` read only at identity are evaluated by their readers. | bitwise |
| | `permrows` | Permutation embedding-gradient bodies run warp-per-rows with many loads in flight and a deferred bijection check. | bitwise |
| | `embunroll` | The scalar embedding gather keeps 8 elements per thread in flight. | bitwise |
| | `rowcopy` | `permutation_gather` copies 16-byte row runs and checks the bijection once per row. | bitwise |
| | `acopy` | Large cat/`slice_backward` copies are assembled into 16-byte vectors. | bitwise |
| `graph_rules` | `cse` | Exact CSE of identical pure operations in the captured graph. | bitwise |
| | `depth_reads` | Recognizes softmax depth-read mixtures followed by RMSNorm. It recomputes the normalized value in front of each backward reader and stacks the forward logit GEMVs that share an operand into one product. | bitwise |
| | `castshare` | A row gather read only by identical BF16 casts becomes one table cast and one BF16 gather. | bitwise |
| `device_unit_inline` | `"isolate_all"` | Extends `"isolate"`: every entry-resident warpgroup-MMA engine form the program emits is also kept out of the entry unit, and offset-view preludes are accepted. | compile only |
| `device_unit_max_registers` | `255` | Adds `-maxrregcount=255` to every device unit's compile. | compile only |

## Results

These were measured with the research build of these programs, which selected
the same rules through environment switches instead of the options above. The
programs this example builds are identical to it (see Program identity).
All timings come from one balanced job on one H100 80 GB that alternated the
arms and covered an initial and a trained state. The authors' arm ran 48 timed
updates per round and every other arm ran 24.

| execution | mean ms/update | median | authors ÷ arm (mean) |
|---|---:|---:|---:|
| Authors' unchanged code (upstream `9c25256f`, B4 × 16, DDP) | 3,279.8 | 3,234.6 | 1.00× |
| Tuned PyTorch: authors' code + streamed vocabulary head, `gc.freeze`, no single-rank DDP, B8 microbatches | 1,924.3 | 1,923.9 | 1.70× |
| **Latch, this option set (7 `accumulate` + 1 `final` launch per update)** | **1,552.6** | **1,552.2** | **2.11×** |

- Latch is 2.11× faster than the authors' implementation on means and 2.08×
  on medians. It is 1.24× faster than the tuned PyTorch control. It is not 2×
  faster than a tuned conventional implementation.
- **Learning:** 300-update paired loss checks against the conventional
  control pair passed, both from initialization and from a trained checkpoint.
  The Latch run differed from the control mean by +0.024 and +0.010 millinats.
  The control's own run-to-run differences were +0.020 and +0.011 millinats.
  A learning-rate change of ×0.9 moves the loss by about 3.9 millinats.
- With `fused_backward`, runs are nondeterministic at about 1e-4 in the loss
  because of atomic dQ accumulation.

## Program identity

The programs this example builds were compared with the measured research
programs. `program.cu`, every device unit and the SASS of every GPU function are
identical (after renaming some generated identifiers), and so are the memory
layout and arena size. With CUDA 12.8 and torch 2.9.1, `build.json` reports `source_hash`
`b9d0363bbad032acf2266183bb41690733023f1b4f865864d64503f6fc3e0027` (accumulate) and
`2c8dc08ab52c804cd9b118264b1e83556a5d5fb09b67c6442f112e7408e7f2e8` (final).

`quickstart.sh ir-hash` prints the sha256 of each captured graph, also recorded
in `build.json` as `ir_sha256`. It leaves out two fields of
`metadata["optimizer"]`, `module` and `source_sha256`, which record where the
update function was written rather than what was captured.

## Limitations

- `sm_90` (H100) only.
- The `segmented_attention` rules and `tile_readiness` apply only to width-32
  heads (Nestor's 30-wide heads are zero-padded to 32). The packed segmented
  attention op itself accepts heads of width 32 or 64 and pads narrower heads
  to the next native width.
- One static microbatch shape per program. Context and batch are fixed at
  capture.
- `kcat`, `folded_forward`, `fused_backward` and `rows` reassociate
  floating-point work and require `precision_policy="practical"`.
  `fused_backward` is nondeterministic.
- Only the warm-up part of the upstream learning-rate schedule is staged
  (`nestor_optimizer.schedule`).
- Upstream Nestor is unlicensed and not vendored. The inventory stage needs
  your own checkout and its environment.

## Files

| file | role |
|---|---|
| `quickstart.sh` | the stages above |
| `nestor_inventory.py` | exports the upstream step-0 state, parameter tags and optimizer groups (runs in the upstream environment) |
| `nestor_model.py` | `CapturedNestor`: the training forward in Latch-capturable operations, with upstream parameter names |
| `nestor_optimizer.py` | MuonAdamW, global clipping and accumulation as one Latch `tensor_update` per program mode |
| `nestor_build.py` | capture and build with the option set above; writes `build.json` |
