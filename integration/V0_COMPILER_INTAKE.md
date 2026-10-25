# Latch v0 compiler intake

## Immutable inputs and integration base

- Repaired release/base: `6127a450ce897a83eb7c7237b3c340376c2c1779`.
- TinyMoE tested product: `adde6ca52b556df8585edfb6c7ade3a241964620`.
- TinyMoE delivery: `b0bc72fa`; its compiler tree equals the tested product. Its
  corrected square-product fixture in `test_static_shared_reserve.py` is selected.
- Common ancestor: `cecf6af3f672ca767f6cec190c2400079f0070a4`.

The TinyMoE branch is not a descendant of the repaired release. V0 starts from
the repaired release and performs a file-level three-way integration of the
compiler and operation tests. No branch-wide merge or donor deletion is used.
`V0_COMPILER_INTAKE_RAW.json` records the mechanical intake; the final hash
inventory records the resolved candidate. Historical measurements retain their
original source identities. This integration requires fresh qualification.

## Accepted compiler capabilities

- Token-choice ragged expert operations, counts/order/gather/scatter, grouped
  BF16 SDPA including grouped-query accumulation and sliding causal masks:
  `a1f0523d`, plus the tested TinyMoE support at `945470f9`.
- Forward-state tensor optimizer rules and tensor-weight `lerp`, permitting
  unused-expert counters and the supervised/EMA recipe through ordinary capture.
- Explicit practical BF16 permissions, chunk proofs and balanced square-sum
  association. These remain declared numerical choices, not silent defaults.
- Checked elementwise/full-sum/pointer/pure-interval source outlining and saved
  manifest authentication from the tested product. Attention-item outlining
  continues to require explicit assembler O0: the preserved O1 failure is not
  relabeled as a supported build.
- `CapturedModule.build(..., device_units=K)` and the runtime's relocatable
  device-unit build/link/cache path: helper integration `b456da5b`, originating
  in `ada4046a..c014c62f`. Publication boundaries, barriers, WGMMA descriptor
  guards, build identity and runtime admission remain checked. This creates
  separately compiled device functions behind one cooperative kernel entry.
- Explicit static shared-memory reservation through ordinary build/lower APIs:
  `adde6ca5`. Zero preserves the previous planner; 8192 bytes is the TinyMoE B2
  choice. Actual linked static plus dynamic storage still requires admission.
- The tested compiler's existing peer transport and exact layout/body options
  accompany its source, with disabled experimental knobs left disabled. No
  newer practical-50 candidate or Granite compiler patch is imported.

## Repair protections retained during conflicts

The repaired release's `execution_plan.py`, `bf16_retention.py`,
`p6d_retention.py`, `row_fusion.py`, `gradient_audits.py`,
`optimizer_reference.py`, numerical reference module and policy JSON files are
retained. In shared files the integration preserves:

- Automatic common/retained implementation selection and explicit legacy
  diagnostic scopes (`lower.py`, `compose.py`, `api.py`).
- Explicit BF16 projection recognition and preservation of the canonical cast
  graph before optional storage rewrites.
- Carried-register substitution for split row components and TMA epilogues,
  plus refusal of dangling references to elided storage.
- Correct indexed atomic numerical-policy metadata after implementation selection.
- Self-contained numerical-reference imports and packaged numerical policies.

Ten files had mechanical conflicts. API documentation combines both contracts;
`lower.py` keeps the repaired projection bypass and accepts the tested donor's
multi-row LayerNorm extensions. Distributed nesting and WGMMA residue additions
take the tested donor form. CUDA LayerNorm additions are additive. BF16 tests
retain explicit legacy comparisons and the repaired canonical-cast expectation,
while checking the donor's explicit practical metadata. The attention-header
test accepts only the donor's documented finite-mask overflow repair; all other
legacy header bytes remain authenticated.

## Test scope and current verification

No existing repaired-release test is deleted. New codegen tests use independent
brace/pointer proof helpers extracted unchanged into
`tests/operations/fixtures/tinymoe_source_proofs.py`. Thirteen donor tests for
private research controllers, profilers, archived artifact restoration and
saved-source research harnesses are excluded; exact names and reason are in
`V0_RESEARCH_TEST_EXCLUSIONS.json`. Their original evidence remains in the donor.
The ordinary installed build path and device-unit API do not import them.

The local bounded static check parsed the compiler/operation-test Python files;
`git diff --check` passed. These are syntax checks, not numerical qualification.
Cloud CPU regressions and fresh cold-build/training validation are required for
the integrated runtime. The TinyMoE donor's two old matrix expectation failures
and corrected reserve fixture remain documented in its original gate review.

Priority regressions: release repair, execution selection, BF16 retention, row
assembly, group rules, distributed boundaries, ragged experts, grouped BF16
SDPA, forward-state optimizer, tensor lerp, device-unit intervals, static shared
reserve/API, source outlining, attention build policy and TinyMoE product tests.

## Candidates explicitly deferred

### Practical-50

Branch `codex/latch-practical-50` diverges from TinyMoE at
`ee334541c9e0dffd6271b6d55085991ef8edf68a`. Its validated final compiler is
`9a4c8c66`; the validated best short-K candidate is `2e7641c8` (artifacts built
at `7e6632af`). Its `research/latch_practical_50/RESULTS.md` records a distinct
four-H100 E32/H2 B32/T2048 practical-BF16 workload: 60.829 ms weighted update
clock, with exact equivalence to its 14/14-qualified m13u twin. It did not reach
50 or 60 ms. This does not qualify those bytes for TinyMoE or the repaired
release's strict examples.

Deferred: BF16 segment widening/row pulls (`209677d`, `57f511d`), SDPA layout
absorption (`9a3ca0b`, `b13f6db`, `07050b9`), skinny router tiles (`e738b1a`),
async exchanges/SUM/owner broadcast and sharded checkpoint changes, fused
vocabulary stats/gradient, and short-K K-major expert defaults. These are useful
completed improvements but add independent layout, distributed execution and
arithmetic qualification scope to a final packaging lane. The three-line
fused-warp rejection (`5c212044`) applies to that lane's absent fused-warp
implementation and has no corresponding v0 site to repair. Rejected multicast,
copy-stream, warp-count and warp-specialized screens remain deferred too.

### Granite

Delivery `33b4dd3d`, model/program source `b6511e25`, starts from TinyMoE
delivery `b0bc72fa`. It is slower than compiled grouped conventional execution
and has not passed its frozen numerical gates. Keep it as a documented limit,
not a qualified example. Its candidate reusable compiler patches are deferred:
integer histogram counts (`7d50a5a7`), static BF16 logits scale (`f939e998`) and
warp-coalesced row permutation (`567aef16`, unused by final Granite). Each needs
its own focused release tests and new numerical qualification; none is needed
for the pinned TinyMoE quickstart.
