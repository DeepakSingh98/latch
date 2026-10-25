# License and attribution review


The copyright owner selected MIT for original Latch code on October 25,
2026. The root LICENSE and NOTICE apply to that code; upstream notices remain
intact. This document records the third-party source review.

## Included adaptations

Reviewed scope: the exported `operation_compiler`, `fast_examples` and `tinymoe`
packages, operations/researcher/TinyMoE examples, and operations tests. Excluded
MoE4 and legacy frozen runtime bundles are not prerequisites of this package.

| Upstream | Retained material | Notice and verification |
|---|---|---|
| [llm-random, 810e6451a402f9a37c2f6f43c1bb75bc00f37140](https://github.com/llm-random/llm-random/tree/810e6451a402f9a37c2f6f43c1bb75bc00f37140) | Adapted expert-choice model, initialization and conventional-control recipes | Apache-2.0; exact license retained in `provenance/upstream/LLM_RANDOM_LICENSE.txt` and beside both model adaptations. `UPSTREAM.md` documents changes. Pinned root tree checked: no root `NOTICE`. |
| [Muon, f98f1cacc0263b04290753e32be8d498c1efc806](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806) | Muon optimizer recipes and adapted implementations | Exact MIT notice in `provenance/upstream/MUON_LICENSE.txt`; copyright and permission text preserved. |
| [PyTorch v2.9.1](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE) | Explicitly attributed arithmetic/reduction adaptations in compiler Python and CUDA code | Full upstream root license retained in `provenance/upstream/PYTORCH_LICENSE.txt`, including all listed copyright holders and conditions. |
| [Random123](https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h) | Philox4x32-10 algorithm reference | Compiler source declares an independently written implementation; no upstream source/header is bundled. Algorithm attribution retained. |

Upstream license URLs, exact bytes/hashes and checks are recorded in
`provenance/upstream/V0_LICENSE_SOURCES.json`. License files are included in wheel
metadata through `pyproject.toml`. The FFN attribution and Apache license are
also package data. Dependencies installed separately retain their distributions'
licenses and notices; this review does not replace those notices.

## Models and data

The export contains no pretrained weights, training text, corpus shards or
checkpoints. It includes configuration, source identity and preparation helpers.
Downloading public assets does not change their terms.

- The [TinyMoE model card at revision 9b209f7439aadfac282e0bf44f2fda19771aba90](https://huggingface.co/FlameF0X/TinyMoE-200m-2x16/blob/9b209f7439aadfac282e0bf44f2fda19771aba90/README.md)
  declares MIT. That upstream declaration is recorded, not independently vouched
  for as the complete provenance of its training data or weights.
- The [WikiText card at revision b08601e04326c79dfdd32d625aee71d232d685c3](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
  lists CC-BY-SA-3.0 and GFDL in metadata, while its prose says CC-BY-SA-4.0.
  This discrepancy remains unresolved; do not infer an unambiguous redistribution
  grant from the helper. No WikiText text is included in the export.
- The [FineWeb-Edu card at revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md)
  specifies [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and
  [Common Crawl terms](https://commoncrawl.org/terms-of-use). ODC-By covers the
  database rather than all rights in individual contents. Attribution to
  Lozhkov, Ben Allal, von Werra and Wolf (2024), Hugging Face, and underlying web
  authors/Common Crawl is preserved in the reproduction corpus manifest.
  Its compact document index contains coordinates and split labels only.
- Other preparation helpers record upstream dataset identities and attribution.
  Their datasets are downloaded separately; inclusion of a helper is not a
  representation that every upstream document has one uniform license.

## Package completeness review

Compiler and fast-example imports were checked for dependencies on excluded
project namespaces. TinyMoE's only required MoE4 helper, `WeightEMA`, was copied
unchanged into `tinymoe/weight_ema.py`, and its three imports now use the retained
namespace. `provenance/V0_WEIGHT_EMA.json` records the exact source and hash.
Runtime source identity changes remain explicit; no old checkpoint compatibility
or arbitrary native-cache relocation is implied.

The compiler's `kernels/peer_transport.cu`, CUDA headers and numerical-policy
JSON files must remain package data. Keep the full operations example helper
folders and researcher configuration files: their relative imports and data
loads require them. Preserve all adjacent source/license attribution. Optional
learning plots require the `learning-plots` dependency extra.

This review is limited to source inspection and notice verification. Fresh
installed-wheel/sdist correctness results are recorded separately in
`VALIDATION.md`; historical donor qualification does not validate the integrated
release.

## Export review findings

The initial validation-input-01 export was inspected before final documentation
and test-fixture cleanup. Runtime imports no longer require the excluded MoE4
package. Its operations tests still included three MoE4-only dependencies:
`integration/V0_PUBLIC_TEST_SCOPE.json` records their disposition, including the unchanged
small softmax audit fixture extracted to retain compiler coverage. The pure
MoE4 oracle test and two standalone historical harnesses with unbundled helper
imports are excluded from the public test scope; their originals remain in
donor history. No runtime behavior or numerical gates were changed.

The final export must retain `examples/operations/moe_support/UPSTREAM.md`, the
adjacent license, EMA provenance and this review. Initial scans found historical
source labels such as `/private/tmp/megakernel-*`, which are provenance rather
than imports or required runtime paths. The exporter separately rejects private
account identifiers and private-key material. Static review is not a guarantee
that arbitrary sensitive strings can be recognized by a pattern scan.
