# Third-party notices

Latch is licensed under [MIT](LICENSE). Adapted upstream code retains
the notices and terms below.

The expert-choice model and initialization recipes in
`src/megakernel_lab/fast_examples/recovered_release/ffn/` and
`examples/operations/moe_support/` adapt
[llm-random at commit 810e6451a402f9a37c2f6f43c1bb75bc00f37140](https://github.com/llm-random/llm-random/tree/810e6451a402f9a37c2f6f43c1bb75bc00f37140),
licensed under Apache-2.0. The exact license is
`provenance/upstream/LLM_RANDOM_LICENSE.txt`, also retained as
`UPSTREAM_LICENSE.txt` beside each adaptation. The adjacent `UPSTREAM.md` records
source files and modifications. The pinned upstream root contains no `NOTICE`.

Muon optimizer implementations and recipes adapt
[Keller Jordan's Muon at commit f98f1cacc0263b04290753e32be8d498c1efc806](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806).
The MIT copyright and permission notice is preserved in
`provenance/upstream/MUON_LICENSE.txt`. Retained implementations include
`operation_compiler/muon.py`, the conventional fast-example optimizer, and the
operations example optimizer; adaptations are not represented as identical
copies of upstream code.

Several compiler operations and CUDA kernels explicitly adapt PyTorch arithmetic
and reduction order, including LayerNorm forward/backward, softmax, scalar
division and lerp. Their source comments identify the upstream algorithms.
[PyTorch v2.9.1's full root license](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE),
including its copyright holders and BSD-style conditions, is retained as
`provenance/upstream/PYTORCH_LICENSE.txt`. This notice does not imply that the
whole PyTorch source or its binary dependencies are bundled.

The compiler's Philox4x32-10 implementation acknowledges the
[Random123 algorithm](https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h).
Its source states that it was written independently; Random123 source and
headers are not bundled.

The source and wheel include the notices above. Exact upstream URLs, SHA-256
hashes and verification notes are in
`provenance/upstream/V0_LICENSE_SOURCES.json`. Independently installed packages
and toolchains retain their own licenses and notices, including NumPy, PyTorch,
Triton, Transformers, datasets, safetensors, Matplotlib, LLVM and NVIDIA CUDA.

No pretrained weights or training corpus are distributed. Public download
helpers preserve upstream identities; downloaded assets remain subject to their
own terms. The license review records the pinned TinyMoE model and WikiText
sources and a discrepancy in the WikiText card's license metadata. Attribution
and metadata are not a grant of rights in those assets.

The learning reproduction metadata attributes FineWeb-Edu to Lozhkov, Ben Allal,
von Werra and Wolf (2024), Hugging Face, and the underlying web authors/Common
Crawl. Its [pinned source card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md)
specifies [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and
[Common Crawl's terms](https://commoncrawl.org/terms-of-use). ODC-By governs
database rights and does not grant all rights in individual web documents.
`examples/tinymoe/reproduction/corpus.json` retains this attribution, exact source
revision, shard identities and preparation rules. The compact document index
contains coordinates and split labels, not document text.
