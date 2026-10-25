# Examples

Follow [setup](../docs/SETUP.md), then run source examples from the extracted
release root. Compile on CPU with CUDA 12.8; run generated programs on H100
80 GB. Preserve absolute artifact/cache paths between hosts.

| Example | Purpose | Extra |
|---|---|---|
| [MLP quickstart](../README.md) | Compile a training update; change width and learning rate | `gpu` |
| [TinyMoE](tinymoe/README.md) | Training, evaluation, conventional control and restart | `tinymoe` |
| [Selective updates](selective_update/README.md) | Update experts every 16 microbatches | `tinymoe` |
| [Looped TinyMoE](smelt_looped/README.md) | Train with shared-weight recurrence | `tinymoe` |
| [GradLev](gradlev/README.md) | Adapt GPT-2's final MLP after each token | `gradlev` |
| [Editable learning program](researcher/README.md) | Byte transformer, intermediate supervision and recurrence | `gpu` |
| [Direct compiler API](operations/README.md) | Smaller operation examples | `gpu` |
| [Fast examples](../docs/FAST_EXAMPLES.md) | Compare complete updates and audit numerical results | `gpu` |

MLP and the default editable learning program use synthetic data. Other guides
identify their downloads and [numerical limitations](../docs/NOTES.md).
