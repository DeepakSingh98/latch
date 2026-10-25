# Direct compiler API examples

These source examples use the installed `megakernel_lab.operation_compiler`
package. Start with [`../quickstart.py`](../quickstart.py) for the shortest
compile-and-train path. Install `[gpu]`; use CPU plus CUDA 12.8 for preparation
and an H100 for execution. Preserve absolute build and cache paths between hosts.

`train_mlp.py` shows module capture and AdamW; `train_conv.py` shows convolution;
`train_attention_forms.py` and `train_modified_attention.py` show attention;
`train_small_moe.py` uses the accompanying `moe_support/` adapters and notices.
The loop and dynamic-batch scripts demonstrate their respective compiler APIs.
Each script has `--help` with its actual preparation and run arguments. These
are bounded examples of supported operations, not a promise to compile arbitrary
PyTorch programs or a new measurement of historical performance.
