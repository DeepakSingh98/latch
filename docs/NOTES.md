# Known limitations

This preview targets one H100 and a supported set of PyTorch operations. New
architectures may need compiler support. Distributed training is not included.

- **TinyMoE's default 2 × 512 recipe:** the short-run checks pass, but long-trained
  states can fail the BF16 gradient audit. Use the supplied checks when extending
  a run. This is separate from the tuned 2 × 1,024 results.
- **Selective updates:** the slow expert group's AdamW beta1 correction is
  nonstandard. Starting from zero optimizer state can differ from the conventional
  implementation. The example guide describes the rule.
- **GradLev:** adapts GPT-2's final MLP. The rest of the model runs in PyTorch.
  Agreement with the eager reference is approximate; Latch restart is exact.
- **Checkpoints and builds:** keep the generated binary, runtime and absolute
  artifact/cache paths together. Rebuild after changing the compiler.

The fresh Linux/H100 installation check passed the MLP, TinyMoE and GradLev
workflows, including their numerical and restart checks. The bundled CPU test
suite still has ten known failures in matrix admission, expression inlining and
LayerNorm plan assertions. It is not a fully passing suite.

For the optional CPU tests, install the `test` and `tinymoe` extras together;
some tests need Transformers:

```sh
python -m pip install '../megakernel_lab-0.0.0-py3-none-any.whl[test,tinymoe]'
CUDA_VISIBLE_DEVICES= python -m pytest tests
```
