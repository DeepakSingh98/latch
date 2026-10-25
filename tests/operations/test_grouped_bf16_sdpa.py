"""Independent geometry/VJP references and fake native capture for grouped SDPA.

The mathematical tests are deliberately FP64 and tiny: they verify padding,
head membership and the resulting VJP independently of CUDA attention rounding.
GPU primal/VJP qualification is a separate required check of the emitted body.
"""
from dataclasses import replace
import math

import pytest
import torch

from megakernel_lab.operation_compiler import bf16_sdpa
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.standard_ops import causal_mask_constant, causal_constant_key


@pytest.mark.parametrize("batch,query_heads,kv_heads,width,window", [
    (1, 8, 4, 48, None), (2, 6, 2, 40, 4), (1, 4, 1, 64, 1)])
def test_independent_padding_gqa_vjp(batch, query_heads, kv_heads, width, window):
    torch.manual_seed(2981)
    sequence = 9
    q = torch.randn(batch, query_heads, sequence, width, dtype=torch.float64, requires_grad=True)
    k = torch.randn(batch, kv_heads, sequence, width, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cotangent = torch.randn_like(q)
    scale = 1.0 / math.sqrt(width)
    bias = causal_mask_constant(sequence, sequence, torch.float64, "cpu", sliding_window=window)
    operands = bf16_sdpa._expand_grouped_heads(q, k, v, True)
    qp, kp, vp = (bf16_sdpa._pad_head_width(x) for x in operands)
    padded = (torch.softmax((qp @ kp.transpose(-2, -1)) * scale + bias, -1) @ vp)[..., :width]
    # Separate per-query-head loops identify the corresponding original KV
    # head directly, without the candidate's expand/reshape or padding.
    rows = []
    for head in range(query_heads):
        kv = head // (query_heads // kv_heads)
        scores = (q[:, head] @ k[:, kv].transpose(-2, -1)) * scale + bias
        rows.append(torch.softmax(scores, -1) @ v[:, kv])
    expected = torch.stack(rows, 1)
    actual_grads = torch.autograd.grad((padded * cotangent).sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * cotangent).sum(), (q, k, v))
    torch.testing.assert_close(padded, expected, atol=2e-14, rtol=2e-14)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, atol=3e-14, rtol=3e-14)
    assert torch.count_nonzero(qp[..., width:]) == 0
    assert padded.shape == q.shape and actual_grads[1].shape == k.shape


def test_window_constant_boundaries_and_provenance():
    bias = causal_mask_constant(7, 7, torch.float32, "cpu", sliding_window=3)
    for query in range(7):
        for key in range(7):
            assert (bias[query, key] == 0).item() == (max(0, query - 2) <= key <= query)
    assert causal_constant_key(bias.clone()) is not None
    edited = bias.clone()
    edited[6, 0] = 0
    assert causal_constant_key(edited) is None
    assert causal_mask_constant(7, 7, torch.float32, "cpu", sliding_window=7) is causal_mask_constant(
        7, 7, torch.float32, "cpu")
    with pytest.raises(CaptureError, match="positive static integer"):
        causal_mask_constant(7, 7, torch.float32, "cpu", sliding_window=0)


def test_grouped_shape_refusals():
    q = torch.empty(1, 8, 7, 48)
    k = torch.empty(1, 4, 7, 48)
    with pytest.raises(CaptureError, match="multiple of KV heads"):
        bf16_sdpa._expand_grouped_heads(q, k, k, False)
    with pytest.raises(CaptureError, match="multiple of KV heads"):
        bf16_sdpa._expand_grouped_heads(q[:, :7], k, k, True)
    with pytest.raises(CaptureError, match="matching batch"):
        bf16_sdpa._expand_grouped_heads(q, k[:, :, :6], k[:, :, :6], True)
    with pytest.raises(CaptureError, match="head widths"):
        bf16_sdpa._pad_head_width(torch.empty(1, 1, 1, 65))


def grouped_fixture(*, sequence=256, window=1024):
    """Fake capture only; run capture/lowering checks on an admitted CPU worker."""
    from megakernel_lab.operation_compiler.capture import capture_training
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    parameters = {name: torch.zeros(1, heads, sequence, 48) for name, heads in (("q", 8), ("k", 4), ("v", 4))}
    cotangent = torch.zeros(1, 8, sequence, 48, dtype=torch.bfloat16)
    def objective(params, cotangent):
        y = bf16_sdpa.sdpa(*(params[name].to(torch.bfloat16) for name in ("q", "k", "v")),
                          is_causal=True, enable_gqa=True, sliding_window=window)
        return (y.float() * cotangent.float()).sum()
    rule = tensor_update(lambda p, g, s, hp: ({name: value - 0.001*g[name] for name, value in p.items()}, {}),
                         name="grouped_attention_probe")
    with bf16_sdpa.capture_bf16_sdpa():
        return capture_training(objective, parameters, None, inputs=(cotangent,), config=rule)


@pytest.mark.parametrize("window,mask_kind", [(1024, "causal"), (128, "causal_window")])
def test_grouped_native_capture(window, mask_kind):
    from megakernel_lab.operation_compiler.compose_attention import rewrite_tape
    from megakernel_lab.operation_compiler.flat_long_attention import emit, Refused
    captured = grouped_fixture(window=window)
    plan, = bf16_sdpa.plans(captured.ir).values()
    assert plan["hd"] == 64 and plan["bh"] == 8
    assert plan["group"]["scale"] == 48**-0.5
    assert plan["group"]["mask_kind"] == mask_kind
    assert tuple(captured.ir.values[captured.ir.outputs["grads"]["k"]].shape) == (1, 4, 256, 48)
    rewritten, report = rewrite_tape(captured.ir, version=7)
    assert len(report["groups"]) == 1 and report["groups"][0]["plan"]["tape_recipe"] == "bf16"
    assert report["groups"][0]["plan"]["group"]["scale"] == 48**-0.5
    assert any(op.target == "aten.sum.dim_IntList" for op in rewritten.operations)
    with pytest.raises(Refused, match="power of two|causal-window"):
        emit(plan, lambda x: x, (0, 256, 1), which="forward", recipe="bf16", version=13)
    altered = replace(captured.ir, metadata={**captured.ir.metadata, "causal_mask_constants": []})
    with pytest.raises(CaptureError, match="provenance"):
        bf16_sdpa.plans(altered)
