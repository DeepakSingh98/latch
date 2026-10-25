"""Opt-in TinyMoE training precision policies, shared by the Latch adapter and the conventional control.

``release`` (default, ``None``) is the released arithmetic: every product BF16 operands / FP32 accumulation / one
BF16 result rounding, FP32 masters, residuals, normalization, softmax, loss and optimizer.

``readout_split2`` (LATCH-TRAIN-PRECISION) changes one product: the vocabulary readout (final and exit heads) uses
``operation_compiler.split_precision.split_bf16_linear(h, lm_head.weight, weight_terms=2)``. The readout weight is
carried as two BF16 terms (hi + lo; representation error ~2^-16 instead of ~2^-8), in the logits product and in its
input-gradient product; logits are still one BF16-rounded product result, the weight gradient is the released
expression, and every other operation is unchanged. Selected by attribution of the trained-state gradient error
(research/latch_train_precision): the BF16 rounding of the readout weight was the largest single source.

The same policy object drives both backends, so a comparison under the policy is a comparison of executors with
identical declared arithmetic. Evaluation (held-out NLL) is unchanged by these training policies.
"""
from __future__ import annotations

POLICIES = ("release", "readout_split2", "readout_split3", "forward_weights_split2", "forward_weights_split2+readout_split2",
            "forward_weights_split2+readout_split3", "experts_split3", "experts_split3+readout_split3",
            "forward_split3", "forward_split3+readout_split3", "forward_fp32", "forward_fp32+readout_split3")


def components(policy):
    return set(validate(policy).split("+")) - {"release"}


def validate(policy):
    policy = "release" if policy is None else policy
    if policy not in POLICIES:
        raise ValueError(f"unknown TinyMoE training precision {policy!r}; known: {POLICIES}")
    return policy


def readout_terms(policy):
    parts = components(policy)
    return 2 if parts & {"readout_split2", "readout_split3"} else 1


def readout_input_terms(policy):
    return 2 if "readout_split3" in components(policy) else 1


def forward_weight_terms(policy):
    return 2 if components(policy) & {"forward_weights_split2", "forward_split3", "forward_fp32"} else 1


def forward_input_terms(policy):
    return 2 if components(policy) & {"forward_split3", "forward_fp32"} else 1


def forward_fp32(policy):
    """forward_fp32: the split forward products of q/k/o, router and experts keep unrounded FP32 results."""
    return "forward_fp32" in components(policy)


def readout_linear(policy):
    """Callable (hidden [rows, width] FP32, weight [vocab, width] FP32) -> BF16 logits under ``policy``:
    readout_split2 = split weight in the logits and input-gradient products (split_bf16_linear); otherwise
    forward_weights_split2 = split weight in the logits product only (split_forward_linear); else released."""
    from megakernel_lab.operation_compiler.split_precision import split_bf16_linear, split_forward_linear
    if readout_terms(policy) > 1:
        return lambda hidden, weight: split_bf16_linear(hidden, weight, weight_terms=readout_terms(policy),
                                                        input_terms=readout_input_terms(policy))
    terms, inputs = forward_weight_terms(policy), forward_input_terms(policy)
    return lambda hidden, weight: split_forward_linear(hidden, weight, weight_terms=terms, input_terms=inputs)


def describe(policy):
    policy = validate(policy)
    if policy == "release":
        return dict(policy="release", products="BF16 operands, FP32 accumulation, one BF16 result rounding")
    from megakernel_lab.operation_compiler.split_precision import describe as split
    parts = components(policy)
    out = dict(policy=policy, components=sorted(parts))
    if "readout_split2" in parts:
        out["readout"] = dict(split(2), scope="vocabulary readout: logits product and its input-gradient product")
    if "readout_split3" in parts:
        out["readout"] = dict(policy="split_bf16_x3", weight_terms=2, input_terms=2,
                              forward="x_hi.w_hi + x_hi.w_lo + x_lo.w_hi in one BF16 product over 3K, FP32 "
                                      "accumulation, one BF16 rounding of the logits",
                              input_gradient="dy.w_hi + dy.w_lo (widened, FP32 sum)",
                              weight_gradient="dy^T.x_hi + dy^T.x_lo (widened, FP32 sum)",
                              scope="vocabulary readout (final and exit heads)", cost="3x K of the readout products")
    if "forward_weights_split2" in parts:
        out["forward_weights"] = dict(weight_terms=2, scope="every forward product reading a master weight (attention "
                                      "q/k/v/o, router, experts w1/w2/w3" + ("" if parts & {"readout_split2", "readout_split3"} else
                                      ", readout") + "); backward products released (BF16 hi weight)")
    if "forward_split3" in parts:
        out["forward_products"] = dict(
            scope="every forward product reading a master weight: attention q/k/v/o, router, experts w1/w2/w3"
                  + ("" if parts & {"readout_split2", "readout_split3"} else ", readout"),
            forward="weight as BF16 hi + lo terms; an FP32 product input (q/k/v, router, w1/w3) also as hi + lo terms: "
                    "x_hi.w_hi + x_hi.w_lo + x_lo.w_hi in one BF16 product over 3K (BF16 inputs: o_proj, w2: 2K); "
                    "FP32 accumulation, one BF16 result rounding",
            backward="released: dx = dy . bf16(w), dw = dy^T . bf16(x) (widened by the cast VJPs)",
            unchanged="attention scores/softmax/values (BF16 Flash), elementwise, routing, norms, loss, optimizer",
            cost="3x (FP32 inputs) or 2x (BF16 inputs) K extent of the forward products; backward unchanged")
    if "forward_fp32" in parts:
        out["forward_products"] = dict(
            scope="forward products of attention q/k/o, router and experts w1/w3/w2 (FP32 results); attention v "
                  "(split operands, BF16 result)" + ("" if parts & {"readout_split2", "readout_split3"} else ", readout"),
            forward="weight as BF16 hi + lo terms; an FP32 product input (q/k/v, router, w1/w3) also as hi + lo terms: "
                    "x_hi.w_hi + x_hi.w_lo + x_lo.w_hi over 3K (BF16 inputs: o_proj, w2: a.w_hi + a.w_lo over 2K); "
                    "BF16 operands, FP32 accumulation; q/k/o/router/experts store the FP32 accumulators unrounded, v "
                    "rounds its result to BF16",
            expert_elementwise="s = bf16(silu(up)); a = bf16(s * gate) (the released two roundings; up/gate FP32)",
            backward="released: the FP32 cotangent of an FP32-result product is rounded once to BF16, then dx = g . "
                     "bf16(w), dw = g^T . bf16(x) (widened by the cast VJPs)",
            unchanged="attention scores/softmax/values (BF16 Flash, BF16 q/k/v inputs), routing, norms, loss, optimizer",
            cost="3x (FP32 inputs) or 2x (BF16 inputs) K extent of the forward products, FP32 result storage; "
                 "backward unchanged")
    if "experts_split3" in parts:
        out["experts"] = dict(scope="every expert product (w1, w3, w2), forward and backward",
                              weight_terms=2, input_terms="2 for w1/w3 (FP32 input), 1 for w2 (BF16 activation)",
                              forward="w1/w3: x_hi.w_hi + x_hi.w_lo + x_lo.w_hi over 3K; w2: a.w_hi + a.w_lo over 2K; one "
                                      "BF16 result rounding each",
                              input_gradient="sum over weight terms of widened BF16 products, summed in FP32",
                              weight_gradient="sum over input terms of widened BF16 products, summed in FP32",
                              cost="3x (w1/w3) and 2x (w2) K extent of the expert products, forward and backward")
    out["unchanged"] = "product result roundings, attention, routing, norms, loss, optimizer" + (
        "" if parts & {"experts_split3", "readout_split3"} else ", activation casts")
    return out
