"""LATCH-TRAIN-CORRECTNESS: vector_resident output ownership on a weight-updating loop (CPU; lowering only).

The released GradLev loop (examples/gradlev) rewrites its fast weights every iteration and predicts a replicated
per-token NLL. Two variants that only change what leaves the loop used to compile and return NaN on the GPU:
a carried value partitioned with W1 over its m shards (x W1 + b1), and a prediction partitioned over the vocabulary
(head logits in the output region). The first is now refused before execution; the second is stored by its owner
CTAs. The released form keeps its CTA-0 store of the replicated prediction.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "gradlev"))

DIMS, T = (768, 3072, 50257), 16


def _capture(output="released"):
    from gradlev_example.deploy import head_logits
    from gradlev_example.latch_loop import example, loop_functions
    from megakernel_lab.operation_compiler.inference import capture_loop_inference
    params, state, inputs = example(1, T, "cpu", None, DIMS)
    base_init, base_body, out = loop_functions(T)
    init, body = base_init, base_body
    if output == "carry-z":
        def init(p, X, U, TGT, E1, E2):
            return dict(base_init(p, X, U, TGT, E1, E2), z=torch.zeros(X.shape[1], DIMS[1]))

        def body(p, carry, st, X, U, TGT, E1, E2):
            x = X.index_select(0, carry["t"].reshape(1))[0]
            z = torch.bmm(x[:, None, :], st["W1"])[:, 0] + p["b1"]
            new, st2 = base_body(p, {"t": carry["t"], "nll": carry["nll"]}, st, X, U, TGT, E1, E2)
            return dict(new, z=z), st2

        def out(p, carry, *a):
            return carry["nll"], carry["z"]
    elif output == "logits":
        def out(p, carry, X, U, TGT, E1, E2):
            last = X.index_select(0, (carry["t"] - 1).reshape(1))[0]
            return carry["nll"], head_logits(last, p["E"], "mixed")
    return capture_loop_inference(init, body, out, params, inputs, torch.tensor(T), max_steps=T, state=state)


def _lower(cap):
    from megakernel_lab.operation_compiler.inference import lower_inference
    return lower_inference(cap.ir, loop_schedule="vector_resident")


def test_partitioned_carry_is_refused_before_execution():
    from megakernel_lab.operation_compiler.lower import LoweringError
    with pytest.raises(LoweringError, match="loop-carried value .* partitioned"):
        _lower(_capture("carry-z"))


def test_partitioned_prediction_is_stored_by_its_owners():
    released = _lower(_capture("released")).source
    logits = _lower(_capture("logits")).source
    # The replicated NLL keeps its single CTA-0 store; the logits store has no CTA-0 guard.
    assert released.count("if (cta == 0) reinterpret_cast") == 1
    assert logits.count("if (cta == 0) reinterpret_cast") == 1
