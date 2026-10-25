"""CPU-worker scalar-FP64 admission and full optimizer capture checks."""
import copy
import json

import torch

from megakernel_lab.operation_compiler.capture import capture_training
from megakernel_lab.operation_compiler.inference import capture_inference
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from megakernel_lab.operation_compiler.mixed_fma import TARGET, mixed_fma64
from .test_lower_safety import _ref, _value
from megakernel_lab.operation_compiler.user_optimizer import _check_state_tensor


def fixture():
    return dict(inputs=["a", "b", "c", "alpha", "beta"], outputs=["out"],
        values=dict(a=_value((2, 3)), b=_value((2, 3)), c=_value(()),
                    alpha=_value((), "float64"), beta=_value((), "float64"), out=_value((2, 3))),
        operations=[dict(name="out", target=TARGET,
            args=[_ref(x) for x in ("a", "b", "c", "alpha", "beta")], kwargs={})])


def run():
    checks = []
    for schedule, options in (("auto", {}), ("compose", dict(execution="compose")),
                              ("reference", dict(policy="reference"))):
        source = lower(fixture(), **options).source
        assert "__fma_rn(" in source and "__dmul_rn(" in source and "double*" in source
        checks.append(dict(schedule=schedule, admitted=True))
        for key, replacement in (("alpha", _value((1,), "float64")),
                                  ("beta", _value((), "float32")),
                                  ("b", _value((1, 3))), ("out", _value((2, 3), "float64"))):
            bad = copy.deepcopy(fixture()); bad["values"][key] = replacement
            try:
                lower(bad, **options)
            except LoweringError:
                checks.append(dict(schedule=schedule, refused=True))
            else:
                raise AssertionError("malformed mixed_fma64 metadata admitted")
    _check_state_tensor("coefficient", torch.tensor(.1, dtype=torch.float64))
    for value in (torch.ones(1, dtype=torch.float64), torch.tensor(float("nan"), dtype=torch.float64)):
        try:
            _check_state_tensor("coefficient", value)
        except CaptureError:
            checks.append(dict(state_scalar_policy_refused=True))
        else:
            raise AssertionError("invalid FP64 optimizer state admitted")
    def function(_parameters, a, b, c, alpha, beta):
        return mixed_fma64(a, b, c, alpha, beta)
    args = (torch.zeros(2, 3), torch.ones(2, 3), torch.ones(()),
            torch.tensor(.9, dtype=torch.float64), torch.tensor(.1, dtype=torch.float64))
    captured = capture_inference(function, {}, args)
    assert sum(op.target == TARGET for op in captured.ir.operations) == 1
    assert "__fma_rn(" in lower(captured.ir, execution="compose").source
    checks.append(dict(fake_capture=True, dynamic_scalar64=True))

    from .fixtures import moe4_optimizer_fixture as optimizer
    model = torch.nn.Linear(3, 2)
    parameters = dict(model.named_parameters())
    rule = optimizer.make_rule(model)
    def loss(p, cotangent):
        return (p["weight"] * cotangent).sum() + p["bias"].sum()
    trained = capture_training(loss, parameters, None, (torch.ones(2, 3),), config=rule)
    result = lower(trained.ir, execution="compose")
    assert "__fma_rn(" in result.source
    keys = set(trained.ir.metadata["optimizer_state_names"])
    assert optimizer.DIAGNOSTIC_KEYS.issubset(keys) and optimizer.ADAM_DECAY in keys
    assert trained.ir.values[next(v.name for v in trained.ir.values.values()
        if v.role == "optimizer_state" and v.parameter == optimizer.ADAM_DECAY)].dtype == "float64"
    checks.append(dict(full_adam_rule_capture=True, actual_clip_outputs=True, scalar64_state_commit=True))
    return dict(passed=True, checks=checks, contract="FP32 final store; explicit left FP64 FMA; scalar-only FP64 storage")


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(run(), indent=2))
