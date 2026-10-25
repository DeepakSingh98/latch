"""CPU-worker admission/capture checks; GPU same-input probe tests arithmetic."""
import copy
import json

import torch

from megakernel_lab.operation_compiler.inference import capture_inference
from megakernel_lab.operation_compiler.capture import capture_training
from megakernel_lab.operation_compiler.lower import lower, LoweringError
from megakernel_lab.operation_compiler.scaled_add import TARGET, scaled_add
from .test_lower_safety import _value, _ref
from megakernel_lab.operation_compiler.user_optimizer import tensor_update


def fixture():
    return dict(inputs=["input", "other", "alpha"], outputs=["out"],
        values=dict(input=_value((2, 3)), other=_value((2, 3)), alpha=_value(()), out=_value((2, 3))),
        operations=[dict(name="out", target=TARGET,
                         args=[_ref("input"), _ref("other"), _ref("alpha")], kwargs={})])


class _CPUPrimalOracle(torch.fx.Interpreter):
    """Execute a differentiated fake graph without pretending CPU add is FMA.

    Only the opaque primal is substituted. Its two FP32 products fit exactly
    in FP64 for these small dyadic fixtures; every emitted VJP op executes
    unchanged. Native CUDA forward rounding has a separate measured receipt.
    """
    def call_function(self, target, args, kwargs):
        if str(target) == TARGET:
            input, other, alpha = args
            return (input.double() + other.double() * alpha.double()).float()
        return super().call_function(target, args, kwargs)


def differentiated_checks():
    input = torch.tensor([[.5, -1., 2.], [4., .125, -.25]])
    other = torch.tensor([[2., -.5, .25], [1., -4., 8.]])
    cotangent = torch.tensor([[1., -2., 3.], [.5, -1., 2.]])
    def objective(parameters, cotangent):
        return (scaled_add(parameters["input"], parameters["other"],
                           parameters["alpha"]) * cotangent).sum()
    parameters = dict(input=input, other=other, alpha=torch.tensor(.25))
    rule = tensor_update(lambda p, g, s, hp: ({name: value - .125 * g[name]
                                              for name, value in p.items()}, {}),
                         name="scaled_add_vjp_test_sgd")
    captured = capture_training(objective, parameters, None, (cotangent,), config=rule)
    assert sum(op.target == TARGET for op in captured.ir.operations) == 1
    assert "fmaf(" in lower(captured.ir, execution="compose").source
    interpreter = _CPUPrimalOracle(captured.graph_module)
    checks = []
    for alpha in (.25, -.5):
        # Replay the same differentiated graph with a changed scalar, so the
        # forward and its VJP must both consume the runtime scalar binding.
        arguments = list(captured.example_args)
        index = list(parameters).index("alpha")
        arguments[index] = torch.tensor(alpha)
        actual = captured.unpack(interpreter.run(*arguments))
        expected = dict(input=cotangent, other=cotangent * alpha,
                        alpha=(cotangent * other).sum())
        for name, gradient in expected.items():
            torch.testing.assert_close(actual["grads"][name], gradient, rtol=0, atol=0)
        expected_loss = ((input.double() + other.double() * alpha).float() * cotangent).sum()
        torch.testing.assert_close(actual["loss"], expected_loss, rtol=0, atol=0)
        checks.append(dict(differentiated_fake_capture=True, all_three_vjps_exact=True,
                           dynamic_alpha=alpha, primal_oracle="FP64 one-round dyadic CPU fixture"))
    return checks


def run():
    checks = []
    for schedule, options in (("auto", {}), ("compose", dict(execution="compose")),
                              ("reference", dict(policy="reference"))):
        result = lower(fixture(), **options)
        assert "fmaf(" in result.source
        checks.append(dict(schedule=schedule, admitted=True))
        for name, value in (("alpha", _value((1,))), ("alpha", _value((), "bfloat16")),
                            ("other", _value((1, 3))), ("out", _value((2, 3), "bfloat16"))):
            bad = copy.deepcopy(fixture()); bad["values"][name] = value
            try:
                lower(bad, **options)
            except LoweringError:
                checks.append(dict(schedule=schedule, refused=True))
            else:
                raise AssertionError("invalid scaled_add metadata admitted")
    def function(_parameters, input, other, alpha):
        return scaled_add(input, other, alpha)
    captured = capture_inference(function, {}, (torch.zeros(2, 3), torch.ones(2, 3), torch.tensor(.1)))
    assert sum(op.target == TARGET for op in captured.ir.operations) == 1
    assert "fmaf(" in lower(captured.ir, execution="compose").source
    checks.append(dict(fake_capture=True, dynamic_scalar=True))
    checks.extend(differentiated_checks())
    return dict(passed=True, checks=checks, contract="FP32 scalar scaled-add; explicit fmaf; no global contraction change")


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(run(), indent=2))
