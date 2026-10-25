"""Frozen independent SGD oracle used by operation-compiler regression tests.

Only the reference functions are retained; cloud launchers and donor imports
are excluded. See provenance/NUMERICAL_REFERENCE_PACKAGING.json.
"""
import torch

from megakernel_lab.operation_compiler.capture import SGDConfig
from megakernel_lab.operation_compiler.numerical_reference import compare_arithmetic
from megakernel_lab.fast_examples.recovered.transformer import loop_loss, make_loss

OPTIMIZER = SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0001)


def _passed(value):
    if isinstance(value, dict):
        return value.get('passed', True) and all(_passed(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_passed(item) for item in value)
    return True


def reference_step(loss_fn, params, momentum, inputs, *, state=None, dtype=None, optimizer=OPTIMIZER):
    """Separate eager autograd + actual torch.optim.SGD; never execute a traced graph."""
    p = {key: value.detach().to(dtype=dtype or value.dtype).clone().requires_grad_(True)
         for key, value in params.items()}
    data = tuple(value.to(dtype=dtype) if dtype and value.is_floating_point() else value for value in inputs)
    result = loss_fn(p, *data) if state is None else loss_fn(p, state, *data)
    loss, updated_state = (result, {}) if state is None else result
    gradients = torch.autograd.grad(loss, tuple(p.values()), allow_unused=True)
    gradients = tuple(torch.zeros_like(value) if gradient is None else gradient
                      for value, gradient in zip(p.values(), gradients))
    opt = torch.optim.SGD(list(p.values()), lr=optimizer.lr, momentum=optimizer.momentum,
                          weight_decay=optimizer.weight_decay, foreach=False)
    for (name, value), gradient in zip(p.items(), gradients):
        value.grad = gradient.detach().clone()
        opt.state[value]['momentum_buffer'] = momentum[name].detach().to(value.dtype).clone()
    opt.step()
    return {'loss': loss.detach(), 'grads': dict(zip(p, (g.detach().clone() for g in gradients))),
            'params': {key: value.detach().clone() for key, value in p.items()},
            'momentum': {key: opt.state[value]['momentum_buffer'].detach().clone() for key, value in p.items()},
            'state': {key: value.detach().clone() for key, value in updated_state.items()}}


def compare_step(actual, expected, old_params, *, arithmetic='float32'):
    checks = {'loss': compare_arithmetic(actual['loss'], expected['loss'], 'loss', arithmetic)}
    for field, policy in (('grads', 'gradients'), ('params', 'parameters'), ('momentum', 'momentum')):
        checks[field] = {name: compare_arithmetic(value, expected[field][name], policy, arithmetic)
                         for name, value in actual[field].items()}
    checks['updates'] = {name: compare_arithmetic(actual['params'][name] - value,
                                      expected['params'][name] - value, 'updates', arithmetic)
                         for name, value in old_params.items()}
    checks['state'] = {name: {'passed': torch.equal(value, expected['state'][name]), 'exact_integer': True}
                       for name, value in actual.get('state', {}).items()}
    return checks


def _loss_for(model, objective, count=1, *, stochastic=True):
    if objective in ('loop', 'combined'):
        return loop_loss(model, count=count, intermediate=objective == 'combined', stochastic=stochastic)
    if objective not in ('baseline', 'intermediate'):
        raise ValueError('unknown transformer objective ' + objective)
    return make_loss(model, intermediate=objective == 'intermediate')
