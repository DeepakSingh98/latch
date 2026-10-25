"""Forward-only capture/lowering contracts; run on a CPU worker with pinned PyTorch."""
from __future__ import annotations

import traceback

import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.inference import (InferenceIR, LoopInferenceIR, capture_inference, capture_loop_inference,
                        capture_module_inference, lower_inference)
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.lower import LoweringError, lower


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(12, 24, bias=False)
        self.right = torch.nn.Linear(24, 5, bias=False)
        self.register_buffer('offset', torch.linspace(-.2, .2, 24))
        self.register_buffer('classes', torch.arange(5, dtype=torch.int64))

    def forward(self, x, state=None):
        logits = self.right(torch.tanh(self.left(x) + self.offset))
        prediction = {'logits': logits, 'probabilities': torch.softmax(logits, dim=-1),
                      'classes': logits.argmax(dim=-1)}
        if state is None:
            return prediction
        hits = (prediction['classes'].unsqueeze(-1) == self.classes).sum(0)
        return prediction, {'seen': state['seen'] + x.shape[0], 'counts': state['counts'] + hits,
                            'mean': state['mean'] * .5 + logits.mean(0) * .5}


def _ops(ir):
    return {op.target for op in ir.operations}


def run():
    torch.set_num_threads(1)
    results = {}

    def check(name, fn):
        try:
            fn()
            results[name] = {'passed': True}
        except Exception:
            results[name] = {'passed': False, 'traceback': traceback.format_exc()}

    torch.manual_seed(51017)
    x = torch.randn(9, 12)

    def module_forward_matches_eager():
        model = Classifier().eval()
        captured = capture_module_inference(model, (x,))
        actual = captured.unpack(captured.graph_step(*captured.example_args))['predictions']
        expected = model(x)
        assert set(actual) == {'logits', 'probabilities', 'classes'}
        assert torch.allclose(actual['logits'], expected['logits'], atol=1e-6, rtol=1e-6)
        assert torch.allclose(actual['probabilities'], expected['probabilities'], atol=1e-6, rtol=1e-6)
        assert torch.equal(actual['classes'], expected['classes'])
        assert actual['classes'].dtype == torch.int64

    check('module_forward_matches_eager', module_forward_matches_eager)

    def forward_only_graph():
        model = Classifier().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        captured = capture_module_inference(model, (x,))
        targets = _ops(captured.ir)
        assert not any('backward' in t or 'adam' in t.lower() for t in targets), targets
        assert 'aten.argmax.default' in targets
        roles = [captured.ir.values[name].role for name in captured.ir.inputs]
        assert roles.count('parameter') == 2 and roles.count('input') == 3, roles
        assert captured.ir.state_writes == {}
        assert len(captured.ir.outputs) == 3
        program = lower(captured.ir)
        assert program.source.count('__global__') == 1
        assert program.manifest['state_writes'] == {}
        parameter_inputs = {name for name in program.manifest['inputs']
                            if program.manifest['input_roles'][name]['role'] == 'parameter'}
        written = {phase_write for phase in program.manifest['phases'] for phase_write in phase['writes']}
        assert not parameter_inputs & written

    check('forward_only_graph_and_lowering', forward_only_graph)

    def frozen_and_tied_parameters():
        model = Classifier().eval()
        model.right.weight.requires_grad_(False)
        tied = torch.nn.Module()
        tied.a = torch.nn.Linear(6, 6, bias=False)
        tied.b = torch.nn.Linear(6, 6, bias=False)
        tied.b.weight = tied.a.weight
        tied.forward = lambda v: tied.b(torch.tanh(tied.a(v)))
        captured = capture_module_inference(model, (x,))
        assert len(captured.parameter_names) == 2
        z = torch.randn(3, 6)
        captured_tied = capture_inference(lambda p, v: F.linear(torch.tanh(F.linear(v, p['a'])), p['b']),
                                          {'a': tied.a.weight, 'b': tied.b.weight}, (z,))
        assert captured_tied.parameter_names == ('a',) and captured_tied.parameter_aliases['b'] == 'a'
        expected = tied.b(torch.tanh(tied.a(z)))
        assert torch.allclose(captured_tied.graph_step(*captured_tied.example_args)[0], expected, atol=1e-6)

    check('frozen_and_tied_parameters', frozen_and_tied_parameters)

    def buffer_snapshot_frozen():
        model = Classifier().eval()
        captured = capture_module_inference(model, (x,))
        before = captured.graph_step(*captured.example_args)[0].clone()
        model.offset.fill_(3.0)
        assert torch.equal(captured.graph_step(*captured.example_args)[0], before)

    check('buffer_snapshot_frozen', buffer_snapshot_frozen)

    def implicit_randomness_refused():
        model = torch.nn.Sequential(torch.nn.Linear(12, 12), torch.nn.Dropout(.5))
        try:
            capture_module_inference(model.train(), (x,))
        except CaptureError as error:
            assert error.code in {'RNG', 'TRACE'}, error.code
        else:
            raise AssertionError('training-mode dropout admitted')
        capture_module_inference(model.eval(), (x,))

    check('implicit_randomness_refused_eval_admitted', implicit_randomness_refused)

    def runtime_control_refused():
        def branchy(p, v):
            h = v @ p['w']
            return h * 2 if bool(h.sum() > 0) else h
        try:
            capture_inference(branchy, {'w': torch.randn(12, 12)}, (x,))
        except CaptureError as error:
            assert error.code == 'RUNTIME_CONTROL', error.code
        else:
            raise AssertionError('tensor-dependent Python control admitted')

    check('runtime_control_refused', runtime_control_refused)

    def explicit_state_contract():
        model = Classifier().eval()
        state = {'seen': torch.tensor(0, dtype=torch.int64), 'counts': torch.zeros(5, dtype=torch.int64),
                 'mean': torch.zeros(5)}
        captured = capture_module_inference(model, (x,), state=state)
        assert set(captured.ir.state_writes) == set(captured.ir.inputs[2:5])
        flat = list(captured.example_args)
        for _ in range(3):
            result = captured.unpack(captured.graph_step(*flat))
            expected, new = model(x, state)
            assert torch.equal(result['state']['seen'], new['seen'])
            assert torch.equal(result['state']['counts'], new['counts'])
            assert torch.allclose(result['state']['mean'], new['mean'], atol=1e-6)
            state = new
            for position, name in enumerate(captured.state_names):
                flat[2 + position] = result['state'][name]
        assert int(state['seen']) == 27 and int(state['counts'].sum()) == 27
        program = lower(captured.ir)
        assert set(program.manifest['state_writes']) == set(captured.ir.inputs[2:5])

        class Bad(Classifier):
            def forward(self, v, state=None):
                return self.right(torch.tanh(self.left(v))), {'seen': state['seen'].float()}
        try:
            capture_module_inference(Bad().eval(), (x,), state={'seen': torch.tensor(0, dtype=torch.int64)})
        except CaptureError as error:
            assert error.code == 'STATE_RETURN'
        else:
            raise AssertionError('state dtype change admitted')

    check('explicit_state_contract', explicit_state_contract)

    def input_alias_outputs_materialized():
        def passthrough(p, v, state_like):
            return v.t(), (v @ p['w']).sum(-1), state_like
        w = torch.randn(12, 4)
        captured = capture_inference(passthrough, {'w': w}, (x, torch.ones(3)))
        program = lower(captured.ir)
        manifest = program.manifest
        for name in manifest['outputs']:
            assert manifest['layouts'][name]['root'] not in manifest['inputs'], name
        out = captured.graph_step(*captured.example_args)
        assert torch.equal(out[0], x.t()) and torch.equal(out[2], torch.ones(3))

    check('input_alias_outputs_materialized', input_alias_outputs_materialized)

    def argmax_lowering_scope():
        good = capture_inference(lambda p, v: (v @ p['w']).argmax(dim=-1, keepdim=True),
                                 {'w': torch.randn(12, 7)}, (x,))
        lower(good.ir)
        bad = capture_inference(lambda p, v: (v @ p['w']).argmax(dim=0), {'w': torch.randn(12, 7)}, (x,))
        try:
            lower(bad.ir)
        except LoweringError:
            pass
        else:
            raise AssertionError('non-last-dimension argmax admitted')
        ties = torch.tensor([[1., 3., 3.], [float('nan'), 2., float('nan')]])
        captured = capture_inference(lambda p, v: (v * p['s']).argmax(dim=-1), {'s': torch.ones(3)}, (ties,))
        assert captured.graph_step(*captured.example_args)[0].tolist() == [1, 0]

    check('argmax_lowering_scope', argmax_lowering_scope)

    def ir_round_trip():
        captured = capture_module_inference(Classifier().eval(), (x,), state={'seen': torch.tensor(0),
                                            'counts': torch.zeros(5, dtype=torch.int64), 'mean': torch.zeros(5)})
        restored = InferenceIR.from_dict(captured.ir.to_dict())
        assert lower(restored).cache_key == lower(captured.ir).cache_key

    check('ir_round_trip', ir_round_trip)

    def stateless_launch_admission_is_runtime_selected():
        # The maintained baseline replaced the old process-global generated
        # cache with per-Runtime context/binding admission (test_launch_cache).
        # Both checked launch exports are emitted in the same binary.
        import inspect
        from megakernel_lab.operation_compiler.inference import Inference
        from megakernel_lab.operation_compiler.runtime import Runtime
        captured = capture_module_inference(Classifier().eval(), (x,))
        program = lower(captured.ir)
        assert 'int latch_launch(' in program.source and 'int latch_launch_admitted(' in program.source
        assert 'admitted_max_blocks' in program.source and 'cached_max_blocks' not in program.source
        assert 'launch_admission' not in program.manifest
        assert inspect.signature(Inference.load).parameters['launch_cache'].default is True
        assert inspect.signature(Runtime).parameters['launch_cache'].default is True

    check('stateless_launch_admission_is_runtime_selected', stateless_launch_admission_is_runtime_selected)

    def runtime_loop():
        w = {'enc': torch.randn(12, 16) * .3, 'shared': torch.randn(16, 16) * .2, 'head': torch.randn(16, 5) * .3}

        def init(p, v):
            return {'h': torch.tanh(v @ p['enc'])}

        def body(p, carry, state, v):
            return {'h': carry['h'] + .5 * torch.tanh(carry['h'] @ p['shared'])}, {'steps': state['steps'] + 1}

        def head(p, carry, v):
            logits = carry['h'] @ p['head']
            return {'logits': logits, 'classes': logits.argmax(dim=-1)}

        state = {'steps': torch.tensor(0, dtype=torch.int64)}
        captured = capture_loop_inference(init, body, head, w, (x,), torch.tensor(0), max_steps=4, state=state)
        assert isinstance(captured.ir, LoopInferenceIR)
        for region in (captured.ir.init, captured.ir.body, captured.ir.output):
            assert not any('backward' in op.target for op in region.operations)
        for requested in (0, 1, 3, 4, -2, 9):
            flat = (*captured.example_args[:-1], torch.tensor(requested))
            got = captured.unpack(captured.graph_step(*flat))
            executed = min(max(requested, 0), 4)
            carry = init(w, x)
            for _ in range(executed):
                carry, _ = body(w, carry, state, x)
            expected = head(w, carry, x)
            assert torch.allclose(got['predictions']['logits'], expected['logits'], atol=1e-6)
            assert torch.equal(got['predictions']['classes'], expected['classes'])
            assert int(got['executed_count']) == executed and int(got['status']) == int(executed != requested)
            assert int(got['state']['steps']) == executed
        program = lower_inference(captured.ir)
        assert program.source.count('__global__') == 1
        assert set(program.manifest['regions']) == {'init', 'body', 'output'}
        assert len(program.manifest['state_writes']) == 1
        restored = LoopInferenceIR.from_dict(captured.ir.to_dict())
        assert lower_inference(restored).cache_key == program.cache_key

    check('runtime_loop_regions', runtime_loop)

    def loop_contract_refusals():
        w = {'a': torch.randn(12, 12)}
        try:
            capture_loop_inference(lambda p, v: {'h': v}, lambda p, c, s, v: ({'h': c['h'].sum(0)}, s),
                                   lambda p, c, v: c['h'], w, (x,), torch.tensor(0), max_steps=2)
        except CaptureError as error:
            assert error.code == 'LOOP_CARRY'
        else:
            raise AssertionError('carry shape change admitted')
        try:
            capture_loop_inference(lambda p, v: {'h': v}, lambda p, c, s, v: (c, s), lambda p, c, v: c['h'],
                                   w, (x,), torch.tensor(0, dtype=torch.int32), max_steps=2)
        except CaptureError as error:
            assert error.code == 'LOOP_COUNT'
        else:
            raise AssertionError('non-int64 count admitted')

    check('loop_contract_refusals', loop_contract_refusals)
    return {'passed': all(row['passed'] for row in results.values()),
            'contract_count': len(results), 'tests': results}


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))
