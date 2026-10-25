"""LATCH-STEP19-SPEED contracts: bounded dynamic batch through the established flat scheduler.

CPU-only lowering checks (no CUDA). Run: python -c "from .test_dynamic_flat import run; print(run())"
"""
from __future__ import annotations

import traceback

import torch


class _BF16TwoLayer(torch.nn.Module):
    """Explicit BF16 products, FP32 parameters and pointwise work (the STEP19 matrix recipe)."""
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        generator = torch.Generator().manual_seed(1919)
        self.first = torch.nn.Parameter(torch.randn(inputs, hidden, generator=generator) * 0.05)
        self.first_bias = torch.nn.Parameter(torch.zeros(hidden))
        self.head = torch.nn.Parameter(torch.randn(hidden, outputs, generator=generator) * 0.05)
        self.head_bias = torch.nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        h = torch.nn.functional.silu((x.to(torch.bfloat16) @ self.first.to(torch.bfloat16)).float() + self.first_bias)
        return (h.to(torch.bfloat16) @ self.head.to(torch.bfloat16)).float() + self.head_bias


def _objective(prediction, target):
    return (prediction - target).square().mean()


def _capture(dims=(128, 256, 64), batch=32, dynamic=True):
    from megakernel_lab.operation_compiler import DynamicBatch, SGDConfig, capture_module
    inputs, hidden, outputs = dims
    kwargs = {'dynamic_batch': DynamicBatch(maximum=32, axis=0)} if dynamic else {}
    return capture_module(_BF16TwoLayer(inputs, hidden, outputs), _objective,
                          (torch.zeros(batch, inputs),), (torch.zeros(batch, outputs),),
                          optimizer=SGDConfig(lr=.01, momentum=.9, weight_decay=.001), **kwargs)


def run():
    from megakernel_lab.operation_compiler import dynamic_flat
    from megakernel_lab.operation_compiler.lower import lower
    results = {}

    def check(name, fn):
        try:
            fn()
            results[name] = {'passed': True}
        except Exception:
            results[name] = {'passed': False, 'traceback': traceback.format_exc()}

    def flat_schedule_selected():
        program = lower(_capture().ir, execution='compose')
        manifest = program.manifest
        assert manifest['compiler_selection']['schedule'] == 'bounded_dynamic_flat'
        assert manifest['dynamic_batch']['maximum'] == 32 and manifest['dynamic_batch']['input']
        gemms = [p for p in manifest['phases'] if p.get('kind') in ('gemm', 'reduction') and p.get('members')]
        engines = [m['engine'] for p in gemms for m in p['members']]
        assert engines and all('edge_active' in engine for engine in engines), engines
        source = program.source
        assert 'latch_active_batch < 1 || latch_active_batch > 32' in source
        # Generated epilogues use the row-bounded interface and carry the active extent.
        assert 'tile_rows' in source and 'prefetch_rows' in source
        assert 'int latch_active_batch;' in source
        # Fewer barriers than the ordered dynamic lowering of the same graph.
        assert manifest['barrier_count'] < 10, manifest['barrier_count']

    check('bf16_dynamic_program_uses_flat_scheduler', flat_schedule_selected)

    def ordered_lowering_retained_when_disabled():
        captured = _capture()
        saved = dynamic_flat.ENABLED
        try:
            dynamic_flat.ENABLED = False
            program = lower(captured.ir, execution='compose')
        finally:
            dynamic_flat.ENABLED = saved
        assert program.manifest['compiler_selection']['schedule'] == 'bounded_dynamic_dependency'

    check('flat_scheduler_disable_restores_ordered_dynamic_lowering', ordered_lowering_retained_when_disabled)

    def static_programs_unchanged():
        captured = _capture(batch=16, dynamic=False)
        enabled = lower(captured.ir, execution='compose')
        saved = dynamic_flat.ENABLED
        try:
            dynamic_flat.ENABLED = False
            disabled = lower(captured.ir, execution='compose')
        finally:
            dynamic_flat.ENABLED = saved
        assert enabled.cache_key == disabled.cache_key and enabled.source == disabled.source
        assert 'latch_active_batch' not in enabled.source and 'dynamic_flat.cuh' not in enabled.source

    check('static_lowering_independent_of_dynamic_scheduler', static_programs_unchanged)

    def fp32_products_fall_back_with_reason():
        from megakernel_lab.operation_compiler import DynamicBatch, SGDConfig, capture_module
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
        captured = capture_module(model, _objective, (torch.zeros(32, 8),), (torch.zeros(32, 4),),
                                  optimizer=SGDConfig(lr=.01, momentum=.9),
                                  dynamic_batch=DynamicBatch(maximum=32, axis=0))
        program = lower(captured.ir, execution='compose')
        selection = program.manifest['compiler_selection']
        assert selection['schedule'] == 'bounded_dynamic_dependency'
        assert any('flat dynamic schedule refused' in reason for reason in selection['reasons'])

    check('unadmitted_dynamic_programs_keep_ordered_lowering', fp32_products_fall_back_with_reason)

    def retained_diagnostics_store_everything():
        program = lower(_capture().ir, execution='compose', retain_all=True)
        manifest = program.manifest
        assert manifest['compiler_selection']['schedule'] == 'bounded_dynamic_flat'
        assert manifest['retained_all'] and not manifest['materialization']['inlined']
        assert not manifest.get('pops2b', {}).get('matrix_epilogues')

    check('retained_dynamic_programs_store_every_intermediate', retained_diagnostics_store_everything)

    def awkward_capacity_lowered_or_refused():
        # A capacity below 32 with odd feature sizes either lowers through the flat path or records a refusal;
        # it never raises out of normal compilation.
        from megakernel_lab.operation_compiler import DynamicBatch, SGDConfig, capture_module
        captured = capture_module(_BF16TwoLayer(96, 160, 48), _objective, (torch.zeros(24, 96),),
                                  (torch.zeros(24, 48),), optimizer=SGDConfig(lr=.01, momentum=.9),
                                  dynamic_batch=DynamicBatch(maximum=24, axis=0))
        program = lower(captured.ir, execution='compose')
        assert program.manifest['compiler_selection']['schedule'] in ('bounded_dynamic_flat', 'bounded_dynamic_dependency')

    check('other_capacities_compile_through_normal_selection', awkward_capacity_lowered_or_refused)
    return {'passed': all(r['passed'] for r in results.values()), 'contract_count': len(results), 'tests': results}
