"""CPU contracts for the general admission of the split-K generic/skinny BF16 bodies (LATCH-SPLITK-SELECT; cloud CPU).

The split bodies (mixed_gemm: P8-E generic split, P9-E skinny split) were offered only in channels-last convolution
programs. Their contract is the optimized lowering's split-K admission plus emit()'s own shape conditions, so an
ordinary explicit-BF16 linear program with a long-K, few-tile weight gradient selects them too, while the
reference policy, short K and aligned products keep their previous bodies. Numerical and timing evidence for the
generated code is device work (research/latch_release/splitk).
"""
from __future__ import annotations

import torch

SPLIT = ('generic_bf16_simt_16_split', 'p9e_skinny_split')


class Linear(torch.nn.Module):
    def __init__(self, inputs, width, outputs):
        super().__init__()
        generator = torch.Generator().manual_seed(4101)
        self.first = torch.nn.Parameter(torch.randn(inputs, width, generator=generator) * 0.3)
        self.head = torch.nn.Parameter(torch.randn(width, outputs, generator=generator) * 0.1)

    def forward(self, x):
        hidden = torch.nn.functional.silu((x.to(torch.bfloat16) @ self.first.to(torch.bfloat16)).float())
        return (hidden.to(torch.bfloat16) @ self.head.to(torch.bfloat16)).float()


def _program(batch, inputs, width, outputs, **options):
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    generator = torch.Generator().manual_seed(4103)
    x = torch.randn(batch, inputs, generator=generator)
    y = torch.randn(batch, outputs, generator=generator)
    captured = capture_module(Linear(inputs, width, outputs), lambda p, t: (p - t).square().mean(), (x,), (y,),
                              optimizer=SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0))
    return lower(captured.ir, execution='compose', **options)


def _products(program):
    return [p for p in program.manifest['phases'] if p.get('kind') == 'gemm']


def test_linear_long_k_weight_gradients_split():
    """3-wide input and 1-wide head: both weight gradients (K = batch = 2048) take a split body outside conv."""
    program = _program(2048, 3, 64, 1)
    split = [p for p in _products(program) if p.get('engine') in SPLIT]
    assert len(split) == 2, [(p.get('names')[:1], p.get('engine'), p.get('k'), p.get('tiles')) for p in _products(program)]
    assert all(p.get('private_scratch') for p in split)
    zeros = [p for p in program.manifest['phases'] if 'generic_split_counter_zero' in (p.get('targets') or [])]
    assert len(zeros) == len(split)
    assert '/*P2_' not in program.source


def test_reference_policy_keeps_unsplit_bodies():
    program = _program(2048, 3, 64, 1, policy='reference')
    assert not [p for p in _products(program) if p.get('engine') in SPLIT]
    assert '/*P2_' not in program.source


def test_diagnostic_retention_plans_split_scratch():
    """retain_all skips the passes that rebuild producers; the split partials still need their own producer."""
    program = _program(2048, 3, 64, 1, retain_all=True)
    assert len([p for p in _products(program) if p.get('engine') in SPLIT]) == 2


def test_short_k_keeps_generic_body():
    """Below the bodies' long-K threshold (2048) the unsplit generic tile remains."""
    program = _program(1024, 3, 64, 1)
    engines = {p.get('engine') for p in _products(program)}
    assert not engines & set(SPLIT), engines
