"""CPU contracts for matrix_bodies 'splitctr' (entry-zeroed split-K arrival counters; no CUDA execution).

A long-K, few-tile BF16 weight gradient of an ordinary linear program takes a generic split-K body (LATCH-SPLITK-SELECT).
Without the token its arrival counters are an arena root zeroed by its own ``generic_split_counter_zero`` phase (the
release form). With ``matrix_bodies='splitctr'`` the flat dependency-scheduled kernel takes the counters from
per-launch words past the arena, zeroed at kernel entry, and the partials from a private root; no zero phase remains.
"""
from __future__ import annotations

import re

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


def _program(bodies=None, batch=2048, inputs=3, width=64, outputs=1):
    from megakernel_lab.operation_compiler import matrix_bodies
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    generator = torch.Generator().manual_seed(4103)
    x = torch.randn(batch, inputs, generator=generator)
    y = torch.randn(batch, outputs, generator=generator)
    captured = capture_module(Linear(inputs, width, outputs), lambda p, t: (p - t).square().mean(), (x,), (y,),
                              optimizer=SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0))
    with matrix_bodies.scope(bodies):
        return lower(captured.ir, execution='compose')


def _split(program):
    return [p for p in program.manifest['phases'] if p.get('kind') == 'gemm' and p.get('engine') in SPLIT]


def _zero_phases(program):
    return [p for p in program.manifest['phases'] if 'generic_split_counter_zero' in (p.get('targets') or [])]


def _entry_zeroed(phase):
    return any(e.get('storage') == 'entry_zeroed_counter_words' for e in phase.get('counter_effects') or ())


def test_split_counters_use_entry_zeroed_words():
    from megakernel_lab.operation_compiler import step17_intervals
    program = _program('splitctr')
    split = _split(program)
    assert len(split) == 2, [(p.get('names')[:1], p.get('engine')) for p in program.manifest['phases']
                             if p.get('kind') == 'gemm']
    assert not _zero_phases(program)
    assert all(_entry_zeroed(p) for p in split)
    assert all(p.get('private_scratch') and p['private_scratch'][0].startswith('gsk_') for p in split)
    assert '/*GSKC' not in program.source and '/*P2_' not in program.source
    # One kernel-entry statement per 256 words, each in a form the device-unit partitioner admits.
    statements = re.findall(r"if \(blockIdx\.x == 0 && threadIdx\.x < \d+u\) reinterpret_cast<unsigned\*>\(arena \+ \d+LL\)"
                            r"\[threadIdx\.x\] = 0u;", program.source)
    assert statements and all(any(p.fullmatch(s) for p in step17_intervals.TEXT_STATEMENTS) for s in statements)
    words = program.manifest['group_rules']['ecta_queue_counters']['count']
    assert words >= sum(e['words'] for p in split for e in p['counter_effects'])


def test_without_the_token_split_counters_keep_their_zero_phase():
    program = _program()
    split = _split(program)
    assert len(split) == 2
    assert len(_zero_phases(program)) == len(split)
    assert not any(_entry_zeroed(p) for p in split)
    assert '/*GSKC' not in program.source and '/*P2_' not in program.source


def test_split_counter_token_keeps_the_split_bodies():
    """Same split products on the same bodies: only where their arrival counters live changes."""
    on, off = _program('splitctr'), _program()
    key = lambda p: (p.get('names'), p.get('engine'), p.get('matrix_shape'))
    assert [key(p) for p in _split(on)] == [key(p) for p in _split(off)]
