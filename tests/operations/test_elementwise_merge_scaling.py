"""Byte-identical greedy scheduling against the frozen donor implementation."""
import copy
import importlib.util
from pathlib import Path
import random
import time

import pytest

from megakernel_lab.operation_compiler.group_rules import merge_elementwise_chains, _topological
from megakernel_lab.operation_compiler.lower import TensorLayout, _ident, _index

spec = importlib.util.spec_from_file_location("elementwise_merge_donor",
    Path(__file__).parent / "fixtures/elementwise_merge_donor.py")
donor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(donor)


def fixture(seed, chains, length):
    generator = random.Random(seed)
    layouts, phases = {}, []
    inplace = set()
    for chain in range(chains):
        shape = ((4, 4), (16,), ())[chain % 3]
        strides = (4, 1) if len(shape) == 2 else (1,) if shape else ()
        previous = f"input{chain}"
        layouts[previous] = TensorLayout(shape, strides, previous)
        for position in range(length):
            name = f"chain{chain}_{position}"
            layouts[name] = TensorLayout(shape, strides, name)
            alias = name + "_alias"
            layouts[alias] = TensorLayout(shape, strides, name)
            read = previous if position == 0 or generator.random() < .5 else previous + "_alias"
            phase = dict(kind="elementwise", names=[name], targets=["op"], reads=[read], writes=[name],
                shape=shape, body=[f"const float r_{name} = {_ident(read)}[{_index(layouts[read], shape)}] + 1.0f; {_ident(name)}[i] = r_{name};"])
            if generator.random() < .08:
                phase["kind"] = "gemm"
            if generator.random() < .05:
                phase["vprod_inline"] = True
            if generator.random() < .03:
                inplace.add(name)
            phases.append(phase)
            previous = name
    # Interleave independent chains to exercise restarts and nonadjacent merges.
    phases = [phases[chain * length + position] for position in range(length) for chain in range(chains)]
    return phases, layouts, inplace


@pytest.mark.parametrize("seed", range(12))
def test_cached_facts_preserve_complete_donor_schedule(seed):
    phases, layouts, inplace = fixture(seed, 9, 15)
    before = copy.deepcopy(phases)
    kwargs = dict(ident=_ident, index=_index, inplace_roots=inplace)
    expected = donor.merge_elementwise_chains(phases, layouts, **kwargs)
    actual = merge_elementwise_chains(phases, layouts, **kwargs)
    assert actual == expected
    assert phases == before


def test_large_interleaved_optimizer_chains_retain_exact_phase_bytes():
    phases, layouts, inplace = fixture(192, 64, 12)
    kwargs = dict(ident=_ident, index=_index, inplace_roots=inplace)
    began = time.perf_counter()
    expected = donor.merge_elementwise_chains(phases, layouts, **kwargs)
    donor_seconds = time.perf_counter() - began
    began = time.perf_counter()
    actual = merge_elementwise_chains(phases, layouts, **kwargs)
    cached_seconds = time.perf_counter() - began
    assert actual == expected
    # Timings are an observation, never a flaky machine-dependent pass gate.
    print(dict(phases=len(phases), merges=len(actual[1]), donor_seconds=donor_seconds,
               cached_seconds=cached_seconds), flush=True)


@pytest.mark.parametrize("seed", range(12))
def test_topological_heap_retains_smallest_ready_index_with_aliases(seed):
    generator = random.Random(seed)
    phases = [dict(writes=[f"n{i}"], reads=[f"n{j}_alias" for j in generator.sample(
        range(i), min(i, generator.randrange(6)))]) for i in range(100)]
    generator.shuffle(phases)
    root = lambda name: name.removesuffix("_alias")
    assert _topological(phases, root) == donor._topological(phases, root)


def test_topological_heap_retains_last_writer_and_cycle_refusal():
    phases = [dict(writes=["x"], reads=[]), dict(writes=["y"], reads=["x"]),
              dict(writes=["x"], reads=["external"])]
    assert _topological(phases, str) == donor._topological(phases, str)
    phases = [dict(writes=["x"], reads=["y"]), dict(writes=["y"], reads=["x"])]
    for function in (_topological, donor._topological):
        with pytest.raises(ValueError, match="dependency cycle"):
            function(phases, str)


def test_topological_large_chain_keeps_exact_phase_order():
    phases = [dict(writes=[f"n{i}"], reads=[f"n{i-1}"] if i else []) for i in range(4096)]
    began = time.perf_counter()
    expected = donor._topological(phases, str)
    donor_seconds = time.perf_counter() - began
    began = time.perf_counter()
    actual = _topological(phases, str)
    heap_seconds = time.perf_counter() - began
    assert actual == expected == phases
    print(dict(topological_phases=len(phases), donor_seconds=donor_seconds,
               heap_seconds=heap_seconds), flush=True)
