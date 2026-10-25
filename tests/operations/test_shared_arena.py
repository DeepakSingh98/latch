"""CPU contracts for caller-owned arenas; CUDA execution is qualified separately."""
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.operation_compiler.api import ModuleUpdate
from megakernel_lab.operation_compiler.lower import GeneratedProgram
from megakernel_lab.operation_compiler.runtime import _validate_shared_arena


def _arena(words=128):
    allocation = torch.empty(words + 64, dtype=torch.float32)
    offset = (-allocation.data_ptr() // 4) % 64
    result = allocation[offset:offset + words]
    assert result.data_ptr() % 256 == 0
    return result


def _validate(arena, **extra):
    return _validate_shared_arena(arena, device=torch.device("cpu"), words=64, manifest={}, **extra)


def test_shared_arena_admits_disjoint_aligned_storage():
    arena = _arena()
    _validate(arena, input_ranges=[("x", arena.data_ptr() - 32, arena.data_ptr()),
                                  ("y", arena.data_ptr() + 256, arena.data_ptr() + 512)])


@pytest.mark.parametrize("kind", ["short", "dtype", "shape", "strided", "unaligned", "lazy_negative"])
def test_shared_arena_rejects_incompatible_storage(kind):
    arena = _arena()
    bad = {"short": lambda: arena[:32], "dtype": lambda: arena.view(torch.int32),
           "shape": lambda: arena.view(8, 16), "strided": lambda: arena[::2],
           "unaligned": lambda: arena[1:], "lazy_negative": lambda: torch._neg_view(arena)}[kind]()
    with pytest.raises(ValueError, match="shared arena must"):
        _validate(bad)


@pytest.mark.parametrize("role", ["parameter", "optimizer_state", "input", "constant"])
def test_shared_arena_rejects_overlap_with_any_bound_argument(role):
    arena = _arena()
    with pytest.raises(ValueError, match=f"overlaps bound argument {role}"):
        _validate(arena, input_ranges=[(role, arena.data_ptr() + 64, arena.data_ptr() + 320)])


@pytest.mark.parametrize("manifest", [{"persistent_counter_words": 1},
    {"phases": [{"counter_effects": [{"storage": "runtime_counter_arena"}]}]}])
def test_shared_arena_rejects_launch_persistent_counters(manifest):
    with pytest.raises(ValueError, match="launch-persistent counters"):
        _validate_shared_arena(_arena(), device=torch.device("cpu"), words=64, manifest=manifest)


def test_allocate_shared_arena_uses_max_physical_extent(monkeypatch):
    records = {"a": {"arena_words": 3}, "b": {"arena_words": 7}, "empty": {"arena_words": 0}}
    monkeypatch.setattr(GeneratedProgram, "load", lambda directory: SimpleNamespace(manifest=records[directory]))
    arena = ModuleUpdate.allocate_shared_arena("a", "b", "empty", device="cpu")
    assert arena.shape == (8,) and arena.dtype == torch.float32
    with pytest.raises(ValueError, match="at least one"):
        ModuleUpdate.allocate_shared_arena(device="cpu")
    records["b"]["persistent_counter_words"] = 2
    with pytest.raises(ValueError, match="launch-persistent counters"):
        ModuleUpdate.allocate_shared_arena("a", "b", device="cpu")
