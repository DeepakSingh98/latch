"""Latch: compile a PyTorch training step into one GPU kernel.

    captured = latch.capture_module(model, loss_fn, (x,), (y,), optimizer=latch.AdamWConfig())
    captured.build("out")                    # needs nvcc, not a GPU
    update = latch.ModuleUpdate.load("out")  # on the GPU
    update.stage((x,), (y,))
    update.step()                            # forward, loss, backward, optimizer: one launch
"""
from .capture import Captured, capture_module
from .ir import CompileError, Graph
from .lower import lower
from .codegen import generate
from .optim import AdamWConfig, SGDConfig
from .runtime import ModuleUpdate, build

__all__ = ["capture_module", "Captured", "AdamWConfig", "SGDConfig", "ModuleUpdate", "build", "lower",
           "generate", "Graph", "CompileError"]
