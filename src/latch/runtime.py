"""Compile a captured step, then load it and run it.

``build`` needs a compiler but no GPU. ``ModuleUpdate.load`` binds every tensor
once; after that ``step`` is a single kernel launch and nothing else.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

from .codegen import generate
from .ir import CompileError
from .lower import lower

HEADER = Path(__file__).with_name("latch.cuh")
TORCH_DTYPE = {"float32": torch.float32, "int64": torch.int64, "bool": torch.bool}


def build(captured, directory, *, target="cuda", arch="sm_90", cache_dir=None) -> dict:
    """Write source, manifest, initial tensors and the compiled library into ``directory``.

    ``target="cuda"`` compiles with nvcc for ``arch`` (sm_90 is the H100).
    ``target="cpu"`` compiles the same source for the host, with OS threads in
    place of GPU threads: slow, for checking a program without a GPU.
    Identical builds are reused from ``cache_dir`` when it is given.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    program = lower(captured.graph)
    source = generate(program)
    if target == "cuda":
        command = ["nvcc", "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC", "--fmad=false", f"-arch={arch}"]
    elif target == "cpu":
        command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-ffp-contract=off", "-pthread", "-x", "c++"]
    else:
        raise ValueError("target must be 'cuda' or 'cpu'")
    if shutil.which(command[0]) is None:
        raise CompileError(f"{command[0]} was not found; target={target!r} needs it on PATH")
    key = hashlib.sha256("\0".join([source, HEADER.read_text(), *command]).encode()).hexdigest()
    (directory / "step.cu").write_text(source)
    library, cached = directory / "step.so", Path(cache_dir) / f"{key}.so" if cache_dir else None
    if cached is not None and cached.exists():
        shutil.copyfile(cached, library)
    else:
        done = subprocess.run([*command, f"-I{HEADER.parent}", "-o", str(library), str(directory / "step.cu")],
                              capture_output=True, text=True)
        if done.returncode:
            raise CompileError(f"{command[0]} failed:\n{done.stderr[-4000:]}")
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(library, cached)
    graph, loss = captured.graph, program.layout[captured.graph.loss]
    manifest = {
        "target": target, "source_hash": key, "arena_bytes": program.arena_bytes,
        "phases": len(program.phases), "items": sum(len(phase) for phase in program.phases),
        "loss_offset": program.buffers[loss.buffer].offset + 4 * loss.offset,
        "inputs": [{"name": name, "kind": graph.roles[name][0], "key": graph.roles[name][1],
                    "shape": list(graph.values[name].shape), "dtype": graph.values[name].dtype}
                   for name in graph.inputs],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    torch.save({name: tensor.detach().cpu() for name, tensor in captured.tensors.items()}, directory / "tensors.pt")
    return manifest


class ModuleUpdate:
    """A compiled training step bound to its tensors.

    ``parameters`` and ``optimizer_state`` map names to the live tensors the kernel
    updates in place. ``loss`` is a scalar view of the kernel's result.
    """

    @classmethod
    def load(cls, directory, *, device=None, blocks=None) -> "ModuleUpdate":
        """``blocks`` is the number of thread blocks to launch: by default one per GPU multiprocessor.

        More blocks (up to what fits on the device at once) help large matrix products
        and cost a little on small programs.
        """
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        device = torch.device(device or manifest["target"])
        if (device.type == "cuda") != (manifest["target"] == "cuda"):
            raise ValueError(f"this program was built for {manifest['target']}, not {device.type}")
        initial = torch.load(directory / "tensors.pt")
        self = cls()
        self.manifest, self.device = manifest, device
        # Bind before loading the library: on CUDA this creates the context the library launches into.
        self.tensors = [initial[entry["name"]].to(device).contiguous() for entry in manifest["inputs"]]
        self.arena = torch.zeros(max(manifest["arena_bytes"], 64), dtype=torch.uint8, device=device)
        self.table = torch.tensor([t.data_ptr() for t in self.tensors], dtype=torch.int64, device=device)
        self.loss = self.arena[manifest["loss_offset"]:manifest["loss_offset"] + 4].view(torch.float32)[0]
        self.library = ctypes.CDLL(str(directory / "step.so"))
        self.library.latch_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.library.latch_error.restype = ctypes.c_char_p
        default, limit = self.library.latch_blocks(0), self.library.latch_blocks(1)
        if limit <= 0:
            raise RuntimeError("this device cannot run a cooperative kernel")
        self.blocks = default if blocks is None else blocks
        if not 0 < self.blocks <= limit:
            raise ValueError(f"blocks must be between 1 and {limit}")
        by_kind = {}
        for entry, tensor in zip(manifest["inputs"], self.tensors):
            by_kind.setdefault(entry["kind"], {})[entry["key"]] = tensor
        self.parameters, self.optimizer_state = by_kind.get("parameter", {}), by_kind.get("optimizer", {})
        self._inputs, self._targets = by_kind.get("input", {}), by_kind.get("target", {})
        return self

    def stage(self, inputs, targets=()):
        """Copy a minibatch into the bound input and target tensors."""
        for bound, supplied, what in ((self._inputs, tuple(inputs), "inputs"), (self._targets, tuple(targets), "targets")):
            if len(bound) != len(supplied):
                raise ValueError(f"expected {len(bound)} {what}, got {len(supplied)}")
            for position, source in enumerate(supplied):
                if source.shape != bound[position].shape or source.dtype != bound[position].dtype:
                    raise ValueError(f"{what}[{position}] differs in shape or dtype from the captured example")
                bound[position].copy_(source)

    def step(self) -> torch.Tensor:
        """Launch one training step. Returns ``loss`` without waiting for the GPU."""
        stream = torch.cuda.current_stream(self.device).cuda_stream if self.device.type == "cuda" else None
        code = self.library.latch_launch(self.table.data_ptr(), self.arena.data_ptr(), self.blocks, stream)
        if code:
            raise RuntimeError(f"kernel launch failed: {self.library.latch_error(code).decode()}")
        return self.loss

    def state_dict(self) -> dict:
        return {"parameters": {name: t.detach().cpu().clone() for name, t in self.parameters.items()},
                "optimizer_state": {name: t.detach().cpu().clone() for name, t in self.optimizer_state.items()}}

    def load_state_dict(self, state: dict):
        for kind in ("parameters", "optimizer_state"):
            bound = getattr(self, kind)
            if set(bound) != set(state[kind]):
                raise ValueError(f"{kind} names differ from this program's")
            for name, tensor in bound.items():
                tensor.copy_(state[kind][name])
