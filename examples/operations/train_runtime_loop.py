"""Shared-parameter runtime loop, intermediate objective and explicit dropout.

CPU prepare saves source and bindings and builds once. GPU run changes the
bound device count across updates, optionally replaying one CUDA graph.
Full reverse history is differentiated by PyTorch-derived per-iteration VJPs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from megakernel_lab.operation_compiler import SGDConfig, capture_loop_training
from megakernel_lab.operation_compiler.lower import GeneratedProgram, lower
from megakernel_lab.operation_compiler.random import dropout, make_rng_state
from megakernel_lab.operation_compiler.runtime import Runtime, build


def initial(params, x, target):
    hidden = x @ params["encoder"]
    return {"hidden": hidden, "auxiliary": hidden.sum() * 0.0}


def body(params, carry, state, x, target):
    hidden = carry["hidden"] + 0.1 * torch.tanh(carry["hidden"] @ params["shared"])
    hidden, rng = dropout(hidden, state["rng"], p=0.2)
    auxiliary = carry["auxiliary"] + (hidden @ params["head"] - target).square().mean()
    return {"hidden": hidden, "auxiliary": auxiliary}, {"rng": rng, "iterations": state["iterations"] + 1}


def loss(params, carry, x, target):
    return (carry["hidden"] @ params["head"] - target).square().mean() + 0.05 * carry["auxiliary"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(directory, cache_dir, memory_policy):
    torch.set_num_threads(1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(17)
    params = {name: torch.randn(64, 64, generator=generator) * 0.1 for name in ("encoder", "shared", "head")}
    inputs = (torch.randn(64, 64, generator=generator) * 0.2, torch.randn(64, 64, generator=generator) * 0.1)
    state = {"rng": make_rng_state(41), "iterations": torch.zeros((), dtype=torch.int64)}
    captured = capture_loop_training(initial, body, loss, params, None, inputs, torch.tensor(0),
                                     SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.001),
                                     max_steps=3, state=state, memory_policy=memory_policy)
    program = lower(captured.ir, execution="compose")
    program.write(directory)
    library = build(program, cache_dir=cache_dir, arch="sm_90")
    torch.save(tuple(value.detach().clone() for value in captured.example_args), directory / "bindings.pt")
    (directory / "graph.json").write_text(json.dumps(captured.ir.to_dict(), sort_keys=True))
    receipt = {"source_hash": program.cache_key, "library": str(library), "library_sha256": digest(library),
               "bindings_sha256": digest(directory / "bindings.pt"), "graph_sha256": digest(directory / "graph.json"),
               "memory_selection": captured.ir.metadata["memory_selection"]}
    (directory / "example.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def run(directory, counts, cuda_graph):
    torch.set_num_threads(1)
    directory = Path(directory)
    receipt = json.loads((directory / "example.json").read_text())
    for file, key in ((receipt["library"], "library_sha256"), (directory / "bindings.pt", "bindings_sha256"),
                      (directory / "graph.json", "graph_sha256")):
        if digest(file) != receipt[key]:
            raise ValueError("example artifact differs from its CPU build receipt")
    program = GeneratedProgram.load(directory)
    if program.cache_key != receipt["source_hash"]:
        raise ValueError("native source differs from the CPU build")
    saved = torch.load(directory / "bindings.pt", weights_only=True, map_location="cpu")
    runtime = Runtime(program, tuple(value.cuda() for value in saved), library=receipt["library"])
    identity = (program.cache_key, tuple(value.data_ptr() for value in runtime.args))
    graph = None
    if cuda_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            runtime.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runtime.step()
        for target, source in zip(runtime.args, saved):
            target.copy_(source)
    outputs = runtime.manifest["structured_outputs"]
    rows, total = [], 0
    for count in counts:
        runtime.args[-1].fill_(count)
        runtime.step() if graph is None else graph.replay()
        executed = int(runtime.values[outputs["executed_count"]])
        status = int(runtime.values[outputs["status"]])
        assert executed == min(max(count, 0), 3) and status == int(executed != count)
        total += executed
        assert int(runtime.state["iterations"]) == total
        rows.append({"requested": count, "executed": executed, "clamped": bool(status),
                     "loss": float(runtime.loss), "rng": runtime.state["rng"].cpu().tolist()})
    assert identity == (program.cache_key, tuple(value.data_ptr() for value in runtime.args))
    return {"source_hash": program.cache_key, "cuda_graph": cuda_graph, "same_bindings": True, "updates": rows,
            "scope": "synthetic full-history/control/RNG example; no speed or real-data quality claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    cpu = actions.add_parser("prepare")
    cpu.add_argument("directory", type=Path)
    cpu.add_argument("--cache-dir", type=Path, required=True)
    cpu.add_argument("--memory-policy", choices=("default", "low_memory"), default="default")
    gpu = actions.add_parser("run")
    gpu.add_argument("directory", type=Path)
    gpu.add_argument("--counts", type=int, nargs="+", default=[0, 1, 3, 2, -1, 5])
    gpu.add_argument("--cuda-graph", action="store_true")
    options = vars(parser.parse_args())
    action = options.pop("action")
    print(json.dumps((prepare if action == "prepare" else run)(**options), indent=2))


if __name__ == "__main__":
    main()
