"""CPU-built/native-run layout regression; neither command launches resources.

CPU: python -m ops.test_nested_loop_native build --root /out/nested-layout
GPU: python -m ops.test_nested_loop_native gpu --root /out/nested-layout
The CPU artifact includes independent autograd/torch.optim references. GPU work
does no capture, compilation, or reference computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from megakernel_lab.operation_compiler.lower import GeneratedProgram, lower
from .test_nested_loop import assert_close, transposed_fixture


COUNTS = ((0, 0), (1, 0), (1, 1), (3, 3), (3, 1), (0, 3), (1, 2))
MODES = ("save_body_tape", "recompute_boundaries")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_args(args, device="cpu"):
    return tuple(value.detach().to(device).clone(memory_format=torch.preserve_format) for value in args)


def independent_reference(flat):
    """Ordinary autograd and torch.optim; no compiler or captured graph calls."""
    weight = flat[0].detach().clone().requires_grad_(True)
    flag, x = flat[2].clone(), flat[3]
    inner, outer = int(flat[-2].item()), int(flat[-1].item())
    inner_count, outer_count = min(max(inner, 0), 3), min(max(outer, 0), 3)
    h = (x * weight).transpose(0, 1)
    for _ in range(outer_count):
        z = h.transpose(0, 1)
        for _ in range(inner_count):
            z = z + 0.1 * h.transpose(0, 1) * weight
            flag = torch.logical_not(flag)
        h = z.transpose(0, 1).reshape(-1).reshape(3, 2) + 0.03 * weight.transpose(0, 1)
    loss = h.reshape(-1).square().mean()
    (gradient,) = torch.autograd.grad(loss, (weight,))
    optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9, weight_decay=0.0, foreach=False)
    optimizer.state[weight]["momentum_buffer"] = flat[1].clone()
    weight.grad = gradient.clone()
    optimizer.step()
    return copy_args(
        (
            loss,
            gradient,
            weight,
            optimizer.state[weight]["momentum_buffer"],
            flag,
            h,
            flat[-1].new_tensor(outer_count),
            flat[-1].new_tensor(int(outer != outer_count or inner != inner_count)),
        )
    )


def build_artifact(root):
    from megakernel_lab.operation_compiler.runtime import build

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    captured = transposed_fixture(noncontiguous_input=True)
    initial = copy_args(captured.example_args)
    assert not initial[3].is_contiguous()
    snapshots, trajectory = [], []
    state = initial
    for outer, inner in COUNTS:
        args = (*copy_args(initial[:-2]), torch.tensor(inner), torch.tensor(outer))
        expected = independent_reference(args)
        for mode in MODES:
            assert_close(captured.graph_step(*args, storage_mode=mode), expected)
        snapshots.append({"counts": (outer, inner), "args": args, "expected": expected})
        args = (*copy_args(state[:-2]), torch.tensor(inner), torch.tensor(outer))
        expected = independent_reference(args)
        trajectory.append({"counts": (outer, inner), "args": args, "expected": expected})
        state = (expected[2], expected[3], expected[4], initial[3], args[-2], args[-1])
    torch.save({"initial": initial, "snapshots": snapshots, "trajectory": trajectory}, root / "reference.pt")
    summary = {"status": "PASS", "input_stride": list(initial[3].stride()), "counts": COUNTS, "builds": {}}
    for mode in MODES:
        captured.ir.storage_mode = mode
        program = lower(captured.ir, execution="compose", precision_policy="strict")
        directory = root / mode
        directory.mkdir(exist_ok=True)
        program.write(directory)
        library = build(program, cache_dir=root / "cache", arch="sm_90")
        shutil.copyfile(library, directory / "program.so")
        summary["builds"][mode] = {
            "source_hash": program.cache_key,
            "library_sha256": sha(directory / "program.so"),
            "arena_bytes": program.manifest["arena_bytes"],
        }
    (root / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def gpu_check(root):
    from megakernel_lab.operation_compiler.runtime import Runtime

    root = Path(root)
    expected_build = json.loads((root / "build-summary.json").read_text())
    refs = torch.load(root / "reference.pt", weights_only=True)
    summary = {"status": "PASS", "modes": {}}
    for mode in MODES:
        directory = root / mode
        program = GeneratedProgram.load(directory)
        library = directory / "program.so"
        assert program.cache_key == expected_build["builds"][mode]["source_hash"]
        assert sha(library) == expected_build["builds"][mode]["library_sha256"]
        runtime = Runtime(program, copy_args(refs["initial"], "cuda"), library=library)
        pointers = tuple(value.data_ptr() for value in runtime.args)

        def stage(args):
            for dst, src in zip(runtime.args, args):
                dst.copy_(src)

        direct = []
        for row in refs["snapshots"]:
            runtime.arena.fill_(float("nan"))
            stage(row["args"])
            error = assert_close(copy_args(runtime.step()), row["expected"])
            direct.append({"counts": row["counts"], "max_absolute_error": error})
        stage(refs["initial"])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = runtime.step()
        replay = []
        for row in refs["snapshots"]:
            runtime.arena.fill_(float("nan"))
            stage(row["args"])
            graph.replay()
            error = assert_close(copy_args(outputs), row["expected"])
            replay.append({"counts": row["counts"], "max_absolute_error": error})
        stage(refs["initial"])
        trajectory = []
        for row in refs["trajectory"]:
            runtime.args[-2].fill_(row["counts"][1])
            runtime.args[-1].fill_(row["counts"][0])
            graph.replay()
            trajectory.append(assert_close(copy_args(outputs), row["expected"]))
        assert pointers == tuple(value.data_ptr() for value in runtime.args)
        summary["modes"][mode] = {
            "direct": direct,
            "cuda_graph_replay": replay,
            "changing_state_trajectory": trajectory,
            "one_graph_and_same_count_pointers": True,
            "source_hash": program.cache_key,
            "library_sha256": sha(library),
        }
    (root / "gpu-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "gpu"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = build_artifact(args.root) if args.command == "build" else gpu_check(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
