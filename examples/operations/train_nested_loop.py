"""Build and run a nested shared-parameter refinement update.

Build on an admitted CPU worker with nvcc; run the frozen binary on an H100.
No CUDA, cloud provisioning or compilation is attempted by importing this file.
"""
from pathlib import Path
import argparse
import hashlib



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--directory", required=True)
    parser.add_argument("--storage", choices=("save_body_tape", "recompute_boundaries"), default="save_body_tape")
    args = parser.parse_args()
    import json
    import torch
    from megakernel_lab.operation_compiler.lower import GeneratedProgram, lower
    from megakernel_lab.operation_compiler.runtime import Runtime, build
    from nested_workload import CASES, capture

    torch.set_num_threads(1)
    directory = Path(args.directory)
    if args.action == "build":
        directory.mkdir(parents=True, exist_ok=True)
        captured = capture(CASES["tiny"], args.storage)
        program = lower(captured.ir, execution="compose", precision_policy="strict")
        program.write(directory)
        library = build(program, cache_dir=directory / "cache", arch="sm_90")
        torch.save(captured.example_args, directory / "inputs.pt")
        (directory / "example.json").write_text(json.dumps({
            "library": str(library), "source_hash": program.cache_key,
            "library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256((directory / "inputs.pt").read_bytes()).hexdigest(),
        }) + "\n")
        print("Built a complete nested forward/loss/backward/SGD update:", directory)
    else:
        program = GeneratedProgram.load(directory)
        receipt = json.loads((directory / "example.json").read_text())
        if (program.cache_key != receipt['source_hash']
                or hashlib.sha256(Path(receipt['library']).read_bytes()).hexdigest() != receipt['library_sha256']
                or hashlib.sha256((directory / 'inputs.pt').read_bytes()).hexdigest() != receipt['inputs_sha256']):
            raise ValueError('nested example artifact differs from its CPU receipt')
        initial = torch.load(directory / "inputs.pt", weights_only=True)
        runtime = Runtime(program, tuple(v.cuda() for v in initial), library=receipt["library"])
        # Change both counts in stable device storages. The same native program
        # performs every update, including parameter/optimizer/RNG commits.
        rows, visits = [], 0
        for outer, inner in ((0, 0), (1, 1), (3, 2), (2, 4)):
            runtime.args[-2].fill_(inner)
            runtime.args[-1].fill_(outer)
            runtime.step()
            visits += outer * inner
            assert int(runtime.state['visits']) == visits and bool(torch.isfinite(runtime.loss))
            rows.append({"outer": outer, "inner": inner, "loss": float(runtime.loss),
                         "dropout_calls_total": int(runtime.state["visits"])})
        result = {'passed': True, 'source_hash': program.cache_key, 'updates': rows,
                  'scope': 'synthetic installed nested-loop API demonstration; no numerical or speed claim'}
        (directory / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
