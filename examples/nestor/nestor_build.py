"""Capture and build the Nestor accumulate or final training program with Latch on a CPU-only worker.

    python -B nestor_build.py --inventory DIR/inventory-full.json --init DIR/init-full.pt --mode accumulate \
        --batch 8 --out OUT --cache-dir CACHE [--device-unit-jobs 28] [--capture-only] [--ir-sha256]
    python -B nestor_build.py --print-options

The model is ``nestor_model.CapturedNestor`` over the upstream step-0 parameters (``nestor_inventory.py``); the
update is the upstream MuonAdamW / global clip / gradient-accumulation rule (``nestor_optimizer.make_rule``); the
objective is upstream's causal LM loss times a staged coefficient. One training update of B8 x 4096 x 8 microbatches
(262,144 tokens) runs the ``accumulate`` program seven times and the ``final`` program once.

Capture uses zero token, noise and label inputs of the static shape; values are runtime inputs and do not change the
program. Every capture or build refusal is recorded verbatim in ``OUT/build.json`` (stage, type, message, traceback
tail); nothing is retried with another formulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

# Build flags of the measured programs.
BASE = dict(arch="sm_90", execution="compose", precision_policy="practical", bf16_weight_chunk_k=8192,
            square_sum_association="balanced", gradient_outputs="returned", native_code_only=True,
            device_units=16, device_unit_inline="isolate_all", device_unit_split=150000,
            device_unit_shared_pool=False, static_shared_reserve_bytes=8192,
            compact_codegen="elementwise_full_sum_resource_pure_intervals", device_unit_max_registers=255)
# Lowering rules of the measured programs (see README.md for what each token does).
RULES = dict(update_schedule="streamed_prep", attention_rules="streamed",
             segmented_products="flat+tma+rotate+claim+sepi+ktail+fit+bsearch+scatter+route+packsum+kcat",
             matrix_bodies="exact+skinny+skinnybatch+shortk+splitctr",
             accumulation_rules="update+fsload+fsspread+rilp+fsdeepx+rinl+rleaf+rhoist2+rsum1"
                                "+rwarp+rstage+ridx+rearly+rremat+rslot+rnarrow+rmid",
             segmented_attention="folded_forward+fused_backward+claim+rotary",
             tile_readiness="attention", vocab_rules="stream+views+rows",
             elementwise_rules="lanecse+bf16vec+rolling+divc+indexed+permrows+embunroll+rowcopy+acopy",
             graph_rules="cse+depth_reads+castshare")
# Newton-Schulz slice padding and tall orientation of the optimizer rule (nestor_optimizer.make_rule).
NS_PAD, NS_TALL = 8, True
INDEXED_POLICY = "segmented_v1"
MODES = ("accumulate", "final")


def options(device_unit_jobs):
    """The complete keyword set passed to ``CapturedModule.build``."""
    return dict(BASE, **RULES, device_unit_jobs=device_unit_jobs, explain=False)


def load(inventory, init, *, mode):
    import torch
    from nestor_model import CapturedNestor, Geometry, bias_state_from_upstream
    inv = json.loads(Path(inventory).read_text())
    geometry = Geometry.from_record(inv["shape"])
    state_dict = torch.load(init, map_location="cpu", weights_only=True)
    names = [row["name"] for row in inv["parameters"]]
    params = {n: state_dict[n] for n in names}
    model = CapturedNestor(geometry, params, precision="bf16", attention="latch", update=mode)
    return model, geometry, inv, bias_state_from_upstream(geometry, state_dict)


# Fields of ir.metadata["optimizer"] that record where the update function was written (its defining module and
# the sha256 of its source text) rather than what was captured. A copy of the same rule in another module, or with
# a differently spelled source that captures the same operations, changes them, so the identity hash omits them.
OPTIMIZER_PROVENANCE = ("module", "source_sha256")


def graph_identity(ir):
    """``ir.to_dict()`` without the optimizer rule's source provenance (``OPTIMIZER_PROVENANCE``).

    Operations, values, inputs, outputs and every other metadata entry, including the rule's name, function
    qualname, hyperparameters and state schema, are kept.
    """
    data = ir.to_dict()
    optimizer = data.get("metadata", {}).get("optimizer")
    if isinstance(optimizer, dict):
        for key in OPTIMIZER_PROVENANCE:
            optimizer.pop(key, None)
    return data


def ir_sha256(captured):
    """sha256 of ``graph_identity(captured.ir)`` as JSON (keys sorted; non-JSON leaves as ``str``)."""
    text = json.dumps(graph_identity(captured.ir), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inventory", type=Path)
    p.add_argument("--init", type=Path)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=Path, help="new directory: artifact/ and build.json are written here")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--device-unit-jobs", type=int, default=16, help="concurrent device-unit compiles (no effect "
                   "on the generated program)")
    p.add_argument("--capture-only", action="store_true", help="capture and record; do not lower or compile")
    p.add_argument("--ir-sha256", action="store_true", help="also print the captured-graph sha256 (optimizer "
                   "source provenance omitted) as one JSON line")
    p.add_argument("--print-options", action="store_true", help="print the build keywords as JSON and exit")
    a = p.parse_args()
    if a.print_options:
        print(json.dumps(dict(options(a.device_unit_jobs), indexed_reduction_policy=INDEXED_POLICY,
                              ns_pad=NS_PAD, ns_tall=NS_TALL), indent=2, sort_keys=True))
        return
    missing = [flag for flag, value in (("--inventory", a.inventory), ("--init", a.init), ("--mode", a.mode),
                                        ("--out", a.out)) if value is None]
    if not a.capture_only and a.cache_dir is None:
        missing.append("--cache-dir")
    if missing:
        p.error("required: " + ", ".join(missing))
    if a.batch < 1:
        p.error("--batch must be positive")
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise SystemExit("capture and build belong on a CPU-only worker (no visible CUDA device)")
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.bf16_sdpa import capture_bf16_sdpa
    from megakernel_lab.operation_compiler.indexed_reduction import policy as indexed_policy
    from nestor_model import initial_state, objective
    from nestor_optimizer import make_rule
    a.out.mkdir(parents=True, exist_ok=False)
    record = dict(schema="nestor-example-build-v1", mode=a.mode, batch=a.batch,
                  inventory_sha256=hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
                  init_sha256=hashlib.sha256(a.init.read_bytes()).hexdigest(), torch=torch.__version__)
    started = time.perf_counter()
    stage = "load"
    try:
        model, geometry, inv, bias = load(a.inventory, a.init, mode=a.mode)
        if isinstance(inv.get("optimizer_groups"), dict):
            raise RuntimeError("inventory has no upstream optimizer groups: " + inv["optimizer_groups"]["error"])
        record.update(fixture=inv["fixture"], context=geometry.context,
                      parameters=sum(t.numel() for t in model.parameters()),
                      parameter_tensors=len(list(model.parameters())))
        rule = make_rule(model, inv["optimizer_groups"], a.mode, ns_pad=NS_PAD, ns_tall=NS_TALL)
        state = initial_state(geometry, observe=False, batch=a.batch, choice_bias=bias)
        tokens = torch.zeros(a.batch, geometry.context, dtype=torch.int64)
        noise = torch.zeros(geometry.blocks, a.batch, geometry.context, 8, dtype=torch.float32)
        labels = torch.zeros_like(tokens)
        coefficients = torch.ones(1, dtype=torch.float32)
        stage = "capture"
        with capture_bf16_sdpa():
            captured = capture_module(model, objective, (tokens, noise), (labels, coefficients), optimizer=rule,
                                      state=state)
        record.update(capture_seconds=time.perf_counter() - started, operations=len(captured.ir.operations))
        census = {}
        for op in captured.ir.operations:
            target = str(getattr(op, "target", getattr(op, "op", "?")))
            census[target] = census.get(target, 0) + 1
        record["operation_census"] = dict(sorted(census.items(), key=lambda kv: -kv[1]))
        record["ir_sha256"] = ir_sha256(captured)
        optimizer = captured.ir.metadata.get("optimizer") or {}
        record["optimizer_provenance"] = {key: optimizer.get(key) for key in OPTIMIZER_PROVENANCE}
        print(json.dumps(dict(event="capture_complete", seconds=record["capture_seconds"],
                              operations=record["operations"])), flush=True)
        if a.ir_sha256:
            print(json.dumps(dict(mode=a.mode, batch=a.batch, ir_sha256=record["ir_sha256"])), flush=True)
        if a.capture_only:
            stage = "done"
        else:
            stage = "build"
            flags = options(a.device_unit_jobs)
            record["flags"] = dict(flags, indexed_reduction_policy=INDEXED_POLICY, ns_pad=NS_PAD, ns_tall=NS_TALL)
            t0 = time.perf_counter()
            with indexed_policy(INDEXED_POLICY):
                receipt = captured.build(a.out / "artifact", cache_dir=a.cache_dir, **flags)
            record.update(build_seconds=time.perf_counter() - t0, artifact=receipt)
            stage = "done"
    except BaseException as exc:
        record["error"] = dict(stage=stage, type=type(exc).__name__, message=str(exc)[:4000],
                               traceback=traceback.format_exc()[-6000:])
        print(json.dumps(dict(event="error", stage=stage, type=type(exc).__name__, message=str(exc)[:1500])),
              flush=True)
    record["wall_seconds"] = time.perf_counter() - started
    record["build_complete"] = stage == "done"
    here = Path(__file__).resolve().parent
    record["sources"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(here.glob("*.py"))}
    (a.out / "build.json").write_text(json.dumps(record, indent=2, default=repr) + "\n")
    raise SystemExit(0 if stage == "done" else 3)


if __name__ == "__main__":
    main()
