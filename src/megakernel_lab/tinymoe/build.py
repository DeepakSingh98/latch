"""Capture/build TinyMoE through the ordinary single-module training API on CPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .compiled_model import CapturedTinyMoE, objective
from .model import Recipe, load_model
from .optimizer import initial_model_state, make_rule
from .pins import public_identity


def capture_model(model_directory, *, recipe=None, phase="plain", batch_size=1, sequence_length=512,
                  fixed_routes=None, diagnostics=True, optimizer_factory=None, precision=None):
    """Capture the TinyMoE adapter with an optional pure tensor-update factory.

    The factory has the same signature as ``optimizer.make_rule``:
    ``factory(model, recipe, *, ema_active=False)``. It returns a normal
    ``tensor_update`` rule; it does not replace compiler operations.
    """
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.bf16_sdpa import capture_bf16_sdpa
    recipe = recipe or Recipe()
    if optimizer_factory is not None and not callable(optimizer_factory):
        raise TypeError("optimizer_factory must be callable")
    if phase not in ("plain", "auxiliary") or (phase == "auxiliary" and recipe.name != "supervision_ema"):
        raise ValueError("auxiliary phase requires the supervision_ema recipe")
    if batch_size < 1 or not 256 <= sequence_length <= 1024 or sequence_length % 128:
        raise ValueError("requires batch>=1 and sequence 256..1024 divisible by 128")
    original = load_model(model_directory, device="cpu", attention="sdpa")
    # LATCH-TRAIN-PRECISION: opt-in training precision policy (tinymoe/precision.py); None keeps the released program.
    model = (CapturedTinyMoE(original, auxiliary=phase == "auxiliary") if precision is None else
             CapturedTinyMoE(original, auxiliary=phase == "auxiliary", precision=precision))
    ema_active = recipe.name == "supervision_ema" and phase == "plain"
    tokens = torch.zeros(batch_size, sequence_length, dtype=torch.int64)
    labels = torch.zeros_like(tokens)
    weight = 0.25 if phase == "auxiliary" else 0.0
    coefficients = torch.tensor([1 / (1 + weight), weight / (2 * (1 + weight))], dtype=torch.float32)
    model_state = initial_model_state(model)
    if diagnostics:
        for index in range(len(model.model.layers)):
            model_state[f"selected.{index}"] = torch.zeros(batch_size * sequence_length, model.top_k, dtype=torch.int64)
            model_state[f"probabilities.{index}"] = torch.zeros(batch_size * sequence_length, model.num_experts)
    with capture_bf16_sdpa():
        captured = capture_module(model, objective, (tokens,) if fixed_routes is None else (tokens, fixed_routes),
            (labels, coefficients),
            optimizer=(optimizer_factory or make_rule)(model, recipe, ema_active=ema_active), state=model_state)
    captured.ir.metadata["tinymoe"] = dict(public_identity(), recipe=asdict(recipe), phase=phase,
        batch_size=batch_size, sequence_length=sequence_length, ema_active=ema_active,
        routing="unconstrained" if fixed_routes is None else "diagnostic supplied membership",
        routing_diagnostics=bool(diagnostics),
        objective="sum cross-entropies / target tokens, staged normalized coefficients, plus 0.02 global router aux",
        attention="BF16 GQA; real-width RoPE; zero pad 48 to 64; true 1/sqrt(48) scale; causal window1024",
        expert_policy="top2 token choice; no capacity/no dropped tokens; unused-expert AdamW participation")
    if precision is not None:
        from .precision import describe
        captured.ir.metadata["tinymoe"]["training_precision"] = describe(precision)
    from .variant import identity_of
    if identity_of(model_directory) is not None:
        # Edited architecture (variant.py): record it; the public pin fields describe its base.
        captured.ir.metadata["tinymoe"].update(variant_identity=identity_of(model_directory),
            expert_policy=f"top{model.top_k} of {model.num_experts} token choice; no capacity/no dropped tokens; "
                          "unused-expert AdamW participation")
    return model, captured


# LATCH-B32-TRANSFER: automatic selection of the existing exact execution rules for one-update TinyMoE training
# programs (one optimizer update per program step, no accumulation). Measured on recipe Q (B32 x 1024) against the
# frozen L32 donor on one idle H100: every option below was kept only because removing it (COMBINED options) or
# adding it alone (accumulation rules) did not make the complete update slower; the selected program is bitwise
# equal to the donor. gradient_outputs is NOT selected: it changes the API (Runtime.grads) and gave no time alone.
#
# LATCH-TRAIN-INTEGRATE (training compiler wave, candidate c2): two exact rules added to the selection after paired
# measurement of the combined package on two H100s (bitwise traces vs the previous selection at fresh/mid/late states):
#   segmented_products "+claim" (LATCH-TRAIN-EXPERTS; structural admission: ragged products with >= CLAIM_MIN_ROWS
#   packed rows per expert, otherwise byte-identical) - recipe Q 80.9 -> 74.3 ms, B2 not admitted;
#   accumulation_rules "+fsload+fsspread" (LATCH-TRAIN-SYNC; exact full-sum bodies) - B2 14.36 -> 13.69 ms, Q -1.1%;
#   together recipe Q 80.9 -> 73.5 ms. Previous selection: segmented_products="flat+tma+rotate",
#   accumulation_rules="update".
#   c3: matrix_bodies "exact" (LATCH-TRAIN-MATMUL; provider TMA tail/fill bodies, bitwise) - recipe Q -3.8% on the release.
#   final: segmented_products "+sepi" (LATCH-TRAIN-EXPERTS final ef7a68a47: staged epilogue of claimed products;
#   recipe Q -9.7% with claim on the release, bitwise; B2 not admitted).
UPDATE_RULES = dict(update_schedule="streamed_prep", segmented_products="flat+tma+rotate+claim+sepi", norm_fusion="rows",
                    attention_rules="streamed", accumulation_rules="update+fsload+fsspread", matrix_bodies="exact")


def resolve_rules(rules, **explicit):
    """Options for ``prepare``: ``rules=None`` returns ``explicit`` unchanged; ``rules="auto"`` fills every option
    of UPDATE_RULES that the caller left None and refuses an explicit value that differs from the selection."""
    if rules is None:
        return explicit
    if rules != "auto":
        raise ValueError("rules must be None or 'auto'")
    chosen = dict(explicit)
    for name, value in UPDATE_RULES.items():
        if chosen.get(name) is not None and chosen[name] != value:
            raise ValueError(f"rules='auto' selects {name}={value!r}; got {chosen[name]!r}")
        chosen[name] = value
    return chosen


def b32_rules():
    """The automatic selection as explicit prepare options (recipe Q / L32 builders)."""
    return dict(UPDATE_RULES)


def prepare(model_directory, out, cache_dir, *, recipe, phase, batch_size, sequence_length,
            capture_only=False, explain=False, fixed_routes=None, diagnostics=True,
            fast_device_compile=False, compact_elementwise=False, native_code_only=False,
            outline_full_sums=False, disable_expensive_optimizations=False, phase_local_pointers=False,
            ptxas_optimization_level=None, mixed_phase_local_pointers=False,
            resource_phase_local_pointers=False, outline_pure_intervals=False, outline_pure_runs=False,
            outline_attention_items=False, indexed_codegen="off", device_units=None,
            device_unit_inline="wgmma", device_unit_cache=None, device_unit_jobs=None, device_unit_split=None,
            device_unit_shared_pool=False, static_shared_reserve_bytes=0, optimizer_factory=None,
            update_schedule=None, segmented_products=None, norm_fusion=None, attention_rules=None, matrix_schedule=None,
            gradient_outputs="returned", accumulation_rules=None, rules=None, matrix_bodies=None, precision=None):
    # LATCH-SELECTIVE-10: gradient_outputs (release step 15, ops.gradient_consumption) and accumulation_rules
    # (ops.accumulation_rules) pass through to CapturedModule.build; the defaults leave lowering unchanged.
    # LATCH-B32-TRANSFER: rules="auto" selects UPDATE_RULES (resolve_rules); None leaves every option as given.
    selected = resolve_rules(rules, update_schedule=update_schedule, segmented_products=segmented_products,
                             norm_fusion=norm_fusion, attention_rules=attention_rules,
                             accumulation_rules=accumulation_rules, matrix_bodies=matrix_bodies)
    update_schedule, segmented_products = selected["update_schedule"], selected["segmented_products"]
    norm_fusion, attention_rules = selected["norm_fusion"], selected["attention_rules"]
    accumulation_rules, matrix_bodies = selected["accumulation_rules"], selected.get("matrix_bodies")
    if gradient_outputs not in ("returned", "consumed"):
        raise ValueError("gradient_outputs must be 'returned' or 'consumed'")
    if type(static_shared_reserve_bytes) is not int or static_shared_reserve_bytes < 0:
        raise ValueError("static_shared_reserve_bytes must be a nonnegative integer")
    if static_shared_reserve_bytes and capture_only:
        raise ValueError("static shared reserve requires source emission")
    if indexed_codegen not in ("off", "typed_v1"):
        raise ValueError("indexed_codegen must be off or typed_v1")
    if indexed_codegen != "off" and (capture_only or outline_attention_items):
        raise ValueError("indexed codegen requires source emission without attention-item outlining")
    if ptxas_optimization_level is not None and (
            type(ptxas_optimization_level) is not int or ptxas_optimization_level not in range(4)):
        raise ValueError("ptxas_optimization_level must be None or an integer from 0 through 3")
    if ptxas_optimization_level is not None and fast_device_compile:
        raise ValueError("explicit ptxas optimization does not combine with fast_device_compile")
    if type(phase_local_pointers) is not bool:
        raise ValueError("phase_local_pointers must be a boolean")
    if type(mixed_phase_local_pointers) is not bool:
        raise ValueError("mixed_phase_local_pointers must be a boolean")
    if type(resource_phase_local_pointers) is not bool:
        raise ValueError("resource_phase_local_pointers must be a boolean")
    if type(outline_pure_intervals) is not bool:
        raise ValueError("outline_pure_intervals must be a boolean")
    if type(outline_pure_runs) is not bool:
        raise ValueError("outline_pure_runs must be a boolean")
    if type(outline_attention_items) is not bool:
        raise ValueError("outline_attention_items must be a boolean")
    if outline_attention_items and ptxas_optimization_level != 0:
        raise ValueError("attention item outlining requires explicit ptxas_optimization_level=0")
    if outline_attention_items and not outline_pure_runs:
        raise ValueError("attention item outlining requires pure-run outlining")
    if outline_pure_intervals and outline_pure_runs:
        raise ValueError("pure interval and pure run modes are mutually exclusive")
    if (outline_pure_intervals or outline_pure_runs) and not resource_phase_local_pointers:
        raise ValueError("pure interval/run outlining requires resource-phase pointer localization")
    if sum((phase_local_pointers, mixed_phase_local_pointers, resource_phase_local_pointers)) > 1:
        raise ValueError("phase-local pointer modes are mutually exclusive")
    if (phase_local_pointers or mixed_phase_local_pointers or resource_phase_local_pointers) and not (compact_elementwise and outline_full_sums):
        raise ValueError("phase-local pointers require compact elementwise and outlined full sums")
    if torch.cuda.is_available():
        raise RuntimeError("capture/build belongs on an admitted CPU-only worker")
    out, cache_dir = Path(out), Path(cache_dir)
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    model, captured = capture_model(model_directory, recipe=recipe, phase=phase,
                                    batch_size=batch_size, sequence_length=sequence_length, fixed_routes=fixed_routes,
                                    diagnostics=diagnostics, optimizer_factory=optimizer_factory,
                                    **({"precision": precision} if precision is not None else {}))
    captured_seconds = time.perf_counter() - started
    from .precision import forward_fp32
    metadata = captured.ir.metadata["tinymoe"]
    metadata = dict(metadata, **({'update_schedule': update_schedule} if update_schedule is not None else {}),
                    **({'segmented_products': segmented_products} if segmented_products is not None else {}),
                    **({'norm_fusion': norm_fusion} if norm_fusion is not None else {}),
                    **({'attention_rules': attention_rules} if attention_rules is not None else {}),
                    **({'matrix_schedule': matrix_schedule} if matrix_schedule is not None else {}),
                    **({'gradient_outputs': gradient_outputs} if gradient_outputs != "returned" else {}),
                    **({'matrix_bodies': matrix_bodies} if matrix_bodies is not None else {}),
                    **({'accumulation_rules': accumulation_rules} if accumulation_rules is not None else {}),
                    **({'exact_bf16_products': 'tensor'} if forward_fp32(precision) else {}))
    summary = dict(metadata, capture_seconds=captured_seconds, parameters=sum(p.numel() for p in model.parameters()),
                   parameter_tensors=len(list(model.parameters())), operations=len(captured.ir.operations),
                   schema="latch-tinymoe-build-v1")
    print(json.dumps(dict(event="capture_complete", seconds=captured_seconds,
                          operations=summary["operations"], parameters=summary["parameters"])), flush=True)
    if capture_only:
        out.mkdir(parents=True)
        (out / "training_ir.json").write_text(json.dumps(captured.ir.to_dict(), default=repr) + "\n")
    else:
        from megakernel_lab.operation_compiler.indexed_reduction import policy as indexed_policy
        # Repeated vocabulary IDs and packed expert tokens need a fixed
        # occurrence reduction order for exact same-backend checkpoint replay.
        compact_mode = ("elementwise_full_sum" if compact_elementwise else "full_sum") if outline_full_sums else (
            "elementwise" if compact_elementwise else "off")
        if phase_local_pointers:
            compact_mode = "elementwise_full_sum_phase_pointers"
        if mixed_phase_local_pointers:
            compact_mode = "elementwise_full_sum_mixed_phase_pointers"
        if resource_phase_local_pointers:
            compact_mode = "elementwise_full_sum_resource_phase_pointers"
        if outline_pure_intervals:
            compact_mode = "elementwise_full_sum_resource_pure_intervals"
        if outline_pure_runs:
            compact_mode = "elementwise_full_sum_resource_pure_runs"
        if outline_attention_items:
            compact_mode = "elementwise_full_sum_resource_pure_runs_attention_items"
        from contextlib import nullcontext
        from megakernel_lab.operation_compiler.indexed_segmented_outline import preference as indexed_codegen_preference
        codegen_scope = indexed_codegen_preference(indexed_codegen) if indexed_codegen != "off" else nullcontext()
        with indexed_policy("segmented_v1"), codegen_scope:
            receipt = captured.build(out, cache_dir=cache_dir, arch="sm_90", execution="compose",
                precision_policy="practical", bf16_weight_chunk_k=8192, square_sum_association="balanced",
                gradient_outputs=gradient_outputs, explain=explain, fast_device_compile=fast_device_compile,
                native_code_only=native_code_only,
                **({'disable_expensive_optimizations': True} if disable_expensive_optimizations else {}),
                **({'ptxas_optimization_level': ptxas_optimization_level} if ptxas_optimization_level is not None else {}),
                **({'device_units': device_units, 'device_unit_inline': device_unit_inline,
                    'device_unit_cache': device_unit_cache, 'device_unit_jobs': device_unit_jobs,
                    'device_unit_split': device_unit_split, 'device_unit_shared_pool': device_unit_shared_pool}
                   if device_units is not None else {}),
                **({'static_shared_reserve_bytes': static_shared_reserve_bytes}
                   if static_shared_reserve_bytes != 0 else {}),
                **({'update_schedule': update_schedule} if update_schedule is not None else {}),
                **({'segmented_products': segmented_products} if segmented_products is not None else {}),
                **({'norm_fusion': norm_fusion} if norm_fusion is not None else {}),
                **({'attention_rules': attention_rules} if attention_rules is not None else {}),
                **({'matrix_schedule': matrix_schedule} if matrix_schedule is not None else {}),
                **({'accumulation_rules': accumulation_rules} if accumulation_rules is not None else {}),
                **({'matrix_bodies': matrix_bodies} if matrix_bodies is not None else {}),
                # LATCH-TRAIN-PRECISION forward_fp32: its FP32-result dense products are FP32 mm of exact BF16
                # widenings; the opt-in compiler rule runs them on the BF16 tensor-core body (unrounded FP32 result).
                **({'exact_bf16_products': 'tensor'} if forward_fp32(precision) else {}),
                compact_codegen=compact_mode)
        summary["artifact"] = receipt
        summary["indexed_reduction_policy"] = "segmented_v1"
    (out / "tinymoe-build.json").write_text(json.dumps(summary, indent=2, default=repr) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--recipe", choices=("unchanged", "supervision_ema"), default="unchanged")
    parser.add_argument("--phase", choices=("plain", "auxiliary"), default="plain")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--fast-device-compile", action="store_true",
                        help="opt into nvcc maximum device compile preference; requires fresh qualification")
    parser.add_argument("--compact-elementwise", action="store_true",
                        help="share checked repeated elementwise bodies; requires fresh qualification")
    parser.add_argument("--outline-full-sums", action="store_true",
                        help="outline exact balanced full-sum bodies; requires fresh qualification")
    parser.add_argument("--phase-local-pointers", action="store_true",
                        help="derive opaque pointers inside pure phases; requires both compaction options and fresh qualification")
    parser.add_argument("--mixed-phase-local-pointers", action="store_true",
                        help="localize only pure-exclusive pointers in mixed intervals; requires both compaction options")
    parser.add_argument("--resource-phase-local-pointers", action="store_true",
                        help="also localize exact counter-init and concatenation pointer bindings; requires both compaction options")
    parser.add_argument("--outline-pure-intervals", action="store_true",
                        help="outline exact pure publication bodies; requires resource-phase pointer localization")
    parser.add_argument("--outline-pure-runs", action="store_true",
                        help="also outline pure runs within mixed intervals; requires resource-phase pointer localization")
    parser.add_argument("--outline-attention-items", action="store_true",
                        help="wrap audited completed v7 attention items; requires pure-run outlining and explicit ptxas level 0")
    parser.add_argument("--indexed-codegen", choices=("off", "typed_v1"), default="off",
                        help="outline typed deterministic indexed bodies during normal lowering")
    parser.add_argument("--native-code-only", action="store_true",
                        help="emit only native target GPU code, without PTX fallback images")
    parser.add_argument("--disable-expensive-optimizations", action="store_true",
                        help="disable expensive ptxas optimization; requires fresh qualification")
    parser.add_argument("--ptxas-optimization-level", type=int, choices=range(4),
                        help="explicit assembler optimization level; omission preserves defaults")
    parser.add_argument("--device-units", type=int,
                        help="compile publication intervals as K relocatable device units behind one entry")
    parser.add_argument("--device-unit-inline", choices=("wgmma", "none", "isolate"), default="wgmma",
                        help="keep warpgroup-MMA intervals in the entry or isolate their call-free pieces")
    parser.add_argument("--device-unit-cache", type=Path, help="reuse unit cubins by exact unit/header/flag identity")
    parser.add_argument("--device-unit-jobs", type=int, help="concurrent unit compiles (default: one per unit)")
    parser.add_argument("--device-unit-split", type=int,
                        help="split intervals larger than this many source bytes into phase-run pieces")
    parser.add_argument("--device-unit-shared-pool", action="store_true",
                        help="pool block-scope shared flags across sequential interval functions")
    parser.add_argument("--static-shared-reserve-bytes", type=int, default=0,
                        help="reserve shared-memory headroom for linked static storage during common lowering")
    parser.add_argument("--trace-build", action="store_true",
                        help="log the CPU stack every five minutes during long capture/build passes")
    parser.add_argument("--fixed-routes", type=Path, help="diagnostic torch-saved int64 [layers,tokens,top_k] tensor")
    parser.add_argument("--no-routing-observations", action="store_true",
                        help="omit default routing state observations; requires separate functional validation")
    # LATCH-V0-NEXT: the accepted exact execution rules through the normal command (prepare() arguments).
    parser.add_argument("--rules", choices=("auto",),
                        help="select UPDATE_RULES (streamed_prep, flat+tma+rotate+claim+sepi, rows, streamed attention, "
                             "accumulation 'update+fsload+fsspread', matrix_bodies 'exact'); an explicit conflicting "
                             "option is refused")
    parser.add_argument("--update-schedule", help="ops.update_schedule value, e.g. streamed_prep")
    parser.add_argument("--segmented-products", help="e.g. flat+tma+rotate or flat+tma+rotate+narrow")
    parser.add_argument("--norm-fusion", help="e.g. rows or rows+pcse")
    parser.add_argument("--attention-rules", choices=("streamed",))
    parser.add_argument("--accumulation-rules", help="a preset ('update') or '+'-joined rules")
    parser.add_argument("--matrix-schedule", choices=("underfilled", "exact", "exact_split"),
                        help="underfilled reassociates (not bitwise); exact and exact_split are bitwise")
    parser.add_argument("--matrix-bodies", help="ops.matrix_bodies rules: '+'-joined (tailrot, taildeep, fill, p5c, isolate) or 'exact'")
    parser.add_argument("--gradient-outputs", choices=("returned", "consumed"), default="returned",
                        help="consumed leaves Runtime.grads empty")
    from .precision import POLICIES as _POLICIES
    parser.add_argument("--precision", choices=tuple(p for p in _POLICIES if p != "release"),
                        help="opt-in training precision policy (tinymoe/precision.py); omission = released arithmetic")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.trace_build:
        import faulthandler
        faulthandler.dump_traceback_later(300, repeat=True)
    recipe = Recipe(name=args.recipe, horizon=args.horizon, learning_rate=args.learning_rate)
    fixed_routes = None if args.fixed_routes is None else torch.load(args.fixed_routes, map_location="cpu", weights_only=True)
    if fixed_routes is not None and not isinstance(fixed_routes, torch.Tensor):
        raise ValueError("fixed-routes file must contain one Tensor")
    result = prepare(args.model, args.out, args.cache_dir, recipe=recipe, phase=args.phase,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        capture_only=args.capture_only, explain=args.explain, fixed_routes=fixed_routes,
        diagnostics=not args.no_routing_observations, fast_device_compile=args.fast_device_compile,
        compact_elementwise=args.compact_elementwise, native_code_only=args.native_code_only,
        outline_full_sums=args.outline_full_sums,
        phase_local_pointers=args.phase_local_pointers,
        mixed_phase_local_pointers=args.mixed_phase_local_pointers,
        resource_phase_local_pointers=args.resource_phase_local_pointers,
        outline_pure_intervals=args.outline_pure_intervals,
        outline_pure_runs=args.outline_pure_runs,
        outline_attention_items=args.outline_attention_items,
        indexed_codegen=args.indexed_codegen,
        ptxas_optimization_level=args.ptxas_optimization_level,
        disable_expensive_optimizations=args.disable_expensive_optimizations,
        device_units=args.device_units, device_unit_inline=args.device_unit_inline,
        device_unit_cache=args.device_unit_cache, device_unit_jobs=args.device_unit_jobs,
        device_unit_split=args.device_unit_split, device_unit_shared_pool=args.device_unit_shared_pool,
        static_shared_reserve_bytes=args.static_shared_reserve_bytes,
        update_schedule=args.update_schedule, segmented_products=args.segmented_products,
        norm_fusion=args.norm_fusion, attention_rules=args.attention_rules,
        accumulation_rules=args.accumulation_rules, matrix_schedule=args.matrix_schedule,
        matrix_bodies=args.matrix_bodies,
        gradient_outputs=args.gradient_outputs, rules=args.rules, precision=args.precision)
    print(json.dumps(result, indent=2, default=repr), flush=True)


if __name__ == "__main__":
    main()
