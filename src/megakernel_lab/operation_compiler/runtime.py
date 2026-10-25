"""Thin ctypes CUDA binding; source compilation belongs on Modal CPU workers.

Runtime.step launches exactly one cooperative CUDA kernel. Input/state pointers,
output views, and the arena are bound once. It never invokes an ATen fallback.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .lower import GeneratedProgram, _BYTES, cuda_module_state_policy


BUILD_FLAGS = ["-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC", "--fmad=false", "-lineinfo"]
DEFAULT_BLOCK_CAP = 128
THREADS_PER_BLOCK = 256
_ARENA_ALIGNMENT_BYTES = math.lcm(8, *_BYTES.values())


def _arena_allocation_words(logical_words):
    """Pad physical storage for every admitted typed view, without moving roots (OMNI-P1-A 82afa77f7).

    Tensor.view(int64) requires the allocation's byte length to be divisible by eight. The
    manifest remains the logical arena bound; only the backing allocation receives terminal padding.
    """
    if type(logical_words) is not int or logical_words < 0:
        raise ValueError("logical arena size must be a nonnegative integer word count")
    required = 4 * max(1, logical_words)
    aligned = (required + _ARENA_ALIGNMENT_BYTES - 1) // _ARENA_ALIGNMENT_BYTES * _ARENA_ALIGNMENT_BYTES
    return aligned // 4


def _validate_shared_arena(arena, *, device, words, manifest, input_ranges=()):
    """Validate caller-owned temporary storage before binding any generated writes.

    ``input_ranges`` contains (name, start, end) byte spans of bound arguments;
    temporary writes must never alias either persistent state or read-only inputs.
    Sequential ownership of the arena across Runtime objects remains the caller's
    responsibility, including when using multiple CUDA streams.
    """
    import torch
    if (not isinstance(arena, torch.Tensor) or arena.dtype != torch.float32 or arena.device != device
            or arena.dim() != 1 or not arena.is_contiguous() or arena.numel() < words
            or arena.data_ptr() % 256 or arena.is_neg() or arena.is_conj()):
        raise ValueError('shared arena must be a contiguous 256-byte aligned FP32 device vector at '
                         f'least {words} words long')
    if manifest.get('persistent_counter_words') or any(
            e.get('storage') == 'runtime_counter_arena' for ph in manifest.get('phases', [])
            for e in (ph.get('counter_effects') or [])):
        raise ValueError('programs with launch-persistent counters cannot share an arena')
    lo, hi = arena.data_ptr(), arena.data_ptr() + words * arena.element_size()
    for name, start, end in input_ranges:
        if max(lo, start) < min(hi, end):
            raise ValueError(f'shared arena overlaps bound argument {name}')


# LATCH-TINYMOE-25: per-launch binding check. 'pointers' compares the bound tensors' device
# addresses (the values in the kernel's pointer table) on every launch; the full identity (shape,
# strides, dtype, device, offsets) is checked at admission. 'full' recomputes it on every launch
# (about 3 ms of host time for TinyMoE's 2,703 bound tensors, with the GPU idle).
# LATCH-SELECTIVE-10: 'pointers' makes the same comparison of every bound tensor's current device address with
# its admitted address in one C++ call (torch._C._tensors_data_ptrs_at_indices_equal, the CUDA-graph-trees
# input check) instead of a Python loop: ~0.4 ms -> tens of microseconds of idle-GPU host time per launch for
# 3,299 bound tensors. Same refusal on any mismatch. 'pointers_python' keeps the previous loop (the
# LATCH-TINYMOE-25..COMBINED behaviour, e.g. for paired baselines); it is also the fallback when the batched
# helper is unavailable.
LAUNCH_VALIDATION = "pointers"
LAUNCH_VALIDATIONS = ("pointers", "pointers_python", "full")


def dependency_profile_intervals(manifest, stamps):
    """Decode common dependency boundaries without attributing overlap to bodies.

    Each member of a dependency interval receives the *same* CTA0 globaltimer
    stamp after that interval's publication barrier. Phase IDs need not appear
    in execution order. Subtracting consecutive phase IDs would consequently
    invent zero or negative body durations. This decoder follows the declared
    interval order and retains its exact member list.
    """
    phases = manifest['phases']
    intervals = manifest['barrier_intervals']
    members = [phase for interval in intervals for phase in interval]
    if sorted(members) != list(range(len(phases))) or any(not interval for interval in intervals):
        raise ValueError('profile intervals must cover each phase exactly once')
    if len(stamps) < len(phases) + 2:
        raise ValueError('profile timestamp buffer is incomplete')
    rows, previous = [], stamps[0]
    for index, interval in enumerate(intervals):
        end = stamps[interval[0] + 1]
        if any(stamps[phase + 1] != end for phase in interval):
            raise ValueError('dependency interval members have inconsistent boundary stamps')
        if end < previous:
            raise ValueError('dependency boundary timestamps are not monotonic')
        rows.append({'phase': index, 'kind': 'dependency_interval', 'phases': list(interval),
                     'operation_kinds': sorted({phases[phase]['kind'] for phase in interval}),
                     'values': ','.join(name for phase in interval for name in phases[phase]['names']),
                     'nanoseconds': end - previous})
        previous = end
    end = stamps[len(phases) + 1]
    if end < previous:
        raise ValueError('state commit timestamp precedes the last dependency boundary')
    rows.append({'phase': len(rows), 'kind': 'state_commit', 'phases': [],
                 'operation_kinds': ['state_commit'], 'values': 'state_commit',
                 'nanoseconds': end - previous})
    return rows


def _has_engine(node, engine):
    """P-OPS2-A 4dcee751f: inspect emitted engines across branches and regions."""
    if isinstance(node, dict):
        return node.get('engine') == engine or any(_has_engine(value, engine) for value in node.values()
                                                   if isinstance(value, (dict, list, tuple)))
    if isinstance(node, (list, tuple)):
        return any(_has_engine(item, engine) for item in node)
    return False


def select_launch_policy(manifest, max_blocks, blocks=None, *, multiprocessors=None):
    """Choose a resident grid from emitted work; no tensor values or GPU calls.

    Counts describe independent CTA jobs in one phase, not total work across
    sequential phases or loop iterations. The cap was measured on H100; it is
    a bounded default, not an autotuner or a claim about other devices.
    """
    if type(max_blocks) is not int or max_blocks <= 0:
        raise ValueError("cooperative occupancy limit must be a positive integer")
    selection = manifest.get("execution_selection", {})
    selected_body_limit = (selection.get("max_grid_blocks")
                           if selection.get("selected") == "legacy_static" else None)
    if selected_body_limit is not None and (type(selected_body_limit) is not int or selected_body_limit <= 0):
        raise ValueError("selected body grid bound must be a positive integer")
    program_limit = manifest.get("max_grid_blocks")
    if program_limit is not None:
        if type(program_limit) is not int or program_limit <= 0:
            raise ValueError("program grid bound must be a positive integer")
        selected_body_limit = (program_limit if selected_body_limit is None
                               else min(program_limit, selected_body_limit))
    if selected_body_limit is not None:
        if type(selected_body_limit) is not int or selected_body_limit <= 0:
            raise ValueError("selected body grid bound must be a positive integer")
        # Retained full reductions have 1024 partial slots. Preserve their
        # native body and check this resource precondition on both normal and
        # explicitly overridden API launches, including the cached path.
        max_blocks = min(max_blocks, selected_body_limit)
    if blocks is not None and (type(blocks) is not int or not 0 < blocks <= max_blocks):
        raise ValueError(f"cooperative grid {blocks} must be an integer in [1, {max_blocks}]")
    estimates = {kind: 0 for kind in
                 ("gemm", "pointwise", "indexed", "reduction", "softmax", "random", "copies")}

    def record(kind, jobs):
        estimates[kind] = max(estimates[kind], int(jobs))

    def thread_jobs(elements):
        return (elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    def visit(node, *, boundary_copies=False):
        layouts = node.get("layouts", {})

        def numel(name):
            layout = layouts.get(name)
            return math.prod(layout["shape"]) if layout is not None else 0

        # Preserve support for earlier manifests carrying only the aggregate.
        record("gemm", node.get("max_gemm_tiles", 0))
        for phase in node.get("phases", ()):
            kind = phase["kind"]
            elements = max((numel(name) for name in phase.get("writes", phase.get("names", ()))), default=0)
            if "shape" in phase:
                elements = max(elements, math.prod(phase["shape"]))
            if "work_items" in phase:
                record("pointwise", phase["work_items"])
            elif kind == "gemm":
                record("gemm", phase.get("tiles", 0))
            elif kind == "reduction":
                record("reduction", elements)  # One CTA per output.
            elif kind in {"softmax", "log_softmax"}:
                # A fused phase may also write independent pointwise tensors
                # whose lengths are unrelated to the softmax row width.
                heavy = [item for item in phase.get("items", ()) if item.get("kind") == kind]
                if heavy:
                    elements = max(numel(item["name"]) for item in heavy)
                    for item in phase["items"]:
                        if item.get("kind") == "elementwise":
                            record("pointwise", thread_jobs(item["elements"]))
                width = int(phase["reduction_elements"])
                if width <= 0 or elements % width:
                    raise ValueError("softmax launch metadata needs complete positive-width rows")
                record("softmax", elements // width)
            elif kind in {"embedding", "embedding_gradient", "gather", "scatter_add"}:
                record("indexed", thread_jobs(elements))
            elif kind in {"random_mask", "random_state"}:
                record("random", 1 if kind == "random_state" else thread_jobs(elements))
            elif kind in {"copy", "state_commit"}:
                record("copies", thread_jobs(elements))
            else:
                # Pointwise/fill/cast phases have one output element per thread.
                # Composite region labels have no local layout and contribute 0.
                record("pointwise", thread_jobs(elements))
        for target in node.get("state_writes", {}):
            record("copies", thread_jobs(numel(target)))
        if boundary_copies:
            for output in node.get("outputs", ()):
                record("copies", thread_jobs(numel(output)))
        # Branch and loop wrappers copy each called region's outputs. Take a
        # maximum: mutually exclusive branches and loop steps are not parallel.
        for container in ("branches", "regions"):
            for child in node.get(container, {}).values():
                visit(child, boundary_copies=True)

    visit(manifest)
    useful = max(1, *estimates.values())
    full_device_matrix = (_has_engine(manifest, 'pops2_fp32')
                          or manifest.get('full_grid', False))
    resource = manifest.get("resource_policy", {})
    residency = int(resource.get("resident_blocks_per_sm", 1))
    if residency not in (1,2):
        raise ValueError("unsupported whole-program cooperative residency")
    if residency == 2:
        if (not (resource.get("has_popsb") or resource.get("pops2_two_cta")) or resource.get("target_arch") != "sm_90"
                or resource.get("shared_bytes") != manifest.get("shared_bytes")
                or int(resource["shared_bytes"]) > int(resource.get("two_cta_shared_limit", 98304))
                or int(resource.get("two_cta_shared_limit", 98304)) > 110592):
            raise ValueError("two-CTA launch lacks a complete whole-program resource proof")
        if multiprocessors is not None and (type(multiprocessors) is not int or multiprocessors<=0):
            raise ValueError("multiprocessor count must be a positive integer")
        cap = 2*multiprocessors if multiprocessors is not None else 264
    elif full_device_matrix:
        # Qualified A engine plans one CTA per physical SM. Actual occupancy and
        # useful work still bound admission; this does not enable two-CTA default.
        if multiprocessors is not None and (type(multiprocessors) is not int or multiprocessors <= 0):
            raise ValueError('multiprocessor count must be a positive integer')
        cap = multiprocessors if multiprocessors is not None else 132
    elif manifest.get("grid_preference") == "power_of_two_occupancy":
        # LATCH-STEP-04 (indexed-engine programs): largest power of two within the queried cooperative occupancy,
        # at most two CTAs per SM (H100: 256 when the kernel admits two CTAs per SM, else 128).
        if multiprocessors is not None and (type(multiprocessors) is not int or multiprocessors <= 0):
            raise ValueError("multiprocessor count must be a positive integer")
        limit = min(max_blocks, 2 * multiprocessors if multiprocessors is not None else 264)
        cap = 1 << (limit.bit_length() - 1)
    else:
        cap = DEFAULT_BLOCK_CAP
    selected = min(cap, useful, max_blocks) if blocks is None else blocks
    return {"version": ("useful-work-v1" if residency == 1 else "useful-work-v1-residency")
                       + ("+pow2-occupancy" if manifest.get("grid_preference") == "power_of_two_occupancy" else "")
                       + ("+pops2-full-device" if full_device_matrix and residency == 1 else ""), "cap": cap,
            **({"resident_blocks_per_sm":residency,"resource_policy":resource} if residency==2 else {}),
            "threads_per_block": THREADS_PER_BLOCK, "useful_work_blocks": useful,
            "occupancy_limit": max_blocks, "selected_blocks": selected,
            **({"selected_body_grid_limit": selected_body_limit} if selected_body_limit is not None else {}),
            "explicit_override": blocks, "estimates": estimates}


def build(program: GeneratedProgram, *, cache_dir="/out/cache", arch="sm_90", allow_compile=True, nvcc="nvcc",
          step17=None, fast_device_compile=False, native_code_only=False,
          disable_expensive_optimizations=False, ptxas_optimization_level=None, device_units=None,
          device_unit_inline="wgmma", device_unit_cache=None, device_unit_jobs=None, device_unit_split=None,
          device_unit_shared_pool=False) -> Path:
    """CPU-only nvcc build, cached by source/headers/flags/toolchain identity.

    Run on the CPU adapter, then pass the returned library path to Runtime on a
    GPU worker mounting the same owned cloud volume. No torch extension is built.
    Failure logs are preserved alongside generated source and the manifest.
    ``fast_device_compile=True`` opts into nvcc's maximum device compilation
    preference. It changes optimization choices, not numerical flags; the
    resulting binary requires independent numerical/performance qualification.
    ``native_code_only=True`` emits only the selected real GPU architecture,
    omitting PTX fallback images and their extra front-end compilation. This
    preserves the ordinary CUDA host launcher and uses a separate cache key.
    ``disable_expensive_optimizations=True`` explicitly disables ptxas's
    memory/compile-time intensive optimizations. Source, arithmetic flags and
    default builds remain unchanged; the candidate needs fresh qualification.
    ``ptxas_optimization_level`` explicitly selects assembler level 0..3.
    Omission preserves the default command/cache; arithmetic flags are unchanged.
    Lower levels can change native resources/performance and require qualification.
    ``device_units=K`` (opt-in) compiles whole publication intervals as ``__noinline__`` device
    functions in K relocatable units concurrently (K=1: one translation unit, no device link) and
    nvlinks them behind the one ``latch_ops`` entry, loaded by the STEP17 driver shim; see
    ``step17_intervals``. ``device_unit_inline`` keeps inline-wgmma intervals in the entry ("wgmma"),
    outlines every interval ("none"), or outlines every interval and moves each inline-wgmma phase into a
    call-free piece of its own ("isolate"; relocatable calls otherwise serialize a function's wgmma, C7509). ``device_unit_cache`` optionally reuses unit cubins keyed by
    unit source, transitive header, flag and toolchain identity; ``device_unit_jobs`` bounds concurrent
    unit compiles (default: one per unit). ``device_unit_split=B`` splits intervals larger than B source
    bytes into phase-run pieces (part of the cache key). ``device_unit_shared_pool=True`` lets block-scope
    4-byte ``__shared__`` flags of the sequentially executed interval functions share one static pool
    (``step17_intervals.pool_shared_flags``; for shared plans with little static headroom). Requires the anchor host template; refusals raise.
    """
    if device_units is not None:
        if type(device_units) is not int or not 1 <= device_units <= 256:
            raise ValueError("device_units must be an integer from 1 through 256")
        if device_unit_inline not in ("wgmma", "none", "isolate"):
            raise ValueError("device_unit_inline must be 'wgmma', 'none' or 'isolate'")
        if step17 is not None or fast_device_compile:
            raise ValueError("device_units selects its own host-shim build; do not combine with step17/fast compile")
    if type(fast_device_compile) is not bool:
        raise ValueError("fast_device_compile must be a boolean")
    if type(native_code_only) is not bool:
        raise ValueError("native_code_only must be a boolean")
    if type(disable_expensive_optimizations) is not bool:
        raise ValueError("disable_expensive_optimizations must be a boolean")
    if ptxas_optimization_level is not None and (
            type(ptxas_optimization_level) is not int or ptxas_optimization_level not in range(4)):
        raise ValueError("ptxas_optimization_level must be None or an integer from 0 through 3")
    if ptxas_optimization_level is not None and fast_device_compile:
        raise ValueError("explicit ptxas optimization does not combine with fast_device_compile")
    if ("attention_item_codegen" in program.manifest or
            "attention_item_wrappers.cuh" in program.manifest.get("header_hashes", {})):
        if ptxas_optimization_level != 0:
            raise ValueError("attention-item outlining requires explicit ptxas_optimization_level=0; only this level is qualified")
    flags = [*BUILD_FLAGS, *(["--Ofast-compile=max"] if fast_device_compile else []),
             *(["-Xptxas=--allow-expensive-optimizations=false"] if disable_expensive_optimizations else []),
             *([f"-Xptxas=--opt-level={ptxas_optimization_level}"] if ptxas_optimization_level is not None else [])]
    if arch not in {"sm_80", "sm_86", "sm_89", "sm_90"}:
        raise ValueError(f"unqualified CUDA architecture {arch}")
    if program.manifest.get("target_arch", arch) != arch:
        raise ValueError("lowering target and build architecture differ; lower with target_arch=arch")
    if any(marker in program.source for marker in ("P5C_TMAP(", "POPS4B_TMAP(")):
        raise ValueError("tensor-map placeholders were not finalized by this lowering path")
    # The version-only invocation also refuses toolchains without this opt-in
    # spelling before writing or compiling a candidate. Default argv is unchanged.
    compiler_version = subprocess.run([nvcc, *(["--Ofast-compile=max"] if fast_device_compile else []),
                                       "--version"], text=True, capture_output=True, check=True).stdout
    # P-OPS5-C (af1b34c93): warpgroup-MMA engines need the architecture-specific sm_90a target (runs only on sm_90
    # devices); every other program keeps its unchanged sm_90 command and cache key.
    from .p5c_matrix import ARCH_HEADERS
    if arch == "sm_90" and any(name in (*ARCH_HEADERS, "pops4b_wg.cuh", "pops4b_tma.cuh", "p6e_tma.cuh",
                                                   "pops4b_tma_p7a.cuh", "pops4b_wg_p7a.cuh") for name in program.manifest.get("header_hashes", {})):
        arch = "sm_90a"
    # P9-D: panel-engine warpgroup-MMA products (sampled programs) need sm_90a as well.
    if arch == "sm_90" and str(program.manifest.get("panel_engine", {}).get("products_engine", "")).startswith("wgmma"):
        arch = "sm_90a"
    key_data = {"source_hash": program.cache_key, "arch": arch, "flags": flags, "nvcc": compiler_version}
    arch_flags = [f"-arch={arch}"]
    if native_code_only:
        # CUDA 12.8 nvcc manual 4.2.7: explicit --gpu-code requires a virtual
        # --gpu-architecture. In particular -arch=sm_90a otherwise requests
        # sm_90a plus BOTH compute_90 and compute_90a PTX images.
        virtual_arch = "compute_" + arch.removeprefix("sm_")
        arch_flags = [f"--gpu-architecture={virtual_arch}", f"--gpu-code={arch}"]
        key_data["code_generation"] = dict(virtual_arch=virtual_arch, native_arch=arch, ptx_fallback=False)
    # STEP17 (opt-in): device-only nvcc plus a driver-API host shim when the host code is the anchor template.
    # Its key extends the anchor key, so the two build modes never share a cache entry.
    from . import step17_build
    if native_code_only and step17_build.enabled(step17):
        raise ValueError("native_code_only does not combine with STEP17 host-shim builds")
    if disable_expensive_optimizations and step17_build.enabled(step17):
        raise ValueError("disable_expensive_optimizations does not combine with STEP17 host-shim builds")
    if ptxas_optimization_level is not None and step17_build.enabled(step17):
        raise ValueError("explicit ptxas optimization does not combine with STEP17 host-shim builds")
    step17_refusal = None
    intervals = None
    if device_units is not None:
        from . import step17_intervals
        step17_build.host_parameters(program.source)  # raises Unsupported for non-template host code
        keep = step17_intervals.wgmma_inline(program.source) if device_unit_inline == "wgmma" else ()
        if device_unit_split is not None and (type(device_unit_split) is not int or device_unit_split < 1000):
            raise ValueError("device_unit_split must be an integer of at least 1000 source bytes")
        isolate = step17_intervals.wgmma_inline(program.source) if device_unit_inline == "isolate" else ()
        if isolate:
            # LATCH-TRAIN-MATMUL (opt-in matrix_bodies 'isolate'): P5C/P8WS bodies and offset-view preludes.
            from .matrix_bodies import isolate_patterns
            isolate = isolate + isolate_patterns(program.manifest)
        intervals = step17_intervals.identity(device_units, keep, split_bytes=device_unit_split,
                                              isolate=isolate,
                                              shared_pool=bool(device_unit_shared_pool))
        key_data["device_units"] = {**step17_build.toolchain_identity(nvcc), **intervals}
    elif step17_build.enabled(step17):
        try:
            step17_build.host_parameters(program.source)
            units = 1 if step17 == "qualified" else step17_build.partition_units()
            key_data["step17"] = {**step17_build.toolchain_identity(nvcc), "units": units,
                                  **({"partition": step17_build.partition_identity()} if units > 1 else {})}
        except step17_build.Unsupported as error:
            step17_refusal = str(error)
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    # Cloud workers may scope the default shared cache to their owned attempt.
    # Explicit API cache directories keep their usual meaning.
    if str(cache_dir) == "/out/cache" and os.environ.get("LATCH_OPS_BUILD_CACHE"):
        cache_dir = os.environ["LATCH_OPS_BUILD_CACHE"]
    directory = Path(cache_dir) / key
    library = directory / "program.so"
    if library.exists():
        return library
    if not allow_compile:
        raise FileNotFoundError(f"prebuilt CUDA library missing: {library}")
    # Child workers sharing one cache use a local filesystem lock. Independent
    # cloud containers receive distinct cache roots from their controller; a
    # remote volume must not be assumed to provide cross-container flock.
    lock_key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"latch-ops-build-{lock_key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if library.exists():
            return library
        directory.mkdir(parents=True, exist_ok=True)
        if intervals is not None:
            started = time.perf_counter()
            fast = step17_build.build(program, directory, arch=arch, flags=flags, nvcc=nvcc, jobs=device_unit_jobs,
                                      intervals={**intervals, **({"unit_cache": str(device_unit_cache)}
                                                                 if device_unit_cache else {})})
            seconds = time.perf_counter() - started
            temporary = Path(fast.pop("temporary"))
            receipt = {**key_data, "cache_key": key, "command": fast["commands"], "returncode": 0,
                       "build_seconds": seconds, "step17_build": fast}
            (directory / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            temporary.replace(library)
            return library
        if "step17" in key_data:
            started = time.perf_counter()
            fast = step17_build.build(program, directory, arch=arch, flags=flags, nvcc=nvcc,
                                      units=key_data["step17"]["units"], jobs=os.environ.get("LATCH_STEP17_JOBS"))
            seconds = time.perf_counter() - started
            temporary = Path(fast.pop("temporary"))
            receipt = {**key_data, "cache_key": key, "command": fast["commands"], "returncode": 0,
                       "build_seconds": seconds, "step17_build": fast}
            (directory / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            (directory / "step17.json").write_text(json.dumps(fast, indent=2, sort_keys=True) + "\n")
            temporary.replace(library)
            return library
        source = program.write(directory)
        temporary = directory / f"program-{os.getpid()}.so"
        command = [nvcc, *flags, *arch_flags, str(source), "-o", str(temporary)]
        started = time.perf_counter()
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - started
        (directory / "build.log").write_text(proc.stdout)
        receipt = {**key_data, "cache_key": key, "command": command, "returncode": proc.returncode, "build_seconds": seconds}
        if step17_refusal is not None:
            receipt["step17_refusal"] = step17_refusal
        (directory / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if proc.returncode:
            raise RuntimeError(f"nvcc failed ({proc.returncode}); preserved {directory / 'build.log'}\n{proc.stdout[-8000:]}")
        temporary.replace(library)
    return library


def _attention_target_sms(manifest):
    """Only explicit deterministic Flash contracts bind an SM-count target."""
    targets = set()
    for row in manifest.get("attention_groups", ()):
        if row.get("backward_arithmetic") != "flash-bf16-deterministic-v1":
            continue
        target = row.get("deterministic_dq_target_sms")
        if type(target) is not int or target <= 0:
            raise ValueError("deterministic attention SM-count metadata is invalid")
        targets.add(target)
    if len(targets) > 1:
        raise ValueError("deterministic attention groups require different SM-count targets")
    return next(iter(targets)) if targets else None


class Runtime:
    def __init__(self, program: GeneratedProgram, args, *, library, blocks=None, launch_cache=True, arena=None,
                 admitted_aliases=frozenset(), launch_validation=None):
        import torch
        self.torch = torch
        self.launch_validation = LAUNCH_VALIDATION if launch_validation is None else launch_validation
        if self.launch_validation not in LAUNCH_VALIDATIONS:
            raise ValueError("launch_validation must be 'pointers', 'pointers_python' or 'full'")
        self.program = program
        self.manifest = program.manifest
        self.args = tuple(args)
        self.library = str(library)
        inputs = self.manifest["inputs"]
        if len(self.args) != len(inputs):
            raise ValueError(f"expected {len(inputs)} flat tensor arguments, got {len(self.args)}")
        if not self.args or not self.args[0].is_cuda:
            raise ValueError("Runtime requires CUDA FP32 arguments")
        self.device = self.args[0].device
        dynamic = self.manifest.get('dynamic_batch')
        if dynamic is not None:
            index = dynamic.get('input_index')
            if (type(index) is not int or not 0 <= index < len(inputs)
                    or inputs[index] != dynamic.get('input')):
                raise ValueError('dynamic batch scalar ABI metadata differs from bound inputs')
            self.active_batch_tensor = self.args[index]
            if self.active_batch_tensor.dtype != torch.int64 or self.active_batch_tensor.shape != ():
                raise ValueError('dynamic batch input must be one scalar int64 tensor')
            batch = int(self.active_batch_tensor.item())
            if not 1 <= batch <= dynamic['maximum']:
                raise ValueError('initial active batch is outside captured capacity')
            self.active_batch = batch
        target = self.manifest.get("target_arch")
        if target is not None:
            major, minor = torch.cuda.get_device_capability(self.device)
            if target != f"sm_{major}{minor}":
                raise ValueError(f"program target {target} differs from runtime device sm_{major}{minor}")
        attention_sms = _attention_target_sms(self.manifest)
        if attention_sms is not None:
            actual_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            if actual_sms != attention_sms:
                raise ValueError(f"deterministic attention requires {attention_sms} SMs, got {actual_sms}")
        from .torch_column_sum import required_sms
        column_sms = required_sms(self.manifest)
        if column_sms is not None:
            actual_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
            if actual_sms != column_sms:
                raise ValueError(f"column-sum arithmetic requires {column_sms} SMs, got {actual_sms}")
        for name, tensor in zip(inputs, self.args):
            layout = self.manifest["layouts"][name]
            expected_dtype = getattr(torch, layout.get("dtype", "float32"))
            if tensor.device != self.device or tensor.dtype != expected_dtype:
                raise ValueError(f"{name}: expected {expected_dtype} on {self.device}")
            if tensor.is_neg() or tensor.is_conj():
                raise ValueError(f"{name}: resolve lazy negative/conjugate views before runtime binding")
            if tuple(tensor.shape) != tuple(layout["shape"]) or tuple(tensor.stride()) != tuple(layout["strides"]):
                raise ValueError(f"{name}: expected shape/stride {layout['shape']}/{layout['strides']}, got {tuple(tensor.shape)}/{tuple(tensor.stride())}")
            if tensor.data_ptr() % 16:
                raise ValueError(f"{name}: 16-byte aligned input pointer required")
        optimizer_schema = {entry["name"]: entry for entry in self.manifest.get("optimizer_state_schema", ())}
        for name, tensor in zip(inputs, self.args):
            role = self.manifest["input_roles"][name]
            schema = optimizer_schema.get(role.get("parameter")) if role["role"] == "optimizer_state" else None
            if schema is not None and schema["kind"] == "step":
                # One setup-time synchronization validates the admitted lifetime.
                # Native steps themselves never read a counter back to the host.
                step = int(tensor.item())
                if not 0 <= step < (1 << 24):
                    raise ValueError(f"{name}: AdamW step must be in [0, 2**24) at runtime binding")
        # Aliasful state inputs are not distinct ownership domains. Parameter
        # sharing has already been canonicalized to one input by capture.
        state_names = set(self.manifest["state_writes"])
        ranges = []
        for name, tensor in zip(inputs, self.args):
            span = 1 + sum((int(d) - 1) * int(s) for d, s in zip(tensor.shape, tensor.stride()))
            lo, hi = tensor.data_ptr(), tensor.data_ptr() + tensor.element_size() * span
            for other, olo, ohi in ranges:
                # LATCH-MOE-EMA-100 (b01): a caller may admit declared nested boundary pairs (an input stored inside
                # its own slice of a written concatenation, which copies only the pieces not already in place).
                if ((name in state_names or other in state_names) and max(lo, olo) < min(hi, ohi)
                        and frozenset((name, other)) not in admitted_aliases):
                    raise ValueError(f"external state alias {name}/{other} is not admitted")
            ranges.append((name, lo, hi))
        library_manifest = Path(self.library).parent / "manifest.json"
        if library_manifest.exists() and json.loads(library_manifest.read_text())["source_hash"] != program.cache_key:
            raise ValueError("CUDA library and requested program manifest disagree")
        library_to_load = self.library
        self._private_library = None
        self.cuda_module_state = cuda_module_state_policy(self.manifest)
        if self.cuda_module_state["requires_private_module"]:
            # Constant bindings and long-attention scratch/counters belong to
            # the CUDA module. Independent arenas/streams must not share them.
            # Preserve every byte of the CPU-built binary, including concurrent
            # constructors; no new compilation or arbitrary cache relocation.
            import shutil
            import tempfile
            self._private_library = tempfile.TemporaryDirectory(prefix="latch_binding_")
            private = Path(self._private_library.name) / "program.so"
            shutil.copyfile(self.library, private)
            if hashlib.sha256(private.read_bytes()).digest() != hashlib.sha256(Path(self.library).read_bytes()).digest():
                raise RuntimeError("private constant-binding library copy differs")
            library_to_load = str(private)
        self._lib = ctypes.CDLL(library_to_load)
        self._lib.latch_query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self._lib.latch_query.restype = ctypes.c_int
        self._lib.latch_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self._lib.latch_launch.restype = ctypes.c_int
        self._lib.latch_error.argtypes = [ctypes.c_int]
        self._lib.latch_error.restype = ctypes.c_char_p
        max_blocks, sms = ctypes.c_int(), ctypes.c_int()
        with torch.cuda.device(self.device):
            self._check(self._lib.latch_query(ctypes.byref(max_blocks), ctypes.byref(sms)))
            self.max_blocks = max_blocks.value
            self.multiprocessors = sms.value
            self.launch_policy = select_launch_policy(self.manifest, self.max_blocks, blocks,
                                                      multiprocessors=self.multiprocessors)
            self.blocks = self.launch_policy["selected_blocks"]
            self.arena_words = int(self.manifest["arena_words"])
            # OMNI-P1-A 82afa77f7 / LATCH-MATRIX defect: an odd FP32 word count cannot back int64 views.
            self.arena_allocation_words = _arena_allocation_words(self.arena_words)
            if arena is None:
                self.arena = torch.empty(self.arena_allocation_words, device=self.device, dtype=torch.float32)
            else:
                # LATCH-MOESUP: programs that never run concurrently (e.g. phase variants of one training run) may
                # share one temporary arena. Flat programs initialize every arena-resident counter/scratch inside
                # each launch and keep persistent state in bound arguments, so only temporaries live here.
                _validate_shared_arena(arena, device=self.device, words=self.arena_allocation_words,
                                       manifest=self.manifest, input_ranges=ranges)
                self.arena = arena[:self.arena_allocation_words]
            self._pointers = torch.tensor([a.data_ptr() for a in self.args], device=self.device, dtype=torch.int64)
            # P-OPS5-C: TMA operand maps for this Runtime's bound storages, one extra pointer-table entry.
            from .p5c_matrix import encode_maps
            self._tensor_maps = encode_maps(self.manifest, self.args, self.arena, self.device)
            if self._tensor_maps is not None:
                self._pointers = torch.tensor([a.data_ptr() for a in self.args] + [self._tensor_maps.data_ptr()],
                                              device=self.device, dtype=torch.int64)
            self._ticks = torch.empty(self.manifest.get("tick_words", self.manifest["phase_count"] + 1), device=self.device, dtype=torch.int64)
        self.allocated_bytes = self.arena.numel() * self.arena.element_size() + self._pointers.numel() * 8 + self._ticks.numel() * 8
        if self._tensor_maps is not None:
            self.allocated_bytes += self._tensor_maps.numel() * self._tensor_maps.element_size()
        self.values = dict(zip(inputs, self.args))
        for name, layout in self.manifest["layouts"].items():
            if name in self.values:
                continue
            root = layout["root"]
            if root in inputs:
                storage = self.values[root]
                offset = storage.storage_offset() + int(layout["offset"])
            else:
                dtype = getattr(torch, layout.get("dtype", "float32"))
                storage = self.arena.view(dtype)
                byte_offset = 4 * self.manifest["offsets"][root]
                if byte_offset % storage.element_size():
                    raise ValueError(f"{name}: misaligned typed arena allocation")
                offset = byte_offset // storage.element_size() + int(layout["offset"])
            self.values[name] = torch.as_strided(storage, tuple(layout["shape"]), tuple(layout["strides"]), offset)
        self.outputs = tuple(self.values[name] for name in self.manifest["outputs"])
        self.parameters = {self.manifest["input_roles"][name]["parameter"]: self.values[name] for name in inputs if self.manifest["input_roles"][name]["role"] == "parameter"}
        self.momentum = {self.manifest["input_roles"][name]["parameter"]: self.values[name] for name in inputs if self.manifest["input_roles"][name]["role"] == "momentum"}
        self.optimizer_state = {self.manifest["input_roles"][name]["parameter"]: self.values[name] for name in inputs if self.manifest["input_roles"][name]["role"] == "optimizer_state"}
        if not self.optimizer_state:
            self.optimizer_state = self.momentum
        self.state = {self.manifest["input_roles"][name]["parameter"]: self.values[name] for name in inputs if self.manifest["input_roles"][name]["role"] == "state"}
        # LATCH-SELECTIVE-10 dcast: derived copies are runtime-owned state, not model state (derived_casts.py).
        self.derived = {name: self.state.pop(name) for name in [n for n in self.state if n.startswith("derived:")]}
        self._derived_records = [r for r in self.manifest.get("derived_state", ()) if r["key"] in self.derived]
        self._derived_sources = [self.parameters[n] for r in self._derived_records for n in r["pieces"]]
        self._derived_sync = None
        self.grads = {name: self.values[value] for name, value in self.manifest["structured_outputs"].get("grads", {}).items()} if isinstance(self.manifest["structured_outputs"], dict) else {}
        self.loss = self.values[self.manifest["structured_outputs"]["loss"]] if isinstance(self.manifest["structured_outputs"], dict) else self.outputs[0]
        self._bind_constant_args()
        self.launch_count = 0
        self._configure_launch_cache(launch_cache)

    def set_active_batch(self, batch):
        """Stage B in-place before launch/replay, preserving all graph pointers.

        This host action belongs outside the captured training graph and its
        timed replay. Call on the same CUDA stream as input staging and replay.
        """
        dynamic = self.manifest.get('dynamic_batch')
        if dynamic is None:
            raise ValueError('this program has a static batch')
        if type(batch) is not int or not 1 <= batch <= dynamic['maximum']:
            raise ValueError(f'active batch must be an integer in [1, {dynamic["maximum"]}]')
        with self.torch.cuda.device(self.device):
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError('stage active batch outside CUDA graph capture')
            self.active_batch_tensor.fill_(batch)
        self.active_batch = batch

    def _bind_constant_args(self):
        """P7-A: bind the immutable pointer table to this Runtime's private module."""
        n = self.manifest.get("p7_const_args")
        if not n:
            return
        values = [a.data_ptr() for a in self.args]
        if self._tensor_maps is not None:
            values.append(self._tensor_maps.data_ptr())
        if len(values) > n:
            raise ValueError("constant binding table smaller than the bound pointers")
        host = (ctypes.c_ulonglong * len(values))(*values)
        bind = self._lib.latch_bind_cargs
        bind.argtypes, bind.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
        with self.torch.cuda.device(self.device):
            self._check(bind(ctypes.cast(host, ctypes.c_void_p), len(values)))

    def _configure_launch_cache(self, requested):
        """Bind admission to this runtime, native function, context and storages.

        Replacing a bound tensor's storage requires a new Runtime. Copying values
        into existing tensors, including CUDA-graph input staging, stays valid.
        Libraries built before this interface keep their existing query path.
        """
        from .launch_cache import ContextIdentity, binding_identity, function_identity
        if type(requested) is not bool:
            raise ValueError('launch_cache must be a boolean')
        self._admitted_device = self.device
        self._bound_tensors = (*self.args, self.arena, self._pointers, self._ticks,
                               *((self._tensor_maps,) if self._tensor_maps is not None else ()))
        self._bound_identity = tuple(binding_identity(t) for t in self._bound_tensors)
        self._bound_pointers = [t.data_ptr() for t in self._bound_tensors]
        self._bound_list = list(self._bound_tensors)
        self._bound_indices = list(range(len(self._bound_list)))
        self._batched_pointer_check = getattr(self.torch._C, "_tensors_data_ptrs_at_indices_equal", None)
        # LATCH-SELECTIVE-10: a native guard holding the same TensorImpls makes the identical comparison in C++.
        self._native_guard = None
        if self.launch_validation == "pointers":
            from .binding_guard import guard_class
            guard = guard_class()
            if guard is not None:
                try:
                    self._native_guard = guard(self._bound_list)
                    if not self._native_guard.check():
                        self._native_guard = None
                except Exception:
                    self._native_guard = None
        self.launch_validation_path = ("native" if self._native_guard is not None else
                                       "batched" if self._batched_pointer_check is not None else "python")
        self._context_identity = None
        self._admitted_context = None
        self._fast_launch = None
        self._fast_function_identity = None
        reason = 'explicit diagnostic disable'
        if requested:
            try:
                fast = self._lib.latch_launch_admitted
                fast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                fast.restype = ctypes.c_int
                identity = ContextIdentity()
                with self.torch.cuda.device(self.device):
                    context = identity()
                self._fast_launch = fast
                self._fast_function_identity = function_identity(fast)
                self._context_identity, self._admitted_context = identity, context
                reason = 'bound runtime, unique CUDA context, native function and stable storages'
            except (OSError, AttributeError, RuntimeError) as error:
                reason = f'always-query compatibility fallback: {type(error).__name__}: {error}'
        self.launch_cache_policy = {
            'version': 'per-runtime-context-v1', 'requested': requested,
            'enabled': self._fast_launch is not None, 'reason': reason,
            'device_index': self.device.index, 'context_id': self._admitted_context,
            'scope': 'one bound Runtime; no process-global admission cache',
            'storage_change': 'refuse; construct a new Runtime',
            'context_change': 'refuse; construct a new Runtime',
            'setup_queries': 1, 'step_queries': 0, 'cached_launches': 0}

    def _derived_record(self):
        """The refresh record shared by every Runtime bound to these derived tensors (shared phase bindings).

        It lives on the first derived tensor object itself, so programs sharing that binding see one refresh and
        a new binding (new tensor) never inherits an old record."""
        if self._derived_sync is None:
            holder = self.derived[self._derived_records[0]["key"]]
            sync = getattr(holder, "_latch_derived_sync", None)
            if sync is None or sync["sources"] != [id(t) for t in self._derived_sources]:
                from .binding_guard import version_guard_class
                guard = version_guard_class()
                try:
                    guard = guard(self._derived_sources) if guard is not None else None
                except Exception:
                    guard = None
                torch = self.torch
                targets, sources = [], []
                for record in self._derived_records:
                    flat, start = self.derived[record["key"]].view(-1), 0
                    for name in record["pieces"]:
                        piece = self.parameters[name]
                        targets.append(flat[start:start + piece.numel()].view(piece.shape))
                        sources.append(piece)
                        start += piece.numel()
                sync = dict(sources=[id(t) for t in self._derived_sources], versions=None, guard=guard,
                            targets=targets, pieces=sources)
                holder._latch_derived_sync = sync
            self._derived_sync = sync
        return self._derived_sync

    def refresh_derived(self):
        """Recompute every derived copy from the current parameters (FP32 -> BF16 round to nearest even, the same
        values as the in-program cast), once for all Runtimes sharing the derived bindings."""
        torch, sync = self.torch, self._derived_record()
        with torch.no_grad(), torch.cuda.device(self.device):
            try:
                torch._foreach_copy_(sync["targets"], sync["pieces"])
            except (AttributeError, RuntimeError, TypeError):
                for record in self._derived_records:
                    pieces = [self.parameters[n] for n in record["pieces"]]
                    self.derived[record["key"]].copy_(torch.cat(pieces, record["dim"]))
        sync["versions"] = [t._version for t in self._derived_sources]
        if sync["guard"] is not None:
            sync["guard"].sync()

    def _sync_derived(self):
        """Recast when the host has written a source parameter since the last refresh (version counters).

        Device writes by programs never change a version; a program that updates the sources commits their copy."""
        sync = self._derived_record()
        if sync["versions"] is None:
            self.refresh_derived()
        elif sync["guard"] is not None:
            if sync["guard"].changed():
                self.refresh_derived()
        elif [t._version for t in self._derived_sources] != sync["versions"]:
            self.refresh_derived()

    def _validate_launch_binding(self):
        from .launch_cache import binding_identity, function_identity
        if self.device != self._admitted_device:
            raise RuntimeError('bound Runtime device changed; construct a new Runtime')
        if self.launch_validation == 'full':
            current = tuple(binding_identity(t) for t in self._bound_tensors)
            if current != self._bound_identity:
                raise RuntimeError('bound Runtime storage or layout changed; construct a new Runtime')
        elif self.launch_validation == 'pointers' and self._native_guard is not None:
            if not self._native_guard.check():
                raise RuntimeError('bound Runtime storage changed; construct a new Runtime')
        elif self.launch_validation == 'pointers' and self._batched_pointer_check is not None:
            if not self._batched_pointer_check(self._bound_list, self._bound_pointers, self._bound_indices):
                raise RuntimeError('bound Runtime storage changed; construct a new Runtime')
        elif [t.data_ptr() for t in self._bound_tensors] != self._bound_pointers:
            # The kernel reads exactly these device addresses through its pointer table; a replaced
            # storage changes them. Layout metadata is fixed at admission ('full' rechecks it per launch).
            raise RuntimeError('bound Runtime storage changed; construct a new Runtime')
        if self._fast_launch is not None:
            if function_identity(self._lib.latch_launch_admitted) != self._fast_function_identity:
                raise RuntimeError('bound native launch function changed; construct a new Runtime')
            if self._context_identity() != self._admitted_context:
                raise RuntimeError('bound Runtime CUDA context changed; construct a new Runtime')

    def _check(self, error):
        if error:
            raise RuntimeError(f"CUDA error {error}: {self._lib.latch_error(error).decode()}")

    def step(self, *, profile=False):
        """Launch one whole update asynchronously on PyTorch's current stream.

        Returned outputs are persistent arena views, overwritten next step.
        Packed-mode intermediate views can overlap; only graph outputs and input
        state are valid after step. Use retain_all=True for intermediate audits.
        """
        with self.torch.cuda.device(self.device):
            self._validate_launch_binding()
            if self._derived_sources:
                self._sync_derived()
            stream = self.torch.cuda.current_stream(self.device).cuda_stream
            args = (self._pointers.data_ptr(), self.arena.data_ptr(),
                    self._ticks.data_ptr() if profile else None, self.blocks, stream)
            if self._fast_launch is not None:
                self._check(self._fast_launch(*args, self.max_blocks, self.device.index))
                self.launch_cache_policy['cached_launches'] += 1
            else:
                self._check(self._lib.latch_launch(*args))
                self.launch_cache_policy['step_queries'] += 1
        self.launch_count += 1
        return self.outputs

    def profile_step(self):
        """One diagnostic update; returns boundary times in ns, separate from timing.

        CTA0 samples after each cooperative barrier. These intervals include
        synchronization and are diagnostic, not isolated arithmetic timings.
        """
        self.step(profile=True)
        stamps = self._ticks.cpu().tolist()
        if "profile_intervals" in self.manifest:
            return [{"phase": i, "kind": row["kind"], "values": row["name"],
                     "nanoseconds": stamps[row["end_tick"]] - stamps[row["start_tick"]]}
                    for i, row in enumerate(self.manifest["profile_intervals"])]
        if "predicate_input" in self.manifest:
            return [{"phase": 0, "kind": "uniform_conditional", "values": "selected forward/AD/state region",
                     "nanoseconds": stamps[1] - stamps[0]}]
        if self.manifest.get('dynamic_batch') and self.manifest.get('barrier_schedule') == 'dependency':
            return dependency_profile_intervals(self.manifest, stamps)
        labels = [",".join(phase["names"]) for phase in self.manifest["phases"]] + ["state_commit"]
        return [{"phase": i, "kind": self.manifest["phases"][i]["kind"] if i < len(self.manifest["phases"]) else "state_commit", "values": label, "nanoseconds": stamps[i+1] - stamps[i]} for i, label in enumerate(labels)]
