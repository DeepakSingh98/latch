"""Recursive device control around ordinary captured operation regions."""

from __future__ import annotations

import hashlib
import json
import math

from .lower import GeneratedProgram, LoweringError, TensorLayout, _BYTES, _CTYPES, _index, _translation_unit


MERGE_BOUNDARY_BARRIERS = True


def lower_nested(region, *, retain_all=False, fuse_elementwise=True, policy="auto",
                 output_bindings=None, region_body=False, loop_counter=False, **options):
    from .lower import ordered_loop_regions
    fast = (output_bindings is not None and region_body and not retain_all
            and region.metadata["mode"] in ("save_body_tape", "recompute_boundaries"))
    if fast and not loop_counter:
        raise LoweringError("recursive staged execution needs its enclosing launch barrier")
    with ordered_loop_regions():
        return _lower_nested(region, retain_all=retain_all, fuse_elementwise=fuse_elementwise, policy=policy,
                             output_bindings=output_bindings if fast else None, fast=fast, **options)


def _lower_nested(region, *, retain_all=False, fuse_elementwise=True, policy="auto",
                  output_bindings=None, fast=False, **options):
    from .lower import lower
    from .resources import select_program_resources
    from .loop_region_executor import lower_region, wrap_region, _RECOMPUTE_BOUNDARY_ELISION

    m = region.metadata
    n, no, nc, ns, ni = (
        m[k] for k in ("n_parameters", "n_outer_carry", "n_inner_carry", "n_state", "n_inputs")
    )
    na, primal = n + no, n + no + ns + ni
    saved = m["mode"] == "save_body_tape"
    reverse = m["direction"] == "reverse"
    merge_recompute = (fast and m["mode"] == "recompute_boundaries"
                       and MERGE_BOUNDARY_BARRIERS and _RECOMPUTE_BOUNDARY_ELISION.get())
    merged_children = []
    if MERGE_BOUNDARY_BARRIERS and saved and not reverse:
        merged_children = ["init_saved", "finish_saved"]
    elif merge_recompute:
        merged_children = ["init"] if reverse else ["init", "finish"]
    names = (
        ("finish_vjp_saved", "body_vjp_saved", "init_vjp_saved")
        if saved and reverse
        else (
            ("init_saved", "body_saved", "finish_saved")
            if saved
            else (
                ("init", "body", "finish_vjp", "body_vjp", "init_vjp")
                if reverse
                else ("init", "body", "finish")
            )
        )
    )
    programs = {}
    child_bindings = {}
    fallbacks = []
    for name in names:
        count = len(region.regions[name].outputs)
        accumulate = na if fast and name in ("body_vjp_saved", "init_vjp_saved", "body_vjp", "init_vjp") else 0
        child_bindings[name] = ["accumulate"] * accumulate + ["store"] * (count - accumulate)
        def lower_child(selected_policy):
            if fast:
                return lower_region(region.regions[name], child_bindings[name],
                                    fuse_elementwise=fuse_elementwise, policy=selected_policy, **options)
            return lower(region.regions[name], retain_all=retain_all,
                         fuse_elementwise=fuse_elementwise, policy=selected_policy, **options)
        program = lower_child(policy)

        def needs_tensor_maps(candidate):
            return (
                candidate.manifest.get("tensor_maps")
                or "P5C_TMAP(" in candidate.source
                or "P5C_TMAPR(" in candidate.source
            )

        if needs_tensor_maps(program):
            # The ordered recursive ABI binds pointers to runtime-selected tape
            # slices. A flat tensor-map root cannot be rebound to those slices.
            # Keep the existing reference body until recursive map enumeration
            # is implemented; never finalize an inner root against outer args.
            program = lower_child("reference")
            if needs_tensor_maps(program):
                raise LoweringError(f"nested region {name} needs unsupported recursive tensor-map bindings")
            fallbacks.append(
                {
                    "region": name,
                    "selected": "reference",
                    "reason": "ordered recursive tape pointers do not have enumerated tensor-map bindings",
                }
            )
        programs[name] = program
    align = lambda words: (words + 63) // 64 * 64
    scratch = align(max(p.manifest.get("executor_scratch_words", p.manifest["arena_words"])
                        if fast else p.manifest["arena_words"] for p in programs.values()))
    top = scratch

    def allocate(desc):
        nonlocal top
        start = top
        top += align((math.prod(desc["shape"]) * _BYTES[desc["dtype"]] + 3) // 4)
        return f"arena + {start}LL", start

    # Counter storage is raw four-byte arena words, reinterpreted as unsigned
    # by the generated protocol; it is not an additional captured IR dtype.
    counter_offsets = ({name: allocate({"shape": [program.manifest["persistent_counter_words"]],
                                       "dtype": "float32"})[1]
                        for name, program in programs.items() if program.manifest.get("persistent_counter_words")}
                       if fast else {})
    inputs, outputs = list(region.inputs), list(region.outputs)
    layouts = {
        name: {
            "shape": list(v.shape),
            "strides": list(v.strides),
            "dtype": v.dtype,
            "root": name,
            "offset": v.storage_offset,
        }
        for name, v in region.values.items()
    }
    offsets = {}
    out = []
    placed = []
    if fast and (len(output_bindings) != len(outputs) or
                 any(binding not in ("store", "accumulate") for binding in output_bindings)):
        raise LoweringError("nested executor output binding arity/mode mismatch")
    for i, name in enumerate(outputs):
        # An inner parameter VJP is summed locally before one addition into the
        # enclosing gradient. Directly accumulating every inner contribution
        # into that gradient would reassociate floating-point additions.
        if fast and output_bindings[i] == "store":
            ptr = f"out[{i}]"
            placed.append(i)
        else:
            ptr, offsets[name] = allocate(layouts[name])
        out.append(ptr)

    def work(desc):
        return allocate(desc)[0]

    p = [f"args[{i}]" for i in range(na)]
    state = [f"args[{na+i}]" for i in range(ns)]
    data = [f"args[{na+ns+i}]" for i in range(ni)]
    bd = m["boundary_desc"]
    cd = bd[:nc]
    pd = [layouts[name] for name in inputs[:na]]

    def slice_ptr(ptr, desc, index):
        size = math.prod(desc["shape"]) * _BYTES[desc["dtype"]]
        return f"reinterpret_cast<float*>(reinterpret_cast<char*>({ptr}) + ({index}) * {size}LL)"

    if saved:
        packed = [f"args[{primal+i}]" for i in range(len(m["tape"]))] if reverse else out[no + ns :]
        h = nc + ns
        b = h + len(m["inner_body_tape"])
        e = b + len(m["inner_init_tape"])
        histories = packed[:h]
        bt, init_tape, finish_tape = packed[h:b], packed[b:e], packed[e:]
    else:
        histories = [work({"shape": [3, *d["shape"]], "dtype": d["dtype"]}) for d in bd]
        bt, init_tape, finish_tape = [], [], []

    def boundary(index):
        return [slice_ptr(ptr, desc, index) for ptr, desc in zip(histories, bd)]

    def tape(index):
        return [slice_ptr(ptr, desc, index) for ptr, desc in zip(bt, m["inner_body_tape"])]

    def copy(dest, source, desc, op="="):
        if len(dest) != len(source) or len(dest) != len(desc):
            raise LoweringError("nested storage-copy arity mismatch")
        return "\n".join(
            f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; "
            f'i < {math.prod(d["shape"])}LL; i += (long long)gridDim.x * blockDim.x) '
            f'reinterpret_cast<{_CTYPES[d["dtype"]]}*>({a})[i] {op} '
            f'reinterpret_cast<const {_CTYPES[d["dtype"]]}*>({b})[i];'
            for a, b, d in zip(dest, source, desc)
        )

    functions = []
    for name, program in programs.items():
        if fast:
            # Init carry stores and the following state copies are disjoint;
            # their existing caller barrier publishes both, including count 0.
            # Forward finish stores likewise join the caller's state exports.
            # Replay/body/VJP and initial counter publication barriers remain.
            functions.append(wrap_region(program, f"nested_{name}", child_bindings[name], counter=True,
                                         counter_offset=counter_offsets.get(name), initialize_counters=False,
                                         final_barrier=name not in merged_children))
            continue
        mf = program.manifest
        if mf["state_writes"]:
            raise LoweringError("a nested operation may not commit parameter, optimizer, or explicit state")
        copies = []
        for i, output in enumerate(mf["outputs"]):
            d = mf["layouts"][output]
            root = d["root"]
            source = (
                f'args[{mf["inputs"].index(root)}]'
                if root in mf["inputs"]
                else f'arena + {mf["offsets"][root]}LL'
            )
            layout = TensorLayout(tuple(d["shape"]), tuple(d["strides"]), root, d["offset"])
            ctype = _CTYPES[d["dtype"]]
            copies.append(
                f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layout.numel}LL; "
                f"i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>(out[{i}])[i] = "
                f"reinterpret_cast<const {ctype}*>({source})[{layout.offset}LL + {_index(layout,layout.shape)}];"
            )
        functions.append(
            f"namespace nested_{name} {{\n{program.functions}\n"
            "__device__ __noinline__ void run(float** args, float* arena, float** out) {\n"
            "unsigned long long* ticks = nullptr;\n"
            + program.kernel_body
            + "\n"
            + "\n".join(copies)
            + "\ngrid.sync();\n}\n}\n"
        )

    def call(name, args, outs):
        mf = programs[name].manifest
        if len(args) != len(mf["inputs"]) or len(outs) != len(mf["outputs"]):
            raise LoweringError(
                f'nested {name} ABI: {len(args)}/{len(outs)} vs {len(mf["inputs"])}/{len(mf["outputs"])}'
            )
        return (
            "{\nfloat* region_args[] = {"
            + ", ".join(args)
            + "};\nfloat* region_outputs[] = {"
            + ", ".join(outs)
            + "};\n"
            + (f"nested_{name}(region_args,arena,region_outputs,loop_counter);\n}}" if fast
               else f"nested_{name}::run(region_args,arena,region_outputs);\n}}")
        )

    body = [
        "cg::grid_group grid = cg::this_grid();",
        f'const int64_t inner_requested = *reinterpret_cast<const int64_t*>({data[m["count_input"]]});',
        f'const int inner_count = int(inner_requested < 0 ? 0 : (inner_requested > {m["max_steps"]}LL ? {m["max_steps"]}LL : inner_requested));',
    ]
    for name, offset in counter_offsets.items():
        words = programs[name].manifest["persistent_counter_words"]
        body.append(f"for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<{words}LL; "
                    f"i+=(long long)gridDim.x*blockDim.x) "
                    f"reinterpret_cast<unsigned*>(arena + {offset}LL)[i]=0u;")
    if counter_offsets and (reverse or not saved):
        body.append("grid.sync();")
    if not reverse or not saved:
        if saved:
            # Unexecuted slots are not read by AD, but initialize them so exported
            # state/tape and exact restart never expose prior-launch arena bytes.
            for ptr, desc in zip(packed, m["tape"]):
                size = math.prod(desc["shape"]) * _BYTES[desc["dtype"]]
                body.append(
                    f"for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<{size}LL; i+=(long long)gridDim.x*blockDim.x) reinterpret_cast<unsigned char*>({ptr})[i]=0;"
                )
            body.append("grid.sync();")
        b0 = boundary("0")
        body += [
            call("init_saved" if saved else "init", p + data, b0[:nc] + init_tape),
            copy(b0[nc:], state, bd[nc:]),
            "grid.sync();",
        ]
        before = boundary(
            "inner_iteration" if saved else "inner_iteration == 0 ? 0 : 1 + ((inner_iteration-1)%2)"
        )
        after = boundary("inner_iteration+1" if saved else "1 + (inner_iteration%2)")
        body += [
            "for (int inner_iteration=0; inner_iteration<inner_count; ++inner_iteration) {",
            call(
                "body_saved" if saved else "body",
                p + before + data,
                after + (tape("inner_iteration") if saved else []),
            ),
            "}",
        ]
    final = boundary("inner_count" if saved else "inner_count == 0 ? 0 : 1 + ((inner_count-1)%2)")
    if not reverse:
        body += [
            call("finish_saved" if saved else "finish", p + final[:nc] + data, out[:no] + finish_tape),
            copy(out[no : no + ns], final[nc:], bd[nc:]),
            "grid.sync();",
        ]
    else:
        dc = [[work(d) for d in cd] for _ in range(2)]
        contribution = out if fast else [work(d) for d in pd]
        cot = [f"args[{len(inputs)-no+i}]" for i in range(no)]
        body += [
            call(
                "finish_vjp_saved" if saved else "finish_vjp",
                p + final[:nc] + data + finish_tape + cot,
                out + dc[0],
            ),
            "for (int inner_reverse=inner_count-1; inner_reverse>=0; --inner_reverse) {",
        ]
        if saved:
            before = boundary("inner_reverse")
        else:
            rb = boundary("inner_replay == 0 ? 0 : 1 + ((inner_replay-1)%2)")
            ra = boundary("1 + (inner_replay%2)")
            body += [
                "for (int inner_replay=0; inner_replay<inner_reverse; ++inner_replay) {",
                call("body", p + rb + data, ra),
                "}",
            ]
            before = boundary("inner_reverse == 0 ? 0 : 1 + ((inner_reverse-1)%2)")
        current = [f"((inner_count-1-inner_reverse)%2 == 0 ? {a} : {b})" for a, b in zip(*dc)]
        nextcot = [f"((inner_count-1-inner_reverse)%2 == 0 ? {b} : {a})" for a, b in zip(*dc)]
        body += [
            call(
                "body_vjp_saved" if saved else "body_vjp",
                p + before + data + (tape("inner_reverse") if saved else []) + current,
                contribution + nextcot,
            ),
            *([] if fast else [copy(out, contribution, pd, "+="), "grid.sync();"]),
            "}",
        ]
        firstcot = [f"(inner_count%2 == 0 ? {a} : {b})" for a, b in zip(*dc)]
        body += [
            call("init_vjp_saved" if saved else "init_vjp", p + data + init_tape + firstcot, contribution),
            *([] if fast else [copy(out, contribution, pd, "+="), "grid.sync();"]),
        ]
    kernel = "\n".join(body)
    if fast:
        kernel = kernel.replace("grid.sync();", "latch_loop_barrier(loop_counter);")
    helpers = "\n".join(dict.fromkeys(p.helpers for p in programs.values() if p.helpers))
    functions = "\n".join(functions)
    shared = max(p.manifest.get("shared_bytes", 17408) for p in programs.values())
    resource = select_program_resources(
        shared, children=[p.manifest for p in programs.values()], arch=options.get("target_arch", "sm_90")
    )
    grid_limits = [p.manifest["max_grid_blocks"] for p in programs.values() if p.manifest.get("max_grid_blocks")]
    max_grid_blocks = min(grid_limits) if grid_limits else None
    source = _translation_unit(
        kernel,
        helpers=helpers,
        functions=functions,
        shared_bytes=shared,
        resident_blocks_per_sm=resource["resident_blocks_per_sm"],
        max_grid_blocks=max_grid_blocks,
    )
    hashes = {}
    for program in programs.values():
        for name, digest in program.manifest["header_hashes"].items():
            if name in hashes and hashes[name] != digest:
                raise LoweringError(f"nested regions disagree on header identity {name}")
            hashes[name] = digest
    manifest = {
        "format": "latch-ops-cuda-nested-region-v1",
        "source_hash": hashlib.sha256((source + json.dumps(hashes, sort_keys=True)).encode()).hexdigest(),
        "header_hashes": hashes,
        "inputs": inputs,
        "outputs": outputs,
        "structured_outputs": outputs,
        "input_roles": {
            name: {"role": region.values[name].role, "parameter": region.values[name].parameter}
            for name in inputs
        },
        "state_writes": {},
        "layouts": layouts,
        "offsets": offsets,
        "arena_words": top,
        "arena_bytes": top * 4,
        "region_scratch_bytes": scratch * 4,
        "shared_bytes": shared,
        **({"max_grid_blocks": max_grid_blocks} if max_grid_blocks is not None else {}),
        "retained_all": retain_all,
        "phase_count": 1,
        "phases": [{"kind": "nested_reverse" if reverse else "nested_forward", "names": list(names)}],
        "max_gemm_tiles": max(p.manifest["max_gemm_tiles"] for p in programs.values()),
        "regions": {k: p.manifest for k, p in programs.items()},
        "numerical_policy": next(iter(programs.values())).manifest["numerical_policy"],
        "runtime_loop": {
            "max_steps": m["max_steps"],
            "count_input": inputs[na + ns + m["count_input"]],
            "storage_mode": m["mode"],
            "direction": m["direction"],
            "count_policy": "clamp to [0,max_steps]",
        },
        "execution": "pure bounded device loop; no optimizer or mutable-input commit",
        "body_fallbacks": fallbacks,
    }
    if resource["has_popsb"]:
        manifest["resource_policy"] = resource
    if fast:
        manifest.update(
            format="latch-ops-cuda-nested-staged-region-v1",
            recursive_executor={"storage_mode": m["mode"], "direction": m["direction"],
                                "gradient_order": "local finish + reversed body contributions + init; one enclosing addition",
                                "barrier_context": "forwarded from enclosing launch; no reset",
                                "staged_children": list(names), "region_bindings": child_bindings,
                                "persistent_counters": {name: {"word_offset": offset,
                                    "words": programs[name].manifest["persistent_counter_words"],
                                    "initialization": "once per nested invocation before the first child"}
                                    for name, offset in counter_offsets.items()},
                                "merged_boundary_barriers": bool(merged_children),
                                **({"merged_boundary_children": merged_children} if merge_recompute else {}),
                                "private_accumulated_outputs": [i for i, mode in enumerate(output_bindings)
                                                                if mode == "accumulate"]},
            region_body=True, placed_outputs=placed,
            output_placement={outputs[i]: {"output": i, "mode": "store"} for i in placed},
            execution="recursive runtime loop; existing dependency stages; direct child stores and ordered gradient epilogues",
        )
    return GeneratedProgram(source, manifest, kernel, helpers, functions)
