"""Direct-output staged regions for recursive and replay loop executors.

Operation lowering, matrix bodies, and dependency intervals are the existing
compiler's. This adapter owns only output placement and collective invocation.
One launch owns the counter barrier; embedded regions never initialize it.
"""
from __future__ import annotations

from contextvars import ContextVar
from types import SimpleNamespace

from .lower import LoweringError, TensorLayout, _CTYPES, _index


# Existing fast-pipeline operation-lowering rules; diagnostic region_options
# override any entry. These do not reorder contributions across invocations.
RECURSIVE_REGION_OPTIONS = {
    "row_fusion": True, "warp_softmax": True, "batched_reduction_loads": True,
    "defer_accumulate": True, "epilogue_fusion": True, "write_through": True,
    "split_counters": "persistent",
}
_INLINE_STAGE_CALLS = ContextVar("recursive_inline_stage_calls", default=False)
_RECOMPUTE_BOUNDARY_ELISION = ContextVar("recursive_recompute_boundary_elision", default=False)


def lower_region(graph, bindings, *, fuse_elementwise=True, policy="auto", **options):
    """Lower a pure region whose declared outputs have explicit destinations."""
    from .lower import lower
    from .loop_pipeline import PIPELINED_MATRIX_ENGINE

    if len(bindings) != len(graph.outputs):
        raise LoweringError("executor output binding arity mismatch")
    if any(binding not in ("store", "accumulate") for binding in bindings):
        raise LoweringError("recursive executor supports store and accumulate bindings")
    supplied = dict(options)
    inline_stages = supplied.pop("recursive_inline_stages", _INLINE_STAGE_CALLS.get())
    if type(inline_stages) is not bool:
        raise LoweringError("recursive_inline_stages must be boolean")
    merge_recompute = supplied.pop("recursive_recompute_boundary_elision", _RECOMPUTE_BOUNDARY_ELISION.get())
    if type(merge_recompute) is not bool:
        raise LoweringError("recursive_recompute_boundary_elision must be boolean")
    for key in ("retain_all", "output_bindings", "region_body", "staged", "barrier_schedule", "loop_counter"):
        supplied.pop(key, None)
    # Reuse the established pipeline's exact accumulating matrix epilogues.
    # Without this contract a placed gradient product falls back to SIMT merely
    # because its destination is an accumulation rather than a fresh store.
    supplied.setdefault("matrix_engine", dict(PIPELINED_MATRIX_ENGINE))
    # The ordered donor uses the full cooperative grid for long scalar FP32
    # reductions. Preserve those chunk and combine trees when placing outputs;
    # a single-CTA tree would silently reassociate the same calculation.
    supplied.setdefault("preserve_full_reductions", True)
    supplied.setdefault("preserve_matrix_plans", True)
    for key, value in RECURSIVE_REGION_OPTIONS.items():
        supplied.setdefault(key, value)
    if getattr(graph, "nested_operation", False):
        # Executor-only options never enter ordinary operation lowering.
        # Nested children inherit them while preserving explicit child overrides.
        token = _INLINE_STAGE_CALLS.set(inline_stages)
        boundary_token = _RECOMPUTE_BOUNDARY_ELISION.set(merge_recompute)
        try:
            return lower(graph, retain_all=False, fuse_elementwise=fuse_elementwise, policy=policy,
                         output_bindings=bindings, region_body=True, loop_counter=True, **supplied)
        finally:
            _RECOMPUTE_BOUNDARY_ELISION.reset(boundary_token)
            _INLINE_STAGE_CALLS.reset(token)
    program = lower(graph, retain_all=False, fuse_elementwise=fuse_elementwise, policy=policy,
                    output_bindings=bindings, region_body=True, staged=True,
                    barrier_schedule="dependency", elide_stores=True, **supplied)
    mf = program.manifest
    if inline_stages:
        mf["recursive_stage_calls"] = "inline_local_tables"
    if mf.get("persistent_counter_words"):
        # Default flat wrapper storage. Nested callers instead allocate one
        # dedicated block per child outside their shared operation scratch and
        # initialize it once before executing any inner iterations.
        base = (mf["arena_words"] + 63) // 64 * 64
        words = (mf["persistent_counter_words"] + 63) // 64 * 64
        mf.update(executor_scratch_words=mf["arena_words"], executor_counter_word_offset=base,
                  arena_words=base + words, arena_bytes=4 * (base + words))
    return program


def wrap_region(program, name, bindings, *, counter=False, barrier=None,
                counter_offset=None, initialize_counters=True, final_barrier=True):
    """Define NAME(args, arena, out[, loop_counter]) with typed residual copies.

    Flat stages own no grid barrier and execute in their recorded interval
    order. Recursive regions own whole-grid control and are never co-scheduled
    with another region. They forward the enclosing launch's barrier counter.
    """
    mf = program.manifest
    if mf["state_writes"]:
        raise LoweringError("an executor region may not commit mutable inputs")
    if len(bindings) != len(mf["outputs"]):
        raise LoweringError("executor wrapper output binding arity mismatch")
    counter_words = mf.get("persistent_counter_words", 0)
    if counter_words:
        if counter_offset is None:
            counter_offset = mf.get("executor_counter_word_offset")
        if type(counter_offset) is not int or counter_offset < 0:
            raise LoweringError("recursive stage wrapper requires region-owned split counters")
    sync = barrier or ("latch_loop_barrier(loop_counter);" if counter else "grid.sync();")
    placed = set(mf.get("placed_outputs", ()))
    placement = mf.get("output_placement", {})
    copies = []
    for i, value_name in enumerate(mf["outputs"]):
        if i in placed:
            continue
        d = mf["layouts"][value_name]
        root = d["root"]
        if root in mf["inputs"]:
            ptr = f'args[{mf["inputs"].index(root)}]'
        elif root in placement:
            # An unplaced view can alias another directly placed result. Its
            # producer has caller storage and deliberately has no arena slot.
            binding = placement[root]
            mode = binding["mode"]
            ptr = f'args[{mode[8:]}]' if mode.startswith("inplace:") else f'out[{binding["output"]}]'
        else:
            ptr = f'arena + {mf["offsets"][root]}LL'
        layout = TensorLayout(tuple(d["shape"]), tuple(d["strides"]), root, d["offset"])
        ctype = _CTYPES[d["dtype"]]
        op = "+=" if bindings[i] == "accumulate" else "="
        copies.append(
            f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; "
            f"i < {layout.numel}LL; i += (long long)gridDim.x * blockDim.x) "
            f"reinterpret_cast<{ctype}*>(out[{i}])[i] {op} "
            f"reinterpret_cast<const {ctype}*>({ptr})[{layout.offset}LL + {_index(layout, layout.shape)}];"
        )
    arg = ", unsigned long long* loop_counter" if counter else ""
    impl = f"{name}_impl"
    if mf.get("recursive_executor"):
        if not counter:
            raise LoweringError("embedded recursive executor requires the launch barrier context")
        declarations = program.functions
        body = program.kernel_body
    elif "stages" in mf:
        if program.functions:
            raise LoweringError("flat staged executor region unexpectedly contains child functions")
        declarations = program.kernel_body.replace("__REGION__", "stage").replace("__REGION_ID__", "0")
        inline_stages = mf.get("recursive_stage_calls") == "inline_local_tables"
        if inline_stages:
            from .loop_pipeline import P9_OPQ_HELPER, _local_tables
            # Reuse the established P9 stage-call policy. These helpers live
            # in this wrapper's namespace: a translation-unit macro guard
            # would incorrectly hide them from subsequent nested namespaces.
            opaque_helpers = "\n".join(line for line in P9_OPQ_HELPER.splitlines() if not line.startswith("#"))
            declarations = opaque_helpers + "\n" + declarations.replace(
                "__device__ __noinline__ void stage_s", "__device__ __forceinline__ void stage_s")
            track = SimpleNamespace(args=[(value, f"args[{i}]") for i, value in enumerate(mf["inputs"])],
                                    outs=[(value, f"p9_og[{i}]") for i, value in enumerate(mf["outputs"])])
        body = "cg::grid_group grid = cg::this_grid();\n"
        if counter_words and initialize_counters:
            body += (f"for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<{counter_words}LL; "
                     f"i+=(long long)gridDim.x*blockDim.x) "
                     f"reinterpret_cast<unsigned*>(arena + {counter_offset}LL)[i]=0u;\n{sync}\n")
        calls = []
        for i in range(len(mf["stages"])):
            if inline_stages:
                # Tables and derived pointers end their lexical lifetime
                # before the unchanged collective publication barrier.
                tables = "float** p9_og = p9_opq(out); " + _local_tables(track, 0, None, "arena")
                call = (f"stage_s{i}(a0, x0, o0, 0, 0, (int)gridDim.x"
                        + (f", x0 + {counter_offset}LL" if counter_words else "") + ");")
                calls.append("{ " + tables + call + " }\n" + sync)
            else:
                calls.append(f"stage_s{i}(args, arena, out, 0, 0, (int)gridDim.x"
                             + (f", arena + {counter_offset}LL" if counter_words else "") + f");\n{sync}")
        body += "\n".join(calls)
    else:
        raise LoweringError("executor wrapper requires staged or declared recursive control")
    if copies:
        body += "\n" + "\n".join(copies) + "\n" + sync
    if not final_barrier and body.rstrip().endswith(sync):
        # The caller immediately performs independent copies and then owns the
        # publication barrier. All internal producer/consumer barriers remain.
        body = body.rstrip()[:-len(sync)]
    return (f"namespace {impl} {{\n{declarations}\n"
            f"__device__ __forceinline__ void run(float** args, float* arena, float** out{arg}) {{\n"
            "unsigned long long* ticks = nullptr;\n" + body + "\n}\n}\n"
            f"__device__ __forceinline__ void {name}(float** args, float* arena, float** out{arg}) {{\n"
            f"{impl}::run(args, arena, out" + (", loop_counter" if counter else "") + ");\n}\n")
