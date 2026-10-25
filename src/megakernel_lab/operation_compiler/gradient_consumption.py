"""Release step 15: optimizer-consumed gradients in flat training programs.

Capture returns ``(loss, gradients, new parameters, new optimizer/model state)``. Public outputs stay live in
the arena until the final commit, so every FP32 parameter gradient occupies its own storage for the rest of the
update even when the optimizer already consumed it -- for example inside a fused P5C dW epilogue, where the
gradient is otherwise only a register value. With ``gradient_outputs="consumed"`` the caller declares that it
does not read gradients after the update. Gradient values remain ordinary internal values of the same program:
the optimizer reads exactly the same computed gradient, but storage follows normal producer-to-last-reader
lifetimes, and a dW product read only by its optimizer epilogue is never stored.

This changes outputs and storage, never arithmetic: expressions, roundings and reduction orders are unchanged,
so new parameters and optimizer state must equal the ``"returned"`` program's bit for bit (checked on GPU).
Admission is structural (flat TrainingIR with structured outputs); no model, shape or benchmark names.
Regions, loops and conditionals keep their existing gradient bindings and are refused explicitly.
"""
from __future__ import annotations

from dataclasses import replace

POLICIES = ("returned", "consumed")


def consume(graph, policy):
    """Return ``(graph, report)``; ``report`` is None when the existing ABI is kept."""
    from .ir import TrainingIR
    from .lower import LoweringError
    if policy not in POLICIES:
        raise LoweringError(f"gradient_outputs must be one of {POLICIES}")
    if policy == "returned":
        return graph, None
    if type(graph) is not TrainingIR:
        raise LoweringError("gradient_outputs='consumed' requires a flat TrainingIR; regions, loops and "
                            "conditionals keep their existing gradient bindings")
    outputs = graph.outputs
    names = graph.metadata.get("parameter_names")
    if (not isinstance(outputs, dict) or not isinstance(outputs.get("grads"), dict)
            or not isinstance(outputs.get("params"), dict) or not names
            or set(outputs["grads"]) != set(names) or set(outputs["params"]) != set(names)):
        raise LoweringError("gradient_outputs='consumed' requires structured loss/grads/params outputs")
    kept = {kind: group for kind, group in outputs.items() if kind != "grads"}
    still_public = set()
    for kind, group in kept.items():
        still_public.update(group.values() if isinstance(group, dict) else [group])
    gradients = {name: outputs["grads"][name] for name in names}
    readers = {}
    for op in graph.operations:
        for value in op.reads:
            readers.setdefault(value, []).append(op.name)
    unread = sorted(name for name, value in gradients.items() if value not in still_public and not readers.get(value))
    if unread:
        # A gradient nobody reads is not consumed by the update; dropping it would hide work, not remove traffic.
        raise LoweringError(f"gradient_outputs='consumed' requires every gradient to have an in-program reader: {unread}")
    report = {"policy": "consumed", "version": 1,
              "removed_public_outputs": len(gradients),
              "gradient_values": gradients,
              "still_public": sorted(value for value in gradients.values() if value in still_public),
              "arithmetic": "unchanged expressions, roundings and reduction orders; outputs/storage only",
              "dispatch": "flat TrainingIR structure; no model, shape or benchmark names"}
    # LATCH-SELECTIVE-10: the consumed gradient values stay schedule targets (ops.update_schedule late_updates keeps
    # bodies no loss/gradient value depends on after the gradients, as for returned gradients).
    metadata = dict(graph.metadata or {}, consumed_gradient_values=sorted(set(gradients.values())))
    return replace(graph, outputs=kept, metadata=metadata), report


def record(program, report):
    """Attach the declaration and the storage actually removed to the lowered manifest."""
    if report is None:
        return program
    manifest = program.manifest
    offsets, layouts = manifest.get("offsets", {}), manifest.get("layouts", {})
    width = {"float32": 4, "bfloat16": 2, "float16": 2, "int64": 8, "int32": 4}
    stored, unstored = {}, []
    for parameter, value in report["gradient_values"].items():
        layout = layouts.get(value)
        if layout and layout.get("root") in offsets:
            numel = 1
            for size in layout["shape"]:
                numel *= int(size)
            stored[parameter] = numel * width.get(layout.get("dtype", "float32"), 4)
        else:
            unstored.append(parameter)
    manifest["gradient_outputs"] = {**report,
        "stored_internal_gradients": len(stored), "stored_internal_gradient_bytes": sum(stored.values()),
        "unstored_gradients": len(unstored),
        "scope": "stored internal gradients follow producer-to-last-reader arena lifetimes; unstored ones never "
                 "leave registers or are aliased by their consumer. Runtime.grads is empty for this program."}
    return program
