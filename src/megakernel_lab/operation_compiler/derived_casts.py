"""Persistent BF16 operand copies of parameters (LATCH-SELECTIVE-10, ops.accumulation_rules ``dcast``).

A ``cat_cast`` of FP32 parameters (the BF16 product operand of stacked per-expert weights, ops/cast_concat.py) is a
pure function of those parameters. When several programs share one persistent state and only some of them change
the parameters (a selective-cadence accumulate/apply pair: the slow experts change only at apply), recomputing it
in every launch re-reads the same FP32 masters and stores the same BF16 values.

With ``dcast`` the capture keeps that copy as persistent derived state ``derived:bf16cat:<first piece>+<n>``
(role ``state``, so ordinary state binding, sharing and the end-of-update commit apply):

* every reader of the cast reads the derived state instead (the copy of the current parameters);
* a program that returns the pieces unchanged (clone of the binding) returns the derived state unchanged, which
  ``update_schedule`` ustate commits as nothing;
* a program that updates the pieces commits ``cat_cast`` of the updated values (one pass, the same
  ``__float2bfloat16_rn`` rounding) after all reads of the old copy, as every state commit;
* the runtime (``Runtime.derived``) recomputes the copy with ``torch.cat(...).to(bfloat16)`` (round to nearest
  even, the same values) whenever the host has written a piece (PyTorch version counters), so loads and restores
  stay consistent. The copy is not model state: it is excluded from ``Runtime.state`` and from checkpoints.

Values are identical to recasting in every launch; no arithmetic changes. Programs without such casts are unchanged.
"""
from __future__ import annotations

import copy
import hashlib

PREFIX = "derived:"
TARGET = "latch_ops.cat_cast.default"


def _name(x):
    return getattr(x, "name", x)


def _replace_refs(obj, mapping):
    from .ir import Ref
    if isinstance(obj, Ref):
        return Ref(mapping.get(obj.name, obj.name))
    if isinstance(obj, list):
        return [_replace_refs(x, mapping) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_replace_refs(x, mapping) for x in obj)
    if isinstance(obj, dict):
        return {k: _replace_refs(v, mapping) for k, v in obj.items()}
    return obj


def apply(ir, example_args):
    """(ir', example_args', records). ``ir`` and ``example_args`` are not modified."""
    import torch
    from .ir import Operation, Ref, Value
    ops = list(ir.operations)
    candidates = [op for op in ops if str(op.target) == TARGET]
    if not candidates or not isinstance(ir.outputs, dict):
        return ir, example_args, []
    values = dict(ir.values)
    position = {name: i for i, name in enumerate(ir.inputs)}
    producer = {out: op for op in ops for out in op.outputs}
    params_out = {k: _name(v) for k, v in ir.outputs["params"].items()}
    records, mapping, removed, appended = [], {}, set(), []
    inputs, args = list(ir.inputs), list(example_args)
    state_out = dict(ir.outputs.get("state", {}))
    state_names = list(ir.metadata.get("state_names", ()))
    for op in candidates:
        if len(op.args) != 3 or not isinstance(op.args[0], (list, tuple)) or int(op.args[1]) != 0:
            continue
        pieces = [_name(a) for a in op.args[0]]
        if not all(values[p].role == "parameter" and p in position for p in pieces):
            continue
        names = [values[p].parameter for p in pieces]
        new = [params_out.get(n) for n in names]

        def unchanged(piece, value):
            seen = set()
            while value in producer and value not in seen:
                seen.add(value)
                source = producer[value]
                if str(source.target).removeprefix("aten.").split(".")[0] not in ("clone", "detach", "alias") \
                        or len(source.inputs) != 1:
                    return False
                value = source.inputs[0]
            return value == piece

        same = [unchanged(p, v) for p, v in zip(pieces, new)]
        if any(v is None for v in new) or len(set(same)) != 1:
            continue
        out = op.outputs[0]
        result = values[out]
        key = f"{PREFIX}bf16cat:{names[0]}+{len(names) - 1}"
        if key in state_out or key in state_names:
            continue
        index = len(records)
        binding = f"derived_state_{index}"
        values[binding] = Value(binding, tuple(result.shape), result.dtype, tuple(result.strides), role="state",
                                parameter=key)
        inputs.append(binding)
        with torch.no_grad():
            initial = torch.cat([args[position[p]].detach() for p in pieces], 0).to(torch.bfloat16)
        args.append(initial.contiguous())
        mapping[out] = binding
        removed.add(op.name)
        commit = f"derived_commit_{index}"
        if same[0]:
            appended.append(Operation(commit, "aten.clone.default", (Ref(binding),), {}, (binding,), (commit,), (),
                                      (binding,), (commit,)))
            mode = "read (pieces unchanged in this program; commits nothing)"
        else:
            appended.append(Operation(commit, TARGET, ([Ref(v) for v in new], 0, op.args[2]), {}, tuple(new),
                                      (commit,), tuple(dict.fromkeys(producer[v].name for v in new if v in producer)), tuple(new),
                                      (commit,)))
            mode = "refresh (pieces updated here; cat_cast of the updated values committed after all reads)"
        values[commit] = Value(commit, tuple(result.shape), result.dtype, tuple(result.strides), producer=commit,
                               output_roles=(f"state:{key}",))
        state_out[key] = commit
        state_names.append(key)
        records.append(dict(key=key, binding=binding, replaced=out, pieces=names, dim=0, dtype=str(result.dtype),
                            shape=list(result.shape), mode=mode,
                            pieces_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest()))
    if not records:
        return ir, example_args, []
    kept = []
    for op in ops:
        if op.name in removed:
            continue
        if any(name in mapping for name in (*op.inputs, *op.reads)):
            op = copy.copy(op)
            op.args = _replace_refs(op.args, mapping)
            op.kwargs = _replace_refs(op.kwargs, mapping)
            op.inputs = tuple(mapping.get(n, n) for n in op.inputs)
            op.reads = tuple(mapping.get(n, n) for n in op.reads)
            op.dependencies = tuple(d for d in op.dependencies if d not in mapping)
        kept.append(op)
    for name in mapping:
        del values[name]
    for name, value in list(values.items()):
        if value.alias_of in mapping:
            values[name] = copy.copy(value)
            values[name].alias_of = mapping[value.alias_of]
    outputs = dict(ir.outputs)
    outputs["state"] = state_out
    metadata = dict(ir.metadata)
    metadata["state_names"] = state_names
    metadata.setdefault("state_derivatives", "parameter derivatives of loss only; returned state is auxiliary output")
    metadata["derived_state"] = records
    result = copy.copy(ir)
    result.values, result.operations, result.inputs = values, kept + appended, tuple(inputs)
    result.outputs, result.metadata = outputs, metadata
    return result, tuple(args), records
