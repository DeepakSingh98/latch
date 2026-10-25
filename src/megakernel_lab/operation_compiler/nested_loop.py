"""A pure bounded inner loop embedded in a differentiated outer loop.

Arithmetic and its VJPs use the existing operation-region capture and tape
partitioner. The new IR describes only recursive control and typed history.
No optimizer runs in the inner region: shared weights have one old version for
the entire update and receive one accumulated gradient and one final update.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch
from torch.func import grad_and_value, vjp

from .ir import CaptureError, Value, _contiguous_strides
from .loop import OperationIR, _capture_region, _partition_body_tape


@dataclass(frozen=True)
class NestedLoopBody:
    """Explicit inner loop used as ``capture_loop_training``'s outer body.

    init_fn(params, outer_carry, *inputs) -> inner_carry
    body_fn(params, outer_carry, inner_carry, state, *inputs) -> (inner_carry, state)
    finish_fn(params, outer_carry, inner_carry, *inputs) -> outer_carry

    ``count_input`` selects a scalar int64 device tensor in outer ``inputs``.
    Both counts change in-place between launches, including CUDA graph replays.
    Count clamping at either level sets the complete update's status to one.
    Initial scope: two levels, FP32 carry and momentum SGD. Ordinary supported
    PyTorch operations, tied parameters, and explicit Philox state are reused.
    """

    init_fn: Callable
    body_fn: Callable
    finish_fn: Callable
    count_input: int
    max_steps: int

    def __call__(self, params, carry, state, *inputs):
        inner = self.init_fn(params, carry, *inputs)
        count = min(max(int(inputs[self.count_input].item()), 0), self.max_steps)
        for _ in range(count):
            inner, state = self.body_fn(params, carry, inner, state, *inputs)
        return self.finish_fn(params, carry, inner, *inputs), state


@dataclass
class NestedOperationIR:
    regions: dict[str, OperationIR]
    values: dict[str, Value]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: dict[str, Any]
    state_writes: dict[str, str] = field(default_factory=dict)
    nested_operation: bool = True

    def __post_init__(self):
        m = self.metadata
        if self.state_writes or any(child.state_writes for child in self.regions.values()):
            raise CaptureError("REGION_EFFECT", "nested regions cannot commit mutable state")
        if any(type(child) is not OperationIR for child in self.regions.values()):
            raise CaptureError("NESTED_SCOPE", "the inner loop currently requires flat operation regions")
        if (
            m.get("mode") not in ("save_body_tape", "recompute_boundaries")
            or m.get("direction") not in ("forward", "reverse")
            or type(m.get("max_steps")) is not int
            or not 0 <= m["max_steps"] <= 1024
        ):
            raise CaptureError("NESTED_BOUND", "invalid nested operation policy or bound")
        sizes = [
            m.get(key) for key in ("n_parameters", "n_outer_carry", "n_inner_carry", "n_state", "n_inputs")
        ]
        if any(type(size) is not int or size < 0 for size in sizes) or not all(sizes[:3]):
            raise CaptureError("NESTED_ABI", "invalid nested region argument counts")
        n, no, nc, ns, ni = sizes
        if type(m.get("count_input")) is not int or not 0 <= m["count_input"] < ni:
            raise CaptureError("NESTED_COUNT", "nested count_input must name a data input")
        saved, reverse = m["mode"] == "save_body_tape", m["direction"] == "reverse"
        nt = len(m["tape"])
        expected_inputs = n + no + ns + ni + (nt if saved and reverse else 0) + (no if reverse else 0)
        expected_outputs = n + no if reverse else no + ns + (nt if saved else 0)
        if len(self.inputs) != expected_inputs or len(self.outputs) != expected_outputs:
            raise CaptureError(
                "NESTED_ABI", "nested region input/output arity differs from its control metadata"
            )
        count = self.values[self.inputs[n + no + ns + m["count_input"]]]
        if count.dtype != "int64" or count.shape != ():
            raise CaptureError("NESTED_COUNT", "nested count must be one scalar int64 input")

    def output_names(self):
        return self.outputs

    def to_dict(self):
        return {
            "format_version": 6,
            "kind": "nested_operation",
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "values": {k: asdict(v) for k, v in self.values.items()},
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("format_version") != 6 or data.get("kind") != "nested_operation":
            raise CaptureError("NESTED_VERSION", "unknown nested-operation format")
        if data.get("state_writes"):
            raise CaptureError("REGION_EFFECT", "nested operation regions cannot commit mutable state")
        values = {}
        for key, raw in data["values"].items():
            raw = dict(raw)
            for name in ("shape", "strides", "output_roles"):
                raw[name] = tuple(raw[name])
            values[key] = Value(**raw)
        return cls(
            {k: OperationIR.from_dict(v) for k, v in data["regions"].items()},
            values,
            tuple(data["inputs"]),
            tuple(data["outputs"]),
            data["metadata"],
        )


def _desc(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}


class NestedRegionModule:
    """CPU interpreter of the exact native schedule, with no runtime AD."""

    def __init__(self, ir, modules):
        self.ir, self.modules = ir, modules

    @property
    def code(self):
        return f"bounded nested {self.ir.metadata['direction']} {self.ir.metadata['mode']}"

    def __call__(self, *flat):
        m, gm = self.ir.metadata, self.modules
        n, no, nc, ns, ni = (
            m[k] for k in ("n_parameters", "n_outer_carry", "n_inner_carry", "n_state", "n_inputs")
        )
        na, primal = n + no, n + no + ns + ni
        # The native region-copy ABI materializes carry/tape/cotangent values.
        # Preserve arbitrary data-input strides, but model those boundary copies
        # explicitly in this interpreter.
        p = (*flat[:n], *(value.contiguous() for value in flat[n:na]))
        state, data = tuple(flat[na : na + ns]), tuple(flat[na + ns : primal])
        count = min(max(int(data[m["count_input"]].item()), 0), m["max_steps"])
        saved = m["mode"] == "save_body_tape"
        reverse = m["direction"] == "reverse"
        if not reverse or not saved:
            init = tuple(gm["init_saved" if saved else "init"](*p, *data))
            carry, init_tape = tuple(value.contiguous() for value in init[:nc]), tuple(
                value.contiguous() for value in init[nc:]
            )
            initial = carry, state
            histories = [[] for _ in range(nc + ns)]
            tapes = [[] for _ in m["inner_body_tape"]]
            for i, value in enumerate((*carry, *state)):
                histories[i].append(value)
            for _ in range(count):
                result = tuple(gm["body_saved" if saved else "body"](*p, *carry, *state, *data))
                carry, state = tuple(value.contiguous() for value in result[:nc]), tuple(
                    value.contiguous() for value in result[nc : nc + ns]
                )
                for i, value in enumerate((*carry, *state)):
                    histories[i].append(value)
                for values, value in zip(tapes, result[nc + ns :]):
                    values.append(value)
            if not reverse:
                result = tuple(gm["finish_saved" if saved else "finish"](*p, *carry, *data))
                if not saved:
                    return (*(value.contiguous() for value in result), *state)
                packed = []
                for desc, values in zip(m["boundary_desc"], histories):
                    output = torch.zeros(
                        (m["max_steps"] + 1, *desc["shape"]),
                        dtype=getattr(torch, desc["dtype"]),
                        device=p[0].device,
                    )
                    for i, value in enumerate(values):
                        output[i] = value
                    packed.append(output)
                for desc, values in zip(m["inner_body_tape"], tapes):
                    output = torch.zeros(
                        (max(1, m["max_steps"]), *desc["shape"]),
                        dtype=getattr(torch, desc["dtype"]),
                        device=p[0].device,
                    )
                    for i, value in enumerate(values):
                        output[i] = value
                    packed.append(output)
                return (
                    *(value.contiguous() for value in result[:no]),
                    *state,
                    *packed,
                    *init_tape,
                    *(value.contiguous() for value in result[no:]),
                )
        else:
            tape = flat[primal : primal + len(m["tape"])]
            h = nc + ns
            histories = tape[:h]
            b = h + len(m["inner_body_tape"])
            tapes = tape[h:b]
            e = b + len(m["inner_init_tape"])
            init_tape, finish_tape = tape[b:e], tape[e:]
            carry = tuple(value[count] for value in histories[:nc])
        cot = tuple(value.contiguous() for value in flat[-no:])
        result = tuple(
            gm["finish_vjp_saved" if saved else "finish_vjp"](
                *p, *carry, *data, *((*finish_tape, *cot) if saved else cot)
            )
        )
        gradients, dc = result[:na], tuple(value.contiguous() for value in result[na:])
        for t in range(count - 1, -1, -1):
            if saved:
                carry, state = tuple(value[t] for value in histories[:nc]), tuple(
                    value[t] for value in histories[nc:]
                )
                bt = tuple(value[t] for value in tapes)
            else:
                carry, state = initial
                for _ in range(t):
                    result = tuple(gm["body"](*p, *carry, *state, *data))
                    carry, state = tuple(value.contiguous() for value in result[:nc]), tuple(
                        value.contiguous() for value in result[nc:]
                    )
                bt = ()
            result = tuple(gm["body_vjp_saved" if saved else "body_vjp"](*p, *carry, *state, *data, *bt, *dc))
            gradients = tuple(a + b for a, b in zip(gradients, result[:na]))
            dc = tuple(value.contiguous() for value in result[na:])
        result = tuple(gm["init_vjp_saved" if saved else "init_vjp"](*p, *data, *init_tape, *dc))
        return tuple((a + b).contiguous() for a, b in zip(gradients, result))


def capture_nested_loop_training(
    init_fn, nested, loss_fn, params, momentum, inputs, count, config, **options
):
    from .loop import capture_loop_training
    from .optim import SGDConfig, prepare_optimizer

    inputs = tuple(inputs)
    config = config or SGDConfig()
    if not isinstance(config, SGDConfig):
        raise CaptureError(
            "NESTED_OPTIMIZER",
            "nested loops currently qualify momentum SGD; other optimizer connectivity is not yet admitted",
        )
    if (
        type(nested.max_steps) is not int
        or not 0 <= nested.max_steps <= 1024
        or type(nested.count_input) is not int
        or not 0 <= nested.count_input < len(inputs)
    ):
        raise CaptureError(
            "NESTED_BOUND", "inner max_steps must be in [0,1024] and count_input must name an input"
        )
    inner_count = inputs[nested.count_input]
    if not isinstance(inner_count, torch.Tensor) or inner_count.dtype != torch.int64 or inner_count.ndim != 0:
        raise CaptureError("NESTED_COUNT", "inner runtime count must be an explicit scalar int64 input")
    if options.get("storage_mode", "save_body_tape") not in ("save_body_tape", "recompute_boundaries"):
        raise CaptureError("NESTED_STORAGE", "nested loops support save_body_tape or recompute_boundaries")
    if (
        options.get("sequence_fn") is not None
        or options.get("precision") is not None
        or options.get("memory_policy", "default") != "default"
        or options.get("cast_reuse") is True
        or options.get("tape_options")
        or options.get("tape_policy", "auto") not in ("auto", "stage3")
    ):
        raise CaptureError(
            "NESTED_SCOPE",
            "nested loops currently require FP32, no sequences/cast reuse, and the saved stage3 tape policy",
        )
    # The ordinary outer capture still owns initial/terminal AD and the exact
    # optimizer ABI. Its identity body is replaced in all four execution modes.
    opts = {**options, "cast_reuse": False, "tape_policy": "stage3"}
    outer = capture_loop_training(
        init_fn, lambda p, c, s, *x: (c, s), loss_fn, params, momentum, inputs, count, config, **opts
    )
    n, no, ns, ni = len(outer.parameter_names), len(outer.carry_names), len(outer.state_names), len(inputs)
    aliases = outer.ir.metadata["parameter_aliases"]
    pvalues = outer.example_args[:n]
    cvalues = tuple(value.contiguous() for value in outer.graph_modules["init"](*pvalues, *inputs))
    if any(v.dtype != torch.float32 for v in cvalues):
        raise CaptureError("NESTED_CARRY", "nested outer carry must be FP32")
    extra_names = [f"__outer_carry_{i}" for i in range(no)]
    if set(extra_names).intersection(params):
        raise CaptureError("NESTED_PARAMETER", "parameter name collides with reserved outer carry binding")
    augmented = {k: v.detach().clone().contiguous() for k, v in zip(outer.parameter_names, pvalues)}
    augmented.update({k: v.detach().clone().contiguous() for k, v in zip(extra_names, cvalues)})

    def separate(p):
        return {a: p[k] for a, k in aliases.items()}, dict(
            zip(outer.carry_names, (p[k] for k in extra_names))
        )

    def initial(p, *data):
        pp, cc = separate(p)
        return nested.init_fn(pp, cc, *data)

    def body(p, c, s, *data):
        pp, cc = separate(p)
        return nested.body_fn(pp, cc, c, s, *data)

    inner = capture_loop_training(
        initial,
        body,
        lambda p, c, *x: next(iter(c.values())).sum(),
        augmented,
        None,
        inputs,
        inner_count.detach().clone(),
        SGDConfig(),
        max_steps=nested.max_steps,
        state=options.get("state"),
        storage_mode="save_body_tape",
        tape_policy="stage3",
        cast_reuse=False,
        init_storage="save_tape",
    )
    if any(inner.ir.values[inner.ir.outputs["carry"][k]].dtype != "float32" for k in inner.carry_names):
        raise CaptureError("NESTED_CARRY", "nested inner carry must be FP32")
    nc, na = len(inner.carry_names), n + no
    ivals = tuple(value.contiguous() for value in inner.graph_modules["init"](*augmented.values(), *inputs))
    dc = tuple(torch.zeros_like(v) for v in cvalues)

    def finish(p, c, *data):
        pp, cc = separate(dict(zip(augmented, p)))
        result = nested.finish_fn(pp, cc, dict(zip(inner.carry_names, c)), *data)
        if not isinstance(result, dict) or set(result) != set(outer.carry_names):
            raise CaptureError("NESTED_CARRY", "finish must preserve the outer carry keys")
        values = tuple(result[k] for k in outer.carry_names)
        if any((a.shape, a.dtype) != (b.shape, b.dtype) for a, b in zip(values, cvalues)):
            raise CaptureError("NESTED_CARRY", "finish must preserve the outer carry shapes and dtypes")
        return values

    def forward(*flat):
        return tuple(v.clone() for v in finish(tuple(flat[:na]), tuple(flat[na : na + nc]), *flat[na + nc :]))

    def joint(*flat):
        p, c, data = tuple(flat[:na]), tuple(flat[na : na + nc]), tuple(flat[na + nc : na + nc + ni])
        out, pb = vjp(lambda pp, cc: finish(pp, cc, *data), p, c)
        dp, di = pb(tuple(flat[na + nc + ni :]))
        return tuple(v.clone() for v in (*out, *dp, *di))

    def reverse(*flat):
        return joint(*flat)[no:]

    labels = [
        *(f"parameter:{k}" for k in augmented),
        *(f"carry:{k}" for k in inner.carry_names),
        *(f"input:{i}" for i in range(ni)),
    ]
    args = (*augmented.values(), *ivals, *inputs)
    regions = {k: v for k, v in inner.ir.regions.items() if k not in ("optimizer", "terminal_vjp")}
    modules = {k: v for k, v in inner.graph_modules.items() if k in regions}
    regions["finish"], modules["finish"] = _capture_region(forward, args, "finish", labels)
    ir, gm = _capture_region(
        joint, (*args, *dc), "finish_joint", labels + [f"cotangent:{k}" for k in outer.carry_names]
    )
    regions["finish_vjp"], modules["finish_vjp"] = _capture_region(
        reverse, (*args, *dc), "finish_vjp", labels + [f"cotangent:{k}" for k in outer.carry_names]
    )
    (
        regions["finish_saved"],
        modules["finish_saved"],
        regions["finish_vjp_saved"],
        modules["finish_vjp_saved"],
        ftape,
    ) = _partition_body_tape(ir, gm, n_primals=len(args), n_forward_outputs=no, policy="stage3")
    for owner, group in (("outer", outer.ir.regions), ("inner", regions)):
        for label, region in group.items():
            unsupported = {
                value.dtype
                for value in region.values.values()
                if value.dtype in ("float16", "bfloat16", "float64")
            }
            if unsupported:
                raise CaptureError(
                    "NESTED_PRECISION",
                    f"{owner} {label}: nested floating arithmetic requires FP32; found {sorted(unsupported)}",
                )
    bd = [_desc(v) for v in (*ivals, *(options.get("state") or {}).values())]
    bt, it = inner.ir.metadata["body_tape"], inner.ir.metadata["init_tape"]
    tape = [
        {"name": f"inner_boundary_{i}", "shape": [nested.max_steps + 1, *d["shape"]], "dtype": d["dtype"]}
        for i, d in enumerate(bd)
    ]
    tape += [
        {"name": f"inner_body_{i}", "shape": [max(1, nested.max_steps), *d["shape"]], "dtype": d["dtype"]}
        for i, d in enumerate(bt)
    ]
    tape += [{**d, "name": f"inner_init_{i}"} for i, d in enumerate(it)]
    tape += [{**d, "name": f"inner_finish_{i}"} for i, d in enumerate(ftape)]
    meta = {
        "n_parameters": n,
        "n_outer_carry": no,
        "n_inner_carry": nc,
        "n_state": ns,
        "n_inputs": ni,
        "count_input": nested.count_input,
        "max_steps": nested.max_steps,
        "boundary_desc": bd,
        "inner_body_tape": bt,
        "inner_init_tape": it,
        "finish_tape": ftape,
        "tape": tape,
    }
    primal = (*pvalues, *cvalues, *((options.get("state") or {}).values()), *inputs)
    primal_labels = [
        *(f"parameter:{k}" for k in outer.parameter_names),
        *(f"carry:{k}" for k in outer.carry_names),
        *(f"state:{k}" for k in outer.state_names),
        *(f"input:{i}" for i in range(ni)),
    ]
    for label, mode, direction in (
        ("body", "recompute_boundaries", "forward"),
        ("body_saved", "save_body_tape", "forward"),
        ("body_vjp", "recompute_boundaries", "reverse"),
        ("body_vjp_saved", "save_body_tape", "reverse"),
    ):
        saved, rev = mode == "save_body_tape", direction == "reverse"
        in_desc = (
            [{**_desc(v), "strides": list(v.stride())} for v in primal]
            + (tape if saved and rev else [])
            + ([_desc(v) for v in dc] if rev else [])
        )
        out_desc = (
            [_desc(v) for v in (*pvalues, *cvalues)]
            if rev
            else [_desc(v) for v in (*cvalues, *((options.get("state") or {}).values()))]
            + (tape if saved else [])
        )
        values = {}

        def make(desc, index, prefix):
            name = f"{prefix}_{index}"
            values[name] = Value(
                name,
                tuple(desc["shape"]),
                desc["dtype"],
                (
                    tuple(desc["strides"])
                    if prefix == "arg" and "strides" in desc
                    else _contiguous_strides(desc["shape"])
                ),
                role="region_input" if prefix == "arg" else "region_output",
            )
            return name

        ins = tuple(make(d, i, "arg") for i, d in enumerate(in_desc))
        outs = tuple(make(d, i, "result") for i, d in enumerate(out_desc))
        region = NestedOperationIR(
            regions,
            values,
            ins,
            outs,
            {
                **meta,
                "name": label,
                "mode": mode,
                "direction": direction,
                "input_labels": primal_labels
                + ([f"tape:{d['name']}" for d in tape] if saved and rev else [])
                + ([f"carry_cotangent:{k}" for k in outer.carry_names] if rev else []),
            },
        )
        setattr(outer.ir, label, region)
        outer.graph_modules[label] = NestedRegionModule(region, modules)
    outer.ir.metadata.update(
        body_tape=tape,
        storage_modes=["save_body_tape", "recompute_boundaries"],
        body_tape_policy="save_body_tape recursively saves inner boundaries and inner init/body/finish VJP dependencies; recompute_boundaries replays both histories from old state",
        tape_selection={
            "requested": options.get("tape_policy", "auto"),
            "selected": "stage3",
            "reason": "recursive saved-state partition; flat rematerialization passes do not rewrite region history",
        },
        nested_loop={
            "count_input": nested.count_input,
            "max_steps": nested.max_steps,
            "parameter_version": "shared old parameters at both levels; one optimizer commit after both reverse histories",
            "storage": "typed inner boundaries, body/init/finish VJP tape per outer iteration",
            "scope": "two levels; FP32 carry; momentum SGD; explicit int64/bool state",
        },
    )
    # Independent eager definition keeps ordinary autograd through both Python
    # histories for reference consumers; graph_step interprets captured VJPs.
    prepared = prepare_optimizer(
        config,
        dict(zip(outer.parameter_names, pvalues)),
        momentum=momentum,
        optimizer_state=options.get("optimizer_state"),
        aliases=aliases,
    )
    nm = len(outer.optimizer_state_names)

    def step(*flat):
        p, m = tuple(flat[:n]), tuple(flat[n : n + nm])
        s = dict(zip(outer.state_names, flat[n + nm : n + nm + ns]))
        data, requested = flat[n + nm + ns : -1], int(flat[-1].item())
        steps = min(max(requested, 0), outer.ir.max_steps)

        def objective(pp):
            mapped = {a: pp[outer.parameter_names.index(k)] for a, k in aliases.items()}
            carry, state = init_fn(mapped, *data), s
            for _ in range(steps):
                carry, state = nested(mapped, carry, state, *data)
            return loss_fn(mapped, carry, *data), (
                tuple(state[k] for k in outer.state_names),
                tuple(carry[k] for k in outer.carry_names),
            )

        grads, (loss, (state, carry)) = grad_and_value(objective, has_aux=True)(p)
        newp, newm = prepared.update(p, grads, m)
        inner_requested = int(data[nested.count_input].item())
        status = int(requested != steps or not 0 <= inner_requested <= nested.max_steps)
        return (
            loss,
            *grads,
            *newp,
            *newm,
            *state,
            *carry,
            flat[-1].new_tensor(steps),
            flat[-1].new_tensor(status),
        )

    outer.step_fn = step
    return outer
