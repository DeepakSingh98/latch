"""Adapt attributed flat attention bodies to the common phase contract.

The matchers and emitted arithmetic are unchanged. Long BF16-boundary and
mixed BF16/FP32 groups are selected independently, alongside ordinary matrix,
reduction and pointwise phases. A refusal affects that group only.
"""
from __future__ import annotations

from . import attention_lower, attention_pack, attention_rules, flat_long_attention, mixed_attention, padding_masks
from .execution_plan import current_precision

TAPE_SCHEMA = "composed-flat-long-attention-tape-v1"
TAPE_TARGET = "latch_ops.composed_attention_tape.default"


def _graph_maps(ir):
    """(producers {value: op}, consumers {value: {op name}}) of an IR."""
    producers, consumers = {}, {}
    for op in ir.operations:
        for name in op.inputs:
            consumers.setdefault(name, set()).add(op.name)
        for name in op.outputs:
            producers[name] = op
    return producers, consumers


def rewrite_tape(ir, *, version=7):
    """Expose an existing joint attention body's saved values before partitioning.

    These are compiler IR markers, never a new eager operator or derivative.
    Admission consumes the original captured primal/VJP and uses the same
    matcher, packed-layout proof, phase emitter and exact3 recipe as
    an unsplit composed program. Inadmissible groups keep their ordinary ops.
    The old loop_attention rewrite has a different (natural-log LSE) tape and
    cannot bind the flat body's log2-LSE, live bits and proof tables.
    """
    from dataclasses import replace
    import json
    from .ir import CaptureError, Operation, Ref, Value, _contiguous_strides
    from .lower import TensorLayout, _field, _ident, _ref

    if type(version) is not int or version not in (3, 7, 13):
        raise CaptureError("ATTENTION_TAPE", "attention tape supports schedules 3, 7 and explicit BF16 schedule 13")
    if ir.metadata.get("composed_attention_tape"):
        raise CaptureError("ATTENTION_TAPE", "attention tape prepass was already applied")
    names = set(ir.values)
    out_name = lambda op: op.outputs[0]
    parts = (ir.operations, ir.values, names, _ref, _field, out_name)
    static = {name: TensorLayout(v.shape, v.strides, name) for name, v in ir.values.items()}
    groups, refusals = attention_lower.plan(*parts[:3], ref=_ref, field=_field, output_name=out_name,
        protected=set(ir.output_names()), masks=True)
    plans, rejected = flat_long_attention.admit(groups, parts, static, set(ir.output_names()))
    from . import bf16_sdpa
    explicit = bf16_sdpa.plans(ir)
    if version == 13 and (plans or not explicit):
        raise CaptureError("ATTENTION_TAPE", "schedule 13 requires exclusively explicit BF16 SDPA groups")
    if set(plans).intersection(explicit):
        raise CaptureError("ATTENTION_TAPE", "opaque attention plan name collision")
    plans.update(explicit)
    refusals += rejected
    values = {name: replace(v) for name, v in ir.values.items()}
    before, skipped, records, mask_owner, graph_maps = {}, set(), [], {}, None
    order = {name: i for i, op in enumerate(ir.operations) for name in op.outputs}

    def serial(value):
        if isinstance(value, dict):
            return {key: serial(item) for key, item in value.items()}
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, (tuple, list)):
            return [serial(item) for item in value]
        return value

    for label, plan in plans.items():
        group = plan["group"]
        try:
            if skipped.intersection(plan["skip"]):
                raise flat_long_attention.Refused("attention groups overlap")
            # Material boundaries use contiguous logical storage. A packed root
            # itself must retain its captured addressing across the cut.
            for name in set(plan["sources"].values()):
                v = ir.values[name]
                if name not in ir.inputs and (v.alias_of is not None or v.strides != _contiguous_strides(v.shape)):
                    raise flat_long_attention.Refused(f"{name}: cross-region source is not a contiguous material root")
            mask = None
            if group.get("mask") is not None:
                mask = padding_masks.mask_strides(group, static, plan["bh"], plan["t"], attention_lower._mask_strides)
                if mask is None or ir.values[group["mask"]].dtype != "float32":
                    raise flat_long_attention.Refused("tape mask is not FP32 broadcastable")
            recipe = "bf16" if plan.get("arithmetic_contract") == bf16_sdpa.CONTRACT else "exact3"
            # ops.attention_rules (opt-in): reuse an earlier group's mask proof tables, and compute key
            # norms inside forward items when that leaves the group without a prepare phase.
            mask_key = None if mask is None else (group["mask"], plan["bh"], plan["t"], tuple(mask),
                                                  getattr(mask, "heads", 1), getattr(mask, "batches", None))
            mask_tables = mask_owner.get(mask_key) if (mask_key is not None and attention_rules.enabled("tables")) else None
            pack = None
            if attention_rules.enabled("pack") and version == 7 and plan.get("arithmetic_contract") == bf16_sdpa.CONTRACT:
                if graph_maps is None:
                    graph_maps = _graph_maps(ir)
                try:
                    pack = attention_pack.plan_group(ir, plan, *graph_maps, set(ir.output_names()))
                except attention_pack.Refused as exc:
                    refusals.append({"group": label, "reason": "attention pack: " + str(exc)})
            if pack is not None:
                removed = {name for item in pack.values() for name in item["removed"]}
                if skipped.intersection(removed):
                    raise flat_long_attention.Refused("attention operand chains overlap")
                plan = dict(plan, skip=set(plan["skip"]) | removed, removed=set(plan["removed"]) | removed)
            keynorm_item = attention_rules.enabled("kn_item") and version == 7 and pack is None
            merged = attention_rules.enabled("merged_backward") and version == 7
            emissions = {}
            for which, phases in (("forward", ("prepare", "forward")),
                                  ("backward", flat_long_attention.backward_phases(version, merged))):
                outputs = {}
                if which == "forward" and pack is not None:
                    # The pack phase re-produces the padded operands and writes the key-norm table.
                    outputs.update({plan["sources"][key]: None for key in attention_pack.KEYS})
                    outputs[flat_long_attention.aux_names(group)["kn"]] = ((plan["bh"], plan["t"] // 64), "float32")
                for phase in phases:
                    emitted = flat_long_attention.emit(plan, _ident, mask, which=phase, recipe=recipe,
                                                       version=version, keynorm_epilogue=False,
                                                       mask_tables=mask_tables, keynorm_in_item=keynorm_item,
                                                       keynorm_external=pack is not None)
                    if emitted is not None:
                        outputs.update(emitted[2]["outputs"])
                emissions[which] = outputs
            synthetic = {n for outputs in emissions.values() for n, desc in outputs.items() if desc is not None}
            if synthetic.intersection(values):
                raise flat_long_attention.Refused("synthetic attention state name collision")
            base_inputs = list(dict.fromkeys(plan["sources"][key] for key in ("q", "k", "v")))
            if pack is not None:
                base_inputs = [pack[key]["src"] for key in attention_pack.KEYS]
            if group.get("mask") is not None:
                base_inputs.append(group["mask"])
            if mask_tables is not None:
                base_inputs += [mask_tables[key] for key in ("mmax", "mzero", "rm")]
            halves = {}
            for which, outputs in emissions.items():
                # Retain every pointer field used by the original body, even
                # when a particular schedule only uses its value in forward.
                inputs = list(base_inputs)
                if which == "backward":
                    if pack is not None:  # the backward reads the packed operands, not their sources
                        inputs = [plan["sources"][key] for key in attention_pack.KEYS] + inputs[3:]
                    inputs += [plan["sources"]["do"], *emissions["forward"]]
                inputs = list(dict.fromkeys(inputs))
                anchor = plan["anchor_" + which]
                if any(n in order and order[n] >= order[anchor] for n in inputs if n not in synthetic
                       and n not in emissions["forward"]):
                    raise flat_long_attention.Refused(f"{which}: body source is not ready at its anchor")
                halves[which] = {"anchor": next(iter(outputs)), "outputs": list(outputs), "inputs": inputs}
            extra = {}
            if mask_tables is not None:
                extra["mask_tables"] = dict(mask_tables)
            if keynorm_item:
                extra["tape_keynorm_item"] = True
            if merged:
                extra["tape_merged_backward"] = True
            if pack is not None:
                extra["tape_pack"] = {key: {k: v for k, v in pack[key].items() if k != "removed"}
                                      for key in attention_pack.KEYS}
            resolved = serial(dict(plan, tape_recipe=recipe, tape_version=version, **extra))
            record = {"plan": resolved, **halves}
            # JSON serialization is part of the artifact/region ABI.
            json.dumps(record)
        except flat_long_attention.S.Refused as exc:
            if label in explicit:
                raise CaptureError("BF16_SDPA", f"{label}: explicit BF16 contract refused: {exc}") from exc
            refusals.append({"group": label, "reason": "composed attention tape: " + str(exc)})
            continue
        for which, outputs in emissions.items():
            inputs = tuple(halves[which]["inputs"])
            markers = []
            for name, desc in outputs.items():
                if desc is None:
                    values[name] = replace(values[name], alias_of=None, storage_offset=0, producer=name,
                                           strides=_contiguous_strides(values[name].shape))
                else:
                    shape, dtype = tuple(desc[0]), desc[1]
                    values[name] = Value(name, shape, dtype, _contiguous_strides(shape), producer=name)
                markers.append(Operation(name, TAPE_TARGET, tuple(Ref(n) for n in inputs),
                    {"group": label, "which": which}, inputs, (name,), (), inputs, (name,)))
            before[plan["anchor_" + which]] = markers
        skipped.update(plan["skip"])
        records.append(record)
        if mask_key is not None and mask_tables is None and attention_rules.enabled("tables"):
            mask_owner.setdefault(mask_key, {key: name for key, name in flat_long_attention.aux_names(group).items()
                                             if key in ("mmax", "mzero", "rm")})
    operations = []
    available = set(ir.inputs)
    for op in ir.operations:
        replacement = before.get(op.outputs[0], [])
        for item in [*replacement, *([] if op.outputs[0] in skipped else [op])]:
            missing = set(item.inputs) - available
            if missing:
                raise CaptureError("ATTENTION_TAPE", f"{item.name}: missing SSA dependencies {sorted(missing)}")
            operations.append(replace(item, dependencies=tuple(dict.fromkeys(
                values[n].producer for n in item.inputs if values[n].producer is not None))))
            available.update(item.outputs)
    required = set(ir.inputs) | set(ir.output_names()) | {n for op in operations for n in (*op.inputs, *op.outputs)}
    report = {"schema": TAPE_SCHEMA, "groups": records, "refusals": refusals,
              "schedule_version": version,
              "selection": f"joint captured AD, existing flat-long admission; explicit BF16 or exact3/v{version}; no model dispatch",
              "fallback": "ordinary captured operations for refused groups",
              **({"attention_rules": attention_rules.current(), "attention_rules_version": attention_rules.VERSION}
                 if attention_rules.current() is not None else {})}
    return replace(ir, values={n: values[n] for n in required}, operations=operations,
                   metadata={**ir.metadata, "composed_attention_tape": report}), report


class ComposedAttention:
    def __init__(self, *, operations, values, names, inputs, outputs, state_writes,
                 placement, layouts, dtypes, roots, phases, producers, helper_parts,
                 extra_headers, decisions, ref, field, output_name, groups,
                 excluded_members, refusals, allow_short=False, tape=None):
        from .lower import TensorLayout, _contiguous
        self.__dict__.update(locals())
        self.parts = (operations, values, names, ref, field, output_name)
        self.protected = set(outputs) | set(state_writes.values()) | set(placement)
        self.static = {}
        for name, value in values.items():
            shape = tuple(int(x) for x in field(value, "shape"))
            self.static[name] = TensorLayout(shape, tuple(field(value, "strides", _contiguous(shape))), name)
        long, rejected = flat_long_attention.admit(groups, self.parts, self.static, self.protected)
        refusals.extend(rejected)
        accumulating = {name for name, (_, mode) in placement.items() if mode == "accumulate"}
        self.long = {}
        for label, plan in long.items():
            if set(plan["skip"]) & (set(excluded_members) | accumulating):
                refusals.append({"group": label, "reason": "composed long body overlaps an existing group or an accumulated output"})
            else:
                self.long[label] = plan
        self.remaining_groups = [g for g in groups if g["label"] not in self.long]
        occupied = set(excluded_members) | {n for p in self.long.values() for n in p["skip"]}
        self.short = {}
        if allow_short:
            from . import short_attention
            short, rejected = short_attention.admit(self.remaining_groups, self.parts, self.static, self.protected,
                                                    splits=3, cast_first=True)
            refusals.extend(rejected)
            for label, plan in short.items():
                if plan["skip"] & (occupied | accumulating):
                    refusals.append({"group": label, "reason": "composed short body overlaps another group or an accumulated output"})
                elif plan["splits"] != 3:
                    refusals.append({"group": label, "reason": "flat short adapter requires the existing exact three-piece recipe"})
                else:
                    self.short[label] = plan
                    occupied |= plan["skip"]
            self.remaining_groups = [g for g in self.remaining_groups if g["label"] not in self.short]
        occupied |= {n for g in self.remaining_groups for n in g["skip"]}
        mixed, rejected = mixed_attention.admit(self.parts,
            lambda name: (self.static[name].shape, self.static[name].strides), self.protected,
            exclude=attention_lower.excluded_groups())
        refusals.extend(rejected)
        self.mixed = {}
        for label, plan in mixed.items():
            finals = set(plan["finals"].values()) | {v[0] for v in plan.get("reduce", {}).values()}
            if plan["removed"] & occupied or (plan["removed"] | finals) & accumulating:
                refusals.append({"group": label, "reason": "composed mixed attention overlaps another group or an accumulated output"})
            else:
                self.mixed[label] = plan
                occupied |= plan["removed"]
        self.members = {n: ("long", p) for p in self.long.values() for n in p["skip"]}
        self.members.update({n: ("short", p) for p in self.short.values() for n in p["skip"]})
        self.members.update({n: ("mixed", p) for p in self.mixed.values() for n in p["removed"]})
        self.mask_owner = {}
        self.tape_outputs = set()
        if tape:
            if tape.get("schema") != TAPE_SCHEMA:
                raise attention_lower.AttentionRefused("tape", "unknown composed attention tape schema")
            by_output = {output_name(op): op for op in operations}
            for record in tape["groups"]:
                for which in ("forward", "backward"):
                    half = record[which]
                    if half["anchor"] not in by_output:
                        continue
                    for name in half["outputs"]:
                        op = by_output.get(name)
                        if (op is None or str(field(op, "target")) != TAPE_TARGET
                                or tuple(field(op, "inputs")) != tuple(half["inputs"])):
                            raise attention_lower.AttentionRefused(record["plan"]["group"]["label"],
                                                                   "incomplete attention tape half")
                    resolved = dict(record["plan"], tape_which=which, tape_anchor=half["anchor"])
                    self.members.update({name: ("tape_long", resolved) for name in half["outputs"]})
                    self.tape_outputs.update(half["outputs"])

    def _include(self, header, defines=""):
        text = defines + f'#include "{header}"\n'
        if text not in self.helper_parts:
            self.helper_parts.append(text)
        if header not in self.extra_headers:
            self.extra_headers.append(header)

    def _phase(self, kind, code, extra, reads, *, group, header):
        from .lower import TensorLayout, _contiguous, _dtype, _ident
        outputs = dict(extra["outputs"])
        split = extra.get("p2_split")
        if split:
            # Scratch/counter lifetime is exactly this body. Initialization is
            # explicit, and a full-grid fence precedes any counter reader.
            prefix = group["o"] + "__compose_split_"
            for suffix, words in (("counters", split["counter_words"]), ("partials", split["partial_words"])):
                if words:
                    outputs[prefix + suffix] = ((int(words),), "float32")
            ctr = _ident(prefix + "counters")
            code = code.replace("/*P2_CNT*/", f"reinterpret_cast<unsigned*>({ctr})")
            code = code.replace("/*P2_PART*/", _ident(prefix + "partials") if split["partial_words"] else "nullptr")
            code = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; "
                    f"i < {split['counter_words']}LL; i += (long long)gridDim.x * blockDim.x) {ctr}[i] = 0.0f;\n"
                    "grid.sync();\n" + code)
        for name, desc in outputs.items():
            if desc is None:
                shape, dtype = tuple(self.field(self.values[name], "shape")), _dtype(self.values[name])
            else:
                shape, dtype = tuple(desc[0]), desc[1]
                if name in self.names or name in self.layouts:
                    if (name not in self.tape_outputs or name not in self.values
                            or tuple(self.field(self.values[name], "shape")) != shape
                            or _dtype(self.values[name]) != dtype):
                        raise attention_lower.AttentionRefused(group["label"], f"synthetic name collision {name}")
            self.layouts[name] = TensorLayout(shape, _contiguous(shape), name)
            self.dtypes[name] = dtype
            if name not in self.placement and name not in self.roots:
                self.roots.append(name)
        metadata = {k: v for k, v in extra.items() if k not in ("outputs", "reads", "aux_reads", "bias_phase")}
        metadata.update(group=group["label"], attention_output=group["o"], header=header, threads=256,
                        body_interface="compose-v1", source_commit=extra.get("source_commit",
                            "P-OPS4-D" if "long" in kind else "P-OPS5-D"))
        if split:
            metadata["internal_grid_barrier"] = True
            metadata["scratch_effects"] = {name: "body-private zeroed counter or partials" for name in outputs
                                             if name.startswith(group["o"] + "__compose_split_")}
            metadata["private_scratch"] = [{"name": name, "shape": desc[0], "dtype": desc[1],
                "role": "counter" if name.endswith("counters") else "partials",
                "initialization": "zero counter before grid barrier; body defines partials before reads"}
                for name, desc in outputs.items() if name.startswith(group["o"] + "__compose_split_")]
        self.phases.append(dict(kind=kind, names=list(outputs), targets=["compose." + kind],
            reads=list(dict.fromkeys(reads)), writes=list(outputs), code=code, **metadata))
        for name in outputs:
            self.producers[name] = len(self.phases) - 1

    def install(self, name):
        family, plan = self.members[name]
        if family == "tape_long":
            if name == plan["tape_anchor"]:
                self._long(plan, plan["tape_which"] == "forward")
            return
        if name not in (plan["anchor_forward"], plan["anchor_backward"]):
            return
        if family == "long":
            self._long(plan, name == plan["anchor_forward"])
        elif family == "short":
            self._short(plan, name == plan["anchor_forward"])
        else:
            self._mixed(plan, name == plan["anchor_forward"])

    def _short(self, plan, forward):
        from . import attention_adapters, short_attention
        from .lower import _ident
        group = plan["group"]
        try:
            fresh = short_attention.finalize(plan, self.parts, {**self.static, **self.layouts}, self.protected,
                                             cast_first=True)
        except short_attention.Refused as exc:
            raise attention_lower.AttentionRefused(group["label"], f"short attention: {exc}") from exc
        for key in (("q", "k", "v") if forward else ("q", "k", "v", "do")):
            source = fresh["sources"][key]
            if (source not in self.layouts or source != plan["sources"][key]
                    or fresh["in_strides"][key] != plan["in_strides"][key]
                    or self.dtypes.get(source) != "bfloat16"):
                raise attention_lower.AttentionRefused(group["label"], f"{key}: lowered short-attention operand differs")
        mask = None
        if group.get("mask") is not None:
            mask = padding_masks.mask_strides(group, self.layouts, plan["bh"], plan["t"], attention_lower._mask_strides)
            if mask is None or self.dtypes[group["mask"]] != "float32":
                raise attention_lower.AttentionRefused(group["label"], "short mask is not FP32 broadcastable")
        portable = attention_adapters.emit_phases("bf16b_short", fresh, layouts=self.layouts, dtypes=self.dtypes,
            values=self.values, field=self.field, ident=_ident, mask_strides=mask)
        for phase in portable:
            if (phase["role"] == "forward") != forward:
                continue
            for header in phase["headers"]:
                self._include(header, "".join(f"#define {k} {v}\n" for k, v in sorted(phase["defines"].items())))
            extra = {key: phase[key] for key in ("engine", "role", "shared_bytes", "work_items", "arithmetic",
                                               "storage", "arch", "internal_grid_barrier")}
            extra.update(phase["layouts"], outputs={name: None if origin == "captured_final" else (shape, dtype)
                         for name, (shape, dtype, origin) in phase["outputs"].items()},
                         source_commit=attention_adapters.BODIES["bf16b_short"].lineage,
                         requires_full_grid=True, execution_grid_scope="physical_cooperative_grid")
            self._phase(phase["kind"], phase["code"], extra, phase["reads"], group=group, header=phase["headers"][-1])
        if forward:
            self.decisions.append(dict(group=group["label"], decision="adapter:bf16b_short", splits=3,
                sources=fresh["sources"], outputs=fresh["finals"], head_width=plan["hd"], sequence=plan["t"],
                removed=sorted(fresh["removed"]), source_commit=attention_adapters.BODIES["bf16b_short"].lineage))

    def _long(self, plan, forward):
        from .lower import _ident
        group = plan["group"]
        pack = plan.get("tape_pack") if "tape_which" in plan else None
        for key in (("q", "k", "v") if forward else ("q", "k", "v", "do")):
            if forward and pack:
                continue  # produced by this half's pack phase
            name = plan["sources"][key]
            if self.layouts.get(name) != self.static[name]:
                raise attention_lower.AttentionRefused(group["label"], f"{name}: lowered BF16 source layout differs")
        mask = None
        if group.get("mask") is not None:
            mask = padding_masks.mask_strides(group, self.layouts, plan["bh"], plan["t"], attention_lower._mask_strides)
            if mask is None or self.dtypes[group["mask"]] != "float32":
                raise attention_lower.AttentionRefused(group["label"], "mask is not FP32 broadcastable")
        recipe = "bf16" if current_precision() == "practical" else "exact3"
        from .bf16_sdpa import CONTRACT
        if plan.get("arithmetic_contract") == CONTRACT:
            recipe = "bf16"
        previous = getattr(self, "long_recipe", recipe)
        if previous != recipe:
            raise attention_lower.AttentionRefused(group["label"], "one region cannot mix flat-long header arithmetic recipes")
        self.long_recipe = recipe
        # One header instance serves this complete region, so all eligible
        # long groups use the explicitly bound tape schedule. Historical tape
        # artifacts without this field retain their original default v7.
        version = self.tape.get("schedule_version", 7) if self.tape else flat_long_attention.DEFAULT_VERSION
        if self.tape and version not in (3, 7, 13):
            raise attention_lower.AttentionRefused(group["label"], "unsupported attention tape schedule")
        if "tape_which" in plan:
            if recipe != plan["tape_recipe"] or version != plan["tape_version"]:
                raise attention_lower.AttentionRefused(group["label"], "attention tape arithmetic/schedule differs")
        self._include(flat_long_attention.HEADER, flat_long_attention.defines(recipe, version))
        if forward and mask is not None and "tape_which" not in plan:
            key = (group["mask"], plan["bh"], plan["t"], tuple(mask), getattr(mask, "heads", 1))
            if key in self.mask_owner:
                plan["mask_tables"] = self.mask_owner[key]
            else:
                self.mask_owner[key] = {k: v for k, v in flat_long_attention.aux_names(group).items()
                                        if k in ("mmax", "mzero", "rm")}
        if forward and pack:
            self._include(attention_pack.HEADER)
            try:
                kind, code, extra = attention_pack.emit(plan, pack, _ident, self.layouts,
                                                        flat_long_attention.aux_names(group)["kn"])
            except attention_pack.Refused as exc:
                raise attention_lower.AttentionRefused(group["label"], f"attention pack: {exc}") from exc
            # A mask-table owner's remaining prepare units (constant-mask tables only) run in the pack phase:
            # alone they would depend on the constant alone and be placed in the program's first interval.
            prepare = flat_long_attention.emit(plan, _ident, mask, which="prepare", recipe=recipe, version=version,
                                               mask_tables=plan.get("mask_tables"), keynorm_epilogue=False,
                                               keynorm_external=True)
            if prepare is not None:
                # Only constant-mask units remain (unit u + bh*nt of prepare_unit is mask unit u); the pack
                # header's compact mask_unit computes the same exact maxima and flags.
                skip = plan["bh"] * (plan["t"] // 64)
                body = prepare[1].replace(f"pflatlong::prepare_unit(A, u + {skip}LL);", "pattnpack::mask_unit(A, u);")
                if body == prepare[1]:
                    raise attention_lower.AttentionRefused(group["label"], "attention pack: unexpected prepare body")
                code = "{\n" + code + "\n}\n{\n" + body + "\n}"
                extra = dict(extra, outputs={**extra["outputs"], **prepare[2]["outputs"]},
                             reads=list(dict.fromkeys(extra["reads"] + prepare[2]["reads"])),
                             work_items=max(extra["work_items"], prepare[2]["work_items"]))
            self._phase(kind, code, extra, extra["reads"], group=group, header=attention_pack.HEADER)
        for which in (("forward",) if forward and pack else ("prepare", "forward") if forward else
                      flat_long_attention.backward_phases(version, bool(plan.get("tape_merged_backward")))):
            emitted = flat_long_attention.emit(plan, _ident, mask, which=which, recipe=recipe, version=version,
                mask_tables=plan.get("mask_tables"), keynorm_epilogue=False,
                keynorm_in_item=bool(plan.get("tape_keynorm_item")), keynorm_external=bool(pack))
            if emitted is not None:
                kind, code, extra = emitted
                self._phase(kind, code, extra, extra["reads"], group=group, header=flat_long_attention.HEADER)
        if forward or "tape_which" in plan:
            self.decisions.append(dict(group=group["label"], decision="compose_long_bf16_boundary_attention",
                recipe=recipe, version=version, sources=plan["sources"], outputs=plan["finals"],
                head_width=plan["hd"], sequence=plan["t"], removed=sorted(plan["removed"]),
                source_commit="P-OPS4-D",
                keynorm="forward items (ops.attention_rules kn_item)" if plan.get("tape_keynorm_item")
                else "operand pack phase (ops.attention_rules pack)" if plan.get("tape_pack")
                else "explicit prepare body",
                **({"operand_pack": plan["tape_pack"]} if plan.get("tape_pack") else {}),
                **({"shared_mask_tables": plan["mask_tables"]} if plan.get("mask_tables") and "tape_which" in plan else {}),
                **(dict(deterministic_dq_target_sms=flat_long_attention.FLASH_TARGET_SMS,
                        deterministic_dq_partitions=min(
                            (flat_long_attention.FLASH_TARGET_SMS + plan["bh"] - 1) // plan["bh"], plan["t"] // 128),
                        backward_arithmetic="flash-bf16-deterministic-v1") if version == 13 else {}),
                **({"arithmetic_contract": plan["arithmetic_contract"]} if "arithmetic_contract" in plan else {}),
                **({"tape_half": plan["tape_which"]} if "tape_which" in plan else {})))

    def _mixed(self, plan, forward):
        from .lower import _ident
        group = plan["group"]
        def layout(name):
            value = self.layouts.get(name, self.static[name])
            return value.shape, value.strides
        try:
            fresh = mixed_attention.refresh(plan, self.parts, layout, self.protected)
        except mixed_attention.Refused as exc:
            raise attention_lower.AttentionRefused(group["label"], f"mixed attention: {exc}") from exc
        used = ("q", "k", "v") if forward else ("q", "k", "v", "do")
        if any(fresh["strides"][f"in:{k}"] != plan["strides"][f"in:{k}"] for k in used):
            raise attention_lower.AttentionRefused(group["label"], "lowered mixed operand layouts differ")
        for key in used:
            if fresh["sources"][key] not in self.layouts or (key in fresh["biases"] and fresh["biases"][key] not in self.layouts):
                raise attention_lower.AttentionRefused(group["label"], f"{key}: operand not ready")
        kind, code, extra = mixed_attention.emit(fresh, _ident, self.dtypes.__getitem__,
            which="forward" if forward else "backward", bias_phase=True)
        defines = {"PMIX_LONG_V2": int(mixed_attention.LONG_V2), "PMIX_LONG_V3": int(mixed_attention.LONG_V3),
                   "PMIX_LONG_FWD_V2": int(mixed_attention.LONG_FWD_V2), "PMIX_FA_DELTA": int(mixed_attention.FA_DELTA)}
        self._include(mixed_attention.HEADER, "".join(f"#define {k} {v}\n" for k, v in sorted(defines.items())))
        reads = [fresh["sources"][key] for key in used] + [fresh["biases"][key] for key in used if key in fresh["biases"]]
        if not forward:
            reads.append(mixed_attention.stats_name(group))
        self._phase(kind, code, extra, reads + extra["aux_reads"], group=group, header=mixed_attention.HEADER)
        if extra.get("bias_phase"):
            bias_code, bias = extra["bias_phase"]
            self._phase("attention_mixed_bias", bias_code, bias, bias["aux_reads"], group=group, header=mixed_attention.HEADER)
        if forward:
            self.decisions.append(dict(group=group["label"], decision="compose_mixed_attention",
                sources=fresh["sources"], outputs=fresh["finals"], head_width=fresh["hd"], sequence=fresh["t"],
                removed=sorted(fresh["removed"]), source_commit="P-OPS5-D"))
