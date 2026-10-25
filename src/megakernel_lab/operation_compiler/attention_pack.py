"""Attention operand pack (ops.attention_rules ``pack``, LATCH-TINYMOE-ATTN).

An explicit BF16 SDPA group lowered to the P-OPS4-D flat-long body reads padded contiguous BF16 operands
[B, H, T, 64]. The captured graph materializes each one through a chain of exact layout/conversion copies:

    source y [B, Hs, T, W] (any strides, unit channel stride; FP32 or BF16)
      -> optional FP32 -> BF16 cast            aten._to_copy
      -> optional grouped-query repetition     unsqueeze(2) -> expand -> clone -> _unsafe_view/view/reshape
      -> zero channels W..63                   cat((x, zeros), -1)

(the cast and repetition in either order). When every intermediate is read only by the next step, the chain is
replaced by one pack phase that writes the same padded values from y (cast with __float2bfloat16_rn, the
compiler's own conversion), and key units also write the body's key-norm proof table with prepare_unit's exact
arithmetic. A group that owns mask proof tables computes them in the same phase (its prepare would otherwise
hold only constant-mask units and be placed in the program's first interval). Values are bitwise identical
by construction.
"""
from __future__ import annotations

HEADER = "attention_pack.cuh"
KEYS = ("q", "k", "v")


class Refused(Exception):
    pass


def _target(op):
    return str(op.target)


def _single_reader(name, reader, consumers, protected):
    if name in protected or consumers.get(name, set()) != {reader}:
        raise Refused(f"{name} is read outside the operand chain")


def trace(ir, start, producers, consumers, protected):
    """Return the chain spec of one padded operand ``start`` (a cat output), or raise Refused."""
    values = ir.values
    cat = producers.get(start)
    if cat is None or _target(cat) != "aten.cat.default" or tuple(cat.outputs) != (start,):
        raise Refused(f"{start}: operand is not a zero-channel concatenation")
    tensors = list(cat.args[0]) if cat.args else []
    dim = int(cat.args[1]) if len(cat.args) > 1 else 0
    shape = tuple(values[start].shape)
    if len(tensors) != 2 or len(shape) != 4 or dim not in (-1, 3) or cat.kwargs:
        raise Refused(f"{start}: concatenation is not (x, zeros) along channels")
    x, zeros = tensors[0].name, tensors[1].name
    zop = producers.get(zeros)
    if (zop is None or _target(zop) != "aten.zeros.default" or values[zeros].dtype != "bfloat16"
            or values[x].dtype != "bfloat16" or tuple(values[x].shape[:3]) != shape[:3]
            or tuple(values[zeros].shape) != shape[:3] + (shape[3] - values[x].shape[3],)):
        raise Refused(f"{start}: padding operand is not BF16 zeros")
    for name in (x, zeros):
        _single_reader(name, cat.name, consumers, protected)
    removed, casts, group = [start, zeros], 0, 1
    current, reader = x, cat.name
    views = ("aten._unsafe_view.default", "aten.view.default", "aten.reshape.default")
    while True:
        op = producers.get(current)
        target = _target(op) if op is not None else ""
        if target == "aten._to_copy.default":
            source = op.inputs[0]
            if (casts or values[current].dtype != "bfloat16" or values[source].dtype != "float32"
                    or set(op.kwargs) - {"dtype", "layout", "device", "pin_memory"}
                    or str(op.kwargs.get("dtype", "bfloat16")).removeprefix("torch.") != "bfloat16"
                    or tuple(values[source].shape) != tuple(values[current].shape)):
                raise Refused(f"{current}: conversion is not one FP32 -> BF16 cast")
            casts += 1
            following = source, op.name
        elif target in views:
            clone = producers.get(op.inputs[0])
            expand = producers.get(clone.inputs[0]) if clone is not None and _target(clone) == "aten.clone.default" else None
            unsq = producers.get(expand.inputs[0]) if expand is not None and _target(expand) == "aten.expand.default" else None
            if (group != 1 or unsq is None or _target(unsq) != "aten.unsqueeze.default"
                    or len(unsq.args) < 2 or int(unsq.args[1]) != 2):
                raise Refused(f"{current}: layout copy is not a grouped-query head repetition")
            source = unsq.inputs[0]
            b, hs, t, w = (int(n) for n in values[source].shape)
            repeated = tuple(int(n) for n in values[expand.outputs[0]].shape)
            if (len(repeated) != 5 or repeated[:2] != (b, hs) or repeated[3:] != (t, w)
                    or tuple(values[current].shape) != (b, hs * repeated[2], t, w)
                    or tuple(values[clone.outputs[0]].shape) != repeated
                    or values[source].dtype != values[current].dtype):
                raise Refused(f"{current}: repetition shapes differ")
            for name, by in ((clone.outputs[0], op.name), (expand.outputs[0], clone.name),
                             (unsq.outputs[0], expand.name)):
                _single_reader(name, by, consumers, protected)
            group = repeated[2]
            removed += [clone.outputs[0], expand.outputs[0], unsq.outputs[0]]
            following = source, unsq.name
        else:
            break
        # ``current``'s operation is removed: only the next chain step may read it.
        _single_reader(current, reader, consumers, protected)
        removed.append(current)
        current, reader = following
    value = values[current]
    if value.dtype not in ("float32", "bfloat16") or (value.dtype == "float32") != (casts == 1):
        raise Refused(f"{current}: source dtype does not match the chain's conversions")
    b, hs, t, w = (int(n) for n in value.shape)
    if (b, hs * group, t) != shape[:3] or not 1 <= w <= 64 or shape[3] != 64:
        raise Refused(f"{current}: source shape does not match the padded operand")
    rope = _rope(ir, current, producers, consumers, protected) if casts == 1 else None
    if rope is not None:
        # The FP32 rotary product is computed in the pack phase from its BF16 input and the shared tables.
        _single_reader(current, reader, consumers, protected)
        removed += [current, *rope.pop("removed")]
        current = rope.pop("x")
    # The removed list names values whose producing operations disappear (x and every intermediate);
    # the padded operand itself is re-produced by the pack phase.
    return dict(src=current, fp32=casts == 1 and rope is None, group=group, width=w, rope=rope,
                removed=list(dict.fromkeys(removed)))


def _rope(ir, y, producers, consumers, protected):
    """Half-rotation rotary embedding y = x * cos + cat(-x[..., W/2:], x[..., :W/2]) * sin, or None.

    x is a BF16 [B, H, T, W] value; cos and sin are FP32 tables broadcast over (batch, head). The captured
    elementwise arithmetic is fl(fl(x * cos) + fl(r * sin) * 1) with r the BF16 rotation (negation is exact).
    """
    values = ir.values
    add = producers.get(y)
    if add is None or _target(add) != "aten.add.Tensor" or len(add.args) != 2 or (
            add.kwargs and add.kwargs.get("alpha", 1) != 1):
        return None
    left, right = (producers.get(getattr(item, "name", None)) for item in add.args)
    if left is None or right is None or _target(left) != "aten.mul.Tensor" or _target(right) != "aten.mul.Tensor":
        return None
    if left.kwargs or right.kwargs or len(left.args) != 2 or len(right.args) != 2:
        return None
    x, cos = (getattr(item, "name", None) for item in left.args)
    rotated, sin = (getattr(item, "name", None) for item in right.args)
    cat = producers.get(rotated)
    if None in (x, cos, rotated, sin) or cat is None or _target(cat) != "aten.cat.default":
        return None
    shape = tuple(int(n) for n in values[x].shape)
    if values[x].dtype != "bfloat16" or len(shape) != 4 or shape[3] % 16:
        return None
    half = shape[3] // 2
    parts = list(cat.args[0]) if cat.args else []
    if len(parts) != 2 or (int(cat.args[1]) if len(cat.args) > 1 else 0) not in (-1, 3):
        return None
    neg = producers.get(parts[0].name)
    first = producers.get(neg.inputs[0]) if neg is not None and _target(neg) == "aten.neg.default" else None
    second = producers.get(parts[1].name)

    def is_slice(op, start, end):
        return (op is not None and _target(op) == "aten.slice.Tensor" and tuple(op.inputs) == (x,)
                and int(op.args[1]) % 4 == 3 and int(op.args[2]) == start
                and min(int(op.args[3]) if len(op.args) > 3 and op.args[3] is not None else shape[3], shape[3]) == end
                and (len(op.args) < 5 or int(op.args[4]) == 1))
    if not (is_slice(first, half, shape[3]) and is_slice(second, 0, half)):
        return None
    for table in (cos, sin):
        v = values[table]
        if v.dtype != "float32" or tuple(int(n) for n in v.shape) not in ((1, 1, shape[2], shape[3]), (shape[2], shape[3])):
            return None
    if any(values[n].dtype != "float32" for n in (left.outputs[0], right.outputs[0], y)) or values[rotated].dtype != "bfloat16":
        return None
    try:
        for name, by in ((left.outputs[0], add.name), (right.outputs[0], add.name), (rotated, right.name),
                         (neg.outputs[0], cat.name), (first.outputs[0], neg.name), (second.outputs[0], cat.name)):
            _single_reader(name, by, consumers, protected)
    except Refused:
        return None
    return dict(x=x, cos=cos, sin=sin, half=half,
                removed=[left.outputs[0], right.outputs[0], rotated, neg.outputs[0], first.outputs[0], second.outputs[0]])


def plan_group(ir, plan, producers, consumers, protected):
    """Pack spec {q, k, v} for one explicit SDPA plan (sources already clone-elided), or raise Refused."""
    spec = {key: trace(ir, plan["sources"][key], producers, consumers, protected) for key in KEYS}
    if len({plan["sources"][key] for key in KEYS}) != 3:
        raise Refused("operands share a padded buffer")
    if len({spec[key]["src"] for key in KEYS}) != 3:
        raise Refused("operands share a source")
    return spec


def emit(plan, spec, ident, layouts, kn_name):
    """(kind, code, extra) of the pack phase, from the lowered source layouts."""
    lines = ["pattnpack::Args P{};"]
    reads = []
    t, bh, heads = plan["t"], plan["bh"], plan["heads"]
    for index, key in enumerate(KEYS):
        item = spec[key]
        layout = layouts.get(item["src"])
        if layout is None or len(layout.shape) != 4 or int(layout.strides[3]) != 1 and int(layout.shape[3]) != 1:
            raise Refused(f"{item['src']}: lowered source layout is not unit-channel 4-D")
        if tuple(int(n) for n in layout.shape) != (bh // heads, heads // item["group"], t, item["width"]):
            raise Refused(f"{item['src']}: lowered source shape differs")
        sb, sh, sr = (int(s) for s in layout.strides[:3])
        chunk = 4 if item["fp32"] else 8  # elements per 16-byte vector
        if item["width"] % 8 or not all(x % chunk == 0 for x in (sb, sh, sr, int(layout.offset))):
            raise Refused(f"{item['src']}: source rows are not 16-byte chunk aligned")
        offset = f" + {int(layout.offset)}LL" if layout.offset else ""
        rope_fields = "nullptr, nullptr, 0LL, 0"
        rope = item.get("rope")
        if rope:
            tables = []
            for name in (rope["cos"], rope["sin"]):
                table = layouts.get(name)
                if table is None or tuple(int(n) for n in table.shape[-2:]) != (t, item["width"]):
                    raise Refused(f"{name}: lowered rotary table layout differs")
                strides = tuple(int(s) for s in table.strides)
                if strides[-1] != 1 or strides[-2] % 4 or int(table.offset) % 4:
                    raise Refused(f"{name}: rotary table rows are not 16-byte aligned")
                tables.append((table, strides[-2]))
                reads += [name, table.root]
            if tables[0][1] != tables[1][1] or rope["half"] % 8:
                raise Refused("rotary tables differ in layout")
            pointer = lambda table: ident(table.root) + (f" + {int(table.offset)}LL" if table.offset else "")
            rope_fields = (f"(const float*)({pointer(tables[0][0])}), (const float*)({pointer(tables[1][0])}), "
                           f"{tables[0][1]}LL, {rope['half']}")
        lines.append(f"P.src[{index}] = pattnpack::Src{{(const void*)({ident(layout.root)}{offset}), {sb}LL, {sh}LL, "
                     f"{sr}LL, {int(item['fp32'])}, {item['group']}, 1, {item['width']}, {rope_fields}}};")
        lines.append(f"P.out[{index}] = {ident(plan['sources'][key])};")
        reads += [item["src"], layout.root]
    nt = t // 64
    units = bh * nt
    lines.append(f"P.kn = {ident(kn_name)}; P.heads = {heads}; P.bh = {bh}; P.t = {t};")
    lines.append("// Warps 0 and 4 of each 256-thread CTA write the key-norm table (one warp per (head batch, 64-key tile)\n"
                 "// unit); the other six warps copy the padded BF16 operands concurrently, one 16-byte chunk each.\n"
                 "{\n"
                 "const unsigned warp_ = threadIdx.x >> 5, lane_ = threadIdx.x & 31;\n"
                 "if ((warp_ & 3u) == 0) {\n"
                 f"  for (unsigned u = blockIdx.x * 2u + (warp_ >> 2); u < {units}u; u += gridDim.x * 2u)\n"
                 f"    pattnpack::key_norm_unit<{t}, {heads}>(P, u);\n"
                 "} else {\n"
                 "  const unsigned slot_ = warp_ - 1u - (warp_ >> 2);  // 0..5\n"
                 f"  pattnpack::pack_copy<{bh}, {t}, {heads}>(P, (blockIdx.x * 6u + slot_) * 32u + lane_, gridDim.x * 192u);\n"
                 "}\n"
                 "}")
    if 3 * bh * t * 8 >= 2 ** 31:
        raise Refused("operands exceed the pack's 32-bit chunk index")
    outputs = {plan["sources"][key]: None for key in KEYS}
    outputs[kn_name] = ((bh, nt), "float32")
    items = (3 * bh * t * 8 + 255) // 256
    return "attention_pack", "\n".join(lines), dict(work_items=items, shared_bytes=0, head_width=64,
                                                    sequence=t, head_batch=bh, outputs=outputs,
                                                    reads=list(dict.fromkeys(reads)),
                                                    pack={key: {k: v for k, v in spec[key].items() if k != "removed"}
                                                          for key in KEYS})
