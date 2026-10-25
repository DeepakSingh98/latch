"""Emit the bounded unpadded CE connection; matching lives in exact_vocab."""
import math

HEADERS = ("torch_logsoftmax.cuh", "exact_vocab.cuh")


def emit(metadata, target, args, kwargs, name, shape, layouts, dtypes, names):
    from .lower import LoweringError, _ident, _index, _ref
    from .torch_logsoftmax import KEY, CONTRACT
    from .exact_vocab import TARGETS
    if target not in TARGETS.values():
        return None
    kind = next(k for k, v in TARGETS.items() if v == target)
    if ((kind == "stats" and len(args) != 1) or (kind == "pick" and len(args) != 3)
            or (kind == "grad" and len(args) < 4)):
        raise LoweringError("exact vocabulary operation argument arity differs")
    if metadata.get(KEY) != CONTRACT:
        raise LoweringError("exact vocabulary rows require the explicit ILP4 arithmetic contract")
    ref = lambda arg: _ref(arg, names)
    product = ref(args[0])
    lp = layouts[product]
    if (len(lp.shape) != 2 or not lp.contiguous or lp.offset
            or dtypes[product] != "bfloat16"):
        raise LoweringError("exact vocabulary rows require unpadded contiguous BF16 products")
    rows, width = lp.shape
    if width <= 9216 or width % 4 == 0 or width >= 2**31 or rows <= 0:
        raise LoweringError("exact vocabulary rows require the regular unaligned ILP4 dispatch")
    if int(kwargs.get("vocab", width)) != width:
        raise LoweringError("exact vocabulary logical width mismatch")
    output_layout = layouts[name]
    from .matrix_adapters.padding import pitch
    padded_gradient = (kind == "grad" and tuple(output_layout.shape) == (rows, width)
                       and tuple(output_layout.strides) == (pitch(width), 1))
    if (not output_layout.contiguous and not padded_gradient) or output_layout.offset:
        raise LoweringError("exact vocabulary output must be contiguous or use the proved BF16 operand pitch")
    x, out = _ident(product), _ident(name)
    extra = dict(engine="exact_vocab_" + kind,
                 vocab_rows=rows, reduction_elements=width, work_items=rows, threads=256,
                 row_arithmetic=CONTRACT,
                 exact_vocab_contract=dict(contract=CONTRACT, rows=rows, width=width,
                   padding=False, padding_scope="logical vocabulary and matrix contraction",
                   output_row_pitch=output_layout.strides[0],
                   arithmetic="virtual FP32 ILP4 alignment and ordered sums; original FP32 chain/FMA and BF16 rounding"))
    if target == TARGETS["stats"]:
        if tuple(shape) != (rows, 2) or dtypes[name] != "float32":
            raise LoweringError("exact vocabulary statistics must be FP32[R,2]")
        extra["shared_bytes"] = 128
        code = f"for(long long row=blockIdx.x;row<{rows}LL;row+=gridDim.x) {{ latch_exact_vocab::stats_row({x},{out},{width},row,smem); }}"
        return "exact_vocab_stats", code, extra
    stats_name = ref(args[1]); stats = _ident(stats_name)
    if (layouts[stats_name].shape != (rows, 2) or not layouts[stats_name].contiguous
            or layouts[stats_name].offset or dtypes[stats_name] != "float32"):
        raise LoweringError("exact vocabulary requires contiguous FP32 row statistics")
    index = ref(args[2])
    if dtypes[index] != "int64" or tuple(layouts[index].shape) != (rows, 1):
        raise LoweringError("exact vocabulary row targets must be int64[R,1]")
    idx = f"{_ident(index)}[{_index(layouts[index], (rows, 1), 'row')}]"
    if target == TARGETS["pick"]:
        if tuple(shape) != (rows, 1) or dtypes[name] != "float32":
            raise LoweringError("exact vocabulary pick must be FP32[R,1]")
        extra.update(shared_bytes=0, reduction_elements=1)
        code = f"""for(long long row=(long long)blockIdx.x*blockDim.x+threadIdx.x;row<{rows}LL;row+=(long long)gridDim.x*blockDim.x) {{
  const int64_t t={idx}; assert(t>=0 && t<{width}LL);
  {out}[row]=__fsub_rn(__fsub_rn(__bfloat162float({x}[row*{width}LL+t]),{stats}[2*row]),{stats}[2*row+1]);
}}"""
        return "exact_vocab_pick", code, extra
    if tuple(shape) != (rows, width) or dtypes[name] != "bfloat16":
        raise LoweringError("exact vocabulary gradient must retain original BF16[R,V]")
    chain = kwargs.get("chain")
    if (not isinstance(chain, str) or len(args[4:]) != len(chain)
            or any(c not in "*~" for c in chain) or type(kwargs.get("leaf_add", False)) is not bool):
        raise LoweringError("exact vocabulary gradient supports only proven mul/neg sparse chains")
    leaf = args[3]
    if isinstance(leaf, (int, float)):
        if not math.isfinite(float(leaf)):
            raise LoweringError("exact vocabulary scatter scalar must be finite")
        selected = f"{float(leaf)!r}f"
    else:
        leaf_name = ref(leaf)
        if dtypes[leaf_name] != "float32" or tuple(layouts[leaf_name].shape) != (rows, 1):
            raise LoweringError("exact vocabulary scatter source must be FP32[R,1]")
        selected = f"{_ident(leaf_name)}[{_index(layouts[leaf_name], (rows, 1), 'row')}]"
    if kwargs.get("leaf_add"):
        selected = f"__fadd_rn(0.0f,{selected})"
    steps = []
    for i, (operation, factor) in enumerate(zip(chain, args[4:])):
        if operation == "~":
            if factor is not None:
                raise LoweringError("exact vocabulary negation cannot have a factor")
            steps.append("selected=-selected; other=-other;")
        else:
            f = ref(factor)
            factor_shape = tuple(layouts[f].shape)
            if (dtypes[f] != "float32" or len(factor_shape) > 2
                    or factor_shape not in ((), (1,), (1, 1), (rows, 1))):
                raise LoweringError("exact vocabulary chain factors must be FP32 row broadcasts")
            value = f"{_ident(f)}[{_index(layouts[f], (rows, width), f'(row*{width}LL)')}]"
            steps.append(f"const float factor{i}={value}; selected=__fmul_rn(selected,factor{i}); other=__fmul_rn(other,factor{i});")
    extra.update(shared_bytes=2048)
    code = f"""for(long long row=blockIdx.x;row<{rows}LL;row+=gridDim.x) {{
  const int64_t target={idx}; assert(target>=0 && target<{width}LL);
  float selected={selected},other=0.0f;
  {' '.join(steps)}
  latch_exact_vocab::grad_row({x},{stats},{out},{width},{output_layout.strides[0]},row,(int)target,selected,other,smem);
}}"""
    return "exact_vocab_grad", code, extra


# LATCH-MOE-EMA-FAST: the statistics row pass also emits the sparse-cotangent gradient of its rows
# (exact_vocab_fused.cuh) when every other gradient operand is produced before that pass. The logits are read
# once instead of twice; each statistic, cotangent leaf, element expression and rounding is unchanged.
# False keeps the separate statistics and gradient passes.
FUSED_ROWS = True
FUSED_HEADERS = ("exact_vocab_cached.cuh", "exact_vocab_fused.cuh")
# Two row buffers: the CTA's next row is copied (cp.async) while the current row is reduced and emitted.
FUSED_ROW_BUFFERS = 2


def fused_row_shared_bytes(width, buffers):
    """Launch storage of a fused row pass (exact_vocab_fused.cuh layouts)."""
    if buffers == 1:
        return max(2048, 128 + ((width + 3 + 3) // 4) * 8)
    if buffers == 2:
        return 128 + 2 * (((width + 3) * 2 + 127) // 128 * 128) + 2048
    raise ValueError("fused exact vocabulary rows use one or two row buffers")


def fuse_rows(kind, code, extra, name, args, refs, names, layouts, phases, producers):
    """Fold an emitted gradient phase into its statistics phase; returns its headers or None (not admitted)."""
    if not FUSED_ROWS or kind != "exact_vocab_grad":
        return None
    from .lower import _ref
    stats, product = _ref(args[1], names), _ref(args[0], names)
    q = producers.get(stats)
    if q is None or code.count("latch_exact_vocab::grad_row(") != 1:
        return None
    phase = phases[q]
    if (phase.get("kind") != "exact_vocab_stats" or phase.get("engine") != "exact_vocab_stats"
            or phase.get("names") != [stats] or phase.get("reads") != [product]
            or phase.get("code", "").count("latch_exact_vocab::stats_row(") != 1
            or phase.get("reduction_elements") != extra.get("reduction_elements")
            or phase.get("vocab_rows") != extra.get("vocab_rows")):
        return None
    roots = {layouts[r].root if r in layouts else r for r in refs if r != stats}
    if any(producers.get(root) is not None and producers[root] >= q for root in roots):
        return None
    if any(set(later.get("writes", ())) & roots for later in phases[q + 1:]):
        return None
    width = extra["reduction_elements"]
    rows = extra["vocab_rows"]
    from .exact_vocab import TARGETS
    if FUSED_ROW_BUFFERS == 2:
        fused = code.replace("latch_exact_vocab::grad_row(", "latch_exact_vocab::stats_grad_row_db(", 1)
        if fused.count(",smem);") != 1:
            return None
        fused = fused.replace(",smem);", f",smem,(long long)gridDim.x,{rows}LL);")
    else:
        fused = code.replace("latch_exact_vocab::grad_row(", "latch_exact_vocab::stats_grad_row(", 1)
    phase.update(
        kind="exact_vocab_stats_grad", engine="exact_vocab_stats_grad",
        names=[stats, name], writes=[stats, name], targets=[TARGETS["stats"], TARGETS["grad"]],
        reads=list(dict.fromkeys([product, *refs])),
        code=fused,
        shared_bytes=fused_row_shared_bytes(width, FUSED_ROW_BUFFERS),
        exact_vocab_gradient_contract=extra.get("exact_vocab_contract"),
        exact_vocab_fused_rows=dict(statistics=stats, gradient=name, logits_reads_per_row=1,
                                    row_buffers=FUSED_ROW_BUFFERS,
                                    storage="BF16 row in program shared storage",
                                    preserves="ILP1024 leaves and reduction tree; gradient element expression, "
                                              "word routing and stores"))
    producers[name] = q
    return FUSED_HEADERS
