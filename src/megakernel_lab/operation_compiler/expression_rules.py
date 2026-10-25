"""Shared C-OPS4-A scalar rules, independent of the selected execution schedule.

Adapted from frozen 2f04ba284 (full donor recorded by the integration manifest).
Extra-ops handles clamp first, including its bounded/NaN contract.
"""


def standard_expression(base, target, args, kwargs, dtypes, names, rd, *, layouts=None, name=None):
    from .lower import LoweringError, _float, _ref
    if target == "latch_ops.scaled_add_f32.default":
        from .scaled_add import expression
        return expression(target, args, kwargs, name, dtypes, names, rd, layouts)
    if target == "latch_ops.mixed_fma64.default":
        from .mixed_fma import expression
        return expression(target, args, kwargs, name, dtypes, names, rd, layouts)
    if base == "masked_fill":
        if target != "masked_fill.Scalar" or dtypes[_ref(args[1], names)] != "bool":
            raise LoweringError("masked_fill admits a bool mask and a static scalar value")
        return f"({rd(1)} ? {_float(args[2])} : {rd(0)})"
    if base == "rsub":
        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        return f"({rd(1)} - ({rd(0)} * {_float(alpha)}))"
    if base in {"erf", "sin", "cos"}:
        return f"{base}f({rd(0)})"
    if base in {"minimum", "maximum"}:
        symbol = "<" if base == "minimum" else ">"
        return (f"(({rd(0)} != {rd(0)}) || ({rd(1)} != {rd(1)}) ? "
                f"__int_as_float(0x7fc00000) : ({rd(0)} {symbol} {rd(1)} ? {rd(0)} : {rd(1)}))")
    if base == "pow" and target == "pow.Tensor_Scalar" and args[1] == 0.5:
        return f"sqrtf({rd(0)})"
    return None
