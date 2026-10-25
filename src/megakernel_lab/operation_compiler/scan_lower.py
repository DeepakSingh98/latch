"""Lowering rules for the selective-scan primitive (ops/scan.py), C-OPS2-C.

One captured ``latch_ops.selective_scan_fwd`` or ``..._bwd`` operation becomes one
heavy schedule item. Its results (and private scratch: chunk carries, dB/dC and
vector-gradient partials) are packed into one FP32-word allocation; each result
is a typed view of it, so the central scheduler keeps authority over phase
placement, storage lifetimes and arena reuse. The item contains its own
grid-wide barriers between sub-passes (carry -> main [-> reduce]); every CTA
executes the item unconditionally, so it is valid in flat cooperative programs.
Staged/partitioned regions refuse it.

Engine selection uses operand dtype, layout, alignment, state count and the
declared target only, never model names:

* ``p1a``: the accepted P1-A selective-scan body (vendor/scan_p1a.cuh), BF16
  operands with unit inner stride and 16-byte aligned rows, N in {8, 16, 32},
  channels a multiple of 32, steps a multiple of 8, softplus step.
* ``generic``: vendor/scan_generic.cuh for every other admitted case and for the
  ``reference`` compiler policy (any FP32/BF16 strides, any N <= 64).
"""
from __future__ import annotations

import math

from .scan import BWD_TARGET, CHECKPOINT_INTERVAL, FWD_TARGET, TARGETS

P1A_HEADER = "scan_p1a.cuh"
P1A_COPS2C_HEADER = "scan_p1a_cops2c.cuh"   # diagnostic: the unmodified C-OPS2-C body (scan option body="cops2c")
GENERIC_HEADER = "scan_generic.cuh"
P1A_SOURCE = {"lane": "P1-A", "tag": "p1a-final", "commit": "96f20b9f4",
              "files": {"scan2.cuh": "23e8e5ce52ce3d21bda91ac54c8edad3641740b5fb6daef72674507ac71cb43c",
                        "scan.cuh": "2470be09c84e579b1756913d8be4501a150b469ae2cc206cdbbd55fca9d5fb91",
                        "common.cuh": "10b4ac62d8a1fc5352dee9ab25308dafd9fbdf4669182e07be7500420c7eeb2a"},
              "options": ["P1A_NOSG", "S109D_FLAGS=0", "S109D_CTRIM=2", "S109D_BPRO=0"]}
SHARED_LIMIT = 200704          # donor dynamic-shared budget (196 KiB carveout); also <= the 227 KiB opt-in limit
MAX_WARPS = 8                  # 256-thread LATCH-OPS CTA
GRID_BOUND = 264               # largest admitted cooperative grid (two CTAs on 132 SMs); bounds scratch only
ENGINES = ("auto", "p1a", "generic")
# P-OPS4-G P1-A body options (scan execution only; arithmetic order documented in the header):
#   contract: explicit single-rounding multiply-adds at the body's a*b+c sites (LSP_CONTRACT). Default on: P1-A was
#     accepted as compiled with nvcc's default contraction, which the program-wide --fmad=false had removed;
#   backward_group: states per backward-main group (LSP_BG; 2 = P1-A; 4/8 are bitwise-equal ILP variants);
#   body: "cops2c" selects the unmodified C-OPS2-C header (diagnostic baseline only).
SCAN_OPTIONS = {"contract": (False, True), "backward_group": (2, 4, 8), "body": ("current", "cops2c"),
                "backward_chunks": (1, 2, 3, 4),   # P6-E: floor on backward time chunks (LSP_NT_MIN_BWD)
                "forward_unroll": (2, 4, 8, 16),   # P6-E: forward step-loop unroll depth (LSP_FUNROLL; same arithmetic)
                "forward_buffers": (2, 3, 4),      # P6-E: forward staging ring depth (LSP_FNB; same copies, issued earlier)
                "backward_buffers": (2, 3, 4)}     # P6-E: backward-main staging ring depth (LSP_BNB)
DEFAULT_SCAN_OPTIONS = {"contract": True}


def option_defines(options):
    options = dict(options or {})
    if options.get("body") != "cops2c":
        options = {**DEFAULT_SCAN_OPTIONS, **options}
    unknown = set(options) - set(SCAN_OPTIONS)
    if unknown:
        raise ScanRefused(f"unknown scan options {sorted(unknown)}")
    for key, value in options.items():
        if value not in SCAN_OPTIONS[key]:
            raise ScanRefused(f"scan option {key}={value!r} not in {SCAN_OPTIONS[key]}")
    defines = []
    if options.get("body") == "cops2c" and (options.get("contract") or options.get("backward_group", 2) != 2):
        raise ScanRefused("the diagnostic C-OPS2-C body takes no other scan options")
    if options.get("contract"):
        defines.append("LSP_CONTRACT")
    if options.get("backward_group", 2) != 2:
        defines.append(f"LSP_BG {options['backward_group']}")
    if options.get("forward_unroll", 4) != 4:
        defines.append(f"LSP_FUNROLL {options['forward_unroll']}")
    if options.get("backward_chunks", 1) != 1:
        defines.append(f"LSP_NT_MIN_BWD {options['backward_chunks']}")
    if options.get("forward_buffers", 2) != 2:
        defines.append(f"LSP_FNB {options['forward_buffers']}")
    if options.get("backward_buffers", 2) != 2:
        defines.append(f"LSP_BNB {options['backward_buffers']}")
    return defines


class ScanRefused(ValueError):
    pass


def _a16(x):
    return -(-x // 16) * 16


def _a64w(words):
    return -(-words // 64) * 64


def staging_bytes(N, fnb=2, bnb=2):
    """Per-warp shared bytes of scan_p1a.cuh sub-passes (donor model.scan_warps footprints; fnb forward buffers)."""
    fwd = _a16(fnb * _a16(16 * (3 * 64 + 4 * N) + 16) + 16 * 2 * N * 4)
    bwd_carry = _a16(2 * _a16(8 * (4 * 64 + 4 * N) + 8) + 8 * 2 * N * 4)
    bwd_main = _a16(bnb * _a16(N * 128 + 8 * (4 * 64 + 4 * N) + 8) + (8 * 2 * N + (96 if N <= 16 else 64) * N) * 4)
    return fwd, bwd_main, bwd_carry


def _warps(need):
    return max(0, min(MAX_WARPS, SHARED_LIMIT // need))


def ntmax(B, T, E, warps):
    lanes32 = B * (E // 32)
    return max(1, min(max(1, T // 16), max(1, (warps * GRID_BOUND) // lanes32)))


def _layout(layouts, ref):
    return layouts[ref]


def _operand_names(op, names, ref, field):
    args = list(field(op, "args", []))
    target = str(field(op, "target", field(op, "op")))
    keys = ["u", "delta", "A", "B", "C", "D", "z", "delta_bias", "reset"]
    if target == BWD_TARGET:
        keys += ["dout", "ckpt"]
    if len(args) != len(keys) + 1 or field(op, "kwargs", {}):
        raise ScanRefused(f"{target}: unexpected argument encoding")
    operands = {}
    for key, arg in zip(keys, args):
        name = ref(arg, names)
        if arg is None and key == "reset":
            operands[key] = None
            continue
        if name is None:
            raise ScanRefused(f"{target}: {key} must be a tensor value")
        operands[key] = name
    softplus = args[-1]
    if type(softplus) is not bool:
        raise ScanRefused(f"{target}: delta_softplus must be a static bool")
    return operands, softplus


def _check_shapes(operands, layouts, dtypes):
    u = layouts[operands["u"]]
    if len(u.shape) != 3:
        raise ScanRefused("u must be rank 3")
    Bt, T, E = u.shape
    A = layouts[operands["A"]]
    if len(A.shape) != 2 or A.shape[0] != E:
        raise ScanRefused("A must be [channels, states]")
    N = A.shape[1]
    expect = {"delta": (Bt, T, E), "z": (Bt, T, E), "B": (Bt, T, N), "C": (Bt, T, N), "D": (E,),
              "delta_bias": (E,), "dout": (Bt, T, E), "ckpt": (Bt, -(-T // CHECKPOINT_INTERVAL), N, E)}
    for key, shape in expect.items():
        if key in operands and tuple(layouts[operands[key]].shape) != shape:
            raise ScanRefused(f"{key} must have shape {shape}")
    dtype = dtypes[operands["u"]]
    if dtype not in ("float32", "bfloat16"):
        raise ScanRefused("u/delta/z/B/C must be FP32 or BF16")
    for key in ("delta", "z", "B", "C", "dout"):
        if key in operands and dtypes[operands[key]] != dtype:
            raise ScanRefused(f"{key} dtype differs from u")
    for key in ("A", "D", "delta_bias", "ckpt"):
        if key in operands and dtypes[operands[key]] != "float32":
            raise ScanRefused(f"{key} must be FP32")
    if operands.get("reset") is not None:
        r = layouts[operands["reset"]]
        if dtypes[operands["reset"]] != "bool" or tuple(r.shape) != (Bt, T):
            raise ScanRefused("reset must be bool [batch, time]")
    if N > 64:
        raise ScanRefused("more than 64 states per channel are not admitted")
    return Bt, T, E, N, dtype


def p1a_eligibility(operands, layouts, dtypes, softplus, dims, *, backward):
    """Reasons the P1-A body cannot run this operation (empty list = eligible)."""
    Bt, T, E, N, dtype = dims
    reasons = []
    if dtype != "bfloat16":
        reasons.append("operands are not BF16")
    if not softplus:
        reasons.append("P1-A body applies a softplus step")
    if N not in (8, 16, 32):
        reasons.append(f"N={N} (P1-A staging needs 8, 16 or 32 states)")
    if E % 32:
        reasons.append("channels not a multiple of 32")
    if T % CHECKPOINT_INTERVAL:
        reasons.append("steps not a multiple of 8")
    rows = ["u", "delta", "z", "B", "C"] + (["dout"] if backward else [])
    for key in rows:
        lay = layouts[operands[key]]
        sb, st, si = lay.strides
        if si != 1:
            reasons.append(f"{key}: inner stride is not 1")
        if sb % 8 or st % 8 or lay.offset % 8:
            reasons.append(f"{key}: rows/offset are not 16-byte aligned")
    A = layouts[operands["A"]]
    if tuple(A.strides) != (N, 1):
        reasons.append("A is not contiguous [E][N]")
    for key in ("D", "delta_bias"):
        if tuple(layouts[operands[key]].strides) != (1,):
            reasons.append(f"{key} is not contiguous")
    if backward:
        ck = layouts[operands["ckpt"]]
        if not ck.contiguous or ck.offset % 4:
            reasons.append("checkpoint is not a contiguous 16-byte aligned tensor")
    if operands.get("reset") is not None:
        r = layouts[operands["reset"]]
        if tuple(r.strides) != (T, 1) or r.offset % 8:
            reasons.append("reset is not contiguous [B][T] with 8-byte aligned rows")
    if Bt * T % 4:
        reasons.append("batch x steps not a multiple of 4")
    fwd, bm, bc = staging_bytes(N) if N in (8, 16, 32) else (10**9,) * 3
    if min(_warps(fwd), _warps(bm), _warps(bc)) < 1:
        reasons.append("per-warp staging exceeds the shared-memory budget")
    return reasons


class Emission:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _strides(layouts, name, rank):
    lay = layouts[name]
    if len(lay.strides) != rank or any(s < 0 for s in lay.strides):
        raise ScanRefused(f"{name}: unsupported strides {lay.strides}")
    return lay.strides


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, engine="auto", reserved=(), options=None,
         placement=None):
    """Plan one scan operation. Returns an Emission with root, result views, code, reads, headers, metadata."""
    target = str(field(op, "target", field(op, "op")))
    if target not in TARGETS:
        raise ScanRefused(f"not a scan operation: {target}")
    if engine not in ENGINES:
        raise ScanRefused(f"unknown scan engine {engine}")
    backward = target == BWD_TARGET
    operands, softplus = _operand_names(op, names, ref, field)
    dims = _check_shapes(operands, layouts, dtypes)
    Bt, T, E, N, dtype = dims
    M = Bt * T
    results = list(field(op, "outputs"))
    if len(results) != (8 if backward else 2):
        raise ScanRefused(f"{target}: unexpected result count")
    reasons = p1a_eligibility(operands, layouts, dtypes, softplus, dims, backward=backward)
    defines = option_defines(options)
    if dims[3] % max(2, dict(options or {}).get("backward_group", 2)):
        reasons.append("backward state group does not divide N")
    if engine == "p1a" and reasons:
        raise ScanRefused("P1-A scan body refused: " + "; ".join(reasons))
    selected = "p1a" if engine in ("auto", "p1a") and not reasons else "generic"
    op_name = str(field(op, "name"))
    root = f"{op_name}_scan_pack"
    if root in names or root in layouts or root in reserved:
        raise ScanRefused(f"packed result name collision: {root}")
    # ---- result and scratch segments (FP32 words, 64-word aligned)
    width = {"float32": 4, "bfloat16": 2}
    if backward:
        shapes = [(Bt, T, E), (Bt, T, E), (E, N), (Bt, T, N), (Bt, T, N), (E,), (Bt, T, E), (E,)]
        rdtypes = [dtype, dtype, "float32", dtype, dtype, "float32", dtype, "float32"]
    else:
        shapes = [(Bt, T, E), (Bt, -(-T // CHECKPOINT_INTERVAL), N, E)]
        rdtypes = [dtype, "float32"]
    words, outputs = 0, []
    # P6-E placement: the BF16 gradients du | dz | ddelta | dB | dC of a backward scan whose only use is a disjoint
    # full-width column assembly [B, T, W] are written in place into that assembly (first segment of the pack, row
    # stride W), which then needs no separate pass.
    placed, assembly = {}, None
    if (placement and backward and selected == "p1a" and dtype == "bfloat16"
            and dict(options or {}).get("body") != "cops2c"):
        W = int(placement["width"])
        assembly = (placement["name"], (Bt, T, W))
        for index, col in placement["cols"].items():
            placed[results[int(index)]] = int(col)
        words = _a64w(-(-M * W * 2 // 4))
    for name, shape, rdtype in zip(results, shapes, rdtypes):
        if dtypes.get(name, rdtype) != rdtype or tuple(field(values[name], "shape")) != shape:
            raise ScanRefused(f"{name}: result metadata differs from the scan contract")
        if name in placed:
            outputs.append((name, shape, rdtype, placed[name]))
            continue
        count = math.prod(shape)
        outputs.append((name, shape, rdtype, words * 4 // width[rdtype]))
        words += _a64w(-(-count * width[rdtype] // 4))
    scratch = {}
    fwd_b, bm_b, bc_b = (staging_bytes(N, dict(options or {}).get("forward_buffers", 2),
                                       dict(options or {}).get("backward_buffers", 2)) if selected == "p1a"
                         else (0, 0, 0))
    wf, wbm, wbc = (_warps(fwd_b), _warps(bm_b), _warps(bc_b)) if selected == "p1a" else (0, 0, 0)
    nt_bound = ntmax(Bt, T, E, max(wf, wbm)) if selected == "p1a" else 1

    def reserve(key, count):
        nonlocal words
        scratch[key] = words
        words += _a64w(count)
    if selected == "p1a":
        reserve("carry", 3 * Bt * nt_bound * (N + 1) * E)
    if backward:
        EG = -(-E // 32)
        reserve("krp", EG * 2 * N * M)
        reserve("vp", Bt * nt_bound * (E * N + 2 * E))
    # ---- code
    v = {key: ident(name) for key, name in operands.items() if name is not None}
    r = [ident(name) for name in results]
    pack = f"reinterpret_cast<float*>({ident(root)})"
    reads = [name for name in operands.values() if name is not None]
    if selected == "p1a":
        f = "latch_scan_p1a"
        lines = [f"{f}::Args sa{{}};"]
        for key, pointer, prefix in (("u", "u", "u"), ("z", "z", "z"), ("delta", "dl", "dl"),
                                     ("B", "Bm", "B"), ("C", "Cm", "C")):
            sb, st, _ = layouts[operands[key]].strides
            lines.append(f"sa.{pointer} = {v[key]}; sa.{prefix}_b = {sb}LL; sa.{prefix}_t = {st}LL;")
        if backward:
            sb, st, _ = layouts[operands["dout"]].strides
            lines.append(f"sa.dout = {v['dout']}; sa.do_b = {sb}LL; sa.do_t = {st}LL;")
        else:
            lines.append("sa.dout = sa.u; sa.do_b = 0; sa.do_t = 0;")
        lines.append("sa.rst = " + (f"reinterpret_cast<const unsigned char*>({v['reset']});" if operands.get("reset") else "nullptr;"))
        lines.append(f"sa.A = {v['A']}; sa.D = {v['D']}; sa.bias = {v['delta_bias']};")
        if backward:
            lines.append(f"sa.ckpt = {v['ckpt']}; sa.du = {r[0]}; sa.ddl = {r[1]}; sa.dA = {r[2]}; sa.dB = {r[3]}; "
                         f"sa.dC = {r[4]}; sa.dD = {r[5]}; sa.dz = {r[6]}; sa.dbias = {r[7]};")
            lines.append(f"sa.krp = {pack} + {scratch['krp']}LL; sa.vp = {pack} + {scratch['vp']}LL;")
        else:
            lines.append(f"sa.out = {r[0]}; sa.ckpt = {r[1]};")
        lines.append(f"sa.carry = {pack} + {scratch['carry']}LL;")
        lines.append(f"sa.B = {Bt}; sa.T = {T}; sa.E = {E}; sa.M = {M}; sa.NCK = {-(-T // 8)}; sa.ntmax = {nt_bound};")
        if dict(options or {}).get("body") != "cops2c":
            ot, bt = (assembly[1][2], assembly[1][2]) if assembly else (E, N)
            lines.append(f"sa.o_t = {ot}; sa.bc_t = {bt}; sa.canon0 = {int(bool(assembly))};")
        lines.append(f"sa.wf = {wf}; sa.bytes_f = {fwd_b}; sa.wbm = {wbm}; sa.bytes_bm = {bm_b}; sa.wbc = {wbc}; sa.bytes_bc = {bc_b};")
        sm = "reinterpret_cast<unsigned char*>(smem)"
        if backward:
            lines += [f"{f}::backward<{N}, true>(sa, {sm});", "grid.sync();",
                      f"{f}::backward<{N}, false>(sa, {sm});", "grid.sync();", f"{f}::reduce<{N}>(sa, {sm});"]
            shared = max(wbm * bm_b, wbc * bc_b, MAX_WARPS * 32 * 4)
            barriers = 2
        else:
            lines += [f"{f}::forward<{N}, true>(sa, {sm});", "grid.sync();", f"{f}::forward<{N}, false>(sa, {sm});"]
            shared = wf * fwd_b
            barriers = 1
        header = P1A_COPS2C_HEADER if dict(options or {}).get("body") == "cops2c" else P1A_HEADER
        work = Bt * (E // 32) * nt_bound
    else:
        f = "latch_scan_generic"
        ctype = "__nv_bfloat16" if dtype == "bfloat16" else "float"
        lines = [f"{f}::Args<{ctype}> sa{{}};"]
        for key, prefix, field_name in (("u", "u", "u"), ("delta", "dl", "dl"), ("z", "z", "z"),
                                        ("B", "B", "Bm"), ("C", "C", "Cm")):
            sb, st, si = _strides(layouts, operands[key], 3)
            inner = "e" if key in ("u", "delta", "z") else "n"
            lines.append(f"sa.{field_name} = {v[key]}; sa.{prefix}_b = {sb}LL; sa.{prefix}_t = {st}LL; sa.{prefix}_{inner} = {si}LL;")
        if backward:
            sb, st, si = _strides(layouts, operands["dout"], 3)
            lines.append(f"sa.dout = {v['dout']}; sa.do_b = {sb}LL; sa.do_t = {st}LL; sa.do_e = {si}LL;")
        ae, an = _strides(layouts, operands["A"], 2)
        (de,), (be,) = _strides(layouts, operands["D"], 1), _strides(layouts, operands["delta_bias"], 1)
        lines.append(f"sa.A = {v['A']}; sa.A_e = {ae}LL; sa.A_n = {an}LL; sa.D = {v['D']}; sa.D_e = {de}LL; "
                     f"sa.bias = {v['delta_bias']}; sa.bias_e = {be}LL;")
        if operands.get("reset") is not None:
            rb, rt = _strides(layouts, operands["reset"], 2)
            lines.append(f"sa.rst = {v['reset']}; sa.r_b = {rb}LL; sa.r_t = {rt}LL;")
        else:
            lines.append("sa.rst = nullptr;")
        lines.append(f"sa.softplus = {int(softplus)}; sa.B = {Bt}; sa.T_ = {T}; sa.E = {E}; sa.M = {M}; sa.NCK = {-(-T // 8)};")
        if backward:
            lines.append(f"sa.ckpt = {v['ckpt']}; sa.du = {r[0]}; sa.ddl = {r[1]}; sa.dA = {r[2]}; sa.dB = {r[3]}; "
                         f"sa.dC = {r[4]}; sa.dD = {r[5]}; sa.dz = {r[6]}; sa.dbias = {r[7]};")
            lines.append(f"sa.krp = {pack} + {scratch['krp']}LL; sa.vp = {pack} + {scratch['vp']}LL;")
            lines += [f"{f}::backward<{ctype}, {N}>(sa);", "grid.sync();", f"{f}::reduce<{ctype}, {N}>(sa);"]
            barriers = 1
        else:
            lines.append(f"sa.out = {r[0]}; sa.ckpt = {r[1]};")
            lines.append(f"{f}::forward<{ctype}, {N}>(sa);")
            barriers = 0
        shared = 0
        header = GENERIC_HEADER
        work = Bt * (-(-E // 32))
    code = "{\n" + "\n".join(lines) + "\n}"
    kind = "selective_scan_backward" if backward else "selective_scan_forward"
    metadata = {
        "engine": f"scan_{selected}", "scan_engine": selected, "internal_grid_barriers": barriers,
        "work_items": max(1, -(-work // MAX_WARPS)), "exclusive_phase": True,
        "scan_dims": {"batch": Bt, "steps": T, "channels": E, "states": N, "dtype": dtype,
                      "reset": operands.get("reset") is not None, "delta_softplus": softplus},
        "packed_results": {name: {"word_offset": off * width[dt] // 4, "shape": list(shape), "dtype": dt}
                           for name, shape, dt, off in outputs},
        "scratch_words": scratch, "packed_words": words,
    }
    if shared:
        metadata["shared_bytes"] = shared
    if selected == "p1a":
        metadata.update(source=P1A_SOURCE, warps={"forward": wf, "backward_main": wbm, "backward_carry": wbc},
                        staging_bytes={"forward": fwd_b, "backward_main": bm_b, "backward_carry": bc_b},
                        ntmax=nt_bound)
    defines = defines if selected == "p1a" else []
    include = "".join(f"#define {d}\n" for d in defines) + f'#include "{header}"'
    if selected == "p1a":
        metadata["scan_options"] = dict(options or {}) if dict(options or {}).get("body") == "cops2c" else \
            {**DEFAULT_SCAN_OPTIONS, **dict(options or {})}
        metadata["scan_defines"] = list(defines)
    decision = {"operation": op_name, "target": target, "selected": selected,
                "p1a_refusals": reasons, "requested_engine": engine,
                **({"scan_options": metadata["scan_options"], "scan_defines": metadata["scan_defines"]}
                   if selected == "p1a" else {})}
    if assembly:
        decision["p6e_placement"] = {"assembly": assembly[0], "width": assembly[1][2],
                                     "columns": {name: col for name, col in placed.items()}}
    return Emission(placed={name: (Bt * T * assembly[1][2] // Bt, assembly[1][2], 1) for name in placed} if assembly else {},
                    assembly=assembly, root=root, root_words=words, outputs=outputs, code=code, reads=reads, header=header, include=include,
                    kind=kind, metadata=metadata, decision=decision, target=target)
