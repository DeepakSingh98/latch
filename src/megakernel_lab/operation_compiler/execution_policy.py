"""Frozen execution selections, chosen by lower() from operation contracts.

P7-C FINAL1 plus selected matrix/rounding rules; P7-D adds sink only for its
explicit storage-rematerialization policy. P8-B adds colreg (coalesced exact column reductions). No workload identity is inspected.
P9-B drops alap: optimizer stages run as soon as their gradients are final (gpu-53: -64 us, bitwise).
"""
RECURRENT_RULES = (
    "rot+defer+jointbig+place+chains+softmax+wrow+part+wsoft+ilp+deepred+rngel+vfill+"
    "vec4+wide+idx32+idxid+rsink+epiearly+epipair+epismh+vec4u+lpt+lptl+zmerge+msum+"
    "p5c+p5cw+p5ce+powx+zelide+colreg+rhoist+ilp8"   # P9-B gpu-79: +rhoist+ilp8 7.080 vs 7.148 ms, bitwise
)

# The selected C sequence rules do not inherit B recurrent scheduling changes.
_SEQUENCE_BASE_RULES = (
    "rot+defer+jointbig+place+chains+softmax+wrow+alap+part+wsoft+ilp+deepred+rngel+vfill+"
    "vec4+wide+idx32+idxid+rsink+epiearly+epipair+epismh+vec4u+lpt+lptl+zmerge+msum+"
    "p5c+p5cw+p5ce+powx+zelide"
)

# P8-C: sequence-input (scan "xs") loop regions. The C-OPS4-A rejection of the P-OPS5-B V4 set was traced to one
# feature, "part" (phase CTA partitioning rewrites blockIdx/gridDim only in the phase text, not inside helper
# bodies that index work themselves): every other feature passed bitwise-repeat/NaN/gate checks individually and
# combined (P8-C gpu-02/gpu-03). "part" stays rejected for these regions.
# P8-C adds vecbf (16-byte FP32/BF16 elementwise vectors, bitwise), ilp8 (bitwise) and the P-OPS5-B deterministic
# split-K for long-K BF16 products (FP32 association; gates pass, gpu-19).
# The tensor-map (wgmma) product rules p5c/p5cw/p5ce/p5ck, with P8-C's 16-byte register epilogues (p5cv, bitwise)
# and short-K admission, are kept by loop_pipeline only in once-per-update regions of sequence loops (init/sequence
# trunk, its VJP and the optimizer); gates pass (P8-C gpu-23), 25.10 -> 23.61 ms on S106B m8k (gpu-21/gpu-26).
SEQUENCE_RULES_EXACT_ORDER = "+".join({"ilp": "ilp8"}.get(f, f) for f in _SEQUENCE_BASE_RULES.split("+")
                                      if f not in ("part", "p5c", "p5cw", "p5ce")) + "+vecbf+splitk"
SEQUENCE_RULES = SEQUENCE_RULES_EXACT_ORDER + "+p5c+p5cw+p5ce+p5ck+p5cv"
# P9-C: race-free disjoint CTA ranges for independent self-contained elementwise / gather phases of a stage.
SEQUENCE_RULES = SEQUENCE_RULES   # "+pelem": measured no gain, and a fill-weight regression (r9/r10); opt-in only
# P9-C: coalesced two-stage column reductions for long [R, C] row sums (norm-gain gradients).
SEQUENCE_RULES = SEQUENCE_RULES + "+colsplit"   # pred measured harmful (r9)
# P9-C diagnostic: extra sequence-region rules for A/B builds (e.g. "+rowepi"); empty by default.
import os as _os
SEQUENCE_RULES = SEQUENCE_RULES + _os.environ.get("P9C_RULES_EXTRA", "")
