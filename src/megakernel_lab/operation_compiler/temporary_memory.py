"""Release step 15: admission policies for late-operand rematerialization (ops/remat.py).

``schedule_options={'remat': True}`` keeps the existing behaviour (every exact candidate). The step-15 modes are
structural filters over the same exact candidates (operation inputs, input roles and consumer kinds; never model,
shape or benchmark names):

``'activations'``  refuse groups whose copied producers read a trainable parameter. Recomputing a BF16 parameter
                   cast in the backward pass rereads the FP32 master weights (4 bytes/parameter) and rewrites it.
``'absorbed'``     'activations', and additionally refuse groups with a late matrix-product consumer. A product
                   reads its operands from memory, so the copy becomes a separate stored elementwise pass; other
                   consumers (reductions, elementwise chains) can evaluate the copy in registers in their own phase.

Recomputation is exact (same ATen operation on the same stored inputs), so results stay bitwise equal; the added
work is listed in the remat report. Whether a policy pays for itself is a measured question, not assumed here.
"""
from __future__ import annotations

MODES = (True, "all", "activations", "absorbed")
MATRIX = ("mm", "bmm", "addmm", "matmul", "baddbmm")


def remat_admission(mode):
    if mode in (True, "all"):
        return None
    if mode not in MODES:
        from .lower import LoweringError
        raise LoweringError(f"remat must be one of {MODES}")

    def admit(graph, sources, late):
        parameters = {name for name in graph.inputs if graph.values[name].role == "parameter"}
        if any(set(source.reads) & parameters for source in sources):
            return False
        if mode == "absorbed":
            return not any(op.target.removeprefix("aten.").split(".")[0] in MATRIX for op in late)
        return True
    return admit
