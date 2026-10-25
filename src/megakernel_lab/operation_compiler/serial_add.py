"""Stream identity-indexed FP32 addition chains without preloading every leaf.

The ordinary vector emitter loads all operands for all unrolled indices first.
For a long serial sum that creates hundreds of live values. This emitter keeps
one float4 sum and one float4 leaf live, preserving each original FP32 addition
and its operand order. It recognizes the emitted operation and storage contract,
not parameter names or the workload that produced the chain.
"""
from __future__ import annotations

import re

ENABLED = True
_LINE = re.compile(r'const float r_(\w+) = (.*?);(?:\s+(.*))?$')
_ADD = re.compile(r'^\((.+) \+ \((.+) \* 1\.0f\)\)$')


def emit(phase, numel, texts):
    """Return a streaming float4 loop only for a private left/right-linear sum.

``texts`` proves every admitted load and final store is an aligned, contiguous
FP32 identity access. Intermediate materializations, branching sums, alpha
scaling, other arithmetic, and nonidentity accesses retain ordinary lowering.
"""
    names, body = phase.get('names', []), phase.get('body', [])
    if (not ENABLED or numel < 4096 or numel % 4 or len(names) < 4
            or len(body) != len(names)
            or phase.get('targets') != ['aten.add.Tensor'] * len(names)):
        return None
    operations, previous = [], None
    for pos, (name, line) in enumerate(zip(names, body)):
        match = _LINE.fullmatch(line.strip())
        if not match or match[1] != name:
            return None
        addition = _ADD.fullmatch(match[2])
        if not addition:
            return None
        left, right = addition.groups()
        operands = (left, right)
        if previous is None:
            if any(x not in texts for x in operands):
                return None
        elif (operands.count(previous) != 1
                or any(x != previous and x not in texts for x in operands)):
            return None
        final = pos == len(names) - 1
        expected_store = f'v_{name}[i] = r_{name};'
        if (match[3] or '') != (expected_store if final else ''):
            return None
        if final and f'v_{name}[i]' not in texts:
            return None
        operations.append((left, right, previous))
        previous = f'r_{name}'

    code = [f'for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < {numel // 4}LL; '
            'j += (long long)gridDim.x * blockDim.x) {', '  float4 serial_acc_;']
    for pos, (left, right, previous) in enumerate(operations):
        code.append('  {')
        values = {}
        for operand in (left, right):
            if operand == previous:
                values[operand] = 'serial_acc_'
            elif operand not in values:
                local = 'serial_leaf_' + str(len(values))
                code.append(f'    const float4 {local} = reinterpret_cast<const float4*>({texts[operand]})[j];')
                values[operand] = local
        # No reassociation or extra +0: the first operation adds the same two
        # leaves, and every later operation consumes precisely the prior sum.
        for component in 'xyzw':
            code.append(f'    serial_acc_.{component} = __fadd_rn({values[left]}.{component}, {values[right]}.{component});')
        code.append('  }')
    code.extend([f'  reinterpret_cast<float4*>({texts[f"v_{names[-1]}[i]"]})[j] = serial_acc_;', '}'])
    phase['serial_add_stream'] = dict(additions=len(names), vector_width=4,
        arithmetic='original operand order; one FP32 RN addition per captured add; no reassociation',
        live_vector_operands=2)
    return '\n'.join(code)
