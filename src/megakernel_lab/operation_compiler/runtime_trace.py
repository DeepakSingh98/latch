"""Opt-in runtime interval probes: per-CTA land/done times and a shared-memory guard, in the shipped binary.

Selected through the normal build API: ``CapturedModule.build(..., runtime_trace=True)``. Omitting it leaves lowering
and generated source unchanged. The program then carries its probes always; the launch's ``ticks`` argument is the one
runtime flag. A launch without ticks (``Runtime.step()``) takes one branch per probe and writes nothing. A launch with
ticks (``Runtime.set_runtime_trace(True)``) records, for every publication interval and CTA:

* land: thread 0 is about to enter the interval;
* end: thread 0 has left the interval's phases (where the older diagnostic stamps sit);
* done: every thread of the CTA has finished the interval (a CTA barrier just ahead of the closing grid barrier, taken
  only on these launches), so a thread leaving a work queue early does not read as the CTA finishing early;
* a shared-memory guard: GUARD_BYTES past the planned shared allocation are written at the first land and compared by
  thread 0 at each done, once every thread has left the interval. A CTA's first mismatch stores a fault word naming the
  interval.

The interval's own code is left alone. Measured on a routed 100M-parameter model, stamps placed inside a large interval moved ptxas' register
allocation enough to slow that interval by 22% with the flag off, and inline stamps between phase blocks serialize
warpgroup MMA (C7509). So every probe stands at kernel scope around the interval's brace block, and the closing
barrier and its tick stanza move out of the block to kernel scope, between the probes. The pass runs on the finished
source after every code-sharing pass, so those passes see the same text as an unprobed program, and device-unit
outlining sees the same phases; an interval function differs from an unprobed one only by the missing barrier and
stanza at its end. Kernel scope pays one branch per probe. Probe bodies are out-of-line calls, which is why a program
with warpgroup MMA left in the entry needs an isolating ``device_unit_inline`` mode. Macros in this source avoid the
substring the device-unit build refuses in a #define (``step17_build``).

The guard catches an access past the end of the planned allocation, not one that stays below it. Hopper refuses
accesses past the launch allocation by itself; the guard words sit inside it. Layout of the ticks buffer after the
program's own tick words: three slots per interval (land, end, done) of 256 words, one per CTA; then 256 fault words.

``python -m megakernel_lab.operation_compiler.runtime_trace --manifest M --ticks T.npy --blocks N`` prints the per-interval
report; ``Runtime.runtime_trace_read()`` decodes the same record in process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import re

KEY = "runtime_trace"
VERSION = 1
GUARD_BYTES = 64
GUARD_WORDS = GUARD_BYTES // 4
STRIDE = 256                  # words per slot, one per CTA
SLOTS = 3                     # land, end, done
GUARD_FIRST, GUARD_LAST = 0x4c52544f, 0x50524f42
FAULT_FLAG = 1 << 63
INJECT_THREAD = 17            # thread that writes an injected overflow
_MODE = ContextVar("latch_runtime_trace", default=False)
_KERNEL = re.compile(r'extern "C" __global__ (?:__launch_bounds__\([^)]*\) )?void latch_ops\(float\*\* args, '
                     r'float\* arena, unsigned long long\* ticks\) \{')
_BARRIER = re.compile(r'\bgrid\.sync\(\);|\blatch_tr::elided_barrier\(\);')
_STAMP = re.compile(r'\bticks\[(\d+)\] = (?:t|stamp\(\));')
_LEAD = r'if \(ticks && (?:lsr_bx|blockIdx\.x) == 0 && (?:lsr_tx|threadIdx\.x) == 0\) '
_TAIL = re.compile(r'\s*(?:' + _LEAD + r'(?:\{ const unsigned long long t = stamp\(\); (?:ticks\[\d+\] = t; ?)+\}'
                   r'|ticks\[\d+\] = stamp\(\);)\s*)?')


class Refusal(Exception):
    pass


def validate(value):
    """False/None: off. True: probes. 'inject': probes plus the test-only overflow sites (never a shipped program)."""
    if value not in (None, False, True, "inject"):
        raise ValueError(f"runtime_trace must be None, False, True or 'inject'; got {value!r}")
    return value


def current():
    return _MODE.get()


@contextmanager
def scope(value):
    token = _MODE.set(validate(value) or False)
    try:
        yield
    finally:
        _MODE.reset(token)


def guard_extension(shared_bytes):
    """The launch allocation that holds ``shared_bytes`` planned bytes and the guard."""
    return shared_bytes + GUARD_BYTES


def _helpers(guard_word, base, intervals, inject):
    injected = (f'''
__device__ __forceinline__ void inject(unsigned long long* ticks, float* s, unsigned interval) {{
  if (ticks[{base + (SLOTS * intervals + 1) * STRIDE}] == interval + 1ULL && blockIdx.x == interval % gridDim.x) {{
    volatile unsigned* g = reinterpret_cast<volatile unsigned*>(s) + {guard_word}u;
    g[interval % 2u ? {GUARD_WORDS - 1}u : 0u] = 0u;
  }}
}}''' if inject else '')
    return f'''// Runtime interval probes (manifest {KEY}): active only on launches that pass the ticks buffer.
namespace latch_rtp {{
__device__ __forceinline__ unsigned long long now() {{
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}}
static __device__ __noinline__ void land(unsigned long long* ticks, unsigned interval) {{
  const unsigned long long t = now();
  if (blockIdx.x < {STRIDE}u) ticks[{base}ULL + interval * {SLOTS * STRIDE}ULL + blockIdx.x] = t;
}}
static __device__ __noinline__ void arm(unsigned long long* ticks) {{
  extern __shared__ float smem[];
  volatile unsigned* g = reinterpret_cast<volatile unsigned*>(smem) + {guard_word}u;
  g[0] = {GUARD_FIRST}u; g[{GUARD_WORDS - 1}] = {GUARD_LAST}u;
  if (blockIdx.x < {STRIDE}u) ticks[{base + SLOTS * intervals * STRIDE}ULL + blockIdx.x] = 0ULL;
}}
static __device__ __noinline__ void done(unsigned long long* ticks, unsigned interval) {{
  extern __shared__ float smem[];
  const bool lead = threadIdx.x == 0u && blockIdx.x < {STRIDE}u;
  if (lead) ticks[{base + STRIDE}ULL + interval * {SLOTS * STRIDE}ULL + blockIdx.x] = now();
  __syncthreads();
  if (!lead) return;
  ticks[{base + 2 * STRIDE}ULL + interval * {SLOTS * STRIDE}ULL + blockIdx.x] = now();
  volatile unsigned* g = reinterpret_cast<volatile unsigned*>(smem) + {guard_word}u;
  const unsigned first = g[0], last = g[{GUARD_WORDS - 1}];
  if (first != {GUARD_FIRST}u || last != {GUARD_LAST}u) {{
    unsigned long long* fault = ticks + {base + SLOTS * intervals * STRIDE}ULL + blockIdx.x;
    if (*fault == 0ULL) *fault = 0x8000000000000000ULL | ((unsigned long long)(first != {GUARD_FIRST}u) << 62)
                                 | ((unsigned long long)(last != {GUARD_LAST}u) << 61) | interval;
    g[0] = {GUARD_FIRST}u; g[{GUARD_WORDS - 1}] = {GUARD_LAST}u;
  }}
}}
template <unsigned... I> static __device__ __noinline__ void publish(unsigned long long* ticks) {{
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const unsigned long long t = now();
  ((ticks[I] = t), ...);
}}{injected}
}}
// End of runtime interval probes
'''


def _sites(block):
    """(inject offset, barrier start, barrier end, tick words) of one kernel-scope brace block, or None without a closing
    barrier. Offsets are into ``block``; the inject offset is inside the first phase block when it stands alone."""
    from .step17_intervals import Refusal as ScanRefusal, _strip, scan
    inner = block[1:-1]
    try:
        rows = scan(inner)
    except ScanRefusal:
        return None
    clean = _strip(inner)
    barrier = None
    for kind, a, b in rows:
        if kind == 'text':
            for m in _BARRIER.finditer(clean, a, b):
                barrier = m
    if barrier is None:
        return None
    tail = clean[barrier.end():]
    if _TAIL.fullmatch(tail) is None:
        raise Refusal("unrecognized interval tail: " + tail.strip()[:120])
    inject = 1
    for kind, a, _ in rows:
        if a >= barrier.start():
            break
        if kind == 'block':
            before = clean[:a].rstrip()
            if not before or before[-1] in ';}':
                inject = a + 2 + (inner[a + 1] == '\n')
            break
    return inject, barrier.start() + 1, barrier.end() + 1, [int(n) for n in _STAMP.findall(tail)]


def instrument(program):
    """The program with its probes: interval phase text is unchanged; only each closing barrier and stanza moves out."""
    from .lower import GeneratedProgram
    from .step17_intervals import scan
    source, manifest = program.source, program.manifest
    mode = current()
    planned = manifest.get("runtime_trace_planned_shared_bytes")
    if planned is None or manifest.get("barrier_schedule") != "dependency":
        raise Refusal("requires one standalone dependency-scheduled compose program")
    entries = list(_KERNEL.finditer(source))
    if len(entries) != 1:
        raise Refusal("expected exactly one recognized latch_ops entry")
    entry = entries[0]
    tail = source.find('\nextern "C" int latch_query(', entry.end())
    close = source.rfind('}', entry.end(), tail)
    if tail < 0 or close < 0:
        raise Refusal("kernel end marker missing")
    body = source[entry.end():close]
    blocks = [(a, b) for kind, a, b in scan(body) if kind == 'block']
    sited = [(a, b, _sites(body[a:b])) for a, b in blocks]
    sited = [row for row in sited if row[2] is not None]
    if not sited:
        raise Refusal("no publication interval")
    base = max(int(manifest.get("tick_words", 0)), int(manifest["phase_count"]) + 2)
    pieces, intervals, cursor = [], [], 0
    for k, (a, b, (inject, start, end, words)) in enumerate(sited):
        block = body[a:b]
        head = f"if (ticks && threadIdx.x == 0u) latch_rtp::land(ticks, {k}u);\n"
        if k == 0:
            head = "if (ticks && threadIdx.x == 0u) latch_rtp::arm(ticks);\n" + head
        kept = block[:start]
        if mode == "inject":
            kept = (kept[:inject] + f"if (ticks && threadIdx.x == {INJECT_THREAD}u) latch_rtp::inject(ticks, smem, {k}u);\n"
                    + kept[inject:])
        stanza = (f"if (ticks) latch_rtp::publish<{','.join(str(w) for w in words)}>(ticks);\n" if words else "")
        pieces += [body[cursor:a], head, kept, "}\n", f"if (ticks) latch_rtp::done(ticks, {k}u);\n", block[start:end], "\n",
                   stanza]
        cursor = b
        intervals.append(dict(interval=k, phases=sorted({w - 1 for w in words if w >= 1})))
    pieces.append(body[cursor:])
    guard_word = planned // 4
    if planned % 4:
        raise Refusal("planned shared bytes are not a whole number of words")
    helpers = _helpers(guard_word, base, len(sited), mode == "inject")
    emitted = (source[:entry.start()] + helpers + source[entry.start():entry.end()] + ''.join(pieces) + source[close:])
    result = dict(manifest)
    fault_base = base + SLOTS * len(sited) * STRIDE
    result[KEY] = dict(version=VERSION, base=base, stride=STRIDE, slots=SLOTS, fault_base=fault_base, intervals=intervals,
                       kernel_blocks=len(blocks), guard_word=guard_word, guard_bytes=GUARD_BYTES,
                       inject_word=fault_base + STRIDE if mode == "inject" else None,
                       inject_thread=INJECT_THREAD if mode == "inject" else None,
                       source_before=manifest["source_hash"])
    result["tick_words"] = fault_base + STRIDE + 1
    result["source_hash"] = hashlib.sha256((emitted + json.dumps(result["header_hashes"], sort_keys=True)).encode()
                                           ).hexdigest()
    return GeneratedProgram(emitted, result)


def fault_word(word):
    """{'interval', 'first_word', 'last_word'} of one fault word, or None."""
    word = int(word) & ((1 << 64) - 1)
    if not word & FAULT_FLAG:
        return None
    return dict(interval=word & 0xffffffff, first_word=bool(word >> 62 & 1), last_word=bool(word >> 61 & 1))


def decode(manifest, stamps, blocks):
    """Per-interval records from one traced launch's ticks words (``blocks``: that launch's grid)."""
    record = manifest[KEY]
    base, stride = record["base"], record["stride"]
    phases, rows = manifest["phases"], []
    live = min(int(blocks), stride)
    words = [int(x) & ((1 << 64) - 1) for x in stamps]
    for row in record["intervals"]:
        k = row["interval"]
        land, end, done = (words[base + (SLOTS * k + slot) * stride:base + (SLOTS * k + slot) * stride + live]
                           for slot in range(SLOTS))
        seen = [c for c in range(live) if land[c] and done[c]]
        rows.append(dict(interval=k, phases=row["phases"],
                         kinds=sorted({phases[p]["kind"] for p in row["phases"] if p < len(phases)}),
                         names=[n for p in row["phases"][:3] if p < len(phases) for n in phases[p]["names"][:1]],
                         ctas=len(seen), land={c: land[c] for c in seen}, end={c: end[c] for c in seen},
                         done={c: done[c] for c in seen}))
    faults = {}
    for c in range(live):
        decoded = fault_word(words[record["fault_base"] + c])
        if decoded is not None:
            faults[c] = decoded
    return dict(intervals=rows, faults=faults)


def _summary(row, next_row):
    if not row["ctas"]:
        return None
    lands, dones = list(row["land"].values()), list(row["done"].values())
    busy = [row["done"][c] - row["land"][c] for c in row["land"]]
    first, last = min(lands), max(dones)
    span = (min(next_row["land"].values()) - first) if next_row and next_row["ctas"] else last - first
    return dict(span_ns=span, land_skew_ns=max(lands) - first, busy_min_ns=min(busy),
                busy_mean_ns=sum(busy) / len(busy), busy_max_ns=max(busy), done_skew_ns=last - min(dones),
                lead_wait_ns=sum(row["done"][c] - row["end"][c] for c in row["land"]) / len(busy),
                slowest_cta=max(row["done"], key=row["done"].get))


def report(trace, *, top=None):
    """The per-interval report as text lines; ``top``: only the N longest intervals, in program order."""
    rows = trace["intervals"]
    summaries = [(row, _summary(row, rows[k + 1] if k + 1 < len(rows) else None)) for k, row in enumerate(rows)]
    summaries = [(row, s) for row, s in summaries if s is not None]
    total = sum(s["span_ns"] for _, s in summaries)
    chosen = summaries
    if top is not None:
        kept = {row["interval"] for row, _ in sorted(summaries, key=lambda x: -x[1]["span_ns"])[:top]}
        chosen = [x for x in summaries if x[0]["interval"] in kept]
    lines = [f"{'iv':>5} {'span_us':>10} {'%':>6} {'land_skew':>10} {'busy_min':>10} {'busy_mean':>10} {'busy_max':>10} "
             f"{'done_skew':>10} {'lead_wait':>10} {'slow':>5} {'ctas':>5}  kinds / first values"]
    for row, s in chosen:
        lines.append(f"{row['interval']:>5} {s['span_ns'] / 1e3:>10.1f} {100 * s['span_ns'] / max(total, 1):>6.2f} "
                     f"{s['land_skew_ns'] / 1e3:>10.1f} {s['busy_min_ns'] / 1e3:>10.1f} {s['busy_mean_ns'] / 1e3:>10.1f} "
                     f"{s['busy_max_ns'] / 1e3:>10.1f} {s['done_skew_ns'] / 1e3:>10.1f} "
                     f"{s['lead_wait_ns'] / 1e3:>10.1f} {s['slowest_cta']:>5} "
                     f"{row['ctas']:>5}  {','.join(row['kinds'])} {' '.join(row['names'])}")
    lines.append(f"total {total / 1e6:.3f} ms over {len(summaries)} intervals of {len(rows)}")
    if trace["faults"]:
        worst = sorted(trace["faults"].items(), key=lambda kv: kv[1]["interval"])
        c, first = worst[0]
        lines.append(f"FAULT shared-memory guard overwritten in interval {first['interval']} (CTA {c}; "
                     f"{len(worst)} CTAs reported)")
    else:
        lines.append("no fault")
    return lines


def main():
    import numpy as np
    p = argparse.ArgumentParser(description="Per-interval report of one traced launch")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ticks", type=Path, required=True, help=".npy of the launch's int64 ticks words")
    p.add_argument("--blocks", type=int, required=True, help="the launch's grid size")
    p.add_argument("--top", type=int, help="only the N longest intervals")
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if KEY not in manifest:
        raise SystemExit("this program was not built with runtime_trace")
    print("\n".join(report(decode(manifest, np.load(a.ticks).tolist(), a.blocks), top=a.top)))


if __name__ == "__main__":
    main()
