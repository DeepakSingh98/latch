"""Fused vocabulary statistics and logits gradient (``vocab_rules`` ``rows``): one staged row pass.

The streamed vocabulary head (vocab_head.py) reads the BF16 logits [rows, Vp] three times per launch: the online
statistics pass (vocab_rows), the one-element pick and the logits-gradient pass (vocab_gradient). The statistics
pass and the gradient pass each stream the whole tensor, both bound by HBM bandwidth for large vocabularies.

Here the gradient phase is folded into its statistics phase. One CTA owns one row at a time: the row is staged once
into the program's shared storage (16-byte cp.async, one row = Vp BF16 = 2 Vp bytes, admitted only when it fits the
dynamic shared budget), the exact row maximum and the exp-sum are reduced from the staged row, and every
gradient element is emitted from the staged row. While a thread emits its chunks of row r it issues the cp.async of
the same chunks of its CTA's next row into the slots it has just consumed (each thread only ever touches its own
chunks, so no extra synchronisation), so the next row's load overlaps this row's gradient stores. HBM traffic per
launch: one logits read + one gradient write instead of two reads + one write.

Arithmetic: row maximum exact (fmaxf, order-free); exp-sum = sum of expf(x - max) over the first V columns per thread
in ascending chunk order, then a warp xor-butterfly and an ascending 8-warp merge (a declared FP32 reassociation of
the online warp pass; same expression class, hence ``precision_policy='practical'`` only); logsum = logf(sum).
Gradient element: the vocab_gradient expression unchanged ([j==t]c - expf((x - max) - logsum) * c, one rounding to
BF16, exact zero in padded columns). The pick phase is unchanged and reads the statistics this phase writes.
"""
from __future__ import annotations

import math

from .vocab_rules import VOCAB_DYNAMIC_SHARED_BOUND

THREADS = 256


def enabled():
    """vocab_rules ``rows``."""
    from .vocab_rules import active
    return active("rows")


def shared_bytes(vp):
    """Dynamic shared bytes for a padded width ``vp``: 256 bytes of reduction scratch plus one BF16 row. The rule
    admits only ``shared_bytes(vp) <= VOCAB_DYNAMIC_SHARED_BOUND``, i.e. vp <= 112,000."""
    return 256 + 2 * int(vp)


def _terms(args, kwargs, names, layouts, dtypes, rows):
    """The gradient's target index, cotangent leaf and chain steps, spelled exactly as vocab_head.emit does."""
    from .lower import LoweringError, _ident, _index, _ref as ref_of
    vocab = int(kwargs["vocabulary"])
    chain = kwargs["chain"]
    leaf = args[3]
    factors = list(args[4:])
    if len(factors) != len(chain):
        raise LoweringError("vocab_grad chain/factor arity mismatch")
    index = ref_of(args[2], names)
    li = layouts[index]
    idx = f"{_ident(index)}[{_index(li, (rows, 1), 'row')}]"
    if isinstance(leaf, (int, float)):
        if not math.isfinite(float(leaf)):
            raise LoweringError("non-finite scatter value")
        c = f"{float(leaf)!r}f"
    else:
        src = ref_of(leaf, names)
        c = f"{_ident(src)}[{_index(layouts[src], (rows, 1), 'row')}]"
        if kwargs.get("scatter_add"):
            c = f"(0.0f + {c})"
    steps = []
    for op, factor in zip(chain, factors):
        if op == "~":
            steps.append("c = -c;")
            continue
        f = ref_of(factor, names)
        if dtypes[f] != "float32":
            raise LoweringError("vocab_grad factors must be FP32")
        value = f"{_ident(f)}[{_index(layouts[f], (rows, vocab), f'(row * {vocab}LL)')}]"
        steps.append(f"c = c {'*' if op == '*' else '/'} {value};")
    return vocab, idx, c, steps


def code(x, stats, out, rows, vp, vocab, idx, c, steps):
    chunks = vp // 8
    full = vocab // 8          # chunks wholly inside the vocabulary
    step_code = "\n    ".join(steps)
    T = THREADS
    return f"""{{
  // Fused vocab: statistics + logits gradient, one staged row per CTA (ops/vocab_rows.py).
  float* red_ = smem;
  __nv_bfloat16* row_ = reinterpret_cast<__nv_bfloat16*>(smem + 64);
  const int lane_ = threadIdx.x & 31, warp_ = threadIdx.x >> 5;
  __syncthreads();
  if (blockIdx.x < {rows}LL) {{
    const __nv_bfloat16* src_ = {x} + (long long)blockIdx.x * {vp}LL;
    for (int q = threadIdx.x; q < {chunks}; q += {T}) {{
      const unsigned d_ = static_cast<unsigned>(__cvta_generic_to_shared(row_ + 8 * q));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(d_), "l"(src_ + 8LL * q) : "memory");
    }}
    asm volatile("cp.async.commit_group;\\n" ::: "memory");
  }}
  for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
    asm volatile("cp.async.wait_group 0;\\n" ::: "memory");
    float m = -INFINITY;
#pragma unroll 7
    for (int q = threadIdx.x; q < {chunks}; q += {T}) {{   // a thread reads only the chunks it staged
      float v[8]; latch_vocab_unpack(*reinterpret_cast<const uint4*>(row_ + 8 * q), v);
      if (q < {full}) {{
#pragma unroll
        for (int e = 0; e < 8; ++e) m = fmaxf(m, v[e]);
      }} else {{
#pragma unroll
        for (int e = 0; e < 8; ++e) if (8 * q + e < {vocab}) m = fmaxf(m, v[e]);
      }}
    }}
#pragma unroll
    for (int d = 16; d > 0; d /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    if (lane_ == 0) red_[warp_] = m;
    __syncthreads();
    float maximum = red_[0];
#pragma unroll
    for (int w = 1; w < {T // 32}; ++w) maximum = fmaxf(maximum, red_[w]);
    float s = 0.0f;
#pragma unroll 7
    for (int q = threadIdx.x; q < {chunks}; q += {T}) {{
      float v[8]; latch_vocab_unpack(*reinterpret_cast<const uint4*>(row_ + 8 * q), v);
      if (q < {full}) {{
#pragma unroll
        for (int e = 0; e < 8; ++e) s += expf(v[e] - maximum);
      }} else {{
#pragma unroll
        for (int e = 0; e < 8; ++e) if (8 * q + e < {vocab}) s += expf(v[e] - maximum);
      }}
    }}
#pragma unroll
    for (int d = 16; d > 0; d /= 2) s += __shfl_xor_sync(0xffffffffu, s, d);
    if (lane_ == 0) red_[32 + warp_] = s;
    __syncthreads();
    float total_ = red_[32];
#pragma unroll
    for (int w = 1; w < {T // 32}; ++w) total_ += red_[32 + w];
    const float logsum = logf(total_);
    if (threadIdx.x == 0) {{ {stats}[2 * row] = maximum; {stats}[2 * row + 1] = logsum; }}
    const int64_t t = {idx};
    assert(t >= 0 && t < {vocab}LL);
    float c = {c};
    {step_code}
    const int tq_ = (int)(t >> 3);
    const long long next_ = row + gridDim.x;
    const __nv_bfloat16* nsrc_ = {x} + next_ * {vp}LL;
    __nv_bfloat16* dst_ = {out} + row * {vp}LL;
#pragma unroll 7
    for (int q = threadIdx.x; q < {chunks}; q += {T}) {{
      float v[8]; latch_vocab_unpack(*reinterpret_cast<const uint4*>(row_ + 8 * q), v);
      float g[8];
      if (q < {full} && q != tq_) {{
#pragma unroll
        for (int e = 0; e < 8; ++e) g[e] = 0.0f - expf((v[e] - maximum) - logsum) * c;
      }} else {{
#pragma unroll
        for (int e = 0; e < 8; ++e) {{
          const long long j = 8LL * q + e;
          g[e] = j < {vocab}LL ? ((j == t ? c : 0.0f) - expf((v[e] - maximum) - logsum) * c) : 0.0f;
        }}
      }}
      uint4 packed; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
      for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(g[2 * e], g[2 * e + 1]);
      *reinterpret_cast<uint4*>(dst_ + 8LL * q) = packed;
      if (next_ < {rows}LL) {{
        const unsigned d_ = static_cast<unsigned>(__cvta_generic_to_shared(row_ + 8 * q));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(d_), "l"(nsrc_ + 8LL * q) : "memory");
      }}
    }}
    asm volatile("cp.async.commit_group;\\n" ::: "memory");
    __syncthreads();   // red_ is rewritten by the next row
  }}
  asm volatile("cp.async.wait_group 0;\\n" ::: "memory");
  __syncthreads();
}}"""


def annotate(kind, target, name, args, kwargs, refs, names, layouts, dtypes, extra):
    """At emission: record the gradient's terms on its phase (``vocab_rows_terms``) for fuse_pass; no code change."""
    if not enabled() or kind != "vocab_gradient":
        return
    from .lower import _ident, _ref as ref_of
    product, stats = ref_of(args[0], names), ref_of(args[1], names)
    lp = layouts[product]
    rows, vp = lp.shape
    if (dtypes[product] != "bfloat16" or dtypes[name] != "bfloat16" or tuple(lp.strides) != (vp, 1) or lp.offset
            or vp % 8 or tuple(layouts[name].shape) != (rows, vp) or tuple(layouts[name].strides) != (vp, 1)
            or layouts[name].offset or shared_bytes(vp) > VOCAB_DYNAMIC_SHARED_BOUND):
        return
    vocab, idx, c, steps = _terms(args, kwargs, names, layouts, dtypes, rows)
    extra["vocab_rows_terms"] = dict(product=product, stats=stats, target=str(target), rows=rows, vp=vp, vocab=vocab,
                                     code=code(_ident(product), _ident(stats), _ident(name), rows, vp, vocab, idx, c,
                                               steps))


def fuse_pass(phases, layouts):
    """Before barrier intervals: each annotated gradient phase replaces its statistics phase (one row pass).

    The gradient's other operands (target index, cotangent factors) may be emitted after the statistics phase; their
    producer phases are hoisted ahead of it when every value they read is produced before it (moving a producer
    earlier keeps all its readers after it). Returns the list of fused gradient names (phases rewritten in place)."""
    fused = []
    while True:
        g = next((k for k, p in enumerate(phases) if p.get("vocab_rows_terms") and p.get("kind") == "vocab_gradient"), None)
        if g is None:
            return fused
        grad = phases[g]
        terms = grad.pop("vocab_rows_terms")
        root = lambda n: layouts[n].root if n in layouts else n
        writer = {}
        for k, p in enumerate(phases):
            for w in p["writes"]:
                writer.setdefault(root(w), k)
        stats, product = terms["stats"], terms["product"]
        q = writer.get(root(stats))
        if q is None or q >= g:
            continue
        sp = phases[q]
        if sp.get("kind") != "vocab_rows" or sp.get("names") != [stats] or sp.get("reads") != [product] \
                or int(sp.get("reduction_elements", terms["vocab"])) != terms["vocab"]:
            continue
        others = [r for r in grad["reads"] if r not in (stats, product)]
        # transitive closure of the producers emitted between the statistics and the gradient phase
        hoist, ok, work = set(), True, [writer.get(root(r)) for r in others]
        while work:
            k = work.pop()
            if k is None or k < q or k in hoist:
                continue
            if k == q or k >= g:
                ok = False
                break
            hoist.add(k)
            work.extend(w for w in (writer.get(root(x)) for x in phases[k]["reads"]) if w is not None and w != k)
        if not ok:
            continue
        hoist = sorted(hoist)
        name = grad["names"][0]
        new = dict(kind="vocab_rows_gradient", engine="vocab_rows_staged", names=[stats, name], writes=[stats, name],
                   targets=[*sp.get("targets", []), *grad.get("targets", [])],
                   reads=list(dict.fromkeys([product, *others])), code=terms["code"],
                   shared_bytes=shared_bytes(terms["vp"]), threads=THREADS, vocab_rows=terms["rows"],
                   reduction_elements=terms["vocab"],
                   vocab_rows_fused=dict(rows=terms["rows"], padded=terms["vp"], vocabulary=terms["vocab"],
                                         logits_reads_per_row=1, hoisted=[phases[k]["names"] for k in hoist],
                                         storage="one BF16 row in program shared storage, next row prefetched"),
                   arithmetic=("exact row max; per-thread ascending expf(x - max) sums, warp xor butterfly, ascending "
                               "warp merge (declared FP32 reassociation of the online statistics); logsum = logf; "
                               "gradient element [j==t]c - expf((x-max)-logsum)*c, one BF16 rounding, exact zero "
                               "in padded columns"))
        skip = set(hoist) | {q, g}
        phases[:] = ([*phases[:q], *(phases[k] for k in hoist), new]
                     + [p for k, p in enumerate(phases) if q < k and k not in skip])
        fused.append(name)
