"""Bitwise skinny bodies for explicit-BF16 dense products (matrix_bodies ``skinny``, ``skinnybatch``).

Skinny products -- a GEMV-shaped split-K product ``[S, K] x [K, L]`` with S <= 16 (for example the gradient of a
[T, 1] product with respect to its [1, D] operand) and a short-K product ``[M, K] x [K, N]`` with K <= 16 -- are
lowered by the P9-E skinny bodies. Both are latency/issue bound: the split body stalls each chain on one exposed
global load per 128-row slab, and the short-K body writes 2-byte scalars and divides 64-bit indices per element.
The bodies here keep the arithmetic and change only data movement and thread mapping, so every result is bit for
bit the P9-E result.

* ``split``: the same tasks (chunk of L positions x K part), K parts and part length, ascending-k fmaf chains from +0
  per (output, part), FP32 part results in the same private scratch, the same last-arriver part-order sum from +0 and
  one rounding to BF16, and the same arrival-counter restore. New: the task's K slab of the big operand streams through
  an NS-stage cp.async ring (16-byte copies by all 256 threads) and the small operand for the whole part is copied to
  shared memory once.
* ``outer``: short-K (K <= 16) products whose output rows are 16-byte vectors: each item computes 8 adjacent outputs
  of one row with the same per-element ascending fmaf chain from +0 and one RN rounding, stored as one 16-byte write;
  the A values of all items of one unrolled round are loaded before any output store.
* ``batch_interval`` (``skinnybatch``): the same-geometry split products of one dependency interval run up to
  ``BATCH_MAX`` per body; small batches also carry the interval's short-K products on warps 4-7.

Opt-in through ``matrix_bodies``; without ``skinny`` the P9-E bodies and their call forms remain. Dispatch is keyed by
shape/stride/alignment conditions only, never by graph position or value names.
"""
from __future__ import annotations

SPLIT_LC = 16          # L positions per task
SPLIT_ROWS = 256       # K rows per ring stage
SPLIT_STAGES = 8       # ring depth
SPLIT_MAX_SHARED = 196608
OUTER_UNROLL = 8       # short-K items per thread and round


def enabled():
    from .matrix_bodies import active
    return active("skinny")


def _pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def chain_loop(R, LC, S, indent):
    """Full-stage chain over R rows: acc = fmaf(big[z], small[z], acc) for z ascending."""
    pad = " " * indent
    return (f"{pad}#pragma unroll 16\n{pad}for (int z = 0; z < {R}; ++z) "
            f"acc = fmaf(__bfloat162float(bk[z * {LC}]), __bfloat162float(sk[z * {S}]), acc);")


def split_plan(m, n, k, sam, sak, sbk, sbn, la, lb, parts, kp):
    """Shape/layout admission for ``split`` (else None: the caller keeps the P9-E body)."""
    small_n = n <= 16
    S, L = (n, m) if small_n else (m, n)
    if S > 16 or parts < 2:
        return None
    sp = _pow2(S)
    # Small operand: element (k, s) contiguous as [K][S] (s stride 1, k stride S) at a 16-byte aligned offset, so the
    # whole part copies with 16-byte cp.async into the [kq][sp] shared layout (S == sp keeps that layout exact).
    s_stride, k_stride, s_off = (sbn, sbk, lb.offset) if small_n else (sam, sak, la.offset)
    if S != sp or (S > 1 and s_stride != 1) or k_stride != S or s_off % 8 or (kp * S) % 8 or (k * S) % 8:
        return None
    big_l, big_k, big_off = (sam, sak, la.offset) if small_n else (sbn, sbk, lb.offset)
    if big_l != 1 or big_k % 8 or big_off % 8 or L % 8 or L < 64:
        return None
    small_bytes = -(-(kp * S * 2) // 16) * 16
    shared = small_bytes + SPLIT_STAGES * SPLIT_ROWS * SPLIT_LC * 2
    if shared > SPLIT_MAX_SHARED:
        return None
    chunks = -(-L // SPLIT_LC)
    return dict(small_n=small_n, S=S, L=L, sp=sp, chunks=chunks, small_bytes=small_bytes, shared_bytes=shared,
                big_k=big_k)


def split(a, b, out, m, n, k, parts, kp, plan, ident):
    S, L, LC, R, NS = plan["S"], plan["L"], SPLIT_LC, SPLIT_ROWS, SPLIT_STAGES
    chunks, big_k = plan["chunks"], plan["big_k"]
    big, small = (a, b) if plan["small_n"] else (b, a)
    row, col = ("l", "sidx") if plan["small_n"] else ("sidx", "l")
    copies = R * LC // 8                       # 16-byte copies per stage (LC = 16: two per row)
    per = -(-copies // 256)
    small16 = kp * S // 8                       # 16-byte copies of one part's small operand
    return f"""{{ // Alt skinny split (bitwise = P9-E skinny split): {L} x {S} outputs, {parts} K parts of {kp}, {chunks} chunks of {LC}, {NS}x{R}-row cp.async ring
  __shared__ int latch_last;
  unsigned* latch_cnt = /*P2_CNT*/;
  float* latch_part = /*P2_PART*/;                   // [{parts}][{m * n}] FP32 part results
  __nv_bfloat16* smalls = reinterpret_cast<__nv_bfloat16*>(smem);                         // [{kp}][{S}] this part
  __nv_bfloat16* ring = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(smem) + {plan["small_bytes"]});  // [{NS}][{R}][{LC}]
  const int sidx = threadIdx.x / {LC}, lt = threadIdx.x % {LC};
  for (long long task = blockIdx.x; task < {chunks * parts}LL; task += gridDim.x) {{
    const int chunk = (int)(task / {parts}LL), part = (int)(task % {parts}LL);
    const int l = chunk * {LC} + lt;
    const int k0 = part * {kp}, k1 = min({k}, k0 + {kp});
    const int stages = (k1 - k0 + {R - 1}) / {R};
    auto issue = [&](int st) {{
      if (st < stages) {{
        const int kb = k0 + st * {R};
        __nv_bfloat16* dst = ring + (st % {NS}) * {R * LC};
#pragma unroll
        for (int u = 0; u < {per}; ++u) {{
          const int c = threadIdx.x + 256 * u, kk = c / {LC // 8}, l8 = (c % {LC // 8}) * 8, gl = chunk * {LC} + l8;
          if (c < {copies} && kb + kk < k1 && gl < {L})
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"((unsigned)__cvta_generic_to_shared(dst + kk * {LC} + l8)),
                         "l"({ident(big)} + (long long)(kb + kk) * {big_k}LL + gl) : "memory");
        }}
      }}
      asm volatile("cp.async.commit_group;\\n" ::: "memory");
    }};
    __syncthreads();                                 // shared memory of the previous task / phase is free
    for (int c = threadIdx.x; c < {small16}; c += 256)
      if (k0 * {S} + c * 8 < k1 * {S})
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"((unsigned)__cvta_generic_to_shared(smalls + c * 8)),
                     "l"({ident(small)} + (long long)k0 * {S}LL + c * 8) : "memory");
#pragma unroll
    for (int st = 0; st < {NS - 1}; ++st) issue(st);  // the small copies join the first group
    float acc = 0.0f;
    for (int st = 0; st < stages; ++st) {{
      asm volatile("cp.async.wait_group {NS - 2};\\n" ::: "memory");
      __syncthreads();                               // stage st (and the small operand) visible; stage st-1 consumed
      issue(st + {NS - 1});                          // refills the buffer of stage st-1
      if (sidx < {S}) {{
        const __nv_bfloat16* bk = ring + (st % {NS}) * {R * LC} + lt;
        const __nv_bfloat16* sk = smalls + st * {R * S} + sidx;
        const int n_k = min({R}, k1 - (k0 + st * {R}));
        if (n_k == {R}) {{
{chain_loop(R, LC, S, 10)}
        }} else {{
          for (int z = 0; z < n_k; ++z) acc = fmaf(__bfloat162float(bk[z * {LC}]), __bfloat162float(sk[z * {S}]), acc);
        }}
      }}
    }}
    asm volatile("cp.async.wait_group 0;\\n" ::: "memory");
    if (sidx < {S} && l < {L}) __stcg(latch_part + (long long)part * {m * n}LL + (long long){row} * {n}LL + {col}, acc);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) latch_last = atomicAdd(latch_cnt + chunk, 1u) == {parts - 1}u;
    __syncthreads();
    if (latch_last) {{
      __threadfence();
      if (sidx < {S} && l < {L}) {{
        const long long o = (long long){row} * {n}LL + {col};
        float total = 0.0f;
        for (int q = 0; q < {parts}; ++q) total += __ldcg(latch_part + (long long)q * {m * n}LL + o);
        {out}[o] = __float2bfloat16_rn(total);
      }}
      if (threadIdx.x == 0) latch_cnt[chunk] = 0u;   // counters are restored to zero for later phases
    }}
  }}
}}"""


def outer_ok(m, n, k, sbk, sbn, lb):
    """Short-K products with 16-byte output rows (the B rows are staged in shared memory as FP32, as before)."""
    return k <= 16 and k * n * 4 <= 16384 and n % 8 == 0 and m * (n // 8) < (1 << 31)


def outer(a, b, out, m, n, k, sam, sak, sbk, sbn, ident):
    return f"""{{ // Alt skinny-K product (bitwise = P9-E skinny-K): {m}x{n} = ({m}x{k})({k}x{n}), 8 outputs per item, 16-byte stores
  __syncthreads();                                   // earlier users of shared memory are done
  float* bs = smem;
  for (int i = threadIdx.x; i < {k * n}; i += blockDim.x)
    bs[i] = __bfloat162float({ident(b)}[(i / {n}) * {sbk}LL + (i % {n}) * {sbn}LL]);
  __syncthreads();
{_outer_loop(a, out, m, n, k, sam, sak, 0, ident)}
  __syncthreads();
}}"""


def _outer_loop(a, out, m, n, k, sam, sak, off, ident):
    n8 = n // 8
    items = m * n8
    U = OUTER_UNROLL
    # B row z, outputs c..c+7: two float4 shared reads (16-byte lane stride: no bank conflicts); per output the
    # same ascending-z fmaf chain from +0.
    terms = "\n".join(
        f"        {{ const float4 b0 = *reinterpret_cast<const float4*>(bs + {off + z * n} + c), "
        f"b1 = *reinterpret_cast<const float4*>(bs + {off + z * n} + c + 4);\n"
        f"          acc[0] = fmaf(av[{z}], b0.x, acc[0]); acc[1] = fmaf(av[{z}], b0.y, acc[1]); "
        f"acc[2] = fmaf(av[{z}], b0.z, acc[2]); acc[3] = fmaf(av[{z}], b0.w, acc[3]);\n"
        f"          acc[4] = fmaf(av[{z}], b1.x, acc[4]); acc[5] = fmaf(av[{z}], b1.y, acc[5]); "
        f"acc[6] = fmaf(av[{z}], b1.z, acc[6]); acc[7] = fmaf(av[{z}], b1.w, acc[7]); }}" for z in range(k))
    dst = f"reinterpret_cast<uint4*>({out} + (long long)row * {n}LL + c)"
    store = f"*{dst} = v;"
    item = f"""        float acc[8] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
{terms}
        uint4 v;
        __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) h[j] = __halves2bfloat162(__float2bfloat16_rn(acc[2 * j]), __float2bfloat16_rn(acc[2 * j + 1]));
        {store}"""
    # All U items' A values load before any output store (the uint4 stores may alias A for the compiler, which
    # otherwise serializes one global-load latency per item); same per-element arithmetic.
    body = f"""    float avs[{U}][{k}];
#pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      const unsigned e = e0 + u * stride, row = (e < {items}u ? e : 0u) / {n8}u;
#pragma unroll
      for (int z = 0; z < {k}; ++z) avs[u][z] = __bfloat162float({ident(a)}[(long long)row * {sam}LL + z * {sak}LL]);
    }}
#pragma unroll
    for (int u = 0; u < {U}; ++u) {{
      const unsigned e = e0 + u * stride;
      if (e < {items}u) {{
        const unsigned row = e / {n8}u;
        const int c = (int)(e % {n8}u) * 8;
        const float* av = avs[u];
{item}
      }}
    }}"""
    return f"""  if ((reinterpret_cast<unsigned long long>({out}) & 15ull) == 0ull) {{
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned e0 = blockIdx.x * blockDim.x + threadIdx.x; e0 < {items}u; e0 += {U}u * stride) {{
{body}
  }}
  }} else {{                                         // unaligned output view: the P9-E per-element loop
  for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < {m * n}LL; e += (long long)gridDim.x * blockDim.x) {{
    const long long row = e / {n}LL;
    const int col = (int)(e % {n}LL);
    float acc = 0.0f;
#pragma unroll
    for (int z = 0; z < {k}; ++z) acc = fmaf(__bfloat162float({ident(a)}[row * {sam}LL + z * {sak}LL]), bs[{off} + z * {n} + col], acc);
    {out}[row * {n}LL + col] = __float2bfloat16_rn(acc);
  }}
  }}"""


# ---------------------------------------------------------------------------------------------------------------
# Batched split (matrix_bodies ``skinnybatch``): the mutually independent skinny split products of one dependency
# interval with identical geometry run up to BATCH_MAX at a time in one body. A task is still (chunk, part); its
# products' chains run concurrently on G x 16 x S lanes (several warps instead of one), each product with its own
# shared ring and small operand, its own FP32 partials, arrival counters and last-arriver part-order sum (values bit for
# bit as the unbatched body). Placement inside the interval changes only: the batch runs at its first member's position.
# Batches of at most COMBINE_MAX products (and at most what fits in 128 threads) use 128 threads for the split chains
# and carry the interval's short-K products on warps 4-7.
BATCH_MAX = 8
BATCH_ROWS = 128
BATCH_STAGES = 4
COMBINE_MAX = 5


def batching():
    return enabled() and _skinnybatch()


def combining():
    """Batches also carry the interval's skinny-K products on warps 4-7 (always together with batching)."""
    return enabled() and _skinnybatch()


def _skinnybatch():
    from .matrix_bodies import active
    return active("skinnybatch")


def _geometry(phase):
    h = phase.get("skinny_split")
    return None if not h else (h["S"], h["L"], h["k"], h["m"], h["n"], h["parts"], h["kp"], h["small_n"], h["chunks"])


def batch_size(h, threads=256):
    S, LC, R, NS = h["S"], SPLIT_LC, BATCH_ROWS, BATCH_STAGES
    per = h["small_bytes"] + NS * R * LC * 2
    return max(1, min(BATCH_MAX, SPLIT_MAX_SHARED // per, threads // (LC * S)))


def batch_interval(phases, members):
    """{phase index: replacement code} for this interval's batchable skinny split phases (leaders get the batch body,
    the other members of a batch an empty body). Groups of one keep their own code."""
    import re
    groups = {}
    for i in members:
        g = _geometry(phases[i])
        if g is not None:
            groups.setdefault(g, []).append(i)
    combine = combining()
    cmax = COMBINE_MAX                       # combine only batches of at most this many products
    outers = [i for i in members if phases[i].get("skinny_outer")] if combine else []
    plan = []
    for g, items in groups.items():
        h = phases[items[0]]["skinny_split"]
        size = batch_size(h, 256)
        if combine and len(items) <= min(cmax, batch_size(h, 128)):
            size = len(items)
        plan += [items[start:start + size] for start in range(0, len(items), size)]
    shares = {}
    eligible = [b for b in plan if len(b) <= min(cmax, batch_size(phases[b[0]]["skinny_split"], 128))]
    if outers and eligible:
        per_batch = -(-len(outers) // len(eligible))
        start = 0
        for batch in eligible:
            take = min(per_batch, len(batch))   # about one short-K product per split product
            shares[batch[0]] = outers[start:start + take]
            start += take
    out = {}
    for batch in plan:
        h = phases[batch[0]]["skinny_split"]
        mine = shares.get(batch[0], [])
        if len(batch) < 2 and not mine:
            continue
        prods = []
        for i in batch:
            code = phases[i]["code"]
            cnt = re.search(r"unsigned\* latch_cnt = (.+?);\n", code).group(1)
            part = re.search(r"float\* latch_part = (.+?);", code).group(1)
            hi = phases[i]["skinny_split"]
            prods.append(dict(big=hi["big"], small=hi["small"], out=hi["out"], cnt=cnt, part=part,
                              name=phases[i]["names"][0]))
        outs = [dict(phases[i]["skinny_outer"], name=phases[i]["names"][0]) for i in mine]
        out[batch[0]] = batch_split(h, prods, outs)
        for i in mine:
            out[i] = f"// Alt: {phases[i]['names'][0]} runs on warps 4-7 of the batched skinny split of {phases[batch[0]]['names'][0]}"
            phases[i]["skinny_batch"] = dict(leader=phases[batch[0]]["names"][0], size=len(batch), outer=True)
        per = h["small_bytes"] + BATCH_STAGES * BATCH_ROWS * SPLIT_LC * 2
        phases[batch[0]]["shared_bytes"] = max(int(phases[batch[0]].get("shared_bytes", 0)), len(batch) * per)
        for i in batch:
            phases[i]["skinny_batch"] = dict(leader=phases[batch[0]]["names"][0], size=len(batch))
        for i in batch[1:]:
            out[i] = f"// Alt: {phases[i]['names'][0]} runs in the batched skinny split of {phases[batch[0]]['names'][0]}"
    return out


def batch_split(h, prods, outers=()):
    G, S, L, LC, R, NS = len(prods), h["S"], h["L"], SPLIT_LC, BATCH_ROWS, BATCH_STAGES
    T = 128 if outers else 256                 # threads of the split group (warps 4-7 stream the outer products)
    m, n, k, parts, kp, chunks, big_k = h["m"], h["n"], h["k"], h["parts"], h["kp"], h["chunks"], h["big_k"]
    sb = h["small_bytes"]                         # per-product region strides (bytes)
    rb = NS * R * LC * 2
    lanes = LC * S
    row, col = ("l", "sidx") if h["small_n"] else ("sidx", "l")
    copies = R * LC // 8
    per = -(-copies // T)
    small16 = kp * S // 8
    sync = "asm volatile(\"bar.sync 1, 128;\" ::: \"memory\");" if outers else "__syncthreads();"
    issue = []
    for j, p in enumerate(prods):
        issue.append(f"""#pragma unroll
        for (int u = 0; u < {per}; ++u) {{
          const int c = threadIdx.x + {T} * u, kk = c / {LC // 8}, l8 = (c % {LC // 8}) * 8, gl = chunk * {LC} + l8;
          if (c < {copies} && kb + kk < k1 && gl < {L})
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"((unsigned)__cvta_generic_to_shared(ring + {j * rb // 2} + (st % {NS}) * {R * LC} + kk * {LC} + l8)),
                         "l"({p['big']} + (long long)(kb + kk) * {big_k}LL + gl) : "memory");
        }}""")
    smalls = []
    for j, p in enumerate(prods):
        smalls.append(f"""    for (int c = threadIdx.x; c < {small16}; c += {T})
      if (k0 * {S} + c * 8 < k1 * {S})
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"((unsigned)__cvta_generic_to_shared(smalls + {j * sb // 2} + c * 8)),
                     "l"({p['small']} + (long long)k0 * {S}LL + c * 8) : "memory");""")
    store = "\n".join(f"      {'if' if j == 0 else 'else if'} (g == {j}) __stcg(skb_part{j} + (long long)part * {m * n}LL + o, acc);"
                      for j, p in enumerate(prods))
    arrive = "\n".join(f"    {'if' if j == 0 else 'else if'} (threadIdx.x == {j}) skb_last[{j}] = atomicAdd(skb_cnt{j} + chunk, 1u) == {parts - 1}u;"
                       for j, p in enumerate(prods))
    finish = "\n".join(f"""      {'if' if j == 0 else 'else if'} (g == {j}) {{
        float total = 0.0f;
        for (int q = 0; q < {parts}; ++q) total += __ldcg(skb_part{j} + (long long)q * {m * n}LL + o);
        {p['out']}[o] = __float2bfloat16_rn(total);
        if (lane == 0) skb_cnt{j}[chunk] = 0u;   // counters are restored to zero for later phases
      }}""" for j, p in enumerate(prods))
    names = ", ".join(p["name"] for p in prods)
    open_group, close_group = ("", "")
    if outers:
        open_group = ("  __syncthreads();                                 // earlier users of shared memory are done\n"
                      "  if (threadIdx.x < 128) {\n")
        close_group = outer_group(outers)
    # Each product's counter / partial pointer text appears exactly once (entry-zeroed counter markers are
    # resolved per occurrence by the interval emitter).
    decls = "\n".join(f"  unsigned* skb_cnt{j} = {p['cnt']};\n  float* skb_part{j} = {p['part']};" for j, p in enumerate(prods))
    return f"""{{ // Alt batched skinny split ({G} products: {names}): {L} x {S} outputs each, {parts} K parts of {kp}, {chunks} chunks of {LC}, {NS}x{R}-row rings
  __shared__ int skb_last[{G}];
{decls}
  __nv_bfloat16* smalls = reinterpret_cast<__nv_bfloat16*>(smem);                                // [{G}][{kp}][{S}]
  __nv_bfloat16* ring = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(smem) + {G * sb});  // [{G}][{NS}][{R}][{LC}]
  const int g = threadIdx.x / {lanes}, lane = threadIdx.x % {lanes}, sidx = lane / {LC}, lt = lane % {LC};
{open_group}  for (long long task = blockIdx.x; task < {chunks * parts}LL; task += gridDim.x) {{
    const int chunk = (int)(task / {parts}LL), part = (int)(task % {parts}LL);
    const int l = chunk * {LC} + lt;
    const int k0 = part * {kp}, k1 = min({k}, k0 + {kp});
    const int stages = (k1 - k0 + {R - 1}) / {R};
    auto issue = [&](int st) {{
      if (st < stages) {{
        const int kb = k0 + st * {R};
{chr(10).join(issue)}
      }}
      asm volatile("cp.async.commit_group;\\n" ::: "memory");
    }};
    {sync}                                 // shared memory of the previous task is free
{chr(10).join(smalls)}
#pragma unroll
    for (int st = 0; st < {NS - 1}; ++st) issue(st);
    float acc = 0.0f;
    for (int st = 0; st < stages; ++st) {{
      asm volatile("cp.async.wait_group {NS - 2};\\n" ::: "memory");
      {sync}
      issue(st + {NS - 1});
      if (g < {G}) {{
        const __nv_bfloat16* bk = ring + g * {rb // 2} + (st % {NS}) * {R * LC} + lt;
        const __nv_bfloat16* sk = smalls + g * {sb // 2} + st * {R * S} + sidx;
        const int n_k = min({R}, k1 - (k0 + st * {R}));
        if (n_k == {R}) {{
{chain_loop(R, LC, S, 10)}
        }} else {{
          for (int z = 0; z < n_k; ++z) acc = fmaf(__bfloat162float(bk[z * {LC}]), __bfloat162float(sk[z * {S}]), acc);
        }}
      }}
    }}
    asm volatile("cp.async.wait_group 0;\\n" ::: "memory");
    const long long o = (long long){row} * {n}LL + {col};
    if (g < {G} && l < {L}) {{
{store}
    }}
    __threadfence();
    {sync}
{arrive}
    {sync}
    if (g < {G} && skb_last[g < {G} ? g : 0]) {{
      __threadfence();
      if (l < {L}) {{
{finish}
      }} else if (lane == 0) {{
{chr(10).join(f"        {'if' if j == 0 else 'else if'} (g == {j}) skb_cnt{j}[chunk] = 0u;" for j, p in enumerate(prods))}
      }}
    }}
  }}
{close_group}}}"""


def outer_group(outers):
    """Warps 4-7 of every CTA: the paired skinny-K products (same values as ``outer``), B read through L1."""
    parts = ["  } else {                                         // warps 4-7: skinny-K products",
             "    const unsigned tb = blockIdx.x * 128u + (threadIdx.x - 128u), sb = gridDim.x * 128u;"]
    U = OUTER_UNROLL
    for o in outers:
        m, n, k, sam, sak, sbk, sbn = o["m"], o["n"], o["k"], o["sam"], o["sak"], o["sbk"], o["sbn"]
        n8, items = n // 8, m * (n // 8)
        A, B, O = o["a"], o["b"], o["out"]
        terms = "\n".join(
            f"          {{ const float bz[8] = {{{', '.join(f'__bfloat162float({B}[{z * sbk}LL + (long long)(c + {j}) * {sbn}LL])' for j in range(8))}}};\n"
            f"#pragma unroll\n            for (int j = 0; j < 8; ++j) acc[j] = fmaf(av[{z}], bz[j], acc[j]); }}" for z in range(k))
        parts.append(f"""    // {o['name']}: {m}x{n} = ({m}x{k})({k}x{n})
    if ((reinterpret_cast<unsigned long long>({O}) & 15ull) == 0ull) {{
      for (unsigned e0 = tb; e0 < {items}u; e0 += {U}u * sb) {{
        float avs[{U}][{k}];                            // A values of U items load before any output store
#pragma unroll
        for (int u = 0; u < {U}; ++u) {{
          const unsigned e = e0 + u * sb, row = (e < {items}u ? e : 0u) / {n8}u;
#pragma unroll
          for (int z = 0; z < {k}; ++z) avs[u][z] = __bfloat162float({A}[(long long)row * {sam}LL + z * {sak}LL]);
        }}
#pragma unroll
        for (int u = 0; u < {U}; ++u) {{
          const unsigned e = e0 + u * sb;
          if (e < {items}u) {{
        const unsigned row = e / {n8}u;
        const int c = (int)(e % {n8}u) * 8;
        const float* av = avs[u];
        float acc[8] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
{terms}
        uint4 v;
        __nv_bfloat162* hh = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) hh[j] = __halves2bfloat162(__float2bfloat16_rn(acc[2 * j]), __float2bfloat16_rn(acc[2 * j + 1]));
        *reinterpret_cast<uint4*>({O} + (long long)row * {n}LL + c) = v;
          }}
        }}
      }}
    }} else {{
      for (long long e = tb; e < {m * n}LL; e += sb) {{
        const long long row = e / {n}LL;
        const int col = (int)(e % {n}LL);
        float acc = 0.0f;
#pragma unroll
        for (int z = 0; z < {k}; ++z) acc = fmaf(__bfloat162float({A}[row * {sam}LL + z * {sak}LL]), __bfloat162float({B}[z * {sbk}LL + col * {sbn}LL]), acc);
        {O}[row * {n}LL + col] = __float2bfloat16_rn(acc);
      }}
    }}""")
    parts.append("  }\n  __syncthreads();                                   // both groups leave together")
    return "\n".join(parts) + "\n"
