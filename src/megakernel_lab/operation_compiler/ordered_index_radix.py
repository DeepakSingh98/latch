"""Integer-only occurrence construction for the ordered embedding reducer.

Reuse the existing stable radix sorter from indexed_segmented, but none of its
floating-point partial reductions. The final output is the same permutation of
ordinals as sorting unique (row, ordinal) keys. Padding stays in its original row:
the unchanged consumer is responsible for its zero result.
"""
from __future__ import annotations

ENABLED = True
# Research admission until paired device evidence chooses this integer schedule.
LINEAR_BOUNDS = False


def supported(n, rows):
    # Bound per-thread key registers in the existing 128-tile radix body. Small
    # lists retain the low-scratch merge implementation.
    return ENABLED and 4096 <= n <= 262144 and 0 < rows < (1 << 31)


def emit(name, ids, n, rows, id_at, ident, allocate, *, bounds=None):
    from .indexed_segmented import radix_plan, _radix_phases

    plan = radix_plan(n, rows, -1)
    hist_words = (plan["passes"] * plan["tiles"] * plan["bins"] + 1) // 2
    index = allocate("ordered_radix_index", (rows + hist_words,))

    def typed(suffix, shape, dtype):
        if dtype != "int64":
            raise ValueError("ordered radix scratch must be int64")
        return allocate("ordered_" + suffix, shape)

    phases, keys = _radix_phases(index, ids, n, rows, -1, plan, ident, id_at, typed)
    if bounds is not None:
        # Initialize absent rows while the first radix phase is already live.
        # Present rows receive their unique first/end positions below. Integer
        # zero/zero describes the identical empty FP32 loop for absent rows.
        phases[0]["writes"].append(bounds)
        phases[0]["code"] = f"""
for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
     i < {2 * rows}LL; i += (long long)gridDim.x * blockDim.x)
  {ident(bounds)}[i] = 0;
""" + phases[0]["code"]
    ordered = allocate("ordinal_radix", (n,))
    scatter = ""
    if bounds is not None:
        scatter = f"""
  const unsigned long long* keys_ = reinterpret_cast<const unsigned long long*>({ident(keys)});
  const unsigned int row_ = (unsigned int)(keys_[i] >> 32);
  if (i == 0 || (unsigned int)(keys_[i - 1] >> 32) != row_)
    {ident(bounds)}[2LL * row_] = i;
  if (i + 1 == {n}LL || (unsigned int)(keys_[i + 1] >> 32) != row_)
    {ident(bounds)}[2LL * row_ + 1] = i + 1;
"""
    # As in indexed_segmented's bound scatter, this phase's ordinary output
    # supplies readiness for the side effect. Every consumer reads both ordered
    # and bounds, keeping the initialized allocation live through this phase.
    phases.append(dict(kind="index_radix_ordinals", writes=[ordered],
                       reads=[keys] + ([bounds] if bounds is not None else []), code=f"""
for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
     i < {n}LL; i += (long long)gridDim.x * blockDim.x) {{
  {ident(ordered)}[i] = (long long)(unsigned int)
      reinterpret_cast<const unsigned long long*>({ident(keys)})[i];
{scatter}
}}
""", index_sort="stable_lsd_row_digits", ordinal_order="ascending within each row",
                       row_bounds="sorted_neighbor_scatter" if bounds is not None else "separate_binary_search"))
    return phases, ordered, plan, rows + hist_words
