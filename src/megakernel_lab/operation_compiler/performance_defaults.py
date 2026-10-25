"""Frozen selected performance options; diagnostics remain opt-in. No model-specific dispatch."""
P7B_SELECTED = {'matrix_residency': 2,
 'matrix_rule_overrides': {'SHARED_LIMIT': 97280,
                           'WGMMA_ALIGNED': True,
                           'WG_MIN_FLOPS': 268435456,
                           'WG_TILES': ((128, 64, 2), (64, 128, 1)),
                           'mixed_attention.BWD_FIT_LIMIT': 98304,
                           # P9-E amendment A1: long-K TMA products whose few tiles leave each CTA a serial K are
                           # split (existing deterministic split-K class); see research/latch_ops/p9_e
                           'P8E_LATENCY_SPLIT': True,
                           'pops2b_rules.SHARED_BUDGET': 97280},
 'p6b_chunk_multi_loops': True,
 'p6b_cse': True,
 'p6b_epilogue_full_grid_cost': 90,
 'p6b_epilogue_prefetch': True,
 'p6b_generic_unroll': True,
 'p6b_index32': True,
 'p6b_inline_div': True,
 'p6b_loop_balance': True,
 'p6b_loop_batch': 'light',
 'p6b_loop_batch_heavy': 32,
 'p6b_loop_batch_unrolled': True,
 'p6b_node_cse': True,
 'p6b_reduction_share': 2,
 'p6b_rotate_exclude': True,
 'p6b_rotate_grid': True,
 'p6b_row_chain': True,
 'p6b_row_registers': True,
 'p6b_scalar_store': True,
 'p6b_split_grid': 264,
 # P9-E amendment A3: split-K choice minimizes modeled steps plus FP32 slice traffic.
 'p6b_split_objective': 'p9e',
 # P9-E amendment A6: full-column FP32 sums of a staged TMA epilogue value accumulate in the epilogue (slab partials).
 'p9e_epi_colsum': True,
 # P9-E amendment A8: full column sums of [S, T, D] FP32 non-view values (D <= 512) as per-segment column sums inside
 # the producing P7-B segment, then a sum over segments (validated: gpu-41a learning, cpu-77/78/79 audits).
 'p9e_split_colsum': 512,
 # P9-E amendment A9: K-segmented products for sums of BF16 products (q/k/v input gradients), only when the fused
 # product fills the grid (validated: gpu-51a learning, cpu-96 audits; transfers keep their A8 programs).
 'p9e_ksum': True,
 # P9-E: P7-B reduction-phase deferral (schedule only; bitwise to the previous default, gpu-56a: −26 µs on dit_u).
 'p7b_defer_reductions': True,
 # P9-E: long mixed attention -- amendment A2 (fast softmax exponentials/reciprocal) and whole-item claimed backward
 # units (bitwise to the half/row-block units). Amendment A7 (flash forward, PMIX_FA_FWD) stays opt-in: it misses
 # the gradient diagnostic on trained states 200/1999 (see research/latch_ops/p9_e/RESULTS.md).
 'pops5d_defines': {'PMIX_FAST_SOFTMAX': 1, 'PMIX_LONG_V3_HALVES': 0},
 'p6b_splitk': True,
 'p6b_tree_merge': True,
 'p6b_tree_quads': True,
 'p6b_wg_epilogue': True,
 'p6b_wg_split': True,
 'p6b_wg_split_edge': True,
 'p7b_axis_order': True,
 'p7b_companions': True,
 'p7b_cone': True,
 'p7b_cone_recheck': True,
 'p7b_dynamic_items': True,
 'p7b_epi_affine': True,
 'p7b_heavy_batch': 32,
 'p7b_inline_bytes': True,
 'p7b_pass_prefetch': False,
 'p7b_priority': True,
 'p7b_segment': True,
 'p7b_segment_affine': True,
 'p7b_segment_index': True,
 'p7b_segment_parts': 0,
 'p7b_segment_vec': True,
 'p7b_small_epilogue': 131072,
 'p7b_split_generic': True,
 'p7b_staged_epilogue': True,
 'p7b_tma_persist': True,
 'p7b_wave_group': 528,
 'pops4a': True,
 'pops4a_items': 256,
 # P8-E (structural rules; see research/latch_ops/p8_e): one-element scalar chains of one element functor as one
 # multi-tensor loop; generic-fallback products with few tiles and long K split over K parts; declared hardware
 # tanh.approx.f32 for GELU(tanh) (amendment A1: qualified on states 0/8/40 + 2,000-update learning gate, cpu-28).
 'p8e_small_multi': True,
 'p8e_fast_tanh': True,
 'p8e_generic_split': True}
P6E_SELECTED = {'p6e_assembly': True,
 'p6e_early_ld': 1,
 'p6e_emb_pipe': True,
 'p6e_epilogue': True,
 'p6e_fast_act': True,
 'p6e_fin_batch': 16,
 'p6e_gather': True,
 'p6e_inline_div': True,
 'p6e_last_issue': True,
 'p6e_ld_first': True,
 'p6e_live_cap': 128,
 'p6e_quad': True,
 'p6e_scalar_chain': True,
 'p6e_scan_place': True,
 'p6e_scatter_vec': True,
 'p6e_softmax_rows': 4,
 'p6e_split': True,
 'p6e_trap': True,
 'pops3c': True,
 'pops4a': True,
 'pops5e_full': True,
 'pops5e_plan': {'stages': 4, 'tile': 8},
 'pops5e_store': True,
 'pops5e_stream': True,
 'scan_options': {'backward_group': 4, 'forward_unroll': 16},
 'warp_rows': True}
P6E_MATRIX = {'WGMMA': True,
 'WGMMA_ALIGNED': True,
 'WG_BODY': 'p6e',
 'WG_INLINE': True,
 'WG_SHARED_LIMIT': 200704,
 'WG_TMA': True,
 'WG_VEC_STORE': True}

P7A_SELECTED = {
    'pops2b_epi_cost': 40, 'p6a_grid': 132, 'p6a_row_batch': 2,
    'matrix_overrides': {'WGMMA_ALIGNED': True, 'WG_EPI_MAX_WTN': 64,
                         'WG_VEC_EPI_MAX_WTN': 128, 'WG_DYN_TILES': True},
    'p7_rowreg': True, 'p7_row_virtual': True, 'p7_staged_epi': True, 'p7_flat_loops': True,
    'p7_const_args': True, 'p7_scalar_loads': True, 'p7_row_rb_cap': 1,
    'p7_vec_epi': True, 'p7_vec_rb': 4, 'p7_row_prefetch': True,
    'p7_scalar_pin': True, 'p7_quad_sel': True, 'p7_vec_raw': True,
    'pops4a_chunk': 2048, 'p7_trans_store': 20,
}

# Exact scheduling improvements remain automatic; approximate math and changed rounding
# need the public precision_policy="practical" declaration. Preserve the earlier strict recipe.
P7B_STRICT_SELECTED = {'matrix_residency': 2, 'matrix_rule_overrides': {'SHARED_LIMIT': 97280, 'WGMMA_ALIGNED': True, 'WG_MIN_FLOPS': 268435456, 'WG_TILES': ((128, 64, 2), (64, 128, 1)), 'mixed_attention.BWD_FIT_LIMIT': 98304, 'pops2b_rules.SHARED_BUDGET': 97280}, 'p6b_chunk_multi_loops': True, 'p6b_cse': True, 'p6b_epilogue_full_grid_cost': 90, 'p6b_epilogue_prefetch': True, 'p6b_generic_unroll': True, 'p6b_index32': True, 'p6b_inline_div': True, 'p6b_loop_balance': True, 'p6b_loop_batch': 'light', 'p6b_loop_batch_heavy': 32, 'p6b_loop_batch_unrolled': True, 'p6b_node_cse': True, 'p6b_reduction_share': 2, 'p6b_rotate_exclude': True, 'p6b_rotate_grid': True, 'p6b_row_chain': True, 'p6b_row_registers': True, 'p6b_scalar_store': True, 'p6b_split_grid': 264, 'p6b_split_objective': 'steps', 'p6b_splitk': True, 'p6b_tree_merge': True, 'p6b_tree_quads': True, 'p6b_wg_epilogue': True, 'p6b_wg_split': True, 'p6b_wg_split_edge': True, 'p7b_axis_order': True, 'p7b_companions': True, 'p7b_cone': True, 'p7b_cone_recheck': True, 'p7b_dynamic_items': True, 'p7b_epi_affine': True, 'p7b_heavy_batch': 32, 'p7b_inline_bytes': True, 'p7b_pass_prefetch': False, 'p7b_priority': True, 'p7b_segment': True, 'p7b_segment_affine': True, 'p7b_segment_index': True, 'p7b_segment_parts': 0, 'p7b_segment_vec': True, 'p7b_small_epilogue': 131072, 'p7b_split_generic': True, 'p7b_staged_epilogue': True, 'p7b_tma_persist': True, 'p7b_wave_group': 528, 'pops4a': True, 'pops4a_items': 256, 'p8e_small_multi': True, 'p7b_defer_reductions': True, 'pops5d_defines': {'PMIX_LONG_V3_HALVES': 0}}
