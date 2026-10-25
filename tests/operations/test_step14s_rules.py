"""CPU contracts for the step-14 speed flat common-planner rules (no GPU, no nvcc).

* xdiv: FP32 epilogue lines rewrite IEEE divisions / __frcp_rn to the straight-line fast paths with the chunk flag;
  index arithmetic, call arguments and non-FP32 lines are untouched.
* Queue-aware FP32 edge planning (ectaq): the list-scheduling bound picks finer K splits for few long tasks and keeps
  the whole-grid choice where tasks are already short; without the queue the established plan is unchanged.
* Defaults: STEP14S_RULES adds +ectaq+xdiv to the step-14 flat rules; STEP14S_RULES=False reproduces step 14.
* FP32 idle placement model (ectac arm): exact idle slice vs whole grid by estimated makespan.
"""
from __future__ import annotations

from megakernel_lab.operation_compiler import compose, group_rules, matrix_rules


def test_xdiv_rewrites_float_divisions_only():
    lines = ["const float r_silu = (r_x / (1.0f + expf(-r_x))); s80_[j] = r_silu;",
             "const float r_sig = __frcp_rn((1.0f + expf(-r_y)));",
             "const float r_d = (r_a / 33.5f);",
             "const float r_y = (v_a[(i / 1120LL) % 40LL] / r_z);",
             "const float r_w = fmaxf(r_a / r_b, 0.0f);",
             "const __nv_bfloat16 r_q = __float2bfloat16_rn((r_m / r_n));"]
    out, changed = group_rules.xdiv_lines(lines)
    assert changed
    assert out[0] == "const float r_silu = latch_div_f(ok_, r_x, (1.0f + expf(-r_x))); s80_[j] = r_silu;"
    assert out[1] == "const float r_sig = latch_rcp_f(ok_, (1.0f + expf(-r_y)));"
    assert out[2] == "const float r_d = latch_div_f(ok_, r_a, 33.5f);"
    assert out[3] == lines[3] and out[4] == lines[4] and out[5] == lines[5]


def test_xdiv_nested_divisions():
    out, changed = group_rules.xdiv_lines(["const float r_t = ((r_a / r_b) / (r_c / r_d));"])
    assert changed
    assert out[0] == "const float r_t = latch_div_f(ok_, latch_div_f(ok_, r_a, r_b), latch_div_f(ok_, r_c, r_d));"


def test_queue_planning_prefers_short_tasks():
    args = (40, 40, 1120, 27, True, True)
    plain, _ = matrix_rules.fp32_edge_plan(*args)
    token = matrix_rules.FP32_EDGE_QUEUE.set(True)
    try:
        queued, _ = matrix_rules.fp32_edge_plan(*args)
        backward, _ = matrix_rules.fp32_edge_plan(40, 1120, 40, 27, False, False)
    finally:
        matrix_rules.FP32_EDGE_QUEUE.reset(token)
    assert plain['split'] == 4 and queued['split'] == 6
    task = lambda c: c['local_us'] / c['waves']
    assert task(queued) < task(plain)
    # Short-task products keep the whole-grid choice.
    assert (backward['bm'], backward['bn'], backward['split']) == (48, 128, 1)
    # Every K split still covers K exactly.
    assert (queued['split'] - 1) * queued['ks'] < 1120 <= queued['split'] * queued['ks']


def test_defaults_and_switch():
    assert compose.STEP14S_FEATURES == "+ectaq+xdiv"
    previous = compose.STEP14S_RULES
    try:
        compose.STEP14S_RULES = True
        assert compose.STEP14_FLAT_RULES().endswith("+ectaq+xdiv")
        compose.STEP14S_RULES = False
        assert compose.STEP14_FLAT_RULES() == compose.STEP14_FEATURES
    finally:
        compose.STEP14S_RULES = previous
    for feature in ("ectaq", "xdiv", "ectac", "ectaw", "ectai"):
        assert feature in group_rules.FEATURES


def test_idle_placement_model():
    def phase(engine, tiles, local, waves, grid, code=''):
        return {'kind': 'gemm', 'engine': engine, 'tiles': tiles, 'names': [engine + str(tiles)], 'code': code,
                'matrix_plan': {'local_us': local, 'waves': waves, 'grid': grid}}
    # 117 pinned TMA tiles of 132; 486 FP32 tasks: whole grid wins.
    phases = [phase('latch_matrix:pops4b_tma', 36, 3.5, 1, 132, '/*B_VBASE*/'),
              phase('latch_matrix:pops4b_tma', 81, 3.1, 1, 132, '/*B_VBASE*/'),
              phase('popsc_fp32_48x128', 243, 6.6, 2, 128), phase('popsc_fp32_48x128', 243, 6.6, 2, 128)]
    place, record = group_rules.fp32_idle_placement(phases, [0, 1, 2, 3], 117)
    assert record['mode'] == 'whole' and set(place) == {2, 3}
    assert place[3][1] - place[2][1] == 243          # consecutive products continue the rotation
    # Few tasks and many idle CTAs: the idle slice wins.
    phases = [phase('latch_matrix:pops4b_tma', 40, 10.0, 1, 132, '/*B_VBASE*/'), phase('popsc_fp32_48x48', 20, 4.0, 1, 128)]
    _, record = group_rules.fp32_idle_placement(phases, [0, 1], 40)
    assert record['mode'] == 'idle'
