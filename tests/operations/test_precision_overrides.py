"""Metadata-only guards for effective strict/practical compiler options."""
from megakernel_lab.operation_compiler.lower import LoweringError, _validate_flat_precision_policy


def _refused(call, field):
    try:
        call()
    except LoweringError as error:
        assert "precision_policy='practical'" in str(error) and field in str(error)
    else:
        raise AssertionError('strict arithmetic override was admitted: ' + field)


def test_both_public_override_maps_enforce_strict_before_lowering():
    from megakernel_lab.operation_compiler.performance_schedule import lower_selected
    for key in ('matrix_rule_overrides', 'matrix_overrides'):
        for nested in (True, False):
            for schedule, overrides, field in (
                ('p7b', {'mixed_attention.FA_DELTA': True}, 'mixed_attention.FA_DELTA'),
                ('p6d', {'flat_long_attention.DEFAULT_RECIPE': 'bf16'}, 'DEFAULT_RECIPE'),
                ('p6d', {'flat_long_attention.DEFAULT_RECIPE': 'split2'}, 'DEFAULT_RECIPE'),
                ('p6d', {'flat_long_attention.RECIPES': {'exact3': (1, False)}}, 'RECIPES[exact3]')):
                options = {key: overrides}
                kwargs = {'schedule_options': options} if nested else options
                # None cannot lower as a graph. The policy-specific exception
                # proves refusal occurs before any graph lowering or mutation.
                _refused(lambda: lower_selected(None, schedule=schedule, **kwargs), field)


def test_strict_recipe_dictionary_cannot_change_piece_or_output_precision():
    for exact in ((2, True), (3, False), None):
        recipes = {} if exact is None else {'exact3': exact}
        _refused(lambda: _validate_flat_precision_policy('strict', {},
                 {'flat_long_attention.RECIPES': recipes}), 'RECIPES[exact3]')


def test_direct_arithmetic_flags_still_require_practical():
    for key in ('p8e_fast_tanh', 'p9e_ksum', 'p6e_fast_act'):
        _refused(lambda: _validate_flat_precision_policy('strict', {key: True}, {}), key)
    for key in ('PMIX_FAST_SOFTMAX', 'PMIX_FA_FWD', 'PMIX_FA_DELTA'):
        _refused(lambda: _validate_flat_precision_policy('strict', {'pops5d_defines': {key: 1}}, {}), key)
    for recipe in ('bf16', 'split2'):
        _refused(lambda: _validate_flat_precision_policy('strict', {'pops4d_recipe': recipe}, {}),
                 'pops4d_recipe')


def test_practical_keeps_explicit_arithmetic_controls():
    _validate_flat_precision_policy('practical',
        {'p8e_fast_tanh': True, 'p9e_ksum': True, 'p6e_fast_act': True,
         'pops4d_recipe': 'bf16', 'pops5d_defines': {'PMIX_FA_DELTA': 1}},
        {'mixed_attention.FA_DELTA': True, 'flat_long_attention.DEFAULT_RECIPE': 'split2',
         'flat_long_attention.RECIPES': {'exact3': (1, False)}})


def test_effective_exact_settings_and_schedule_only_overrides_remain_valid():
    _validate_flat_precision_policy('strict', {'pops4d_recipe': 'exact3'},
        {'mixed_attention.FA_DELTA': False, 'flat_long_attention.DEFAULT_RECIPE': 'bf16',
         'flat_long_attention.RECIPES': {'exact3': (3, True)},
         'p5c_matrix.P5C_MAX_STAGES': 6, 'SHARED_LIMIT': 97280})


def test_rejected_overrides_leave_module_state_unchanged():
    from megakernel_lab.operation_compiler import flat_long_attention, mixed_attention
    before = (mixed_attention.FA_DELTA, flat_long_attention.DEFAULT_RECIPE,
              dict(flat_long_attention.RECIPES))
    _refused(lambda: _validate_flat_precision_policy('strict', {},
             {'mixed_attention.FA_DELTA': True}), 'mixed_attention.FA_DELTA')
    assert before == (mixed_attention.FA_DELTA, flat_long_attention.DEFAULT_RECIPE,
                      flat_long_attention.RECIPES)


def run():
    tests = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            tests[name] = True
    return {'passed': True, 'tests': tests, 'scope': 'precision option metadata; no tensor computation'}
