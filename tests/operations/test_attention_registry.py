"""Late attention refusal must not retain an un-emitted shared-mask builder."""
from types import SimpleNamespace


def test_retry_preserves_prior_regions_and_discards_failed_builder():
    from megakernel_lab.operation_compiler import long_attention as state
    from megakernel_lab.operation_compiler.attention_lower import AttentionRefused, excluded_groups, retry_lowering
    key = ('readonly-mask', (0, 256, 1), 256)
    attempts = []

    @retry_lowering
    def emit():
        attempts.append(tuple(excluded_groups()))
        if not excluded_groups():
            state.mask_slot('refused', 'forward')
            _, builder = state.mask_table(key, 'forward', 'refused')
            assert builder
            raise AttentionRefused('refused', 'late lowered-layout refusal')
        assert (None, 'prior-region', 'forward') in state._SLOTS
        assert (None, 'refused', 'forward') not in state._SLOTS
        assert (None, key) not in state._MASK_SLOTS
        _, builder = state.mask_table(key, 'forward', 'surviving')
        assert builder, 'retry would read a table whose builder was discarded'
        return SimpleNamespace(manifest={})

    with state.program_state():
        prior = state.mask_slot('prior-region', 'forward')
        result = emit()
        assert state.mask_slot('prior-region', 'forward') == prior
        assert state._MASK_SLOTS[(None, key)][1] == 'surviving'
        assert len(state._SLOTS) == 1
        assert result.manifest['attention_refusals'][0]['group'] == 'refused'
    assert attempts == [(), ('refused',)]


def test_nonretry_failure_rolls_back_only_its_allocations():
    from megakernel_lab.operation_compiler import long_attention as state
    with state.program_state():
        prior = state.mask_slot('prior-region', 'forward')
        try:
            with state.allocation_transaction():
                state.mask_slot('failed-region', 'forward')
                raise ValueError('ordinary lowering failure')
        except ValueError:
            pass
        assert state._SLOTS == {(None, 'prior-region', 'forward'): prior}
        assert state._MASK_SLOTS == {}
        with state.allocation_transaction():
            state.mask_slot('successful-region', 'forward')
        assert (None, 'successful-region', 'forward') in state._SLOTS


def test_distinct_regions_with_same_local_names_do_not_share():
    from megakernel_lab.operation_compiler import long_attention as state
    key = ('arg10_1', (0, 256, 1), 256)
    histories = []
    for _ in range(2):
        with state.program_state():
            rows = []
            for _ in range(2):
                with state.region_namespace() as namespace:
                    phase = state.mask_slot('same-local-label', 'forward')
                    mask, builder = state.mask_table(key, 'forward', 'same-local-label')
                    assert state.mask_slot('same-local-label', 'forward') == phase
                    assert state.mask_table(key, 'forward', 'same-local-label') == (mask, True)
                    assert builder
                    rows.append((namespace, phase, mask))
            assert rows[0][0] != rows[1][0]
            assert len({rows[0][1], rows[0][2], rows[1][1], rows[1][2]}) == 4
            histories.append(rows)
    assert histories[0] == histories[1], 'program-local namespace order must be deterministic'


def test_retry_keeps_its_namespace_and_restores_nested_counter():
    from megakernel_lab.operation_compiler import long_attention as state
    with state.program_state(), state.region_namespace() as outer:
        before = state._NEXT_REGION_NAMESPACE
        try:
            with state.allocation_transaction(), state.region_namespace():
                state.mask_slot('discarded-child', 'forward')
                raise ValueError('discard')
        except ValueError:
            pass
        assert state._REGION_NAMESPACE == outer
        assert state._NEXT_REGION_NAMESPACE == before
        with state.region_namespace() as child:
            assert child == before


def run():
    tests = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            tests[name] = True
    return {'passed': True, 'tests': tests, 'scope': 'provisional compiler registry metadata only'}
