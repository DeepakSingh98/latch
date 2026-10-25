"""Boundary decoding regression: reordered dependency phases are not clocks."""
from megakernel_lab.operation_compiler.runtime import dependency_profile_intervals


def test_dependency_interval_profile_retains_members_and_elapsed_time():
    manifest = {'phases': [{'kind': 'gemm', 'names': ['a']},
                           {'kind': 'elementwise', 'names': ['b']},
                           {'kind': 'reduction', 'names': ['c']}],
                'barrier_intervals': [[0, 2], [1]]}
    # Phase-index differences are +10,+20,-20,+27. Actual intervals are 10,20,7.
    stamps = [100, 110, 130, 110, 137]
    result = dependency_profile_intervals(manifest, stamps)
    assert [row['nanoseconds'] for row in result] == [10, 20, 7]
    assert result[0]['phases'] == [0, 2]
    assert result[0]['operation_kinds'] == ['gemm', 'reduction']
    assert sum(row['nanoseconds'] for row in result) == stamps[-1] - stamps[0]


def test_dependency_interval_profile_rejects_inconsistent_stamps():
    manifest = {'phases': [{'kind': 'gemm', 'names': ['a']},
                           {'kind': 'reduction', 'names': ['b']}],
                'barrier_intervals': [[0, 1]]}
    try:
        dependency_profile_intervals(manifest, [100, 110, 120, 130])
    except ValueError as error:
        assert 'inconsistent' in str(error)
    else:
        raise AssertionError('a fabricated per-body clock was accepted')
