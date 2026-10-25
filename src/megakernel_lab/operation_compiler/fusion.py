"""Public flat operation scheduler; frozen protocols share one structural entry point.

The reference dependency scheduler remains available. Selected P6/P7 schedulers
consume the same operation IR and own an entire effect schedule. See
performance_schedule.py and PERFORMANCE_DONORS.json for admission and attribution.
"""
from . import fusion_reference as _reference
from .performance_schedule import lower_selected, select_schedule


def lower_fused(graph, *, schedule="auto", schedule_options=None, compact_codegen="off", **options):
    return lower_selected(graph, schedule=schedule, schedule_options=schedule_options,
                          compact_codegen=compact_codegen, **options)


def __getattr__(name):
    # Preserve the existing pure helper imports used by diagnostics/tests.
    return getattr(_reference, name)
