"""LATCH-EXPLAIN: a compilation report over existing Latch manifests, plans and profiles.

Entry points (all CPU-only; nothing here launches GPU work):

* ``report_program(program, ir=None, diagnostics=None, profile=None)`` -- report for a lowered
  ``GeneratedProgram`` (``lower(...)``), joined with the captured IR (operation targets, roles,
  optional source locations), collector records and an optional measured profile.
* ``report_artifact(directory, profile=None)`` -- same from a written artifact directory
  (``manifest.json`` + ``program.cu`` [+ ``artifact.json``, ``explain/ir.json``]).
* ``report_manifest(manifest, source=None, ...)`` -- from an in-memory manifest dict.
* ``report_moe4(artifact_json, audit_json, measurement_json=None)`` -- bounded adapter for the
  four-GPU research assembly (``latch-distributed-update-v1`` + assembly audit); read-only.
* ``collect()`` -- context manager that gathers diagnostics the compiler otherwise discards.
* ``render_markdown(report)`` -- readable text; the report dict itself is the structured output.

Every number in a report carries a label: ``measured`` (from a supplied profile or receipt),
``static`` (a count from the plan/manifest), ``estimate`` (the compiler's own planning model),
or ``unknown``. Without a profile, costs are reported as unmeasured.
"""
from .collect import collect, note  # noqa: F401
from .report import (REPORT_SCHEMA, report_artifact, report_manifest, report_program,  # noqa: F401
                     write_report)
from .render import render_markdown  # noqa: F401
from .moe4 import report_moe4  # noqa: F401
