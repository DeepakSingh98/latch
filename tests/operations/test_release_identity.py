"""With every newer option unset, release-operation programs lower to the release's program.cu (CPU; no CUDA).

The opt-in rule families and tokens added since the release are selected only by their options; the release
behaviour they replace when selected is restored when they are absent. ``release_identity_cases.py`` defines the
programs and the release scope stack; the recorded hashes come from running it against the release tree.

The hashes were recorded with torch 2.9.1+cu128; the check accepts any 2.9.1 build (+cu128, +cpu or a plain wheel). With another
torch release the test skips, which means the release programs were not checked; set
LATCH_REQUIRE_RELEASE_IDENTITY=1 to make that a failure. Environment variables that can change lowering are cleared
for the test.

One default-path change is deliberate: phase ordering in ``group_rules._topological`` and ``row_fusion`` now keeps
read-after-write, write-after-read and write-after-write order on roots that are written more than once or mutated
in place. Programs whose values are each written once order exactly as before; ``mutable_roots`` and the full-sum
scratch values of ``copies`` and ``shared_casts`` check that the resulting programs are unchanged.
"""
from __future__ import annotations

import os

import pytest
import torch

from .release_identity_cases import (CASES, FIXTURE, RELEASE_COMMIT, REQUIRE_VARIABLE, SCHEMA, load_hashes,
                                     lowering_environment, program_hash, torch_base)

REGENERATE = ("git worktree add <dir> " + RELEASE_COMMIT + " && PYTHONPATH=<dir>/src python "
              "tests/operations/release_identity_cases.py --write tests/operations/fixtures/release_program_hashes.json")


@pytest.fixture(scope="module")
def recorded():
    record = load_hashes(FIXTURE)
    assert record.get("schema") == SCHEMA and record.get("release") == RELEASE_COMMIT
    missing = sorted(set(CASES) - set(record.get("cases") or {}))
    if missing:
        pytest.fail(f"release program hashes not recorded for {missing}; regenerate with: {REGENERATE}")
    if torch_base(record.get("torch")) != torch_base(torch.__version__):
        message = (f"hashes were recorded with torch {record.get('torch')}, not {torch.__version__}; captured graphs "
                   f"can differ between torch releases, so the release programs were not checked. Install torch "
                   f"{torch_base(record.get('torch'))} or regenerate with: {REGENERATE}")
        if os.environ.get(REQUIRE_VARIABLE) == "1":
            pytest.fail(message)
        pytest.skip(message)
    return record["cases"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_release_programs_are_unchanged_without_new_options(recorded, name, monkeypatch):
    cleared = lowering_environment()
    for variable in cleared:
        monkeypatch.delenv(variable)
    torch.set_num_threads(1)
    note = (f" (these variables were set and were cleared for lowering; a value read when a module was first "
            f"imported still applies: {', '.join(cleared)})" if cleared else "")
    assert program_hash(name) == recorded[name], f"{name}: program.cu differs from release {RELEASE_COMMIT}{note}"


def test_torch_base_ignores_the_local_build_tag():
    assert torch_base("2.9.1+cu128") == torch_base("2.9.1+cpu") == torch_base("2.9.1") == "2.9.1"
    assert torch_base("2.10.0+cu128") != torch_base("2.9.1+cu128")


def test_lowering_environment_lists_compiler_variables_only():
    environ = {"LATCH_CLAIM_HOT": "1", "P9C_BEPI": "0", "P8C_VCOPY_KINDS": "cat", "CXX": "g++", "PATH": "/bin",
               REQUIRE_VARIABLE: "1"}
    assert lowering_environment(environ) == ["LATCH_CLAIM_HOT", "P8C_VCOPY_KINDS", "P9C_BEPI"]
