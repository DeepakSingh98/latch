"""Source hygiene of the compiler package and its tests (CPU; reads files only, imports nothing from the package).

1. Every environment variable the compiler package reads is one the release already read, so new behaviour is
   selected by explicit build options rather than by environment switches.
2. Tests import nothing from a ``research`` tree and build no paths into one.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
PACKAGE = SRC / "megakernel_lab" / "operation_compiler"
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}

# Environment variables the release compiler package reads (string literals, or module-level string constants).
RELEASE_ENVIRONMENT = frozenset({
    "CXX", "LATCH_ARENA_ASSIGN_CHECK", "LATCH_BINDING_GUARD_CACHE", "LATCH_CLAIM_HOT", "LATCH_CLAIM_MERGE",
    "LATCH_CLAIM_MIN_ROWS", "LATCH_CLAIM_WIDE", "LATCH_DIAGNOSTIC_CTRACE", "LATCH_DIAGNOSTIC_FSB", "LATCH_MERGE_CHECK",
    "LATCH_OPS_BATCHED_TMA_EPILOGUES", "LATCH_OPS_BUILD_CACHE", "LATCH_OPS_GATHER_CASTS", "LATCH_OPS_GATHER_PRODUCTS",
    "LATCH_OPS_GRADIENT_INPUT_PLACEMENT", "LATCH_OPS_INPLACE_CAT_PIECES", "LATCH_OPS_K8_SERIAL_CHAINS",
    "LATCH_OPS_K8_SPLIT_INTERVALS", "LATCH_OPS_LN_ROWS_PER_HALF", "LATCH_OPS_PEER_LIBRARY_MANIFEST",
    "LATCH_OPS_PEER_REDISTRIBUTE", "LATCH_OPS_PERMUTATION_GRADIENTS", "LATCH_OPS_ROW_SCATTER_MM",
    "LATCH_OPS_TMA_PAIR_EPILOGUES", "LATCH_OPS_TRANSPORT", "LATCH_RAGGED_FLAT_TILE", "LATCH_SCREEN_DEFINES",
    "LATCH_SCREEN_RULES", "LATCH_STEP17_ASSUME_GLOBAL", "LATCH_STEP17_BUILD", "LATCH_STEP17_INTERNALIZE",
    "LATCH_STEP17_JOBS", "LATCH_STEP17_OUTLINE", "LATCH_STEP17_OUTLINE_MIN", "LATCH_STEP17_PARTITION_MODE",
    "LATCH_STEP17_UNITS", "LATCH_STEP17_UNIT_FLAGS", "P8C_VCOPY_KINDS", "P9C_BEPI", "P9C_BEPI_MODE", "P9C_COLSPLIT",
    "P9C_EMB", "P9C_HEAVYEPI", "P9C_JOINTB", "P9C_PELEM", "P9C_PRED", "P9C_REBATCH", "P9C_RESOFT",
    "P9C_RULES_EXTRA", "P9C_SHARECAST",
})

_IMPORTERS = {"import_module", "__import__", "spec_from_file_location", "run_path", "SourceFileLoader",
              "load_source"}
# Calls that build or open a path from string pieces: Path("research/x"), root.joinpath("research", ...),
# os.path.join(root, "research"), open("research/x").
_PATH_BUILDERS = {"Path", "PurePath", "PosixPath", "joinpath", "join", "open"}
_RESEARCH_PATH = re.compile(r"(^|[/\\.])research([/\\.]|$)")


def _files(top, suffixes=None):
    for directory, subdirs, names in os.walk(top):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS)
        for name in sorted(names):
            path = Path(directory) / name
            if name.endswith((".pyc", ".pyo")) or (suffixes and path.suffix not in suffixes):
                continue
            yield path


def _rel(path):
    return path.relative_to(ROOT).as_posix()


def _research_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [(node.lineno, a.name) for a in node.names if a.name.split(".")[0] == "research"]
        elif isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == "research":
            hits.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _IMPORTERS:
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    for sub in ast.walk(arg):
                        if (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                                and _RESEARCH_PATH.search(sub.value)):
                            hits.append((node.lineno, sub.value))
            elif name in _PATH_BUILDERS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and \
                            arg.value.strip("/").split("/")[0] == "research":
                        hits.append((node.lineno, arg.value))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, str) and \
                        side.value.strip("/").split("/")[0] == "research":
                    hits.append((node.lineno, side.value))
    return hits


def test_tests_do_not_import_research():
    hits = [f"{_rel(path)}:{line}: {what}" for path in _files(TESTS, {".py"}) for line, what in _research_imports(path)]
    assert not hits, "research imports in tests:\n" + "\n".join(hits)


def _environment_keys(path):
    """(literal keys, unresolved key expressions) of os.environ/getenv reads in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                 for target in node.targets if isinstance(target, ast.Name)}

    def environ(node):
        return (isinstance(node, ast.Attribute) and node.attr == "environ") or \
            (isinstance(node, ast.Name) and node.id == "environ")

    keys, unresolved = set(), []

    def key(node, where):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            keys.add(node.value)
        elif isinstance(node, ast.Name) and node.id in constants:
            keys.add(constants[node.id])
        else:
            unresolved.append(f"{_rel(path)}:{where.lineno}: {ast.unparse(where)[:100]}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            reads = (isinstance(func, ast.Attribute) and (func.attr == "getenv" or (
                environ(func.value) and func.attr in ("get", "setdefault", "pop")))) or \
                (isinstance(func, ast.Name) and func.id == "getenv")
            if reads and node.args:
                key(node.args[0], node)
        elif isinstance(node, ast.Subscript) and environ(node.value):
            key(node.slice, node)
        elif isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) \
                and any(environ(c) for c in node.comparators):
            key(node.left, node)
    return keys, unresolved


def test_environment_reads_are_release_names():
    keys, unresolved, where = set(), [], {}
    for path in _files(PACKAGE, {".py"}):
        found, missing = _environment_keys(path)
        keys |= found
        unresolved += missing
        for name in found:
            where.setdefault(name, []).append(_rel(path))
    assert not unresolved, "environment reads with keys that are not string constants:\n" + "\n".join(unresolved)
    extra = sorted(keys - RELEASE_ENVIRONMENT)
    assert not extra, "environment variables the release did not read: " + \
        ", ".join(f"{name} ({', '.join(where[name])})" for name in extra)


@pytest.mark.parametrize("source, hits", [
    ("import research.x\n", 1),
    ("from research.tools import probe\n", 1),
    ("import importlib\nimportlib.import_module('research.study.mod')\n", 1),
    ("from pathlib import Path\np = Path('.') / 'research' / 'study'\n", 1),
    ("import importlib.util as u\nu.spec_from_file_location('m', 'research/study/mod.py')\n", 1),
    ("names = ('research', 'ops')\n# research/study/notes.md\n", 0),
    ("from . import research_helpers\n", 0),
    ("from pathlib import Path\np = Path('research/study/out.json')\n", 1),
    ("from pathlib import Path\np = Path('.').joinpath('research', 'study')\n", 1),
    ("import os\np = os.path.join('.', 'research', 'study')\n", 1),
    ("data = open('research/study/out.json').read()\n", 1),
    ("import os\np = os.path.join('.', 'docs', 'research_notes.md')\nq = ', '.join(['research'])\n", 0),
])
def test_research_import_detector(tmp_path, source, hits):
    path = tmp_path / "case.py"
    path.write_text(source)
    assert len(_research_imports(path)) == hits
