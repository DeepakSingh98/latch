"""Pure regressions for nested quoted includes of interval safety headers."""
import hashlib
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from megakernel_lab.operation_compiler import step17_intervals as SI
from megakernel_lab.operation_compiler import step17_build as BUILD


def resolved_headers(entry, original, copies):
    """Model quoted-include directory precedence, then the build -I directory."""
    seen, result = set(), []

    def visit(name, current):
        directory = current if name in (copies if current == "units" else original) else "build"
        text = (copies if directory == "units" else original)[name]
        key = directory, name
        if key in seen:
            return
        seen.add(key)
        result.append((directory, name, text))
        for child in re.findall(r'#include "([\w.]+\.cuh)"', text):
            visit(child, directory)

    visit(entry, "units")
    return result


class HeaderClosureTests(unittest.TestCase):
    def test_nested_include_cannot_shadow_transformed_leaf(self):
        headers = {"entry.cuh": '#include "middle.cuh"\n',
                   "middle.cuh": '#include "leaf.cuh"\n', "leaf.cuh": "original\n"}
        replacement = {"leaf.cuh": "protected\n"}
        # The old staging behavior demonstrably resolves the original leaf.
        self.assertEqual(resolved_headers("entry.cuh", headers, replacement)[-1],
                         ("build", "leaf.cuh", "original\n"))
        copies = SI._unit_header_copies(headers, replacement)
        self.assertEqual(resolved_headers("entry.cuh", headers, copies)[-1],
                         ("units", "leaf.cuh", "protected\n"))
        self.assertEqual(copies["entry.cuh"], headers["entry.cuh"])
        self.assertEqual(copies["middle.cuh"], headers["middle.cuh"])

    def test_diamond_paths_and_cycles_copy_every_ancestor_once(self):
        headers = {"entry.cuh": '#include "a.cuh"\n#include "b.cuh"\n',
                   "a.cuh": '#include "cycle.cuh"\n#include "leaf.cuh"\n',
                   "b.cuh": '#include "leaf.cuh"\n', "cycle.cuh": '#include "a.cuh"\n',
                   "leaf.cuh": "original", "unrelated.cuh": "unrelated"}
        copies = SI._unit_header_copies(headers, {"leaf.cuh": "protected"})
        self.assertEqual(set(copies), set(headers) - {"unrelated.cuh"})
        self.assertTrue(all(directory == "units" for directory, _, _ in
                            resolved_headers("entry.cuh", headers, copies)))
        self.assertEqual(headers["leaf.cuh"], "original")

    def test_no_transformation_does_not_stage_headers(self):
        headers = {"a.cuh": '#include "b.cuh"\n', "b.cuh": "original"}
        self.assertEqual(SI._unit_header_copies(headers, {}), {})

    def test_unknown_includes_are_not_discovered_or_copied(self):
        headers = {"entry.cuh": '#include "outside.cuh"\n#include "leaf.cuh"\n',
                   "leaf.cuh": "original"}
        copies = SI._unit_header_copies(headers, {"leaf.cuh": "protected"})
        self.assertEqual(set(copies), {"entry.cuh", "leaf.cuh"})

    def test_segmented_wrapper_resolves_exact_existing_safety_transformation(self):
        kernels = Path(SI.__file__).parent / "kernels"
        names = ("attention_segmented.cuh", "attention_long_flat.cuh")
        headers = {name: (kernels / name).read_text() for name in names}
        with patch.object(SI, "_header_names", return_value=list(headers.items())):
            copies = SI.unit_headers('#include "attention_segmented.cuh"\n', {})
        self.assertEqual(set(copies), set(names))
        self.assertEqual(copies[names[0]], headers[names[0]])
        protected = copies[names[1]]
        self.assertEqual(protected.count("renew_a(a, a_in);"), 2)
        restored = protected.replace(SI._RENEW_A, "").replace(SI._WG_NT[1], SI._WG_NT[0])
        restored = restored.replace(SI._WG_NT6[1], SI._WG_NT6[0])
        self.assertEqual(restored, headers[names[1]])
        actual = resolved_headers(names[0], headers, copies)
        self.assertEqual(actual[-1], ("units", names[1], protected))

    def test_malformed_original_wrapper_still_refused(self):
        with patch.object(SI, "_header_names", return_value=[("attention_long_flat.cuh", "different")]):
            with self.assertRaises(SI.Refusal):
                SI.unit_headers("", {})

    def test_independent_unit_cache_binds_header_placement_and_safety_version(self):
        text, headers, flags, nvcc = "unit source", {"leaf.cuh": "protected", "wrapper.cuh": "original"}, [], "nvcc"
        legacy_material = json.dumps(dict(text=hashlib.sha256(text.encode()).hexdigest(), headers=headers,
                                          flags=flags, nvcc=nvcc), sort_keys=True)
        legacy_key = hashlib.sha256(legacy_material.encode()).hexdigest()
        old_placement = dict(version=SI.RENEW_A_VERSION, copied_headers={"leaf.cuh": headers["leaf.cuh"]})
        closed_placement = dict(version=SI.RENEW_A_VERSION, copied_headers=headers)
        keys = [legacy_key]
        for staging in (old_placement, closed_placement, dict(closed_placement, version="older-safety-version")):
            keys.append(BUILD._unit_key(text, headers, flags, nvcc, header_staging=staging))
        self.assertEqual(len(set(keys)), 4)


if __name__ == "__main__":
    unittest.main()
