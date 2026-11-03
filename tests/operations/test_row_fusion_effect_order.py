"""Small row-fusion dependency regressions; no tensors, compilation or CUDA."""
import copy
from types import SimpleNamespace
import unittest

from megakernel_lab.operation_compiler import row_fusion as ROW


def phase(name, reads=(), writes=(), mutations=()):
    return dict(kind="opaque", name=name, names=[name], targets=[name], reads=list(reads),
                writes=list(writes), internal_mutations=list(mutations), code="")


def row_fixture():
    layouts = {name: SimpleNamespace(shape=(128, 128), strides=(128, 1), root=name, offset=0)
               for name in ("x", "square")}
    layouts["mean"] = SimpleNamespace(shape=(128, 1), strides=(1, 1), root="mean", offset=0)
    phases = [dict(kind="elementwise", name="square", names=["square"], targets=["aten.mul.Tensor"],
                   reads=["x"], writes=["square"], shape=[128, 128], dtype="float32",
                   body=["const float r_square = v_x[i] * v_x[i]; v_square[i] = r_square;"]),
              dict(kind="reduction", name="mean", names=["mean"], targets=["aten.mean.dim"],
                   reads=["square"], writes=["mean"], code="",
                   row_reduction=dict(source="square", width=128, rows=128, scale=" * (1.0f / 128.0f)"))]
    return phases, layouts, dict.fromkeys(layouts, "float32")


def fuse(phases, layouts, dtypes, **kwargs):
    return ROW.fuse_rows(copy.deepcopy(phases), layouts, dtypes, ident=lambda name: "v_" + name,
                         index=lambda layout, shape, index: "(" + index + ")", kept={"mean"},
                         inplace_roots=set(), **kwargs)


class RowFusionEffectTests(unittest.TestCase):
    def test_real_row_fusion_with_paired_mutable_counter_phases(self):
        rows, layouts, dtypes = row_fixture()
        counters = [phase("prepare", ("offsets",), ("prefix", "work")),
                    phase("dq", ("prefix", "work"), ("dq", "delta", "work")),
                    phase("kv", ("prefix", "work", "dq", "delta"), ("dk", "dv", "work")),
                    phase("publish", ("work", "dk", "dv"), ("result",))]
        result, records, _ = fuse(rows + counters, layouts, dtypes)
        self.assertEqual(len(records), 1)
        self.assertEqual(result[0]["kind"], "row_fused")
        self.assertEqual([p["name"] for p in result[1:]], ["prepare", "dq", "kv", "publish"])
        self.assertEqual(result[1:], counters)

    def contracted(self, phases, owners, *, extra_reads=None, root=lambda name: name):
        result = []
        for group in range(max(owners) + 1):
            members = [p for i, p in enumerate(phases) if owners[i] == group]
            writes = list(dict.fromkeys(n for p in members for n in p["writes"]))
            produced = {root(n) for n in writes}
            reads = list(dict.fromkeys(n for p in members for n in p["reads"] if root(n) not in produced))
            result.append(phase(str(group), reads + (extra_reads or {}).get(group, []), writes))
        deps = ROW._fusion_dependencies(phases, result, owners, root)
        return ROW._dependency_order(deps), deps

    def test_war_waw_and_global_ssa_survive_contraction(self):
        phases = [phase("init", (), ("work",)), phase("read", ("work", "late"), ("old",)),
                  phase("overwrite", (), ("work",)), phase("late", (), ("late",))]
        order, deps = self.contracted(phases, [0, 1, 2, 3])
        self.assertEqual(order, [0, 3, 1, 2])
        self.assertEqual(deps, [set(), {0, 3}, {0, 1}, set()])

    def test_mutable_read_moved_to_first_member_keeps_original_version(self):
        phases = [phase("init", (), ("work",)), phase("prologue", (), ("p",)),
                  phase("overwrite", (), ("work",)), phase("consumer", ("work", "p"), ("out",))]
        order, _ = self.contracted(phases, [0, 1, 2, 1])
        self.assertEqual(order, [0, 2, 1])

    def test_incoming_version_and_alias_effects_are_preserved(self):
        root = lambda name: name.removesuffix("_view")
        phases = [phase("old", ("work_view", "late"), ("old",)),
                  phase("init", (), ("work",)), phase("mutate", (), (), ("work_view",)),
                  phase("late", (), ("late",)), phase("new", ("work",), ("new",))]
        # Untouched phases retain explicit mutation metadata.
        deps = ROW._fusion_dependencies(phases, phases, list(range(5)), root)
        self.assertEqual(ROW._dependency_order(deps), [3, 0, 1, 2, 4])

    def test_nonconvex_group_still_refuses_cycle(self):
        phases = [phase("a", (), ("a",)), phase("middle", ("a",), ("middle",)),
                  phase("b", ("middle",), ("b",))]
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            self.contracted(phases, [0, 1, 0])

    def test_original_cycle_cannot_disappear_inside_one_group(self):
        phases = [phase("a", ("b",), ("a",)), phase("b", ("a",), ("b",))]
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            self.contracted(phases, [0, 0])

    def test_new_external_gather_keys_keep_global_ssa_producer(self):
        phases = [phase("a", (), ("a",)), phase("keys", (), ("keys",))]
        order, deps = self.contracted(phases, [0, 1], extra_reads={0: ["keys"]})
        self.assertEqual(order, [1, 0])
        self.assertEqual(deps, [{1}, set()])

    def test_actual_gather_fusion_waits_for_new_order_keys(self):
        rows, layouts, dtypes = row_fixture()
        rows[0] = dict(kind="index_select", name="square", names=["square"], targets=["index_select"],
                       reads=["x", "ids"], writes=["square"], code="", engine="index_select_warp_rows",
                       gather_table="x", gather_index="ids")
        for name in ("ids", "keys"):
            layouts[name] = SimpleNamespace(shape=(128,), strides=(1,), root=name, offset=0)
            dtypes[name] = "int64"
        rows.append(phase("make_keys", ("ids",), ("keys",)))
        result, records, _ = fuse(rows, layouts, dtypes, flat=True, orders={"ids": ("keys", 128)})
        self.assertEqual(result[0]["name"], "make_keys")
        self.assertEqual(result[1]["kind"], "row_fused")
        self.assertEqual(records[0]["gather_order"], "keys")
        self.assertIn("keys", result[1]["reads"])

    def test_new_mutable_read_has_no_invented_version(self):
        phases = [phase("a", (), ("a",)), phase("init", (), ("work",)),
                  phase("update", (), ("work",))]
        with self.assertRaisesRegex(ValueError, "new read of mutable root"):
            self.contracted(phases, [0, 1, 2], extra_reads={0: ["work"]})

    def test_pure_ssa_fusion_preserves_body_and_stable_ready_order(self):
        rows, layouts, dtypes = row_fixture()
        baseline = fuse(rows, layouts, dtypes)
        result = fuse([phase("independent", (), ("other",)), *rows], layouts, dtypes)
        self.assertEqual(result[0][0]["name"], "independent")
        self.assertEqual(result[0][1], baseline[0][0])
        self.assertEqual(result[1:], baseline[1:])

    def test_chained_reductions_map_all_members_once(self):
        rows, layouts, dtypes = row_fixture()
        layouts["twice"] = SimpleNamespace(shape=(128, 128), strides=(128, 1), root="twice", offset=0)
        layouts["second"] = SimpleNamespace(shape=(128, 1), strides=(1, 1), root="second", offset=0)
        dtypes.update(twice="float32", second="float32")
        rows.extend([dict(kind="elementwise", names=["twice"], targets=["mul"], reads=["square"],
                          writes=["twice"], shape=[128, 128], dtype="float32",
                          body=["const float r_twice = v_square[i] * 2.0f; v_twice[i] = r_twice;"]),
                     dict(kind="reduction", names=["second"], targets=["mean"], reads=["twice"],
                          writes=["second"], code="", row_reduction=dict(source="twice", width=128,
                          rows=128, scale=" * (1.0f / 128.0f)"))])
        result, records, _ = fuse(rows, layouts, dtypes)
        self.assertEqual(len(result), 1)
        self.assertEqual(records[0]["chained_stages"], ["second"])
        self.assertEqual(result[0]["names"], ["square", "mean", "twice", "second"])

    def test_softmax_group_uses_the_same_effect_ordering(self):
        rows, layouts, dtypes = row_fixture()
        layouts["softmax"] = SimpleNamespace(shape=(128, 128), strides=(128, 1), root="softmax", offset=0)
        dtypes["softmax"] = "float32"
        rows[1] = dict(kind="softmax", names=["softmax"], targets=["softmax"], reads=["square"],
                       writes=["softmax"], code="LOAD", row_softmax=dict(source="square", width=128,
                       rows=128, load_line="LOAD"))
        counters = [phase("init", (), ("work",)), phase("mutate", ("work",), ("work",))]
        result, records, _ = fuse(rows + counters, layouts, dtypes)
        self.assertEqual(records[0]["softmax"], "softmax")
        self.assertEqual(result[0]["names"], ["square", "softmax"])
        self.assertEqual(result[1:], counters)

    def test_mutable_writer_and_internal_mutator_are_not_fused(self):
        for kind in ("repeated_write", "internal_mutation"):
            with self.subTest(kind=kind):
                rows, layouts, dtypes = row_fixture()
                if kind == "repeated_write":
                    rows.insert(0, phase("prior", (), ("square",)))
                else:
                    rows[0]["internal_mutations"] = ["scratch"]
                result, records, _ = fuse(rows, layouts, dtypes)
                self.assertEqual(records, [])
                self.assertEqual(result, rows)


if __name__ == "__main__":
    unittest.main()
