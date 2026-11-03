"""Pure dependency regressions: no tensor runtime or compiler build is needed."""
import unittest

from megakernel_lab.operation_compiler import group_rules as GROUP


def phase(name, reads=(), writes=(), mutations=()):
    return dict(name=name, reads=list(reads), writes=list(writes), internal_mutations=list(mutations))


class EffectOrderTests(unittest.TestCase):
    def order(self, phases, root=lambda name:name):
        return [p["name"] for p in GROUP._topological(phases, root)]

    def test_segmented_counter_chain_preserves_prefix_delta_and_final_reader(self):
        phases = [phase("prepare", ("offsets",), ("prefix","work")),
                  phase("dq", ("q","prefix","work"), ("dq","delta","work")),
                  phase("kv", ("q","prefix","work","dq","delta"), ("dk","dv","work")),
                  phase("publish", ("work","dk","dv"), ("result",))]
        deps = GROUP._topological_dependencies(phases,lambda name:name)
        self.assertEqual(deps,[set(),{0},{0,1},{2}])
        self.assertEqual(self.order(phases),["prepare","dq","kv","publish"])

    def test_write_after_write_is_preserved_when_initialization_waits_on_future_ssa(self):
        phases = [phase("initialize",("seed",),("work",)),
                  phase("overwrite",(),("work",)), phase("seed",(),("seed",))]
        self.assertEqual(self.order(phases),["seed","initialize","overwrite"])

    def test_overwrite_waits_for_intervening_reader_even_when_reader_is_delayed(self):
        phases = [phase("initialize",(),("work",)),
                  phase("snapshot",("work","late"),("snapshot",)),
                  phase("overwrite",(),("work",)), phase("late",(),("late",))]
        self.assertEqual(self.order(phases),["initialize","late","snapshot","overwrite"])

    def test_incoming_version_reader_stays_before_first_write(self):
        phases = [phase("read_old",("work","late"),("old",)),
                  phase("initialize",(),("work",)), phase("overwrite",(),("work",)),
                  phase("late",(),("late",))]
        self.assertEqual(self.order(phases),["late","read_old","initialize","overwrite"])

    def test_aliases_and_internal_read_modify_write_use_preceding_version(self):
        root = lambda name: "work" if name.startswith("work") else name
        phases = [phase("initialize",(),("work_base","work_alias")),
                  phase("mutate",("work_view",),("delta",),("work_alias",)),
                  phase("read",("work_base","delta"),("out",))]
        self.assertEqual(GROUP._topological_dependencies(phases,root),[set(),{0},{1}])
        self.assertEqual(self.order(phases,root),["initialize","mutate","read"])

    def test_single_writer_aliases_keep_global_producer_edge(self):
        root = lambda name: name.removesuffix("_view")
        phases = [phase("read",("value_view",),("out",)),
                  phase("write",(),("value","value_view"))]
        self.assertEqual(self.order(phases,root),["write","read"])

    def test_one_internal_mutator_preserves_incoming_and_outgoing_reads(self):
        phases = [phase("read_old",("work","late"),("old",)),
                  phase("mutate",(),(),("work",)), phase("read_new",("work",),("new",)),
                  phase("late",(),("late",))]
        self.assertEqual(self.order(phases),["late","read_old","mutate","read_new"])

    def test_independent_ssa_tie_order_is_unchanged(self):
        phases = [phase("read",("value",),("out",)), phase("independent",(),("other",)),
                  phase("write",(),("value",)), phase("last",(),("last",))]
        self.assertEqual(self.order(phases),["independent","write","read","last"])

    def test_real_ssa_cycle_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError,"dependency cycle"):
            self.order([phase("a",("b",),("a",)),phase("b",("a",),("b",))])

    def test_effect_order_conflicting_with_ssa_is_rejected(self):
        # Reading the old work value requires a value made by its overwrite.
        # Reordering either side would change the declared effect sequence.
        with self.assertRaisesRegex(ValueError,"dependency cycle"):
            self.order([phase("init",(),("work",)),phase("read",("work","new"),("out",)),
                        phase("overwrite",(),("work","new"))])


if __name__ == "__main__":
    unittest.main()
