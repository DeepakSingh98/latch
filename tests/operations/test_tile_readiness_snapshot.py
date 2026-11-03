"""Structural checks for retained counter snapshots under ``tile_readiness="attention"``; no model execution."""
import copy
from types import SimpleNamespace
import unittest

from megakernel_lab.operation_compiler import tile_ready, tile_readiness


def fixture():
    phases = [
        dict(kind="elementwise", names=["qkv"], reads=[], writes=["qkv"], code="prepare();"),
        dict(kind="segmented_attention_forward", names=["attn", "lse", "work"], reads=["qkv"],
             writes=["attn", "lse", "work"], head_width=32, heads=2, work_counters="work",
             code="latch_segmented::run_forward<32,6>(A);"),
        dict(kind="gemm", names=["projection"], reads=["attn", "weight"], writes=["projection"],
             flat_claim=True, pack_scatter={"unpacked": "projection"}, code="projection();"),
        dict(kind="elementwise", names=["snapshot"], reads=["work"], writes=["snapshot"],
             targets=["aten.clone.default"], dtype="int64", shape=[32, 2, 3],
             body=["v_snapshot[i] = v_work[i];"], code="v_snapshot[i] = v_work[i];"),
        dict(kind="elementwise", names=["tail"], reads=["projection"], writes=["tail"], code="tail();"),
    ]
    layouts = {n: SimpleNamespace(root=n, shape=(256, 2, 32), contiguous=True, offset=0)
               for n in ("qkv", "attn", "lse", "projection", "weight", "tail")}
    for n in ("work", "snapshot", "state"):
        layouts[n] = SimpleNamespace(root="state" if n == "snapshot" else n,
                                     shape=(32, 2, 3), contiguous=True, offset=0)
    return phases, [[0], [1], [2, 3], [4]], layouts, dict(
        state_writes={"state": "snapshot"}, state_inputs={"state"}, returned=["snapshot"])


def plan(data, rules="attention"):
    phases, intervals, layouts, options = data
    with tile_readiness.scope(rules):
        return tile_ready.plan(phases, intervals, layouts, **options)


class CounterSnapshotTests(unittest.TestCase):
    def test_copy_follows_full_completion_without_changing_body_or_state(self):
        data = fixture(); before = copy.deepcopy(data[0]); result = plan(data)
        self.assertIsNotNone(result)
        self.assertEqual(data[0], before)
        self.assertEqual(data[1], [[0], [1], [2], [3, 4]])
        self.assertEqual([b["interval"] for b in result.boundaries], [1])
        self.assertEqual(result.boundaries[0]["edges"], [("attn", "projection")])
        self.assertEqual(result.boundaries[0]["sunk"], ["snapshot"])
        self.assertEqual(result.epoch_of(data[1]), {0: 0, 1: 1, 2: 1, 3: 2, 4: 2})
        metadata = result.record()["boundaries"][0]["counter_snapshots"][0]
        self.assertEqual((metadata["from_interval"], metadata["to_interval"], metadata["completion_barrier_after"]), (2, 3, 2))
        self.assertEqual(metadata["counter"], "work")
        self.assertEqual(metadata["state_target"], "state")

    def test_unobserved_graph_has_no_bookkeeping_move(self):
        data = fixture(); data[1][2].remove(3); data[1][-1].append(3)
        result = plan(data)
        self.assertEqual(result.boundaries[0]["sunk"], [])
        self.assertNotIn("counter_snapshots", result.record()["boundaries"][0])

    def test_without_the_option_nothing_moves(self):
        data = fixture(); before = copy.deepcopy(data[1])
        self.assertIsNone(plan(data, rules=None))
        self.assertEqual(data[1], before)

    def test_non_observer_cases_fail_closed(self):
        cases = ("read_destination", "second_destination_writer", "crossed_source_writer", "concurrent_source_writer",
                 "dtype", "target", "not_returned", "not_state", "not_state_write", "wrong_alias", "offset",
                 "shape", "not_native_counter", "consumer_publisher", "no_later_interval", "internal_mutation",
                 "extra_read", "excluded")
        for case in cases:
            with self.subTest(case=case):
                data = fixture(); phases, intervals, layouts, options = data
                if case == "read_destination": phases[4]["reads"].append("state")
                elif case == "second_destination_writer": phases[4]["writes"].append("state")
                elif case in ("crossed_source_writer", "concurrent_source_writer"):
                    phases[2 if case == "crossed_source_writer" else 4]["writes"].append("work")
                elif case == "dtype": phases[3]["dtype"] = "float32"
                elif case == "target": phases[3]["targets"] = ["aten.add.Tensor"]
                elif case == "not_returned": options["returned"] = []
                elif case == "not_state": options["state_inputs"] = set()
                elif case == "not_state_write": options["state_writes"] = {}
                elif case == "wrong_alias": layouts["snapshot"].root = "snapshot"
                elif case == "offset": layouts["snapshot"].offset = 1
                elif case == "shape": layouts["state"].shape = (32, 2, 4)
                elif case == "not_native_counter": phases[1]["work_counters"] = "other"
                elif case == "consumer_publisher":
                    phases[4] = dict(phases[1], names=["other"], writes=["other"], reads=[])
                    intervals[2].append(4); intervals[3].remove(4)
                elif case == "no_later_interval": intervals.pop()
                elif case == "internal_mutation": phases[3]["internal_mutations"] = ["state"]
                elif case == "extra_read": phases[3]["reads"].append("qkv")
                elif case == "excluded": options["excluded"] = {3}
                before = copy.deepcopy(intervals)
                self.assertIsNone(plan(data), case)
                self.assertEqual(intervals, before, case)


if __name__ == "__main__":
    unittest.main()
