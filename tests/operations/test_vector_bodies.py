"""Vectorized elementwise and row-reduction body rules (operation_compiler/vector_bodies.py): default-off identity,
the exact lane-invariant row index, BF16 uint2 rounds, rolling registers and the unroll cap (``elementwise_rules``),
warp-per-row reductions and staged chained row stages (``accumulation_rules``). Source-level checks (no GPU)."""
import copy
import random
import re
import struct
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

from megakernel_lab.operation_compiler import accumulation_rules, elementwise_rules, group_rules, row_fusion
from megakernel_lab.operation_compiler import vector_bodies

BODY = [
    "const float r_div_1 = (v_expand_1[(long long)((unsigned)(i) / 384u % 8192u) * 1LL] / 6.0f);",
    "const float r_add_1 = (v_a[i] + (__bfloat162float(v_mm_1[i]) * 1.0f));",
    "const float r_mul_2 = ((r_add_1 * r_div_1) + v_w[(long long)((unsigned)(i) % 384u) * 1LL]); v_mul_2[i] = r_mul_2;",
]
PHASE = dict(names=["div_1", "add_1", "mul_2"], body=BODY)
TEXTS = {"v_a[i]": "v_a", "v_mul_2[i]": "v_mul_2"}
BF16 = {"v_mm_1[i]": "v_mm_1"}
N = 8192 * 384


@contextmanager
def rules(elementwise=None, accumulation=None):
    """The option scopes, plus the flat compose context in which broadcast reads stay inside float4 bodies."""
    with ExitStack() as stack:
        try:
            from megakernel_lab.operation_compiler import compose
        except ImportError:          # source-only package without the compose driver
            compose = None
        if compose is not None:
            token = compose.STEP14_FLAT_ACTIVE.set(True)
            stack.callback(compose.STEP14_FLAT_ACTIVE.reset, token)
        stack.enter_context(elementwise_rules.scope(elementwise))
        stack.enter_context(accumulation_rules.scope(accumulation))
        yield


def emit(unroll=4, bf16=True, phase=PHASE):
    return group_rules.vector_elementwise(dict(phase), N, dict(TEXTS), unroll=unroll,
                                          bf16_texts=dict(BF16) if bf16 else None)


class VectorBodies(unittest.TestCase):
    def test_default_off_is_byte_identical(self):
        with rules():
            a = emit(bf16=False)
            b = emit(bf16=True)
            self.assertFalse(vector_bodies.rowcse() or vector_bodies.bf16vec() or vector_bodies.roll())
            self.assertEqual((vector_bodies.unroll_budget(), vector_bodies.rgroup()), (None, 0))
        self.assertIsNotNone(a)
        self.assertEqual(a, b)
        self.assertNotIn("uint2", a)
        self.assertIn("(unsigned)((4 * jc0 + 1)) / 384u", a)

    def test_lanecse_makes_lanes_identical_only_for_quotients(self):
        with rules("lanecse"):
            a = emit()
        self.assertNotIn("(4 * jc0 + 1)) / 384u", a)
        self.assertEqual(a.count("v_expand_1[(long long)((unsigned)((4 * jc0)) / 384u % 8192u) * 1LL]"), 4)
        # The column index (a remainder) keeps its per-lane element.
        self.assertIn("(unsigned)((4 * jc0 + 3)) % 384u", a)

    def test_lane_identity_exhaustive(self):
        for j in list(range(0, 4000)) + [2 ** 30 - 1, 2 ** 30, 2 ** 31 + 7]:
            for c in (4, 8, 384, 768, 8192 * 384):
                for u in range(4):
                    full = ((4 * j + u) & 0xFFFFFFFF) // c
                    self.assertEqual(full, ((4 * j) & 0xFFFFFFFF) // c)

    def test_bf16vec_unpacks_little_endian_lanes(self):
        with rules("bf16vec"):
            a = emit()
        self.assertIn("const uint2 hb0__0 = reinterpret_cast<const uint2*>(v_mm_1)[jc0];", a)
        self.assertNotIn("__bfloat162float(v_mm_1", a)
        for lane, text in enumerate(("hb0__0.x << 16", "hb0__0.x & 0xffff0000u", "hb0__0.y << 16",
                                     "hb0__0.y & 0xffff0000u")):
            self.assertIn(f"r_add_1_{lane}_0 = (q0__0.{'xyzw'[lane]} + (__uint_as_float({text})", a)

    def test_bf16_lane_bits(self):
        for h in (0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F7F, 0x0001, 0x7FC0, 0xFF80):
            lo, hi = h, (h ^ 0x1234) & 0xFFFF
            x = lo | (hi << 16)
            self.assertEqual((x << 16) & 0xFFFFFFFF, lo << 16)
            self.assertEqual(x & 0xFFFF0000, hi << 16)

    def test_bf16_text_used_outside_widening_is_refused(self):
        body = BODY[:1] + ["const float r_add_1 = (v_a[i] + __bfloat162float(v_mm_1[i])); "
                           "v_mm_1[i] = __float2bfloat16(r_add_1);"]
        pattern = re.compile(r'v_\w+\[[^\]]*\]')
        with rules("bf16vec"):
            self.assertEqual(vector_bodies.bf16_operands([body[1].partition('; ')[0]], BF16, pattern),
                             {"v_mm_1[i]": "hb0_"})
            self.assertEqual(vector_bodies.bf16_operands(
                ["const float r_x = (__bfloat162float(v_mm_1[i]) + (float)v_mm_1[i]);"], BF16, pattern), {})
        with rules():
            self.assertEqual(vector_bodies.bf16_operands([body[1].partition('; ')[0]], BF16, pattern), {})

    def test_rolling_caps_the_unroll(self):
        self.assertEqual(vector_bodies.UNROLL_BUDGET, 64)
        with rules():
            self.assertEqual(vector_bodies.choose_unroll(4, 28, 29), 4)
        with rules("rolling"):
            self.assertEqual(vector_bodies.unroll_budget(), 64)
            self.assertEqual(vector_bodies.choose_unroll(4, 28, 29), 1)     # 170 registers: any unroll exceeds 64
            self.assertEqual(vector_bodies.choose_unroll(4, 4, 0), 4)       # 16 x 4 == 64
            self.assertEqual(vector_bodies.choose_unroll(4, 5, 0), 2)       # 20 x 4 > 64 >= 20 x 2
            self.assertEqual(vector_bodies.choose_unroll(4, 10, 1), 1)

    def test_rolling_reloads_after_last_use_and_late_for_inplace_stores(self):
        body = BODY[:2] + ["const float r_mul_2 = ((r_add_1 * r_div_1) + v_w[(long long)((unsigned)(i) % 384u) * 1LL]); "
                           "v_mul_2[i] = v_mul_2[i] + r_mul_2;"]
        phase = dict(names=["div_1", "add_1", "mul_2"], body=body)
        with rules("rolling+bf16vec+lanecse"):
            a = emit(unroll=1, phase=phase)
        lines = a.split("\n")
        store = next(k for k, l in enumerate(lines)
                     if "reinterpret_cast<float4*>(v_mul_2)[j] = make_float4(q1_.x + r_mul_2_0" in l)
        reload_q1 = next(k for k, l in enumerate(lines) if l.strip() == "q1_ = reinterpret_cast<const float4*>(v_mul_2)[jnc_];")
        reload_q0 = next(k for k, l in enumerate(lines) if l.strip() == "q0_ = reinterpret_cast<const float4*>(v_a)[jnc_];")
        last_q0_use = max(k for k, l in enumerate(lines) if "q0_." in l)
        reload_hb = next(k for k, l in enumerate(lines) if l.strip() == "hb0_ = reinterpret_cast<const uint2*>(v_mm_1)[jnc_];")
        last_hb_use = max(k for k, l in enumerate(lines) if "hb0_." in l)
        self.assertGreater(reload_q1, store)                 # in-place accumulate operand: after the store
        self.assertLess(reload_q0, store)
        self.assertEqual(reload_q0, last_q0_use + 1)
        self.assertIn(reload_hb - last_hb_use, (1, 2))       # others: right after their last (lane-interleaved) use
        self.assertIn(f"const long long jnc_ = jn_ < {N // 4}LL ? jn_ : j;", a)
        lane_rows = [l for l in lines if " r_add_1_" in l and "= (" in l]
        self.assertEqual([l.split(" r_add_1_")[1][0] for l in lane_rows], ["0", "1", "2", "3"])
        with rules("bf16vec"):
            b = emit(unroll=1, phase=phase)
        self.assertNotIn("jnc_", b)

    def test_rolling_keeps_lane_invariant_row_reads_in_registers(self):
        with rules("rolling+bf16vec+lanecse"):
            a = emit(unroll=1)
        row = "v_expand_1[(long long)((unsigned)((4 * {})) / 384u % 8192u) * 1LL]"
        self.assertIn("  auto rw0_ = " + row.format("jp_") + ";", a)
        self.assertIn("    rw0_ = " + row.format("jnc_") + ";", a)
        self.assertNotIn(row.format("j"), a)
        self.assertIn("(unsigned)((4 * j + 3)) % 384u", a)      # lane-dependent column reads stay inline
        with rules("rolling+bf16vec"):
            b = emit(unroll=1)
        self.assertNotIn("rw0_", b)                              # without lanecse no read is lane-invariant

    def test_rwarp_replays_the_cta_tree(self):
        read = "v_x[(long long)((unsigned)(r) % 3360u) * 1LL + (long long)((unsigned)(o) % 16384u) * 3360LL]"
        with rules():
            self.assertIsNone(vector_bodies.warp_row_reduction(16384, 3360, read, "v_s[o] = acc;"))
        with rules(accumulation="rwarp"):
            code = vector_bodies.warp_row_reduction(16384, 3360, read, "v_s[o] = acc;")
            self.assertIsNone(vector_bodies.warp_row_reduction(7, 3360, read, "v_s[o] = acc;"))
            self.assertIsNone(vector_bodies.warp_row_reduction(16384, 16384, read, "v_s[o] = acc;"))
        self.assertIn("for (int m_ = 0; m_ < 14; ++m_)", code)
        self.assertIn("const long long r = lane_ + 32 * k_ + 256LL * m_;", code)
        self.assertNotIn("__syncthreads", code)
        # Python model of both trees on random data: identical float32 bits.
        f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
        add = lambda a, b: f32(f32(a) + f32(b))

        def butterfly(vals):
            v = list(vals)
            for off in (16, 8, 4, 2, 1):
                v = [add(v[l], v[l ^ off]) for l in range(32)]
            assert len(set(struct.pack("f", x) for x in v)) == 1
            return v
        rnd = random.Random(7)
        for n in (3360, 2880, 333, 256, 4000):
            x = [f32(rnd.uniform(-3, 3) * 10 ** rnd.randint(-6, 6)) for _ in range(n)]
            seq = []
            for t in range(256):
                acc = 0.0
                for r in range(t, n, 256):
                    acc = add(acc, x[r])
                seq.append(acc)
            warps = [butterfly(seq[32 * w:32 * w + 32])[0] for w in range(8)]
            cta = butterfly([warps[l] if l < 8 else 0.0 for l in range(32)])[0]
            a = [[0.0] * 8 for _ in range(32)]
            for lane in range(32):
                for k in range(8):
                    acc = 0.0
                    for m in range((n + 255) // 256):
                        r = lane + 32 * k + 256 * m
                        if r < n:
                            acc = add(acc, x[r])
                    a[lane][k] = acc
            per = [butterfly([a[lane][k] for lane in range(32)]) for k in range(8)]
            warp = butterfly([per[l][l] if l < 8 else 0.0 for l in range(32)])[0]
            self.assertEqual(struct.pack("f", warp), struct.pack("f", cta))


def _layout(shape, root):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return SimpleNamespace(shape=tuple(shape), strides=tuple(reversed(strides)), root=root, offset=0,
                           contiguous=True, numel=step)


def chained_rows(rows=256, width=128, stages=3):
    """m = mean(x * x, -1); t = x * m; d_j = sum(t * s_j, -1): one row group with chained row stages d_j."""
    layouts = {n: _layout((rows, width), n) for n in ("x", "sq", "t")}
    layouts["m"] = _layout((rows, 1), "m")
    phases = [dict(kind="elementwise", names=["sq"], targets=["aten.mul.Tensor"], reads=["x"], writes=["sq"],
                   shape=[rows, width], body=["const float r_sq = (v_x[(i)] * v_x[(i)]); v_sq[i] = r_sq;"]),
              dict(kind="reduction", names=["m"], targets=["aten.mean.dim"], reads=["sq"], writes=["m"], code="",
                   row_reduction=dict(source="sq", width=width, rows=rows, scale=f" * (1.0f / {width}.0f)")),
              dict(kind="elementwise", names=["t"], targets=["aten.mul.Tensor"], reads=["x", "m"], writes=["t"],
                   shape=[rows, width], body=["const float r_t = (v_x[(i)] * v_m[(i)]); v_t[i] = r_t;"])]
    for j in range(stages):
        for name, shape in ((f"s{j}", (rows, width)), (f"p{j}", (rows, width)), (f"d{j}", (rows, 1))):
            layouts[name] = _layout(shape, name)
        phases += [dict(kind="elementwise", names=[f"p{j}"], targets=["aten.mul.Tensor"], reads=["t", f"s{j}"],
                        writes=[f"p{j}"], shape=[rows, width],
                        body=[f"const float r_p{j} = (v_t[(i)] * v_s{j}[(i)]); v_p{j}[i] = r_p{j};"]),
                   dict(kind="reduction", names=[f"d{j}"], targets=["aten.sum.dim_IntList"], reads=[f"p{j}"],
                        writes=[f"d{j}"], code="", row_reduction=dict(source=f"p{j}", width=width, rows=rows, scale=""))]
    kept = {"t"} | {f"d{j}" for j in range(stages)}
    return phases, layouts, dict.fromkeys(layouts, "float32"), kept


def fuse_chain(rule=None, stages=3):
    phases, layouts, dtypes, kept = chained_rows(stages=stages)
    with rules(accumulation=rule):
        result, records, _ = row_fusion.fuse_rows(copy.deepcopy(phases), layouts, dtypes, ident=lambda n: "v_" + n,
                                                  index=lambda layout, shape, index: "(" + index + ")", kept=kept,
                                                  inplace_roots=set(), flat=True)
    (fused,) = [p for p in result if p["kind"] == "row_fused"]
    return fused["code"], records


class StagedRowChains(unittest.TestCase):
    def test_off_keeps_one_leaf_array_and_stores_between_stages(self):
        code, records = fuse_chain()
        self.assertEqual([r.get("chained_stages") for r in records], [["d0", "d1", "d2"]])
        self.assertNotIn("leaf0_", code)
        lines = code.split("\n")
        stores = [k for k, l in enumerate(lines) if re.fullmatch(r"\s*if \(lane == 0\) v_d\d\[o\] = r_d\d;", l)]
        trees = [k for k, l in enumerate(lines) if "float v_c" in l]
        self.assertEqual(len(stores), 3)
        self.assertLess(stores[0], trees[1])                    # stage 0 stores before stage 1's tree

    def test_rstage_issues_leaf_loops_before_trees_and_defers_stores(self):
        plain, _ = fuse_chain()
        code, records = fuse_chain("rstage")
        self.assertEqual([r.get("chained_stages") for r in records], [["d0", "d1", "d2"]])
        lines = code.split("\n")
        leaves = [lines.index(f"  float leaf{j}_[4];") for j in range(3)]
        trees = [next(k for k, l in enumerate(lines) if f"float v_c{j}[8];" in l) for j in range(3)]
        self.assertLess(max(leaves), min(trees))                # every leaf loop precedes the first tree
        stores = [k for k, l in enumerate(lines) if re.fullmatch(r"\s*if \(lane == 0\) v_d\d\[o\] = r_d\d;", l)]
        self.assertEqual(len(stores), 3)
        self.assertGreater(min(stores), max(trees))             # results stored after the last stage
        self.assertEqual(sorted(re.findall(r"v_d\d\[o\] = r_d\d", plain)), sorted(re.findall(r"v_d\d\[o\] = r_d\d", code)))
        # every stage keeps its tree text; only the leaf array name changes
        for j in range(3):
            self.assertIn(f"v_c{j}[w] = w < 4 ? 0.0f + leaf{j}_[w < 4 ? w : 0] : 0.0f;", code)
            self.assertIn(f"v_c{j}[w] = w < 4 ? 0.0f + leaf_[w < 4 ? w : 0] : 0.0f;", plain)

    def test_rstage_groups_at_most_four_stages(self):
        self.assertEqual(vector_bodies.STAGE_GROUP, 4)
        with rules(accumulation="rstage"):
            self.assertEqual(vector_bodies.rgroup(), 4)
        code, _ = fuse_chain("rstage", stages=6)
        lines = code.split("\n")
        leaves = [lines.index(f"  float leaf{j}_[4];") for j in range(6)]
        trees = [next(k for k, l in enumerate(lines) if f"float v_c{j}[8];" in l) for j in range(6)]
        self.assertLess(max(leaves[:4]), trees[0])
        self.assertGreater(leaves[4], trees[3])                 # the fifth stage opens a new group


if __name__ == "__main__":
    unittest.main()
