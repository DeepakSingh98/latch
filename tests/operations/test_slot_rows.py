"""Token-row groups of slot-axis chains (operation_compiler/slot_rows.py, accumulation_rules 'rslot'): CPU-only
structural tests on a synthetic chain (no CUDA)."""
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import accumulation_rules, slot_rows

T, K = 1024, 4


def _contiguous(shape):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return tuple(reversed(strides))


def _layout(shape, strides, root, offset=0):
    return SimpleNamespace(shape=tuple(shape), strides=tuple(strides), root=root, offset=offset)


def _ident(name):
    return "v_" + name


def _index(layout, shape, flat):
    """Element of ``layout`` read by iteration shape ``shape`` at flat index ``flat`` (broadcasting leading dims)."""
    padded = (1,) * (len(shape) - len(layout.shape)) + tuple(layout.shape)
    strides = (0,) * (len(shape) - len(layout.shape)) + tuple(layout.strides)
    terms, inner = [], 1
    for extent, size, stride in reversed(list(zip(shape, padded, strides))):
        if size != 1 and stride:
            terms.append(f"(({flat}) / {inner}LL % {extent}LL) * {stride}LL")
        inner *= extent
    return " + ".join(([f"{layout.offset}LL"] if layout.offset else []) + list(reversed(terms))) or "0LL"


def chain(W):
    """a = x * g (slot broadcast); m = mean_slots(a); d_j = sel_j(a) - m; s_j = mean(d_j * d_j, -1);
    r_j = rsqrt(s_j + eps); one outside reader of m, d_j and r_j."""
    lay = {}

    def root(name, shape):
        lay[name] = _layout(shape, _contiguous(shape), name)
    root("x", (T, K, W)); root("g", (T, W)); root("a", (T, K, W)); root("m", (T, W)); root("out", (T, 7))
    lay["gb"] = _layout((T, 1, W), (W, W, 1), "g")
    for j in range(K):
        lay[f"sel{j}"] = _layout((T, W), (K * W, 1), "a", j * W)
        root(f"d{j}", (T, W)); root(f"q{j}", (T, W)); root(f"s{j}", (T, 1)); root(f"r{j}", (T, 1))
    dtypes = {n: "float32" for n in lay}
    read = lambda n, shape: f"{_ident(n)}[{_index(lay[n], shape, 'i')}]"
    phases = [dict(kind="elementwise", names=["a"], targets=["aten.mul"], reads=["x", "gb"], writes=["a"], code="",
                   shape=(T, K, W), dtype="float32",
                   body=[f"const float r_a = ({read('x', (T, K, W))} * {read('gb', (T, K, W))}); v_a[i] = r_a;"]),
              dict(kind="reduction", names=["m"], targets=["aten.mean"], reads=["a"], writes=["m"], code="/*slot*/",
                   engine="exact_cta_tree_per_thread", reduction_elements=K,
                   slot_reduction=dict(source="a", axis=1, extent=K, scale=f" / {K}.0f"))]
    body = []
    for j in range(K):
        body.append(f"const float r_d{j} = ({read(f'sel{j}', (T, W))} - {read('m', (T, W))}); v_d{j}[i] = r_d{j};")
        body.append(f"const float r_q{j} = (r_d{j} * r_d{j});")
    names = [n for j in range(K) for n in (f"d{j}", f"q{j}")]
    phases.append(dict(kind="elementwise", names=names, targets=["t"] * 2 * K,
                       reads=[f"sel{j}" for j in range(K)] + ["m"], writes=names, code="", shape=(T, W),
                       dtype="float32", body=body))
    for j in range(K):
        phases.append(dict(kind="reduction", names=[f"s{j}"], targets=["aten.mean"], reads=[f"q{j}"],
                           writes=[f"s{j}"], code="v[w] = 0.0f + x; v[w] = v[w] + y;", engine="exact_cta_tree_per_warp",
                           row_reduction=dict(source=f"q{j}", width=W, rows=T, scale=f" / {W}.0f")))
        phases.append(dict(kind="elementwise", names=[f"r{j}"], targets=["aten.rsqrt"], reads=[f"s{j}"],
                           writes=[f"r{j}"], code="", shape=(T, 1), dtype="float32",
                           body=[f"const float r_r{j} = rsqrtf(({read(f's{j}', (T, 1))} + 1e-06f)); "
                                 f"v_r{j}[i] = r_r{j};"]))
    phases.append(dict(kind="elementwise", names=["out"], targets=["t"],
                       reads=["m"] + [f"d{j}" for j in range(K)] + [f"r{j}" for j in range(K)], writes=["out"],
                       code="", shape=(T, 7), dtype="float32", body=["const float r_out = 0.0f; v_out[i] = r_out;"]))
    return phases, lay, dtypes


def fuse(phases, lay, dtypes, **kwargs):
    return slot_rows.fuse_token_rows(phases, lay, dtypes, ident=_ident, index=_index, kept=set(), inplace_roots=set(),
                                     **kwargs)


def test_rule_selects_the_pass():
    assert not slot_rows.enabled()
    with accumulation_rules.scope("update+rstage"):
        assert not slot_rows.enabled()
    with accumulation_rules.scope("update+rslot"):
        assert slot_rows.enabled()
    assert not slot_rows.enabled()


@pytest.mark.parametrize("W", [320, 512])
def test_slot_chain_becomes_one_token_row_phase(W):
    phases, lay, dtypes = chain(W)
    out, records, dead = fuse(phases, lay, dtypes)
    assert len(out) == 2 and out[0]["kind"] == "slot_token_rows", [p["kind"] for p in out]
    phase = out[0]
    assert phase["engine"] == "slot_token_rows" and phase["slot_rows"]["width"] == W
    assert phase["slot_rows"]["rows"] == T and phase["work_items"] == (T + 7) // 8
    assert sorted(dead) == ["a"] + [f"q{j}" for j in range(K)] + [f"s{j}" for j in range(K)]
    code = phase["code"]
    assert "v_a[" not in code                                      # the slot value stays in registers
    assert f"v_m[(o * {W}LL + tp_c_w0)]" in code                   # stored slot mean, at its own address
    assert "if (lane == 0) v_r0[o]" in code                        # row epilogue stored by lane 0
    assert records[1]["members"] == len(phases) - 1 and records[1]["width"] == W
    assert out[1]["names"] == ["out"] and set(out[1]["reads"]) <= set(phase["writes"])


def test_backward_phases_are_not_candidates():
    phases, lay, dtypes = chain(320)
    phases.insert(0, dict(kind="elementwise_backward", names=["z"], targets=["aten.threshold_backward"], reads=[],
                          writes=["z"], code="", shape=(T,), body=[]))
    lay["z"] = _layout((T,), (1,), "z")
    out, records, dead = fuse(phases, lay, dtypes)
    assert out == phases and dead == set()
    assert records[0]["diagnostics"]["groups"] == 0


def test_row_reduction_prologue_of_an_outside_reader_is_left_to_row_fusion():
    phases, lay, dtypes = chain(320)
    lay["n"] = _layout((T, 1), (1, 1), "n")
    dtypes["n"] = "float32"
    # a row softmax (not a candidate) reads d0 as its source: the phase producing d0 must stay with row_fusion
    phases.insert(len(phases) - 1, dict(kind="softmax", names=["n"], targets=["aten._softmax"], reads=["d0"],
                                        writes=["n"], code="", row_softmax=dict(source="d0", width=320, rows=T)))
    out, records, dead = fuse(phases, lay, dtypes)
    assert records[0]["diagnostics"]["dropped"].get("outside row reduction prologue")
    assert not any(p["kind"] == "slot_token_rows" and "d0" in p["names"] for p in out)


def test_advisory_probe_keeps_row_reductions_that_row_fusion_extends():
    phases, lay, dtypes = chain(320)
    blocked = [dict(reduction="s0", full_epilogue_phases=1)]
    out, records, dead = fuse(phases, lay, dtypes, probe=lambda _: blocked)
    assert records[0]["diagnostics"]["refused"].get("row_fusion full epilogue") == 1
    assert "s0" not in dead
    failing = fuse(phases, lay, dtypes, probe=lambda _: 1 / 0)       # a failing probe keeps every row reduction
    assert failing[1][0]["diagnostics"]["refused"].get("row_fusion full epilogue") == K
