"""CPU contracts for the frozen A/B/C union; no performance measurements.

The cloud contract task imports this normally. Its synthetic emitters require no
tensor arithmetic; CUDA compilation and numerical qualification remain separate.
"""
from pathlib import Path


def _attention_plan():
    stride = (32768, 64, 128)
    return dict(label="test_attention", bh=4, t=256, hd=64, heads=2, splits=1,
                group={"scale_mode": 1, "scale": 0.125, "mask": "mask"},
                sources={"q": "q", "k": "k", "v": "v", "do": "do"},
                finals={"o": "o", "dq": "dq", "dkt": "dk", "dv": "dv"},
                in_strides={k: stride for k in ("q", "k", "v", "do")},
                out_strides={k: stride for k in ("o", "dq", "dkt", "dv")})


def test_explicit_attention_precision_is_per_helper():
    from megakernel_lab.operation_compiler import long_attention as rule
    plan = _attention_plan()
    args = (plan, {}, lambda x: "v_" + x, (0, 256, 1))
    kwargs = dict(which="forward", aux={"o32": "o32", "lse": "lse"})
    strict = rule.emit(*args, **kwargs)[2]
    practical = rule.emit(*args, **kwargs, fast_exp=True)[2]
    assert strict["exponential_policy"] == "accurate"
    assert practical["exponential_policy"] == "approximate_ftz"
    assert "A.practical_exp = false;" in strict["helper"]
    assert "A.practical_exp = true;" in practical["helper"]
    assert strict["helper"].split("(", 1)[0] != practical["helper"].split("(", 1)[0]
    assert not any(x.startswith("P9A_FAST_EXP") for x in rule.P9_DEFINES)
    header = Path(rule.__file__).with_name("kernels") / "attention_p9.cuh"
    text = header.read_text()
    assert "if (!practical) return expf(d);" in text
    assert "p9a_exp((d), a.practical_exp)" in text
    assert "#ifdef P9A_FAST_EXP" not in text


def test_nested_regions_preserve_tape_abi():
    from megakernel_lab.operation_compiler.rng_remat import remat_philox_masks
    from megakernel_lab.operation_compiler.tape_remat import remat_tape_products
    from megakernel_lab.operation_compiler.tape_cse import cse_saved_body
    nested = {"schema": "uniform-operation-v1", "true_branch": {}, "false_branch": {}}
    data = {"storage_mode": "save_body_tape", "metadata": {"body_tape": [{"name": "x"}]},
            "regions": {"body_saved": nested, "body_vjp_saved": nested}}
    for rewrite in (remat_philox_masks, remat_tape_products, cse_saved_body):
        result, receipt = rewrite(data)
        assert result is data
        assert receipt["refused"] == "nested regions retain their original tape ABI"


def test_flat_and_region_warp_specialized_contracts_are_separate():
    from megakernel_lab.operation_compiler import p5c_matrix as rule
    spec = dict(m=1024, n=1024, k=1024, ak=True, bkc=True, ws_ok=True)
    flat = rule.ws_plan(spec, 128)
    assert flat["ws_engine"] == "p8ws_flat"
    assert flat["nst"] == 5
    assert rule.ws_plan(dict(spec, ws_ok=False), 128) is None  # rejected B alternative stays disabled
    assert rule.ws_plan(dict(spec, pointers=("A", "B")), 128) is None
    assert rule.SHORT_MIN_K_STAGES == 4
    assert rule.longk_split_admissible(384, 512, 8192)
    assert rule.split_count(384, 512, 8192) > 4
    assert rule.split_count(768, 768, 2048) == 2


def test_key_norm_requires_actual_producer_admission():
    from megakernel_lab.operation_compiler import flat_long_attention as rule
    import inspect
    assert inspect.signature(rule.emit).parameters["keynorm_epilogue"].default is False
    source = Path(rule.__file__).with_name("fusion_p6d.py").read_text()
    assert 'p5c_matrix.flat_ws_plan(k_spec, 128) is not None' in source
    assert 'key-norm initialization precedes every CTA' in source


def test_program_slots_restore_and_refuse_capacity():
    from megakernel_lab.operation_compiler import long_attention as rule
    old_slots, old_masks = rule._SLOTS, rule._MASK_SLOTS
    with rule.program_state():
        first = rule.mask_slot("phase", "forward")
        shared, builds = rule.mask_table(("mask", (0, 256, 1), 256), "forward", "phase")
        assert first != shared and builds
        for i in range(rule.MT_SLOTS - 2):
            rule.mask_slot(str(i), "backward")
        try:
            rule.mask_slot("overflow", "backward")
        except rule.Refused:
            pass
        else:
            raise AssertionError("slot overflow silently aliased another phase")
    assert rule._SLOTS is old_slots and rule._MASK_SLOTS is old_masks
    with rule.program_state(shared_masks=False):
        plan = dict(_attention_plan(), immutable_mask=True)
        receipt = rule.emit(plan, {}, lambda x: x, (0, 256, 1), which="forward",
                            aux={"o32": "o32", "lse": "lse"})[2]
        assert not receipt["shared_mask_admitted"]


def run():
    tests = {}
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            tests[name] = True
    return {"passed": True, "tests": tests}
