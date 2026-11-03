"""Claimed width-32 segmented-attention items (segmented_attention='claim'; off by default)."""
from pathlib import Path

from megakernel_lab.operation_compiler import attention_claim as claim
from megakernel_lab.operation_compiler import segmented_attention_rules as rules

KV = ("{\nlatch_segmented::Args A{};\nA.q=(v_q + 0LL);\n"
      "latch_segmented::run_kv<32,6>(A, (v_off + 0LL), (v_tiles + 0LL), smem);\n}")
HEADER = Path(claim.__file__).parent / "kernels" / "attention_segmented.cuh"


def alloc_counter():
    calls = []

    def alloc(n):
        calls.append(n)
        return f"reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + {len(calls) - 1}LL)"
    return alloc, calls


def test_default_off_leaves_body():
    alloc, calls = alloc_counter()
    assert claim.attention_claim({"kind": "segmented_attention_kv"}, KV, alloc) == (KV, False)
    with rules.scope("rotary+folded_forward"):
        assert claim.attention_claim({"kind": "segmented_attention_kv"}, KV, alloc) == (KV, False)
    assert calls == []


def test_binds_one_counter_ahead_in_rank_order():
    alloc, calls = alloc_counter()
    with rules.scope("claim"):
        body, bound = claim.attention_claim({"kind": "segmented_attention_kv"}, KV, alloc)
    assert bound and calls == [1]
    inserted = ("A.tkt_claim = reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + 0LL); A.tkt_ahead = 1; "
                "A.tkt_rank = 1;  // attention items claimed by tickets\nlatch_segmented::run_kv")
    assert inserted in body
    assert body.replace(body[body.index("A.tkt_claim"):body.index("latch_segmented::run_kv")], "") == KV
    with rules.scope("claim"):
        for kind, call in (("segmented_attention_forward", "run_forward"), ("segmented_attention_dq", "run_dq")):
            b, ok = claim.attention_claim({"kind": kind}, KV.replace("run_kv", call), alloc)
            assert ok and b.count("A.tkt_claim") == 1
    assert calls == [1, 1, 1]


def test_refusals():
    alloc, calls = alloc_counter()
    wide = KV.replace("<32,6>", "<64,0>")
    twice = KV + KV
    with rules.scope("claim"):
        assert claim.attention_claim({"kind": "segmented_attention_kv"}, wide, alloc) == (wide, False)
        assert claim.attention_claim({"kind": "segmented_attention_prepare"}, KV, alloc) == (KV, False)
        assert claim.attention_claim({"kind": "elementwise"}, KV, alloc) == (KV, False)
        assert claim.attention_claim({"kind": "segmented_attention_kv"}, twice, alloc) == (twice, False)
    assert calls == []


def test_define_and_header_guards():
    assert claim.DEFINE == "#define LATCH_SEG_ATTN_CLAIM 1"
    text = HEADER.read_text()
    assert text.count("#if defined(LATCH_SEG_ATTN_CLAIM)") == 4
    for symbol in ("unsigned* tkt_claim", "tkt_ticket(", "tkt_slot", "tkt_cum", "tkt_next"):
        first = text.index(symbol)
        assert text.rfind("#if defined(LATCH_SEG_ATTN_CLAIM)", 0, first) > text.rfind("#endif", 0, first)
    assert "atom.relaxed.gpu.global.add.u32" in text
