"""Explicit TinyMoE architecture variants (megakernel_lab.tinymoe.variant); CPU, no weights downloaded."""
import hashlib
import json

import pytest

pytest.importorskip("transformers")

from megakernel_lab.tinymoe import model, native, pins, variant  # noqa: E402


def fake_public(root):
    """A public-snapshot stand-in: the pinned config and its identity record (no weights)."""
    root.mkdir()
    config = dict(pins.EXPECTED_CONFIG, architectures=["MixtralForCausalLM"], initializer_range=0.02)
    raw = json.dumps(config, indent=2).encode()
    (root / "config.json").write_bytes(raw)
    identity = dict(schema="latch-tinymoe-public-snapshot-v1", **pins.public_identity(),
                    files={"config.json": dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())})
    (root / "latch_identity.json").write_text(json.dumps(identity))
    return root


def test_finer_variant_counts_and_verification(tmp_path):
    public = fake_public(tmp_path / "public")
    fine = tmp_path / "fine"
    ident = variant.create(public, fine, "fine", dict(num_local_experts=32, intermediate_size=384,
                                                      num_experts_per_tok=4))
    # Expert parameters preserved; only the router grows (12 x 384 x 16 extra).
    assert ident["expected_parameters"] == pins.EXPECTED_PARAMETERS + 12 * 384 * 16
    config = native.evaluator_config(fine)
    assert (config.num_local_experts, config.intermediate_size, config.num_experts_per_tok) == (32, 384, 4)
    assert not variant.is_variant(public)
    text = (fine / "config.json").read_text()
    (fine / "config.json").write_text(text.replace('"num_experts_per_tok": 4', '"num_experts_per_tok": 3'))
    with pytest.raises(ValueError):
        variant.config(fine)


def test_variant_loader_builds_edited_architecture(tmp_path):
    import torch
    public = fake_public(tmp_path / "public")
    small = tmp_path / "small"
    variant.create(public, small, "small", dict(num_hidden_layers=1, num_local_experts=32, intermediate_size=384,
                                                num_experts_per_tok=4))
    state = torch.random.get_rng_state()
    loaded = model.load_model(small)
    assert torch.equal(state, torch.random.get_rng_state())  # placeholders never consume the global RNG
    moe = loaded.model.layers[0].block_sparse_moe
    assert (moe.num_experts, moe.top_k, moe.experts[0].w1.weight.shape[0]) == (32, 4, 384)
    assert sum(p.numel() for p in loaded.parameters()) == variant.identity_of(small)["expected_parameters"]


def test_rejects_unsupported_override(tmp_path):
    public = fake_public(tmp_path / "public")
    with pytest.raises(ValueError):
        variant.create(public, tmp_path / "bad", "bad", dict(hidden_size=512))
    with pytest.raises(ValueError):
        variant.create(public, tmp_path / "bad2", "bad2", dict(num_local_experts=2, num_experts_per_tok=4))
