"""Explicit TinyMoE architecture variants with fresh weights.

The public loaders accept only the pinned FlameF0X/TinyMoE-200m-2x16 checkpoint
(v0 bug B1: architecture edits needed a module-patching wrapper; B3: edited builds
were labelled as the public model). A *variant directory* makes an edit explicit
and verifiable without changing any public pin:

    python -m megakernel_lab.tinymoe.variant create --public MODEL --out DIR \
        --name fine-e32k4 --set num_local_experts=32 intermediate_size=384 num_experts_per_tok=4

DIR holds ``config.json`` (the public config with the overrides applied) and
``latch_variant.json``, which embeds the hash-verified public config text, the
public snapshot identity, the overrides and the expected parameter count. No
weights are stored: a variant always starts from explicitly supplied fresh
weights. Every TinyMoE loader that receives DIR as its model directory
(``model.load_model``, ``native.evaluator_config`` and therefore capture/build,
conventional backends, evaluation and FP64 audits) verifies the variant and
builds the edited architecture; build metadata records ``variant_identity``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VARIANT_FILE = "latch_variant.json"
SCHEMA = "latch-tinymoe-variant-v1"
# Edits whose compiled/conventional semantics are covered by the generic adapter.
ALLOWED = ("num_local_experts", "intermediate_size", "num_experts_per_tok", "num_hidden_layers")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def is_variant(directory):
    return (Path(directory) / VARIANT_FILE).is_file()


def _apply(base, overrides):
    edited = dict(base)
    for key, value in overrides.items():
        if key not in ALLOWED or key not in base:
            raise ValueError("variant override not supported: " + key)
        if type(value) is not int or value < 1:
            raise ValueError("variant overrides must be positive integers")
        edited[key] = value
    if edited["num_experts_per_tok"] > edited["num_local_experts"]:
        raise ValueError("top-k exceeds the expert count")
    return edited


def _expected_parameters(config):
    c = config
    head = c.hidden_size // c.num_attention_heads
    attention = c.hidden_size * head * (c.num_attention_heads * 2 + c.num_key_value_heads * 2)
    experts = c.num_local_experts * 3 * c.hidden_size * c.intermediate_size
    layer = attention + experts + c.hidden_size * c.num_local_experts + 2 * c.hidden_size
    return 2 * c.vocab_size * c.hidden_size + c.num_hidden_layers * layer + c.hidden_size


def create(public, out, name, overrides):
    from .data import digest
    from .pins import EXPECTED_CONFIG, public_identity
    public, out = Path(public), Path(out)
    identity = json.loads((public / "latch_identity.json").read_text())
    if any(identity.get(k) != v for k, v in public_identity().items()):
        raise ValueError("public model/tokenizer identity mismatch")
    path = public / "config.json"
    spec = identity["files"]["config.json"]
    raw = path.read_bytes()
    if len(raw) != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError("public config size/hash mismatch")
    base = json.loads(raw)
    if any(base.get(k) != v for k, v in EXPECTED_CONFIG.items() if k in base):
        raise ValueError("public config differs from the pinned semantics")
    edited = _apply(base, overrides)
    out.mkdir(parents=True, exist_ok=False)
    text = json.dumps(edited, indent=2, sort_keys=True) + "\n"
    (out / "config.json").write_text(text)
    record = dict(schema=SCHEMA, name=name, overrides=dict(sorted(overrides.items())),
                  public_identity=public_identity(), public_config_sha256=spec["sha256"],
                  public_config_text=raw.decode(), config_sha256=_sha(text.encode()))
    record["expected_parameters"] = _expected_parameters(_config_object(out))
    (out / VARIANT_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return identity_of(out)


def _record(directory):
    from .pins import public_identity
    directory = Path(directory)
    record = json.loads((directory / VARIANT_FILE).read_text())
    if record.get("schema") != SCHEMA or record.get("public_identity") != public_identity():
        raise ValueError("variant schema or public identity mismatch")
    raw = record["public_config_text"].encode()
    if _sha(raw) != record["public_config_sha256"]:
        raise ValueError("variant base config text does not match its recorded public hash")
    text = (directory / "config.json").read_bytes()
    if _sha(text) != record["config_sha256"]:
        raise ValueError("variant config.json hash mismatch")
    if json.loads(text) != _apply(json.loads(raw), record["overrides"]):
        raise ValueError("variant config.json is not the public config with the recorded overrides")
    return record


def _config_object(directory):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(Path(directory), local_files_only=True)


def config(directory):
    """Verified edited Transformers config (every unedited field equals the public pin)."""
    from .pins import EXPECTED_CONFIG
    record = _record(directory)
    value = _config_object(directory)
    for key, expected in EXPECTED_CONFIG.items():
        expected = record["overrides"].get(key, expected)
        if getattr(value, key, None) != expected:
            raise ValueError("variant config field differs: " + key)
    return value


def identity_of(directory):
    """Small identity recorded in build/run metadata (None for the public snapshot)."""
    if not is_variant(directory):
        return None
    record = _record(directory)
    return dict(schema=SCHEMA, name=record["name"], overrides=record["overrides"],
                config_sha256=record["config_sha256"], public_config_sha256=record["public_config_sha256"],
                expected_parameters=record["expected_parameters"],
                variant_file_sha256=_sha((Path(directory) / VARIANT_FILE).read_bytes()))


def fresh_model(directory, *, attention="sdpa"):
    """Edited architecture with deterministic placeholder weights (callers load real weights).

    The global RNG is not consumed: placeholders are drawn inside a forked generator.
    """
    import torch
    from transformers import AutoModelForCausalLM
    record = _record(directory)
    value = config(directory)
    value._attn_implementation = attention
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(value, torch_dtype=torch.float32)
    count = sum(p.numel() for p in model.parameters())
    if count != record["expected_parameters"]:
        raise ValueError(f"variant parameter count {count} != expected {record['expected_parameters']}")
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("create")
    s.add_argument("--public", required=True, help="hash-verified public snapshot directory")
    s.add_argument("--out", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--set", nargs="+", required=True, metavar="KEY=INT")
    s = sub.add_parser("verify")
    s.add_argument("directory")
    a = p.parse_args()
    if a.command == "create":
        overrides = {}
        for item in a.set:
            key, _, value = item.partition("=")
            overrides[key] = int(value)
        print(json.dumps(create(a.public, a.out, a.name, overrides), indent=2))
    else:
        fresh_model(a.directory)
        print(json.dumps(identity_of(a.directory), indent=2))


if __name__ == "__main__":
    main()
