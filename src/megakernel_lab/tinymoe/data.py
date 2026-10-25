"""CPU-only pinned download/tokenization and hash-bound deterministic batches.

Invoke preparation on an admitted CPU worker. Token IDs always come from the
model's own pinned tokenizer, never from the historical GPT-2 corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .pins import (DATASET_CONFIG, DATASET_ID, DATASET_REVISION, MODEL_ID,
                   MODEL_REVISION, public_identity)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_model(directory):
    from huggingface_hub import snapshot_download
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=directory,
                      allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "README.md"])
    files = {p.relative_to(directory).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(directory.rglob("*"))
             if p.is_file() and ".cache" not in p.relative_to(directory).parts
             and p.name != "latch_identity.json"}
    row = dict(schema="latch-tinymoe-public-snapshot-v1", **public_identity(), files=files)
    (directory / "latch_identity.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def prepare(model_dir, out, train_tokens=8_388_608, eval_tokens=131_072):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    model_dir, out = Path(model_dir), Path(out)
    identity = download_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    out.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, limit in (("train", train_tokens), ("validation", eval_tokens), ("test", eval_tokens)):
        rows = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION,
                            split=split, streaming=True)
        destination = out / (split + ".u32")
        text_hash, token_count, documents = hashlib.sha256(), 0, 0
        with destination.open("wb") as stream:
            for row in rows:
                text = row["text"]
                if not text.strip():
                    continue
                encoded = text.encode("utf-8")
                text_hash.update(len(encoded).to_bytes(8, "little")); text_hash.update(encoded)
                # Preserve tokenizer policy explicitly: one BOS and EOS per nonempty row.
                ids = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                ids = ids[:limit - token_count]
                np.asarray(ids, dtype="<u4").tofile(stream)
                token_count += len(ids); documents += 1
                if token_count == limit:
                    break
        if token_count < limit:
            raise ValueError(f"{split} has only {token_count} tokens; requested {limit}")
        splits[split] = dict(file=destination.name, tokens=token_count, rows=documents,
                             raw_text_sha256=text_hash.hexdigest(), sha256=digest(destination))
    manifest = dict(schema="latch-tinymoe-data-v1", **public_identity(),
                    model_snapshot_identity_sha256=digest(model_dir / "latch_identity.json"),
                    tokenizer_files={k: v for k, v in identity["files"].items() if "token" in k or k.endswith(".model")},
                    token_dtype="<u4", row_policy="skip blank; BOS + encode(add_special_tokens=False) + EOS",
                    window_policy="contiguous packed stream, causal across row boundaries; each batch consumes B*S new input positions and one lookahead",
                    splits=splits)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class TokenBatches:
    def __init__(self, root, split, batch_size, sequence_length, *, verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["model_revision"] != MODEL_REVISION or self.manifest["tokenizer_revision"] != MODEL_REVISION:
            raise ValueError("data/model tokenizer revision mismatch")
        spec = self.manifest["splits"][split]
        path = self.root / spec["file"]
        if verify and digest(path) != spec["sha256"]:
            raise ValueError("token corpus hash mismatch")
        self.tokens = np.memmap(path, dtype="<u4", mode="r")
        self.batch_size, self.sequence_length = int(batch_size), int(sequence_length)
        self.width = self.batch_size * self.sequence_length
        self.count = (len(self.tokens) - 1) // self.width
        if self.count < 1 or self.sequence_length < 2:
            raise ValueError("insufficient tokens or invalid sequence length")

    def batch(self, update, device="cpu"):
        import torch
        start = (int(update) % self.count) * self.width
        # Independent fixed-length examples; positions restart at zero each row.
        x = np.array(self.tokens[start:start+self.width], dtype=np.int64).reshape(self.batch_size, self.sequence_length)
        y = np.array(self.tokens[start+1:start+self.width+1], dtype=np.int64).reshape(self.batch_size, self.sequence_length)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download"); d.add_argument("--model-dir", required=True)
    d = sub.add_parser("prepare"); d.add_argument("--model-dir", required=True); d.add_argument("--out", required=True)
    d.add_argument("--train-tokens", type=int, default=8_388_608); d.add_argument("--eval-tokens", type=int, default=131_072)
    a = p.parse_args()
    row = download_model(a.model_dir) if a.command == "download" else prepare(a.model_dir, a.out, a.train_tokens, a.eval_tokens)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
