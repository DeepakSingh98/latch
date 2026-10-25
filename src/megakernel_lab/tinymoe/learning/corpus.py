"""TinyMoE-tokenized windows of the pinned learning-scaling FineWeb-Edu documents (CPU worker only).

Source documents: the learning-scaling corpus `fineweb-edu-gpt2-2048-v1` (manifest SHA256
75e67515...8590) lists every selected document with its pinned parquet shard, row group,
row ordinal, raw-content SHA256 and train/validation/test split (content-hash split, global
exact dedup). This job re-reads exactly those rows from the same pinned FineWeb-Edu shards
(HuggingFaceFW/fineweb-edu @ 87f09149..., full-shard SHA256 verified), verifies each text's
raw SHA256, and tokenizes it with TinyMoE's own tokenizer (AutoTokenizer from the hash-pinned
public snapshot): BOS + encode(text, add_special_tokens=False) + EOS, as the donor corpus.

Windows: 1025 consecutive tokens inside ONE document, stride 1024 (context position 0 of a
window repeats the previous window's last target; no target is reused), incomplete tails
discarded. Splits are the source corpus's document splits, so development (validation) and
final test documents stay disjoint from training and each other. Training order is a
PCG64(order seed) permutation applied at load time (protocol.Windows).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import struct
import time
import urllib.request

SOURCE_MANIFEST_SHA256 = "75e67515b176d59a594c4727b0e0e8f7131427d22bec3c56fcd4d3949dba8590"
SPLITS = ("train", "validation", "test")
WINDOW, STRIDE = 1025, 1024
_TOKENIZER = None


def selected_documents(corpus, *, documents=None, selection_index=None, selection_receipt=None):
    """Read original JSONL or a hash-bound compact index of the same public rows.

    Compact records retain document order, shard, row group, row ordinal and
    split. Original text integrity follows from the full-shard SHA256 check.
    The derivation receipt binds the original complete JSONL and source manifest.
    """
    from .protocol import digest
    wanted = {}
    if documents is not None:
        if digest(documents) != corpus["files"]["documents.jsonl"]["sha256"]:
            raise ValueError("documents.jsonl differs from its manifest")
        with Path(documents).open() as stream:
            for line in stream:
                r = json.loads(line)
                wanted.setdefault(r["source_shard"], {})[r["source_row_ordinal"]] = r
        return wanted
    if selection_index is None or selection_receipt is None:
        raise ValueError("provide --documents or both --selection-index and --selection-receipt")
    receipt = json.loads(Path(selection_receipt).read_text())
    if (receipt["source_documents_sha256"] != corpus["files"]["documents.jsonl"]["sha256"]
            or receipt["source_manifest_sha256"] != SOURCE_MANIFEST_SHA256
            or digest(selection_index) != receipt["index_sha256"]
            or receipt["record_format"] != "<QIIIB"):
        raise ValueError("compact selection identity mismatch")
    shards = corpus["source"]["scanned_shards"]
    count, previous = 0, -1
    with gzip.open(selection_index, "rb") as stream:
        while raw := stream.read(21):
            if len(raw) != 21:
                raise ValueError("truncated compact selection record")
            index, shard, group, ordinal, split = struct.unpack("<QIIIB", raw)
            if index <= previous or shard >= len(shards) or split >= len(SPLITS):
                raise ValueError("invalid compact selection record")
            rows = wanted.setdefault(shards[shard]["path"], {})
            if ordinal in rows:
                raise ValueError("duplicate compact selection row")
            rows[ordinal] = dict(document_index=index, source_shard=shards[shard]["path"],
                                 source_row_group=group, source_row_ordinal=ordinal, split=SPLITS[split])
            previous, count = index, count + 1
    if count != receipt["records"]:
        raise ValueError("compact selection record count mismatch")
    return wanted


def _init(model_dir):
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def _encode(text):
    t = _TOKENIZER
    return [t.bos_token_id] + t.encode(text, add_special_tokens=False) + [t.eos_token_id]


def _download(url, path, sha256, size):
    h, n = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={"User-Agent": "latch-tinymoe-smallbatch/1",
                                                   "Accept-Encoding": "identity"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as out:
                h, n = hashlib.sha256(), 0
                while chunk := response.read(8 << 20):
                    n += len(chunk)
                    if n > size:
                        raise ValueError("shard exceeds pinned size")
                    h.update(chunk)
                    out.write(chunk)
            break
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    if n != size or h.hexdigest() != sha256:
        raise ValueError("full shard SHA256/size mismatch: " + url)
    return dict(url=url, bytes=n, sha256=sha256, full_shard_sha256_verified=True)


def main():
    import numpy as np
    import pyarrow.parquet as pq
    from megakernel_lab.tinymoe.data import digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--source-manifest", type=Path, required=True,
                   help="pinned FineWeb-Edu selection corpus.json")
    p.add_argument("--documents", type=Path,
                   help="documents.jsonl bound by the selection manifest")
    p.add_argument("--selection-index", type=Path, help="compact selected-rows.bin.gz alternative to JSONL")
    p.add_argument("--selection-receipt", type=Path, help="hash-bound compact selection receipt")
    p.add_argument("--out", required=True)
    p.add_argument("--scratch", required=True, help="disposable local directory for verified shards")
    p.add_argument("--workers", type=int, default=30)
    a = p.parse_args()
    out, scratch = Path(a.out), Path(a.scratch)
    out.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source = {}
    for name, path in (("corpus.json", a.source_manifest),):
        source[name] = dict(sha256=digest(path), bytes=path.stat().st_size)
    if source["corpus.json"]["sha256"] != SOURCE_MANIFEST_SHA256:
        raise ValueError("source corpus manifest differs from the pinned learning-scaling corpus")
    corpus = json.loads(a.source_manifest.read_text())
    wanted = selected_documents(corpus, documents=a.documents, selection_index=a.selection_index,
                                selection_receipt=a.selection_receipt)
    model_identity = json.loads((Path(a.model_dir) / "latch_identity.json").read_text())
    files = {split: (out / f"{split}.u16").open("xb") for split in SPLITS}
    coords = {split: (out / f"{split}.coordinates.u64").open("xb") for split in SPLITS}
    stats = {split: dict(documents=0, windows=0, tokens_including_bos_eos=0, discarded_tail_tokens=0) for split in SPLITS}
    shard_receipts, last_index = [], -1
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.workers, mp_context=context, initializer=_init, initargs=(a.model_dir,)) as pool:
        for shard in corpus["source"]["scanned_shards"]:
            rows = wanted.get(shard["path"])
            if not rows:
                continue
            local = scratch / Path(shard["path"]).name
            shard_receipts.append(_download(shard["url"], local, shard["sha256"], shard["bytes"]))
            parquet = pq.ParquetFile(local)
            ordinal, texts, records = 0, [], []
            for group in range(parquet.num_row_groups):
                for batch in parquet.iter_batches(batch_size=128, row_groups=[group], columns=["text"], use_threads=False):
                    for text in batch.column(0).to_pylist():
                        r = rows.get(ordinal)
                        if r is not None:
                            raw = text.encode("utf-8")
                            if (r["source_row_group"] != group
                                    or ("utf8_bytes" in r and len(raw) != r["utf8_bytes"])
                                    or ("raw_content_sha256" in r and hashlib.sha256(raw).hexdigest() != r["raw_content_sha256"])):
                                raise ValueError(f"pinned row mismatch {shard['path']}:{ordinal}")
                            texts.append(text)
                            records.append(r)
                        ordinal += 1
            parquet.close()
            if len(records) != len(rows):
                raise ValueError("missing selected rows in " + shard["path"])
            for r, ids in zip(records, pool.map(_encode, texts, chunksize=16)):
                if r["document_index"] <= last_index:
                    raise ValueError("document order differs from source scan order")
                last_index = r["document_index"]
                split = r["split"]
                array = np.asarray(ids, dtype=np.int64)
                if array.max() >= 65536:
                    raise ValueError("token id exceeds uint16")
                take = (len(array) - 1) // STRIDE
                st = stats[split]
                st["documents"] += 1
                st["tokens_including_bos_eos"] += len(array)
                st["discarded_tail_tokens"] += len(array) - 1 - take * STRIDE
                if take:
                    view = np.lib.stride_tricks.sliding_window_view(array.astype("<u2"), WINDOW)[::STRIDE][:take]
                    files[split].write(np.ascontiguousarray(view).tobytes())
                    c = np.empty((take, 2), dtype="<u8")
                    c[:, 0] = r["document_index"]
                    c[:, 1] = np.arange(take, dtype=np.uint64) * STRIDE
                    coords[split].write(c.tobytes())
                    st["windows"] += take
            local.unlink()
            print(json.dumps(dict(event="shard_done", shard=shard["path"], windows={s: stats[s]["windows"] for s in SPLITS},
                                  elapsed=time.time() - started)), flush=True)
    for f in (*files.values(), *coords.values()):
        f.close()
    splits = {}
    for split in SPLITS:
        splits[split] = dict(file=f"{split}.u16", sha256=digest(out / f"{split}.u16"),
                             targets=stats[split]["windows"] * STRIDE, coordinates=f"{split}.coordinates.u64",
                             coordinates_sha256=digest(out / f"{split}.coordinates.u64"), **stats[split])
    tokenizer_files = {k: v for k, v in model_identity["files"].items() if "token" in k or k.endswith(".model")}
    manifest = dict(schema="latch-tinymoe-smallbatch-windows-v1",
                    source=dict(corpus_manifest_sha256=SOURCE_MANIFEST_SHA256,
                                documents_sha256=corpus["files"]["documents.jsonl"]["sha256"],
                                fineweb=corpus["source"]["repo"], revision=corpus["source"]["revision"],
                                shards=shard_receipts, split_policy=corpus["split_policy"],
                                license=corpus["license"], attribution=corpus.get("source_attribution")),
                    tokenizer=dict(model_id=model_identity["model_id"], revision=model_identity["model_revision"],
                                   snapshot_identity_sha256=digest(Path(a.model_dir) / "latch_identity.json"),
                                   files=tokenizer_files, api="transformers.AutoTokenizer (fast), add_special_tokens=False"),
                    token_policy="BOS + exact original text tokens + EOS per selected source document",
                    window_policy="1025 tokens inside one document, stride 1024; 1024 inputs + 1024 supplied next-token targets; tails discarded",
                    order_policy="train: PCG64(order_seed) permutation at load; validation/test: source order",
                    token_dtype="<u2", window=WINDOW, splits=splits, seconds=time.time() - started)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(event="corpus_complete", windows={s: splits[s]["windows"] for s in SPLITS},
                          seconds=manifest["seconds"])), flush=True)


if __name__ == "__main__":
    main()
