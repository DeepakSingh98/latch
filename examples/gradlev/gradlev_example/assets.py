# Provenance: adapted from the Latch repository, commit df184a5f2579 (branch codex/latch-gradlev),
# research/latch_gradlev/gl/assets.py (git blob 515bfb384f56889f4a1244c21174798b75198f0f).
# Kept: the GPT-2 pin (repository, revision, sha256, size) and the download-once/verify rule of ensure().
# Removed: the GCE metadata-token request, the private Google Cloud Storage corpus prefix, upload() and fetch_gs()
# (cloud staging helpers) and the cloud-worker cache default.
# Added: GRADLEV_ROOT handling, the pinned public tokenizer, and build_corpus(), which rebuilds the
# donor's evaluation windows from public FineWeb-Edu. build_corpus() re-implements the validation/test
# selection rule of megakernel_lab/moe4/_support/learning_scaling/prepare_data.py (same commit, git blob
# ce91c9a2779a7c34da2562a1b39d38c8c9b42cc3; the preparation that produced the donor's private
# corpus 'fineweb-edu-gpt2-2048-v1', manifest research/latch_v0/corpus-original.json, git blob
# b767ad189b05b1d379aae7698a61c1dc96f4cd78) restricted to the held-out splits.
"""Pinned, hash-verified public inputs for the GradLev example.

GPT-2 small: openai-community/gpt2 at revision 607a30d783dfa663caf39e06633721c8d4cfcd7e (MIT).
Text: HuggingFaceFW/fineweb-edu (ODC-By 1.0; underlying Common Crawl terms apply) at revision
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, subset sample/10BT, read in the donor's shard order.

The donor read a private, already-tokenized copy of this corpus. Its held-out splits are a
deterministic function of public data: documents are taken in a fixed shard/row order, assigned to
train/validation/test by a hash of their whitespace-collapsed text, exactly deduplicated, tokenized
with the pinned GPT-2 tokenizer (EOS appended) and cut into 2049-token windows (stride 2048) inside
one document. build_corpus() applies that rule to the validation (and optionally test) documents
only. With all 2048 validation windows (or all 8192 test windows) the result is compared with the
donor file's sha256; a shorter prefix cannot be compared against a whole-file hash and is reported
as 'rule-reconstructed, not hash-verified'.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

GPT2_REPO = 'openai-community/gpt2'
GPT2_REV = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
ASSETS = {
    'gpt2.safetensors': (f'https://huggingface.co/{GPT2_REPO}/resolve/{GPT2_REV}/model.safetensors',
                         '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707', 548105171),
    'gpt2-tokenizer.json': (f'https://huggingface.co/{GPT2_REPO}/resolve/{GPT2_REV}/tokenizer.json',
                            '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6', 1355256),
}
TOKENIZER_GIT_BLOB = '4b988bccc9dc5adacd403c00b4704976196548f8'

FINEWEB_REPO = 'HuggingFaceFW/fineweb-edu'
FINEWEB_REV = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
DATA_SEED = 620261006          # shard order = ascending sha256(f'{seed}:{path}')
# The donor preparation's shard order and the full-file identities it verified (git LFS sha256, bytes).
SHARDS = (
    ('sample/10BT/006_00000.parquet', 2152689867, '554fa2613c9261d6c9c396caab881de3160175794c6c3e1856bf28a0e9cb9b76'),
    ('sample/10BT/008_00000.parquet', 2151274846, 'a6d9dcc0c72ecd7c7f0173c2d06b3a4249a2ccaa94961469d1a522ca689bbf1a'),
    ('sample/10BT/007_00000.parquet', 2150686637, '4b7d1f697d4afff7f5f65cb7aa4d83acc6db716da4d46ea23399d0c50608c6e3'),
    ('sample/10BT/001_00000.parquet', 2152222432, '3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693'),
)
SEQUENCE, WINDOW, EOS_ID, VOCAB = 2048, 2049, 50256, 50257
MAX_DOCUMENT_BYTES = 1024 ** 2
# Whole-split identities of the donor's private corpus files (research/latch_v0/corpus-original.json).
DONOR_SPLITS = {'validation': (2048, '0b840cbccc06ec1591eb5fa9613d1f3ba22e086f9f71a59bca5cd213066aaaf8'),
                'test': (8192, '6dfb142e309d03022013cea897cc367d535564c3e005262abba7cfe12a05897f')}


def root():
    value = os.environ.get('GRADLEV_ROOT', '')
    if not value or not os.path.isabs(value):
        raise SystemExit('set GRADLEV_ROOT (or --root) to an absolute output directory')
    return Path(value)


def cache():
    return root() / 'assets'


def data_dir():
    return root() / 'data'


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure(name):
    """Local path of a pinned public asset; downloads and verifies sha256/size once per root."""
    uri, digest, size = ASSETS[name]
    path = cache() / name
    ok = path.with_suffix(path.suffix + '.sha256')
    if path.exists() and ok.exists() and ok.read_text().strip() == digest:
        return path
    cache().mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    h, n = hashlib.sha256(), 0
    request = urllib.request.Request(uri, headers={'User-Agent': 'gradlev-example/1'})
    with urllib.request.urlopen(request, timeout=1800) as response, tmp.open('wb') as out:
        while chunk := response.read(16 << 20):
            h.update(chunk)
            n += len(chunk)
            out.write(chunk)
    if h.hexdigest() != digest or n != size:
        raise ValueError(f'asset {name}: sha256/size mismatch ({h.hexdigest()}, {n})')
    tmp.replace(path)
    ok.write_text(digest + '\n')
    return path


def _tokenizer():
    from tokenizers import Tokenizer
    path = ensure('gpt2-tokenizer.json')
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != TOKENIZER_GIT_BLOB:
        raise ValueError('tokenizer does not match the pinned git blob')
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size(with_added_tokens=True) != VOCAB or tok.token_to_id('<|endoftext|>') != EOS_ID:
        raise ValueError('tokenizer vocabulary mismatch')
    return tok


def split_for(identity):
    """Donor split rule: first 64 bits of sha256(whitespace-collapsed text) mod 10000."""
    bucket = int(identity[:16], 16) % 10000
    return 'train' if bucket < 9000 else 'validation' if bucket < 9500 else 'test'


def build_corpus(validation_windows=128, test_windows=0, log=print):
    """Write data/{split}.u16 ([windows, 2049] little-endian uint16) and data/data.json.

    Reads the pinned shards through HTTP range requests (huggingface_hub HfFileSystem at the pinned
    revision; only the row groups needed are fetched) in the donor's shard and row order and applies
    the donor's held-out selection rule. Train documents are hashed (for the split decision) but never
    tokenized or stored: their windows are not needed for evaluation and do not affect which
    validation/test windows the rule selects (a duplicate always lands in the split of its original).
    """
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    required = {'validation': int(validation_windows), 'test': int(test_windows)}
    for split, n in required.items():
        if not 0 <= n <= DONOR_SPLITS[split][0]:
            raise ValueError(f'{split} windows must be in [0, {DONOR_SPLITS[split][0]}]')
    out = data_dir()
    if (out / 'data.json').exists():
        have = json.loads((out / 'data.json').read_text())
        if {s: have['splits'].get(s, {}).get('windows', 0) for s in required} == required:
            log(f'reusing {out}/data.json')
            return have
        raise SystemExit(f'{out} already holds a different preparation; choose a new GRADLEV_ROOT')
    out.mkdir(parents=True, exist_ok=True)
    tok = _tokenizer()
    fs = HfFileSystem()
    counts = {s: 0 for s in required}
    seen = set()
    streams = {s: (out / f'{s}.u16.partial').open('wb') for s, n in required.items() if n}
    documents = (out / 'documents.jsonl.partial').open('w')
    scanned, started, last = [], time.monotonic(), time.monotonic()
    rows = 0
    try:
        for path, size, lfs_sha in SHARDS:
            if all(counts[s] >= required[s] for s in required):
                break
            handle = fs.open(f'datasets/{FINEWEB_REPO}@{FINEWEB_REV}/{path}', 'rb', block_size=8 << 20)
            parquet = pq.ParquetFile(handle)
            ordinal, groups = 0, 0
            for group in range(parquet.num_row_groups):
                groups += 1
                for records in parquet.iter_batches(batch_size=128, row_groups=[group],
                                                    columns=['text', 'id', 'url'], use_threads=False):
                    for raw in records.to_pylist():
                        row_ordinal = ordinal
                        ordinal += 1
                        rows += 1
                        text = raw.get('text')
                        if not isinstance(text, str):
                            raise ValueError('missing FineWeb text')
                        identity = hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()
                        split = split_for(identity)
                        if split not in required or counts[split] >= required[split]:
                            continue
                        if identity in seen:          # exact normalized-content duplicate
                            continue
                        seen.add(identity)
                        original = text.encode('utf-8')
                        if len(original) > MAX_DOCUMENT_BYTES:
                            continue
                        if len(original) < SEQUENCE:  # byte-level BPE: >= 1 byte per token, cannot fill a window
                            continue
                        ids = tok.encode(text, add_special_tokens=False).ids
                        ids.append(EOS_ID)
                        available = (len(ids) - 1) // SEQUENCE
                        take = min(available, required[split] - counts[split])
                        if take <= 0:
                            continue
                        array = np.asarray(ids, dtype='<u2')
                        view = np.lib.stride_tricks.sliding_window_view(array, WINDOW)[::SEQUENCE][:take]
                        streams[split].write(np.ascontiguousarray(view).tobytes())
                        documents.write(json.dumps({'split': split, 'split_window_start': counts[split],
                                                    'windows': int(take), 'source_shard': path,
                                                    'source_row_group': group, 'source_row_ordinal': row_ordinal,
                                                    'source_id': raw.get('id'), 'source_url': raw.get('url'),
                                                    'raw_content_sha256': hashlib.sha256(original).hexdigest(),
                                                    'tokens_including_eos': len(ids)}, sort_keys=True) + '\n')
                        counts[split] += int(take)
                    if time.monotonic() - last >= 30:
                        log(f'windows {counts} of {required}; rows scanned {rows}; {time.monotonic() - started:.0f} s')
                        last = time.monotonic()
                    if all(counts[s] >= required[s] for s in required):
                        break
                if all(counts[s] >= required[s] for s in required):
                    break
            scanned.append({'path': path, 'git_lfs_sha256': lfs_sha, 'bytes': size,
                            'row_groups_read': groups, 'rows_read': ordinal})
            handle.close()
    finally:
        for stream in (*streams.values(), documents):
            stream.close()
    if counts != required:
        raise ValueError(f'pinned shards exhausted before {required} windows ({counts})')
    splits = {}
    for split, n in required.items():
        if not n:
            continue
        final = out / f'{split}.u16'
        (out / f'{split}.u16.partial').replace(final)
        digest = _sha256_file(final)
        full_n, donor_digest = DONOR_SPLITS[split]
        if n == full_n:
            if digest != donor_digest:
                raise ValueError(f'{split}: rebuilt sha256 {digest} differs from the donor corpus {donor_digest}')
            status = 'hash-verified equal to the donor split'
        else:
            status = f'rule-reconstructed prefix ({n} of {full_n} windows); not hash-verified'
        splits[split] = {'windows': n, 'file': final.name, 'bytes': final.stat().st_size, 'sha256': digest,
                         'donor_identity': status}
    (out / 'documents.jsonl.partial').replace(out / 'documents.jsonl')
    manifest = {'schema': 'gradlev-example-heldout-v1', 'splits': splits,
                'source': {'repo': FINEWEB_REPO, 'revision': FINEWEB_REV, 'subset': 'sample/10BT',
                           'shard_order_seed': DATA_SEED, 'scanned': scanned, 'license': 'ODC-By 1.0',
                           'additional_terms': 'https://commoncrawl.org/terms-of-use'},
                'tokenizer': {'repo': GPT2_REPO, 'revision': GPT2_REV, 'git_blob_sha1': TOKENIZER_GIT_BLOB,
                              'sha256': ASSETS['gpt2-tokenizer.json'][1]},
                'rule': 'donor learning-scaling-fineweb-gpt2-2048-v1 held-out rule: sha256 of whitespace-collapsed '
                        'text, first 64 bits mod 10000 (<9000 train, <9500 validation, else test); exact '
                        'normalized-content dedup; documents > 1 MiB skipped; add_special_tokens=False + EOS; '
                        '2049-token windows, stride 2048, inside one document; fixed shard/row order',
                'rows_scanned': rows, 'elapsed_seconds': time.monotonic() - started}
    (out / 'data.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    log(json.dumps(splits, indent=1))
    return manifest
