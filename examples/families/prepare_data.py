"""Prepare the real datasets used by the family examples (CPU only; no GPU, no credentials).

    python examples/families/prepare_data.py text    --out data/fineweb-bytes        # FineWeb-edu bytes (~64 MiB)
    python examples/families/prepare_data.py cifar10 --out data/cifar10              # 170 MB download
    python examples/families/prepare_data.py fashion-mnist --out data/fashion-mnist  # 30 MB download

text
    UTF-8 bytes of HuggingFaceFW/fineweb-edu (ODC-By 1.0), pinned revision 87f09149ef47, file
    sample/10BT/000_00000.parquet: the same source file as the corpus behind the release's text evidence (S105F).
    Only the needed parquet row groups are fetched (HTTP range reads through huggingface_hub), so a 64 MiB corpus does not
    download the 2 GB file. Needs `pip install -r examples/requirements-data.txt` (pyarrow, huggingface_hub).
    `--text-files a.txt b.txt` (one document per file) or `--jsonl docs.jsonl` (a "text" field per line) use your own
    text instead (numpy only).
    Documents are exact-deduplicated by text sha256 and kept only if they hold at least 2048 bytes. A content-blind ranking
    (sha256 of salt || text sha256) reserves 128 development and 256 held-out documents; training windows never overlap
    them. Unlike the release's S105F corpus, no near-duplicate shingle audit is applied to the reserved documents.
    Output (the layout read by scripts/c4a/common.py Text and the recurrence reader): corpus.npy (uint8 bytes),
    documents.jsonl, SLICES.json, manifest.json.

cifar10
    The official Toronto python archive (md5-checked) -> train.npz / test.npz with uint8 x [N, 3, 32, 32] (CHW) and int64
    y [N], plus SOURCE.json. examples/pytorch/train_timm_vit.py reads this directory.

fashion-mnist
    The official idx archives -> train.npy / test.npy uint8 [N, 784] and train_labels.npy / test_labels.npy, plus
    SOURCE.json. examples/families/train_dit.py reads this directory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

FINEWEB = dict(repo='HuggingFaceFW/fineweb-edu', revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
               file='sample/10BT/000_00000.parquet', license='ODC-By-1.0 (dataset); underlying content rights apply')
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_MD5 = 'c58f30108f718f92721af3b95e74349a'
FMNIST_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
FMNIST_FILES = dict(train='train-images-idx3-ubyte.gz', train_labels='train-labels-idx1-ubyte.gz',
                    test='t10k-images-idx3-ubyte.gz', test_labels='t10k-labels-idx1-ubyte.gz')
SEG = 2048
SALT = b'latch-examples-slices-v1'


def fresh(out):
    out = Path(out)
    if (out / 'manifest.json').exists() or (out / 'SOURCE.json').exists():
        raise SystemExit(f'{out} already holds prepared data; choose a new --out')
    out.mkdir(parents=True, exist_ok=True)
    return out


def download(url, timeout=300):
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read()
    print(f'downloaded {url} ({len(raw) / 2**20:.1f} MiB, {time.time() - t0:.0f} s)', flush=True)
    return raw


# ------------------------------------------------------------------------------------------------------------- text
def fineweb_documents(skip_rows):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    path = f'datasets/{FINEWEB["repo"]}@{FINEWEB["revision"]}/{FINEWEB["file"]}'
    pf = pq.ParquetFile(HfFileSystem().open(path, 'rb', block_size=8 << 20))
    row = 0
    for batch in pf.iter_batches(columns=['text'], batch_size=1024):
        for text in batch.column('text').to_pylist():
            row += 1
            if row > skip_rows:
                yield row - 1, text


def local_documents(text_files, jsonl):
    if text_files:
        for i, f in enumerate(text_files):
            yield i, Path(f).read_text(encoding='utf-8')
    else:
        with open(jsonl, encoding='utf-8') as f:
            for i, line in enumerate(f):
                yield i, json.loads(line)['text']


def prepare_text(a):
    out = fresh(a.out)
    target = int(a.target_mib * 2**20)
    docs = (local_documents(a.text_files, a.jsonl) if (a.text_files or a.jsonl) else fineweb_documents(a.skip_rows))
    rows, chunks, seen, off = [], [], set(), 0
    counts = dict(examined=0, duplicates=0, short=0, retained=0)
    t0 = time.time()
    for source_row, text in docs:
        counts['examined'] += 1
        b = text.encode('utf-8')
        h = hashlib.sha256(b).hexdigest()
        if h in seen:
            counts['duplicates'] += 1
            continue
        seen.add(h)
        if len(b) < SEG:
            counts['short'] += 1
            continue
        rows.append(dict(doc=len(rows), source_row=source_row, offset=off, bytes=len(b), text_sha256=h))
        chunks.append(np.frombuffer(b, dtype=np.uint8))
        off += len(b)
        counts['retained'] += 1
        if counts['retained'] % 2000 == 0:
            print(f'{counts["retained"]} documents, {off / 2**20:.1f} MiB', flush=True)
        if off >= target:
            break
    need = a.dev + a.heldout
    if len(rows) < need + 64:
        raise SystemExit(f'only {len(rows)} documents of >= {SEG} bytes; need more than {need + 64}')
    arr = np.concatenate(chunks)
    ranked = sorted(rows, key=lambda r: hashlib.sha256(SALT + bytes.fromhex(r['text_sha256'])).hexdigest())

    def seg(r):
        return dict(doc=r['doc'], start=r['offset'], document_offset=0, document_sha256=r['text_sha256'])
    slices = dict(schema='latch-examples-slices-v1', salt=SALT.decode(), segment_bytes=SEG,
                  development=[seg(r) for r in ranked[:a.dev]], heldout=[seg(r) for r in ranked[a.dev:need]],
                  excluded_candidates=[],
                  rule='rank documents by sha256(salt || text_sha256); first dev = development, next heldout = held-out; '
                       'reserved documents are excluded from training windows (no near-duplicate audit)')
    slices['slices_sha256'] = hashlib.sha256(json.dumps(slices, sort_keys=True).encode()).hexdigest()
    np.save(out / 'corpus.npy', arr)
    with open(out / 'documents.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    (out / 'SLICES.json').write_text(json.dumps(slices, indent=1, sort_keys=True) + '\n')
    source = (dict(local_files=[str(f) for f in a.text_files]) if a.text_files else dict(local_jsonl=str(a.jsonl))
              if a.jsonl else dict(FINEWEB, skip_rows=a.skip_rows))
    pool = [r for r in rows if r['doc'] not in {s['doc'] for s in slices['development'] + slices['heldout']}]
    manifest = dict(schema='latch-examples-text-v1', source=source, counts=counts, retained_documents=len(rows),
                    corpus_bytes=int(arr.nbytes), corpus_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                    slices_sha256=slices['slices_sha256'], pool_documents=len(pool),
                    pool_segments_2048=sum(r['bytes'] // SEG for r in pool), seconds=time.time() - t0)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: manifest[k] for k in ('corpus_bytes', 'retained_documents', 'pool_segments_2048', 'corpus_sha256')}))


# ------------------------------------------------------------------------------------------------------------ images
def prepare_cifar10(a):
    out = fresh(a.out)
    raw = download(CIFAR_URL)
    if hashlib.md5(raw).hexdigest() != CIFAR_MD5:
        raise SystemExit('CIFAR-10 archive md5 mismatch')
    parts = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as t:
        for m in t.getmembers():
            name = Path(m.name).name
            if name.startswith('data_batch_') or name == 'test_batch':
                parts[name] = pickle.load(t.extractfile(m), encoding='bytes')

    def split(names):
        x = np.concatenate([parts[n][b'data'] for n in names]).reshape(-1, 3, 32, 32).astype(np.uint8)
        y = np.concatenate([np.asarray(parts[n][b'labels']) for n in names]).astype(np.int64)
        return x, y
    files = {}
    for name, members in (('train', [f'data_batch_{i}' for i in range(1, 6)]), ('test', ['test_batch'])):
        x, y = split(members)
        np.savez(out / f'{name}.npz', x=x, y=y)
        files[f'{name}.npz'] = dict(images=len(y), sha256=hashlib.sha256((out / f'{name}.npz').read_bytes()).hexdigest())
    src = dict(schema='latch-examples-cifar10-v1', url=CIFAR_URL, md5=CIFAR_MD5,
               layout='x uint8 [N, 3, 32, 32] CHW, y int64 [N] labels 0-9', files=files)
    (out / 'SOURCE.json').write_text(json.dumps(src, indent=1) + '\n')
    print(json.dumps(files))


def prepare_fashion_mnist(a):
    out = fresh(a.out)
    files = {}
    for key, name in FMNIST_FILES.items():
        raw = gzip.decompress(download(FMNIST_URL + name))
        arr = np.frombuffer(raw, dtype=np.uint8, offset=16 if 'labels' not in key else 8)
        arr = arr.reshape(-1, 784) if 'labels' not in key else arr.astype(np.int64)
        np.save(out / f'{key}.npy', arr)
        files[f'{key}.npy'] = dict(rows=int(arr.shape[0]), source=name, sha256=hashlib.sha256(raw).hexdigest())
    src = dict(schema='latch-examples-fashion-mnist-v1', url=FMNIST_URL, layout='images uint8 [N, 784] (28 x 28)', files=files)
    (out / 'SOURCE.json').write_text(json.dumps(src, indent=1) + '\n')
    print(json.dumps({k: v['rows'] for k, v in files.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', choices=('text', 'cifar10', 'fashion-mnist'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--target-mib', type=float, default=64.0, help='text: stop after this many retained bytes')
    ap.add_argument('--skip-rows', type=int, default=0, help='text: FineWeb rows to skip first')
    ap.add_argument('--text-files', nargs='+', help='text: your own UTF-8 files, one document each')
    ap.add_argument('--jsonl', help='text: your own JSON lines with a "text" field')
    ap.add_argument('--dev', type=int, default=128)
    ap.add_argument('--heldout', type=int, default=256)
    a = ap.parse_args()
    dict(text=prepare_text, cifar10=prepare_cifar10, **{'fashion-mnist': prepare_fashion_mnist})[a.dataset](a)


if __name__ == '__main__':
    main()
