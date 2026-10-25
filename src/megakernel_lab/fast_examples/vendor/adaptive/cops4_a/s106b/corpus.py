"""S106B long-document byte corpus (built on Modal CPU; never on the Mac).

Source: HuggingFaceFW/fineweb-edu @ 87f09149 (the lane family's pinned revision), sample/10BT/000_00000.parquet,
declared sha256 verified. Rows are read in file order from row 20,000 (S21A used rows < 13,826). UTF-8 (strict) bytes
of `text`, no normalization; exact dedupe (first occurrence) by text / id / URL sha256. A document is retained iff it
has >= UNIT = 8193 bytes; its first 8193 bytes form ONE unit (8192 inputs, 8192 next-byte targets). Documents with
>= 16385 bytes also store their 16385-byte prefix (for an optional 16K stage). Reading stops at TARGET_DOCS.

Slices (content-blind): retained documents ranked by sha256(salt || text_sha256); a document is admitted iff its unit's
fraction of distinct 64-byte shingles occurring in any OTHER retained unit is < 0.25 (S100E/S102D/S105F rule). The
first 128 admitted are DEV (selection only), the next 256 are E (targets/confirmation only, read after FREEZE). Slice
documents never enter the training pool. Every training sequence is one unit (8K arms) or one aligned quarter of a
unit (2K arms): no sequence crosses a document, positions restart per sequence.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO_ID = 'HuggingFaceFW/fineweb-edu'
REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
FILENAME = 'sample/10BT/000_00000.parquet'
DECLARED_SHA256 = 'b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871'
SKIP_ROWS = 20000
UNIT = 8193
LONG = 16385
TARGET_DOCS = 13000
SHINGLE = 64
DUP_THRESHOLD = 0.25
SALT = b'S106B-slices-v1'
SALT_STREAM = 'S106B-stream-v1'
DEV_N, E_N = 128, 256
MULT = np.uint64(0x9E3779B97F4A7C15)


def shingle_hashes(a, k=SHINGLE):
    n = len(a) - k + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over='ignore'):
        for j in range(k):
            h = h * MULT + a[j:j + n].astype(np.uint64)
    return h


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def build(out):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise RuntimeError('corpus already built (immutable): ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(REPO_ID, FILENAME, repo_type='dataset', revision=REVISION, local_dir='/tmp/hf')
    file_sha = sha256_file(local)
    if file_sha != DECLARED_SHA256:
        raise RuntimeError('source file sha256 differs from the declared identity')
    seen = {k: set() for k in ('text_sha256', 'id_sha256', 'url_sha256')}
    rows, units, longs = [], [], []
    counts = dict(examined=0, skipped_prefix=0, duplicates=0, short=0, retained=0, long16=0)
    lengths = []
    row = 0
    done = False
    for batch in pq.ParquetFile(local).iter_batches(columns=['text', 'id', 'url'], batch_size=4096):
        cols = batch.to_pydict()
        for text, did, url in zip(cols['text'], cols['id'], cols['url']):
            r, row = row, row + 1
            if r < SKIP_ROWS:
                counts['skipped_prefix'] += 1
                continue
            counts['examined'] += 1
            b = text.encode('utf-8')
            keys = dict(text_sha256=hashlib.sha256(b).hexdigest(), id_sha256=hashlib.sha256(did.encode()).hexdigest(),
                        url_sha256=hashlib.sha256(url.encode()).hexdigest())
            if any(keys[k] in seen[k] for k in seen):
                counts['duplicates'] += 1
                continue
            for k in seen:
                seen[k].add(keys[k])
            lengths.append(len(b))
            if len(b) < UNIT:
                counts['short'] += 1
                continue
            li = -1
            if len(b) >= LONG:
                li = len(longs)
                longs.append(np.frombuffer(b[:LONG], dtype=np.uint8))
                counts['long16'] += 1
            rows.append(dict(doc=len(rows), source_row=r, bytes=len(b), long_index=li, **keys))
            units.append(np.frombuffer(b[:UNIT], dtype=np.uint8))
            counts['retained'] += 1
            if len(rows) >= TARGET_DOCS:
                done = True
                break
        if done:
            break
    if not done:
        raise RuntimeError('source file exhausted before TARGET_DOCS')
    arr = np.stack(units)
    del units
    t_read = time.time()
    # exact shingle universe over all retained units (a shingle counted once per unit)
    hs = [np.unique(shingle_hashes(arr[i])) for i in range(len(arr))]
    H = np.concatenate(hs)
    uniq, count = np.unique(H, return_counts=True)
    del H
    ranked = sorted(rows, key=lambda r: hashlib.sha256(SALT + bytes.fromhex(r['text_sha256'])).hexdigest())
    chosen, excluded, overlaps = [], [], {}
    for r in ranked:
        if len(chosen) == DEV_N + E_N:
            break
        hsd = hs[r['doc']]
        pos = np.searchsorted(uniq, hsd)
        frac = float((count[pos] >= 2).mean())
        overlaps[r['doc']] = frac
        if frac < DUP_THRESHOLD:
            chosen.append(r)
        else:
            excluded.append(dict(doc=r['doc'], overlap=frac))
    dev = [r['doc'] for r in chosen[:DEV_N]]
    heldout = [r['doc'] for r in chosen[DEV_N:]]
    reserved = set(dev) | set(heldout)
    pool = [r['doc'] for r in rows if r['doc'] not in reserved]
    np.save(out / 'units.npy', arr)
    if longs:
        np.save(out / 'long16.npy', np.stack(longs))
    with open(out / 'documents.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    slices = dict(schema='s106b-slices-v1', salt=SALT.decode(), dev=dev, heldout=heldout, excluded=excluded,
                  rule='salted content-blind ranking; admit iff < 0.25 of the unit\'s distinct 64-byte shingles occur '
                       'in any other retained unit; first 128 DEV, next 256 E; slice documents excluded from the pool')
    slices['slices_sha256'] = hashlib.sha256(json.dumps(slices, sort_keys=True).encode()).hexdigest()
    (out / 'SLICES.json').write_text(json.dumps(slices, indent=1, sort_keys=True) + '\n')
    ov = np.array([overlaps[d] for d in reserved])
    L = np.array(lengths)
    manifest = dict(schema='s106b-corpus-v1',
                    source=dict(repo=REPO_ID, revision=REVISION, file=FILENAME, file_sha256=file_sha,
                                declared_match=True, license='ODC-By-1.0 (dataset); underlying content rights apply'),
                    skip_rows=SKIP_ROWS, rows_read=row, counts=counts, unit_bytes=UNIT, long_bytes=LONG,
                    retained_documents=len(rows), pool_documents=len(pool), long16_documents=len(longs),
                    long16_pool_documents=sum(1 for r in rows if r['long_index'] >= 0 and r['doc'] not in reserved),
                    units_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                    index_sha256=sha256_file(out / 'documents.jsonl'), slices_sha256=slices['slices_sha256'],
                    slice_overlap_max=float(ov.max()), slice_overlap_mean=float(ov.mean()),
                    deduplicated_length_quantiles={str(q): float(np.quantile(L, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
                    fraction_ge_unit=float((L >= UNIT).mean()), fraction_ge_long=float((L >= LONG).mean()),
                    seconds=dict(read=t_read - t0, total=time.time() - t0),
                    scope='exact 64-byte windows only; paraphrase/near-template similarity not detected')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


class Corpus:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.units = np.load(self.root / 'units.npy')
        self.rows = [json.loads(x) for x in (self.root / 'documents.jsonl').read_text().splitlines()]
        self.slices = json.loads((self.root / 'SLICES.json').read_text())
        body = {k: v for k, v in self.slices.items() if k != 'slices_sha256'}
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != self.slices['slices_sha256'] \
                or self.slices['slices_sha256'] != self.manifest['slices_sha256']:
            raise ValueError('slice identity mismatch')
        self.reserved = set(self.slices['dev']) | set(self.slices['heldout'])

    def verify(self):
        return hashlib.sha256(self.units.tobytes()).hexdigest() == self.manifest['units_sha256']

    def stream(self, seed):
        """Pool documents in the seed's order: sha256('S106B-stream-v1:{seed}:{text_sha256}')."""
        pool = [r for r in self.rows if r['doc'] not in self.reserved]
        pool.sort(key=lambda r: hashlib.sha256(f'{SALT_STREAM}:{seed}:{r["text_sha256"]}'.encode()).hexdigest())
        return [r['doc'] for r in pool]

    def slice_docs(self, name):
        return list(self.slices[{'dev': 'dev', 'E': 'heldout'}[name]])
