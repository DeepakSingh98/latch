import json
import gzip
import struct

import numpy as np
import pytest

from megakernel_lab.tinymoe.learning import confirm, corpus, protocol, quick_windows


def test_quick_windows_preserve_lookahead_and_reject_corruption(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    splits = {}
    original = np.arange(3073, dtype="<u4")
    for split in ("train", "validation", "test"):
        path = source / (split + ".u32")
        path.write_bytes(original.tobytes())
        splits[split] = dict(file=path.name, tokens=len(original), sha256=protocol.digest(path))
    protocol.write_json(source / "manifest.json", dict(schema="latch-tinymoe-data-v1", token_dtype="<u4", splits=splits))
    out = tmp_path / "windows"
    quick_windows.prepare(source, out)
    windows = protocol.Windows(out, "validation")
    assert windows.count == 3
    assert windows.rows(1, 1).tolist() == [original[1024:2049].tolist()]
    with pytest.raises(ValueError, match="exhausted"):
        windows.rows(3, 1)
    path = out / "validation.u16"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol.Windows(out, "validation")


def test_confirmation_missing_runs_cannot_become_success():
    result = confirm.reduce({})
    assert len(result["missing"]) == 6
    assert all(not pair["complete"] for pair in result["pairs"].values())
    assert result["targets"] == (4.3, 4.1, 3.9, 3.75, 3.6, 3.5, 3.4, 3.3)
    assert all(row["time_ratio_B2_over_A"] is None
               for pair in result["pairs"].values() for row in pair["targets"].values())


def test_confirmation_refuses_an_unfrozen_seed(tmp_path):
    with pytest.raises(ValueError, match="outside the frozen"):
        confirm.reduce({"A:1729": tmp_path})


def test_compact_selection_preserves_coordinates_and_checks_hash(tmp_path):
    index = tmp_path / "rows.bin.gz"
    index.write_bytes(gzip.compress(struct.pack("<QIIIB", 7, 0, 2, 31, 1), mtime=0))
    receipt = tmp_path / "receipt.json"
    protocol.write_json(receipt, dict(source_documents_sha256="original-docs",
        source_manifest_sha256=corpus.SOURCE_MANIFEST_SHA256,
        index_sha256=protocol.digest(index), record_format="<QIIIB", records=1))
    manifest = dict(files={"documents.jsonl": {"sha256": "original-docs"}},
                    source={"scanned_shards": [{"path": "public.parquet"}]})
    rows = corpus.selected_documents(manifest, selection_index=index, selection_receipt=receipt)
    assert rows["public.parquet"][31] == dict(document_index=7, source_shard="public.parquet",
        source_row_group=2, source_row_ordinal=31, split="validation")
    index.write_bytes(index.read_bytes() + b"x")
    with pytest.raises(ValueError, match="identity mismatch"):
        corpus.selected_documents(manifest, selection_index=index, selection_receipt=receipt)
