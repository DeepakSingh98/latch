"""Convert the pinned public quickstart corpus into 1025-token learning smoke windows.

This preserves the quickstart's packed-stream policy (including row boundaries).
It is a small software check, not the document-isolated FineWeb-Edu experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import WINDOW, SEQUENCE, digest, write_json


def prepare(source, out):
    import numpy as np
    source, out = Path(source), Path(out)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema") != "latch-tinymoe-data-v1" or manifest.get("token_dtype") != "<u4":
        raise ValueError("expected the public TinyMoE quickstart corpus")
    out.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split in ("train", "validation", "test"):
        spec = manifest["splits"][split]
        path = source / spec["file"]
        if digest(path) != spec["sha256"]:
            raise ValueError("source corpus hash mismatch: " + split)
        tokens = np.memmap(path, dtype="<u4", mode="r")
        count = (len(tokens) - 1) // SEQUENCE
        if count < 1 or len(tokens) != spec["tokens"] or int(tokens.max()) >= 65536:
            raise ValueError("invalid source tokens: " + split)
        target = out / f"{split}.u16"
        with target.open("xb") as stream:
            for start in range(0, count, 1024):
                stop = min(count, start + 1024)
                selected = tokens[start * SEQUENCE:stop * SEQUENCE + 1]
                rows = np.lib.stride_tricks.sliding_window_view(selected, WINDOW)[::SEQUENCE]
                stream.write(np.asarray(rows, dtype="<u2").tobytes())
        splits[split] = dict(file=target.name, sha256=digest(target), windows=count,
                             targets=count * SEQUENCE)
    result = dict(schema="latch-tinymoe-smallbatch-windows-v1", purpose="software smoke check only",
                  source_manifest_sha256=digest(source / "manifest.json"),
                  window_policy="1025-token packed-stream windows, stride 1024; may cross source rows",
                  order_policy="train PCG64(order_seed); validation/test source order",
                  token_dtype="<u2", window=WINDOW, splits=splits)
    write_json(out / "manifest.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.out), indent=2))


if __name__ == "__main__":
    main()
