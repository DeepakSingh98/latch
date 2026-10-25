"""Evaluate saved final parameters on a held-out split (GPU worker).

Same evaluator as training-time dev evaluation (protocol.evaluate): installed
Transformers modules over the weights, BF16 autocast, FP32 CE, router terms excluded.
Used once per frozen confirmation run for the final test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import protocol


def main():
    import torch
    from megakernel_lab.tinymoe.native import evaluator_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run", action="append", required=True, help="LABEL=local/final-parameters.pt")
    p.add_argument("--split", default="test", choices=("validation", "test"))
    p.add_argument("--windows", type=int, default=4096)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    config = evaluator_config(a.model_dir)
    split = protocol.Windows(a.data_dir, a.split)
    rows = {}
    for item in a.run:
        label, source = item.split("=", 1)
        local = Path(source)
        receipt = dict(path=source)
        receipt["sha256"] = protocol.digest(local)
        parameters = {k: v.to("cuda") for k, v in torch.load(local, map_location="cpu", weights_only=True).items()}
        row = protocol.evaluate(config, parameters, split, a.windows)
        rows[label] = dict(source=source, parameters_sha256=receipt["sha256"], split=a.split, **row)
        del parameters
        print(json.dumps({label: rows[label]["nll"]}), flush=True)
    protocol.write_json(out / "final-evaluation.json", dict(schema="latch-tinymoe-smallbatch-final-eval-v1",
                                                           data_manifest_sha256=split.manifest_sha256, runs=rows))


if __name__ == "__main__":
    main()
