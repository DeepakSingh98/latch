"""Random initial TinyMoE weights for one paired seed (CPU worker only).

The unchanged public architecture (hash-pinned config of FlameF0X/TinyMoE-200m-2x16
@9b209f74) is instantiated with the installed Transformers 4.55.4 `from_config`, whose
`_init_weights` draws Linear/Embedding weights from N(0, initializer_range) and sets
RMSNorm gains to one. The public trained weights are never used. Both arms of a pair
load these exact tensors; only the seed differs between pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from megakernel_lab.tinymoe.native import evaluator_config
    from megakernel_lab.tinymoe.pins import EXPECTED_PARAMETERS
    from .protocol import digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("initial weights are generated on a CPU worker")
    torch.set_num_threads(1)
    config = evaluator_config(a.model_dir)  # hash-verified public config
    config._attn_implementation = "sdpa"
    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    parameters = {name: p.detach().contiguous().clone() for name, p in model.named_parameters()}
    if sum(p.numel() for p in parameters.values()) != EXPECTED_PARAMETERS:
        raise ValueError("parameter count differs from the public architecture")
    stats = {name: dict(shape=list(p.shape), mean=float(p.mean()), std=float(p.std()) if p.numel() > 1 else 0.0)
             for name, p in list(parameters.items())[:6]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    path = out / f"init-s{a.seed}.pt"
    torch.save(dict(schema="latch-tinymoe-smallbatch-init-v1", seed=a.seed, parameters=parameters), path)
    receipt = dict(schema="latch-tinymoe-smallbatch-init-v1", seed=a.seed, bytes=path.stat().st_size,
                   sha256=digest(path), parameters=len(parameters), elements=sum(p.numel() for p in parameters.values()),
                   initializer_range=config.initializer_range, torch=torch.__version__,
                   generator="torch.manual_seed(seed); AutoModelForCausalLM.from_config (Transformers _init_weights)",
                   sample_statistics=stats)
    path.with_suffix(".pt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("seed", "sha256", "elements")}))


if __name__ == "__main__":
    main()
