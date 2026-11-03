"""Export the step-0 state, parameter inventory and optimizer groups of an upstream Nestor model.

Runs in the upstream Nestor environment (not the Latch one), against a clone of
https://github.com/Ma-Lab-Berkeley/Nestor at commit 9c25256fd358f4f3b0b92da175158db20fc035b3:

    python -B nestor_inventory.py --nestor-src /path/to/Nestor --fixture full --out DIR [--seed 42]

Every fixture is built by upstream code (``nestor.experiments.pretrain._backbone`` with the same
SequenceModelConfig/ReferenceMupConfig the registered experiments use, with NESTOR_STACKED_EXPERT_BANKS=1):

* ``full``: the registered ``nestor_100m`` model exactly (d 480, 7 blocks (1,1,2,2,2,3,2), G 8, k (3,4,5,6,6,7,8),
  heads (5,4,3,3,3,2,2) x 30, P 8 top-2, Q 8, vocab 100,263, context 4096).
* ``r1``: a reduced fixture with the same width, head width, column count, public/private pool sizes, gate modes,
  router MLPs/noise, loss-free balancing, depth-read settings and muP rules; 3 blocks k (3,5,8), heads (5,3,2),
  layers (2,1,1) (a multi-layer column block, a top-1 private block and the dense tail), vocab 8,192, context 512.

Writes ``init-<fixture>.pt`` (the upstream state dict after ``set_seed(seed)`` and construction) and
``inventory-<fixture>.json`` (geometry, parameter tags, buffers, optimizer groups, init hash).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

# Set before upstream is imported: stacked expert banks give the [E, in, out] parameter layout the Latch model
# reads. (The ``full`` fixture's DCLM batch variables are set in model_config, before nestor_100m() reads them.)
os.environ.setdefault("NESTOR_STACKED_EXPERT_BANKS", "1")

UPSTREAM_COMMIT = "9c25256fd358f4f3b0b92da175158db20fc035b3"
FIXTURES = {
    "full": None,
    "r1": dict(top_k=(3, 5, 8), heads=(5, 3, 2), layers=(2, 1, 1), vocab=8192, context=512),
}


def checkout_commit(source):
    """The commit of an upstream git checkout; refuses anything but the pinned commit."""
    try:
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"--nestor-src {source} is not a git checkout of Nestor ({exc}); clone "
                         f"https://github.com/Ma-Lab-Berkeley/Nestor and check out {UPSTREAM_COMMIT}")
    if head != UPSTREAM_COMMIT:
        raise SystemExit(f"--nestor-src is at {head}; this example is pinned to {UPSTREAM_COMMIT}")
    return head


def model_config(fixture):
    """The upstream SequenceModelConfig of a fixture (``full`` = the registered nestor_100m job's model config)."""
    from nestor.experiments import pretrain as P
    if fixture == "full":
        os.environ.setdefault("DCLM_GLOBAL_BATCH", "4")
        os.environ.setdefault("DCLM_GRAD_ACCUM", "16")
        return P.nestor_100m().jobs[0].model
    from nestor.configs.adapters import TextAdapterConfig
    from nestor.configs.reference_mup import ReferenceMupConfig
    from nestor.heads import LanguageModelingHeadConfig
    from nestor.models import SequenceModelConfig
    f = FIXTURES[fixture]
    scale = P.SCALES["100m"]
    n = len(f["top_k"])
    private_top_k = tuple(
        2 if P.private_expert_hidden_dim(scale.hidden_size, k, 2, with_public_pool=True) >= P.NESTOR_MIN_PRIVATE_WIDTH
        else 1 for k in f["top_k"])
    backbone = P._backbone(scale, groups=(P.NESTOR_COLUMNS,) * n, top_k=f["top_k"], heads=f["heads"],
                           layers=f["layers"], bank_experts=1, bank_top_k=1, bank_width=scale.mlp_hidden,
                           public_experts=P.NESTOR_PUBLIC_EXPERTS, public_top_k=P.NESTOR_PUBLIC_TOP_K,
                           private_experts=P.NESTOR_PRIVATE_EXPERTS["100m"], private_top_k=private_top_k,
                           **P.NESTOR_READ)
    return SequenceModelConfig(
        adapter_config=TextAdapterConfig(tokenizer=P.TOKENIZER, embed_dim=scale.hidden_size, max_length=f["context"],
                                         vocab_size=f["vocab"]),
        backbone_config=backbone,
        head_config=LanguageModelingHeadConfig(vocab_size=f["vocab"]),
        reference_mup=ReferenceMupConfig(matrix_optimizer="muon", num_layers=sum(f["layers"]), **P.REFERENCE_MUP))


def shape_record(fixture, config):
    """Static geometry the Latch model needs (read from the upstream config, not re-derived)."""
    from nestor.experiments import pretrain as P
    b = config.backbone_config
    if fixture == "full":
        context, vocab = P.BLOCK_SIZE, P.VOCAB_SIZE
    else:
        context, vocab = FIXTURES[fixture]["context"], FIXTURES[fixture]["vocab"]
    return dict(fixture=fixture, width=b.input_dim, vocab=vocab, context=context, columns=list(b.num_groups),
                top_k=list(b.num_groups_per_token), heads=list(b.block_num_heads), head_width=b.block_head_dims,
                layers=list(b.block_num_layers), public_experts=list(b.block_expert_pool_public_experts),
                public_top_k=list(b.block_expert_pool_public_top_k),
                private_experts=list(b.block_expert_pool_private_experts),
                private_top_k=list(b.block_expert_pool_private_top_k),
                rope_base=b.block_rope_bases, bias_rate=b.router_bias_update_rate,
                read=dict(mode=b.block_residual_mode, include_aggregate=b.block_column_include_aggregate,
                          gain_init=b.block_column_gain_init, head_only=b.block_column_head_only,
                          deviation_sources=b.block_column_deviation_sources))


def build(fixture, seed=42):
    """Upstream model on CPU, initialised exactly as the training pipeline does (set_seed before construction)."""
    from nestor.models.base import Model
    from nestor.utils.seed import set_seed
    set_seed(seed)
    config = model_config(fixture)
    model = Model.from_config(config)
    return model, config


def inventory(model):
    from nestor.scaling.blocks import get_scaling
    rows = []
    for name, p in model.named_parameters():
        tag = get_scaling(p)
        rows.append(dict(name=name, shape=list(p.shape), dtype=str(p.dtype), requires_grad=p.requires_grad,
                         optimizer=tag and tag.optimizer, lr_mul=tag and tag.lr_mul, muon_scale=tag and tag.muon_scale,
                         weight_decay=tag and tag.weight_decay))
    buffers = [dict(name=n, shape=list(b.shape), dtype=str(b.dtype),
                    persistent=n in dict(model.state_dict()))
               for n, b in model.named_buffers()]
    return rows, buffers


def optimizer_groups(model):
    """The exact upstream MuonAdamW param groups (built by the upstream optimizer builder)."""
    from nestor.experiments import pretrain as P
    from nestor.configs.optim import MuonAdamWConfig
    from nestor.train.optim import build_optimizer
    config = MuonAdamWConfig(lr_muon=P.LR_MUON, lr_adamw=P.LR_ADAMW, weight_decay=P.WEIGHT_DECAY, betas=(0.9, 0.95))
    optimizer = build_optimizer(config, model)
    names = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for index, g in enumerate(optimizer.param_groups):
        groups.append(dict(index=index, method=g.get("method"), lr=g["lr"], weight_decay=g.get("weight_decay"),
                           betas=list(g.get("betas", ())) or None, eps=g.get("eps"), momentum=g.get("momentum"),
                           nesterov=g.get("nesterov"), ns_steps=g.get("ns_steps"),
                           params=[names[id(p)] for p in g["params"]]))
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nestor-src", type=Path, required=True, help="git checkout of upstream Nestor at the pinned commit")
    p.add_argument("--fixture", choices=sorted(FIXTURES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    source = a.nestor_src.resolve()
    checkout_commit(source)
    sys.path.insert(0, str(source))
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    model, config = build(a.fixture, a.seed)
    rows, buffers = inventory(model)
    try:
        groups = optimizer_groups(model)
    except Exception as exc:  # recorded, not hidden; nestor_build.py then refuses the inventory
        groups = dict(error=repr(exc))
    state = model.state_dict()
    path = out / f"init-{a.fixture}.pt"
    torch.save({k: v.detach().clone() for k, v in state.items()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(schema="nestor-latch-upstream-inventory-v1", fixture=a.fixture, seed=a.seed,
                  shape=shape_record(a.fixture, config), parameters=rows, buffers=buffers, optimizer_groups=groups,
                  total_parameters=sum(r["shape"] and math.prod(r["shape"]) for r in rows),
                  init_state=dict(path=str(path), sha256=digest, tensors=len(state)),
                  torch=torch.__version__)
    (out / f"inventory-{a.fixture}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(fixture=a.fixture, parameters=record["total_parameters"], tensors=len(rows),
                          buffers=len(buffers), init_sha256=digest)), flush=True)


if __name__ == "__main__":
    main()
