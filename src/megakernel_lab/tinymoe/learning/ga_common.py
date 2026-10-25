"""Recovered published TinyMoE training settings and shared timing-harness helpers.

Sources (pinned revision 9b209f7439aadfac282e0bf44f2fda19771aba90), inspected
with pickletools only (no unpickling): training_args.bin, trainer_state.json,
config.json. See docs/v0/LEARNING.md for scope and provenance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

PUBLISHED = dict(
    # training_args.bin (TrainingArguments, transformers 4.55.0 producer)
    per_device_train_batch_size=8, gradient_accumulation_steps=16, n_gpu=1,
    bf16=True, fp16=False, tf32=None, optim="adamw_torch",
    learning_rate=3e-4, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6,
    max_grad_norm=1.0, lr_scheduler_type="cosine", warmup_steps=403, warmup_ratio=0.0, max_steps=8070,
    gradient_checkpointing=False, torch_compile=False, logging_nan_inf_filter=True,
    average_tokens_across_devices=False, seed=42,
    # trainer_state.json: total_flos/(6 * non-embedding params)/global_step = 131072 tokens/update
    sequence_length=1024,
    # config.json
    router_aux_loss_coef=0.02, output_router_logits=True, num_local_experts=16, num_experts_per_tok=2,
)
SEQUENCES_PER_UPDATE = PUBLISHED["per_device_train_batch_size"] * PUBLISHED["gradient_accumulation_steps"]
TOKENS_PER_UPDATE = SEQUENCES_PER_UPDATE * PUBLISHED["sequence_length"]
# Transformers 4.55 Trainer + Mixtral: CE = sum over shifted targets / num_items_in_batch,
# num_items_in_batch = count of UNSHIFTED labels over all 16 microbatches; the router
# term 0.02*aux is added to every microbatch loss without GA division.
CE_COEFFICIENT_PER_MICROBATCH = 1.0 / PUBLISHED["gradient_accumulation_steps"]


def hf_cosine_lr(update_index):
    """transformers.get_cosine_schedule_with_warmup(num_cycles=0.5) at optimizer step k."""
    warmup, total = PUBLISHED["warmup_steps"], PUBLISHED["max_steps"]
    k = int(update_index)
    if k < warmup:
        factor = k / max(1, warmup)
    else:
        progress = (k - warmup) / max(1, total - warmup)
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
    return PUBLISHED["learning_rate"] * factor


def microbatches(data_dir, updates, *, first_update=0):
    """Host (CPU) token windows, loaded before any clock: [update][micro] -> (x, y).

    Each window is 1024 consecutive corpus tokens; y is the supplied next-token
    lookahead. The same windows are reused by every arm and every round.
    """
    import torch
    from megakernel_lab.tinymoe.data import TokenBatches
    source = TokenBatches(data_dir, "train", PUBLISHED["per_device_train_batch_size"], PUBLISHED["sequence_length"])
    steps = PUBLISHED["gradient_accumulation_steps"]
    if (first_update + updates) * steps > source.count:
        raise ValueError("not enough prepared training windows")
    result = []
    for u in range(first_update, first_update + updates):
        row = []
        for m in range(steps):
            x, y = source.batch(u * steps + m, "cpu")
            row.append((x.pin_memory(), y.pin_memory()))
        result.append(row)
    return result, dict(manifest_sha256=__import__("hashlib").sha256(
        (Path(data_dir) / "manifest.json").read_bytes()).hexdigest(), windows_per_update=steps,
        first_update=first_update, updates=updates, window_tokens=PUBLISHED["sequence_length"])


class Clock:
    """Synchronized wall clock around one complete optimizer update."""
    def __init__(self, device):
        import torch
        self.torch, self.device, self.rows = torch, device, []

    def begin(self):
        self.torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()

    def end(self, **extra):
        self.torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - self.start
        self.rows.append(dict(seconds=seconds, **extra))
        return seconds


def summarize(rows, warmup, *, targets_per_update, tokens_per_update):
    import statistics
    measured = [r["seconds"] for r in rows[warmup:]]
    if not measured:
        return {}
    median = statistics.median(measured)
    return dict(updates=len(measured), warmup_updates=warmup,
                median_ms=1000 * median, mean_ms=1000 * statistics.mean(measured),
                min_ms=1000 * min(measured), max_ms=1000 * max(measured),
                stdev_ms=1000 * statistics.pstdev(measured),
                targets_per_update=targets_per_update, tokens_per_update=tokens_per_update,
                targets_per_second_at_median=targets_per_update / median,
                tokens_per_second_at_median=tokens_per_update / median,
                all_ms=[1000 * s for s in measured])


def kernel_census(run_one_update, device):
    """Count device kernels/copies/memsets for one complete update via torch.profiler.

    Profiling runs outside the timed window; names are aggregated, not timed claims.
    """
    import collections
    import torch
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run_one_update()
        torch.cuda.synchronize(device)
    kernels, copies, memsets = collections.Counter(), collections.Counter(), collections.Counter()
    for event in prof.events():
        kind = str(getattr(event, "device_type", ""))
        if "CUDA" not in kind:
            continue
        name = event.name
        if name.startswith("Memcpy"):
            copies[name] += 1
        elif name.startswith("Memset"):
            memsets[name] += 1
        else:
            kernels[name] += 1
    return dict(kernel_launches=sum(kernels.values()), distinct_kernels=len(kernels),
                memcpy=sum(copies.values()), memset=sum(memsets.values()),
                copies_by_kind=dict(copies), memsets_by_kind=dict(memsets),
                top_kernels=kernels.most_common(25))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=repr) + "\n")


def device_identity(device):
    import subprocess
    import torch
    props = torch.cuda.get_device_properties(device)
    try:
        uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version,clocks.max.sm", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # identity only
        uuid = repr(exc)
    return dict(name=props.name, total_memory=props.total_memory, multiprocessors=props.multi_processor_count,
                nvidia_smi=uuid, torch=torch.__version__, cuda=torch.version.cuda)
