"""Bounded conventional screening and alternating paired complete updates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch

from . import conventional as C
from .data import TokenBatches,digest
from .model import Recipe,make_optimizer
from .weight_ema import WeightEMA


def screen(a):
    """Fresh process per candidate preserves failures and isolates compiler caches."""
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for backend in C.BACKENDS:
        directory=out/backend;directory.mkdir(exist_ok=True)
        command=[sys.executable,"-m","megakernel_lab.tinymoe.conventional",
            "--model-dir",a.model_dir,"--data-dir",a.data_dir,"--out",str(directory),
            "--backend",backend,"--recipe",a.recipe,"--horizon",str(a.horizon),
            "--updates",str(a.updates),"--batch-size",str(a.batch_size),
            "--sequence-length",str(a.sequence_length),"--chunk-tokens",str(a.chunk_tokens),
            "--eval-every","0","--timing-warmup",str(a.warmup)]
        started=time.perf_counter();row=dict(backend=backend,command=command)
        with (directory/"process.log").open("w") as log:
            try:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout_seconds,check=False)
                row.update(exit_code=process.returncode,passed=process.returncode==0)
                if process.returncode==0:
                    row["result"]=json.loads((directory/"result.json").read_text())
            except subprocess.TimeoutExpired:
                row.update(passed=False,error="bounded screening timeout",timeout_seconds=a.timeout_seconds)
        row["elapsed_seconds"]=time.perf_counter()-started;rows.append(row)
        (out/"screen.json").write_text(json.dumps(dict(schema="latch-tinymoe-conventional-screen-v1",candidates=rows),indent=2)+"\n")
    valid=[r for r in rows if r["passed"]]
    # This is an execution screen only. Numerical qualification remains a
    # required independent gate before a candidate is called a valid control.
    result=dict(schema="latch-tinymoe-conventional-screen-v1",candidates=rows,
                status="execution screening, not numerical qualification",successful_backends=[r["backend"] for r in valid])
    (out/"screen.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="candidates"},indent=2))


@torch.no_grad()
def import_native_optimizer(optimizer,model,state,recipe,ema):
    """Map canonical native Adam state to stock AdamW at one identical state."""
    opt=state["optimizer_state"]
    optimizer.state.clear()
    for name,p in model.named_parameters():
        completed=int(opt["step:"+name])
        if completed:
            optimizer.state[p]=dict(step=torch.tensor(float(completed),device=p.device),
                exp_avg=opt["exp_avg:"+name].to(p.device,copy=True),
                exp_avg_sq=opt["exp_avg_sq:"+name].to(p.device,copy=True))
    if ema is not None:
        completed=state["next_step"]
        average={name:opt["ema:"+name] for name in dict(model.named_parameters())} if completed>=ema.start else {}
        ema.load_state_dict(dict(policy=ema.policy(),last_step=completed,
            updates=max(0,completed-ema.start),average=average),completed)


def complete_update_wall(update, *, stage=None, device="cuda"):
    """Clock one update after batch readiness, including backend preparation.

    The conventional update prepares its schedule internally. Generated schedule
    and coefficient copies happen in ``stage`` and belong to the same clock.
    There is no extra synchronization between preparation and execution.
    """
    C.synchronize(device)
    began=time.perf_counter()
    if stage is not None: stage()
    result=update()
    C.synchronize(device)
    return result,time.perf_counter()-began


def paired(a):
    from .native import Native,checkpoint_contract
    execution_policy=C.configure_execution()
    torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    recipe=Recipe(name=a.recipe,horizon=a.horizon)
    engine=Native(plain_dir=a.plain_dir,auxiliary_dir=a.auxiliary_dir,recipe=recipe,device="cuda")
    if a.native_state:
        # Read a hash-bound native checkpoint with the original data/model
        # contract. An unlabeled raw state cannot establish continuation identity.
        receipt=json.loads(Path(a.native_state).with_suffix(Path(a.native_state).suffix+".json").read_text())
        if (receipt["bytes"]!=Path(a.native_state).stat().st_size
                or receipt["sha256"]!=digest(a.native_state)):
            raise ValueError("native state size/checksum mismatch")
        payload=torch.load(a.native_state,map_location="cpu",weights_only=False)
        checkpoint_contract(payload["state"],payload.get("contract"),recipe=recipe,data_dir=a.data_dir,
                            batch_size=a.batch_size,sequence_length=a.sequence_length)
        engine.load_state_dict(payload["state"])
    batches=TokenBatches(a.data_dir,"train",a.batch_size,a.sequence_length)
    base=engine.state_dict(host=True)
    # The conventional model views the same parameter buffers. Arms never run
    # concurrently and each is restored before timing, avoiding duplicate weights.
    model=engine.evaluator(a.model_dir)
    model.requires_grad_(True);model.train()
    compiled=C.prepare_backend(model,a.conventional)
    optimizer=make_optimizer(model,recipe,fused=True)
    ema=WeightEMA(model,recipe.horizon) if recipe.name=="supervision_ema" else None
    start=base["next_step"]
    if start+a.warmup+a.updates>recipe.horizon: raise ValueError("paired window exceeds original schedule horizon")
    rows=[]
    torch.cuda.reset_peak_memory_stats()
    for pair in range(a.pairs):
        for arm in (("conventional","latch") if pair%2==0 else ("latch","conventional")):
            engine.load_state_dict(base)
            if arm=="conventional": import_native_optimizer(optimizer,model,base,recipe,ema)
            arm_rows=[]
            for offset in range(a.warmup+a.updates):
                index=start+offset;x,y=batches.batch(index,"cuda")
                if arm=="latch":
                    result,seconds=complete_update_wall(engine.step,stage=lambda:engine.stage(x,y,index))
                else:
                    result,seconds=complete_update_wall(lambda:C.update(model,optimizer,recipe,index,x,y,
                        backend=a.conventional,chunk_tokens=a.chunk_tokens,ema=ema))
                row=dict(pair=pair,arm=arm,update=index+1,warmup=offset<a.warmup,
                         complete_update_wall_seconds=seconds,
                         objective=float(result["loss"]),gradient_norm=float(result["norm"]),
                         supervision_weight=result["supervision_weight"])
                if not torch.isfinite(result["loss"]).item() or not torch.isfinite(result["norm"]).item():
                    raise FloatingPointError("nonfinite paired update")
                arm_rows.append(row)
            rows.extend(arm_rows)
            with (out/"updates.jsonl").open("a") as log:
                for row in arm_rows:log.write(json.dumps(row)+"\n")
    summary={}
    for arm in ("conventional","latch"):
        selected=[r for r in rows if r["arm"]==arm and not r["warmup"]]
        summary[arm]={}
        for phase,active in (("supervision",True),("plain",False)):
            times=[r["complete_update_wall_seconds"] for r in selected if (r["supervision_weight"]>0)==active]
            if times:summary[arm][phase]=dict(updates=len(times),median_ms=statistics.median(times)*1000,
                                             mean_ms=statistics.mean(times)*1000)
    result=dict(schema="latch-tinymoe-alternating-paired-v2",recipe=asdict(recipe),starting_update=start,
        starting_state="fresh public checkpoint" if start==0 else "hash-bound native trained state",
        conventional=a.conventional,compiled_conventional_regions=compiled,pairs=a.pairs,warmup=a.warmup,
        measured_updates_per_arm_per_pair=a.updates,timings=summary,execution=engine.execution_summary(),
        conventional_execution_policy=execution_policy,
        measurement_module_sha256=digest(Path(__file__)),
        primary_clock="complete_update_wall_seconds",
        timing_scope="wall clock after common token-batch readiness sync, before backend-specific staging/schedule preparation, through update completion sync; routing/forward/backward/clipping/optimizer/active EMA included; state restoration, batch loading, scalar reporting and logging excluded; no execution-only clock is reported",
        gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),identity=engine._identity,
        memory_scope="combined resident native and conventional optimizer states; parameters share storage; standalone training receipts measure individual backend peaks",
        scope="same initial raw weights, per-expert Adam moments/steps, EMA and token batches per arm/pair; alternating order on one GPU; numerical qualification is separate")
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("screen","paired"));p.add_argument("--model-dir",required=True)
    p.add_argument("--data-dir",required=True);p.add_argument("--out",required=True)
    p.add_argument("--recipe",choices=("unchanged","supervision_ema"),default="unchanged")
    p.add_argument("--horizon",type=int,default=400);p.add_argument("--updates",type=int,default=16)
    p.add_argument("--warmup",type=int,default=4);p.add_argument("--pairs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=2);p.add_argument("--sequence-length",type=int,default=512)
    p.add_argument("--chunk-tokens",type=int,default=512);p.add_argument("--timeout-seconds",type=int,default=600)
    p.add_argument("--plain-dir");p.add_argument("--auxiliary-dir");p.add_argument("--native-state")
    p.add_argument("--conventional",choices=C.BACKENDS,default="grouped")
    a=p.parse_args()
    if a.mode=="screen": screen(a)
    else:
        if not a.plain_dir:p.error("paired mode requires --plain-dir")
        paired(a)


if __name__=="__main__": main()
