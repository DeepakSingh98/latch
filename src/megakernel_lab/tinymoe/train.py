"""Train/evaluate/resume either conventional or generated public TinyMoE."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

from . import conventional
from .data import TokenBatches,digest
from .model import Recipe
from .pins import public_identity


def native_train(a):
    from .native import Native
    evaluation_execution_policy=conventional.configure_execution()
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    recipe=Recipe(name=a.recipe,horizon=a.horizon,learning_rate=a.learning_rate,weight_decay=a.weight_decay)
    started=time.perf_counter()
    engine=Native(plain_dir=a.plain_dir,auxiliary_dir=a.auxiliary_dir,recipe=recipe,device=a.device)
    batches=TokenBatches(a.data_dir,"train",a.batch_size,a.sequence_length)
    validation=TokenBatches(a.data_dir,"validation",a.batch_size,a.sequence_length)
    contract=dict(**public_identity(),recipe=asdict(recipe),batch_size=a.batch_size,sequence_length=a.sequence_length,
                  data_manifest_sha256=digest(Path(a.data_dir)/"manifest.json"),backend="latch",identity=engine._identity,
                  evaluation=conventional.evaluation_policy(a.eval_windows),
                  evaluation_execution_policy=evaluation_execution_policy)
    if a.resume: engine.load_checkpoint(a.resume,contract)
    raw_model=engine.evaluator(a.model_dir)
    ema_model=None
    conventional.synchronize(a.device)
    setup=time.perf_counter()-started
    (out/"contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    evaluation_start=time.perf_counter()
    if a.eval_every:
        initial=dict(completed_update=engine.next_step,raw=conventional.evaluate(raw_model,validation,windows=a.eval_windows))
        if recipe.name=="supervision_ema" and engine.next_step>=(recipe.horizon+1)//2:
            ema_model=engine.evaluator(a.model_dir,ema=True)
            initial["ema"]=conventional.evaluate(ema_model,validation,windows=a.eval_windows)
        with (out/"evaluation.jsonl").open("a") as stream:stream.write(json.dumps(initial)+"\n")
    initial_evaluation_seconds=time.perf_counter()-evaluation_start
    torch.cuda.reset_peak_memory_stats(a.device)
    started=time.perf_counter();rows=[];checkpoints=[]
    with (out/"updates.jsonl").open("a") as log:
        for index in range(engine.next_step,min(a.updates,a.horizon)):
            x,y=batches.batch(index,a.device)
            observation=None
            if index+1 in a.observe_at:
                observation=dict(before=engine.state_dict(host=True),tokens=x.cpu(),targets=y.cpu(),
                                 update_index=index,recipe=asdict(recipe),identity=engine._identity)
            # Native trajectory timing excludes input and schedule staging.
            # Primary backend comparisons use the staging-inclusive benchmark v2.
            engine.stage(x,y,index);conventional.synchronize(a.device)
            tick=time.perf_counter();result=engine.step();conventional.synchronize(a.device)
            row=dict(completed_update=index+1,seconds=time.perf_counter()-tick,
                     objective=float(result["loss"]),gradient_norm=float(result["norm"]),
                     clip_coefficient=float(result["clip"]),learning_rate=result["learning_rate"],
                     supervision_weight=result["supervision_weight"])
            rows.append(row);log.write(json.dumps(row)+"\n");log.flush()
            if observation is not None:
                observation["after"]=engine.snapshot(host=True)
                observation["gradients"]=observation["after"].pop("gradients_before_clip")
                observation["loss"]=observation["after"]["diagnostics"]["loss"]
                saved_state=observation["after"]["model_state"]
                route_names=sorted((k for k in saved_state if k.startswith("selected.")),key=lambda k:int(k.split(".")[1]))
                if route_names:
                    observation["fixed_routes"]=torch.stack([saved_state[k] for k in route_names])
                path=out/f"observation-{index+1:06d}.pt"
                torch.save(observation,path)
                path.with_suffix(".json").write_text(json.dumps(dict(bytes=path.stat().st_size,
                    sha256=digest(path),completed_update=index+1,identity=engine._identity),indent=2)+"\n")
                del observation
            # Preserve requested failing-update evidence before rejecting it.
            # No later update may overwrite these raw gradients/arena views.
            if not torch.isfinite(result["loss"]).item() or not torch.isfinite(result["norm"]).item():
                raise FloatingPointError("nonfinite generated update: "+str(index+1))
            if a.eval_every and (index+1)%a.eval_every==0:
                val=dict(completed_update=index+1,raw=conventional.evaluate(raw_model,validation,windows=a.eval_windows))
                if recipe.name=="supervision_ema" and index+1>=(recipe.horizon+1)//2:
                    if ema_model is None: ema_model=engine.evaluator(a.model_dir,ema=True)
                    val["ema"]=conventional.evaluate(ema_model,validation,windows=a.eval_windows)
                with (out/"evaluation.jsonl").open("a") as stream: stream.write(json.dumps(val)+"\n")
            if (a.checkpoint_every and (index+1)%a.checkpoint_every==0) or index+1 in a.checkpoint_at:
                checkpoints.append(engine.save_checkpoint(out/f"checkpoint-{index+1:06d}.pt",contract))
    if a.save_final: checkpoints.append(engine.save_checkpoint(out/f"checkpoint-{engine.next_step:06d}.pt",contract))
    phases={}
    for name,active in (("supervision",True),("plain",False)):
        times=[r["seconds"] for r in rows[a.timing_warmup:] if (r["supervision_weight"]>0)==active
               and r["completed_update"] not in a.observe_at]
        if times: phases[name]=dict(updates=len(times),median_ms=statistics.median(times)*1000,mean_ms=statistics.mean(times)*1000)
    summary=dict(schema="latch-tinymoe-native-run-v1",contract=contract,setup_seconds=setup,
        initial_evaluation_seconds=initial_evaluation_seconds,
        training_process_seconds=time.perf_counter()-started,completed_updates=engine.next_step,phases=phases,
        checkpoints=checkpoints,execution=engine.execution_summary(),gpu=torch.cuda.get_device_name(a.device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device),
        timing_scope="synchronized generated complete update; routing/backward/clipping/optimizer/active EMA included; batch/schedule staging, evaluation/checkpoint and scalar reporting excluded; observed updates excluded from phase summaries")
    (out/"result.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=("latch",*conventional.BACKENDS),default="latch")
    p.add_argument("--model-dir",required=True);p.add_argument("--data-dir",required=True);p.add_argument("--out",required=True)
    p.add_argument("--plain-dir");p.add_argument("--auxiliary-dir")
    p.add_argument("--recipe",choices=("unchanged","supervision_ema"),default="unchanged")
    p.add_argument("--horizon",type=int,default=400);p.add_argument("--updates",type=int,default=400)
    p.add_argument("--batch-size",type=int,default=2);p.add_argument("--sequence-length",type=int,default=512)
    p.add_argument("--learning-rate",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--device",default="cuda");p.add_argument("--attention",default="sdpa")
    p.add_argument("--chunk-tokens",type=int,default=512);p.add_argument("--threads",type=int,default=1)
    p.add_argument("--seed",type=int,default=1729);p.add_argument("--fp32",action="store_true")
    p.add_argument("--resume");p.add_argument("--save-final",action="store_true")
    p.add_argument("--checkpoint-every",type=int,default=0);p.add_argument("--eval-every",type=int,default=50)
    p.add_argument("--checkpoint-at",type=int,nargs="*",default=[])
    p.add_argument("--eval-windows",type=int,default=8);p.add_argument("--timing-warmup",type=int,default=4)
    p.add_argument("--observe-at",type=int,nargs="*",default=[])
    a=p.parse_args()
    if a.backend=="latch":
        if not a.plain_dir: p.error("--plain-dir is required for latch")
        if a.fp32: p.error("prepared TinyMoE generated artifacts use explicit BF16 projection policy")
        native_train(a)
    else: conventional.train(a)


if __name__=="__main__":main()
