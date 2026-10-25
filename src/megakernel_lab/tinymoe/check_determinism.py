"""Repeat exact same-state complete conventional updates on the real GPU model."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path

import torch

from . import conventional as C
from .data import TokenBatches
from .model import Recipe,load_model,make_optimizer


def clone(value,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device,copy=True)
    if isinstance(value,dict):return {k:clone(v,device) for k,v in value.items()}
    if isinstance(value,list):return [clone(v,device) for v in value]
    if isinstance(value,tuple):return tuple(clone(v,device) for v in value)
    return copy.deepcopy(value)


def equal(a,b,path=""):
    if isinstance(a,torch.Tensor):
        return [] if (a.shape==b.shape and a.dtype==b.dtype and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))) else [path]
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [path+":keys"]
        return [f for k in a for f in equal(a[k],b[k],path+"/"+str(k))]
    if isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):return [path+":sequence"]
        return [f for i in range(len(a)) for f in equal(a[i],b[i],path+"/"+str(i))]
    return [] if a==b else [path]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",required=True);p.add_argument("--data-dir",required=True);p.add_argument("--out",required=True)
    p.add_argument("--backend",choices=C.BACKENDS,default="compiled-grouped")
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--sequence-length",type=int,default=256)
    a=p.parse_args();policy=C.configure_execution();torch.set_num_threads(1);torch.manual_seed(1729)
    model=load_model(a.model_dir,device="cuda");C.prepare_backend(model,a.backend)
    recipe=Recipe(name="supervision_ema",horizon=400);optimizer=make_optimizer(model,recipe)
    batches=TokenBatches(a.data_dir,"train",a.batch_size,a.sequence_length)
    rows=[]
    for index in range(4):
        x,y=batches.batch(index,"cuda")
        if index in (0,3):
            base=dict(model=clone(model.state_dict(),"cpu"),optimizer=clone(optimizer.state_dict(),"cpu"))
            expected=None;failures=[]
            for repeat in range(3):
                if repeat:
                    model.load_state_dict(base["model"]);optimizer.load_state_dict(base["optimizer"])
                result=C.update(model,optimizer,recipe,index,x,y,backend=a.backend)
                current=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),loss=result["loss"],norm=result["norm"])
                if expected is None:expected=clone(current,"cuda")
                else:failures.extend(dict(repeat=repeat,path=f) for f in equal(expected,current))
            rows.append(dict(update=index+1,repetitions=3,passed=not failures,failures=failures))
            del base,expected
        else:C.update(model,optimizer,recipe,index,x,y,backend=a.backend)
    result=dict(schema="latch-tinymoe-same-state-determinism-v1",passed=all(r["passed"] for r in rows),
                backend=a.backend,execution_policy=policy,recipe=asdict(recipe),cases=rows,
                torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
                scope="three exact replays at fresh and three-update trained states; all parameters, Adam moments/counters, objective and norm bitwise equal; full checkpoint restart remains separate")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
