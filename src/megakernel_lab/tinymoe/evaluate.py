"""Reference final-readout evaluation of a hash-bound training checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import conventional as C
from .data import TokenBatches,digest
from .model import load_model
from .pins import public_identity


def checkpoint_contents(payload,receipt):
    """Reject inconsistent saved metadata before interpreting raw/EMA weights."""
    contract=payload["contract"];recipe=contract["recipe"]
    if recipe.get("name") not in {"unchanged","supervision_ema"}:
        raise ValueError("checkpoint recipe is unknown")
    horizon=recipe.get("horizon")
    if type(horizon) is not int or horizon<2:
        raise ValueError("checkpoint original horizon is invalid")
    completed=payload["state"]["next_step"] if "state" in payload else payload["completed_updates"]
    if type(completed) is not int or not 0<=completed<=horizon:
        raise ValueError("checkpoint completed-update cursor is invalid")
    if "state" in payload:
        state=payload["state"];raw=state["parameters"]
        if (contract.get("backend")!="latch" or state.get("schema")!="latch-tinymoe-native-state-v1"
                or state.get("recipe")!=recipe or state.get("identity")!=contract.get("identity")
                or not isinstance(state.get("identity"),dict) or receipt.get("identity")!=state["identity"]
                or receipt.get("next_step")!=completed):
            raise ValueError("native checkpoint state/receipt/contract differ")
        averages={name:state["optimizer_state"]["ema:"+name] for name in raw} if (
            recipe["name"]=="supervision_ema" and completed>=(horizon+1)//2) else {}
    else:
        raw=payload["model"]
        if (payload.get("schema")!="latch-tinymoe-conventional-checkpoint-v1"
                or contract.get("backend") not in C.BACKENDS
                or receipt.get("completed_updates")!=completed):
            raise ValueError("conventional checkpoint state/receipt/contract differ")
        ema=payload.get("ema")
        if recipe["name"]=="supervision_ema":
            if (not isinstance(ema,dict) or ema.get("last_step")!=completed
                    or ema.get("policy",{}).get("horizon_updates")!=horizon
                    or ema.get("updates")!=max(0,completed-(horizon+1)//2)):
                raise ValueError("checkpoint EMA policy/cursor differs")
            averages=ema["average"]
            expected=raw.keys() if completed>=(horizon+1)//2 else {}.keys()
            if averages.keys()!=expected:
                raise ValueError("checkpoint EMA parameter coverage differs")
        else:
            if ema is not None:raise ValueError("unchanged checkpoint unexpectedly contains EMA")
            averages={}
    return contract,completed,raw,averages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--model-dir",required=True)
    p.add_argument("--data-dir",required=True);p.add_argument("--out",required=True)
    p.add_argument("--split",choices=("validation","test"),default="validation")
    p.add_argument("--windows",type=int,default=128,help="ordered packed windows; 0 evaluates every complete window")
    p.add_argument("--device",default="cuda");p.add_argument("--threads",type=int,default=1)
    a=p.parse_args();execution=C.configure_execution();torch.set_num_threads(a.threads)
    path=Path(a.checkpoint);receipt=json.loads(path.with_suffix(path.suffix+".json").read_text())
    if receipt["bytes"]!=path.stat().st_size or receipt["sha256"]!=digest(path):
        raise ValueError("checkpoint size/hash mismatch")
    payload=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    contract,completed,raw,averages=checkpoint_contents(payload,receipt)
    for key,value in public_identity().items():
        if contract.get(key)!=value:raise ValueError("checkpoint public identity differs: "+key)
    if contract["data_manifest_sha256"]!=digest(Path(a.data_dir)/"manifest.json"):
        raise ValueError("checkpoint corpus differs from supplied data")
    model=load_model(a.model_dir,device=a.device,attention="sdpa")
    model.load_state_dict(raw,strict=True)
    batches=TokenBatches(a.data_dir,a.split,contract["batch_size"],contract["sequence_length"])
    windows=a.windows or batches.count
    if windows<1:raise ValueError("evaluation needs a positive number of windows")
    policy=C.evaluation_policy(windows);policy["split"]=a.split
    policy["implementation"]="installed Transformers expert loops and SDPA"
    policy["parameter_binding"]="checkpoint values copied into one evaluation model"
    result=dict(schema="latch-tinymoe-reference-evaluation-v1",checkpoint_sha256=receipt["sha256"],
                completed_updates=completed,training_backend=contract["backend"],recipe=contract["recipe"],
                execution_policy=execution,evaluation=policy,batch_size=batches.batch_size,
                sequence_length=batches.sequence_length,raw=C.evaluate(model,batches,windows=windows))
    if averages:
        model.load_state_dict(averages,strict=True)
        result["ema"]=C.evaluate(model,batches,windows=windows)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
