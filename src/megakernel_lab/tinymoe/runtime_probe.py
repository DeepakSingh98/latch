"""Observe the loaded public model's actual GPU autocast/loss behavior."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .data import TokenBatches
from .model import load_model
from .pins import public_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",required=True);p.add_argument("--data-dir",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();torch.set_num_threads(1)
    model=load_model(a.model_dir,device="cuda",attention="sdpa")
    batches=TokenBatches(a.data_dir,"train",1,16);x,y=batches.batch(0,"cuda")
    rows=[];original=F.softmax
    def record_softmax(input,*args,**kwargs):
        value=original(input,*args,**kwargs)
        rows.append(dict(shape=list(input.shape),input_dtype=str(input.dtype),
                         requested_dtype=str(kwargs.get("dtype")),output_dtype=str(value.dtype)))
        return value
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16),patch.object(F,"softmax",record_softmax):
        output=model(input_ids=x,labels=y,shift_labels=y,use_cache=False,output_router_logits=True)
        ce=F.cross_entropy(output.logits.float().reshape(-1,model.config.vocab_size),y.reshape(-1))
    result=dict(schema="latch-tinymoe-loaded-cuda-policy-v1",**public_identity(),
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
        parameter_count=sum(p.numel() for p in model.parameters()),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        projection_router_dtypes=[str(t.dtype) for t in output.router_logits],softmax_calls=rows,
        logits_dtype=str(output.logits.dtype),router_loss_dtype=str(output.aux_loss.dtype),
        loss=float(output.loss),final_ce=float(ce),router_aux=float(output.aux_loss),router_coefficient=model.router_aux_loss_coef,
        decomposition_abs=float((output.loss-ce-.02*output.aux_loss).abs()))
    expected=[r for r in rows if r["shape"]==[12*16,16]]
    result["passed"]=len(expected)==1 and expected[0]["output_dtype"]=="torch.float32" and result["decomposition_abs"]<1e-5
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not result["passed"]: raise SystemExit(1)


if __name__=="__main__":main()
