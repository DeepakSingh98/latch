"""Cloud-only conventional observations for the independent CPU audit format."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from .data import digest
from .optimizer import CLIP,NORM


@torch.no_grad()
def persistent(model,optimizer,ema,recipe,completed,*,norm=None):
    """Canonical parameter-named Adam slots, including absent-gradient leaves."""
    parameters={};slots={}
    for name,p in model.named_parameters():
        parameters[name]=p.detach().to("cpu",copy=True)
        live=optimizer.state.get(p,{})
        for key in ("exp_avg","exp_avg_sq"):
            slots[key+":"+name]=(live[key].detach().to("cpu",copy=True) if key in live
                                  else torch.zeros_like(parameters[name]))
        slots["step:"+name]=torch.tensor(int(live.get("step",0)),dtype=torch.int64)
        if ema is not None:
            slots["ema:"+name]=(ema.average[name].detach().to("cpu",copy=True) if ema.ready
                                 else torch.zeros_like(parameters[name]))
    value=0. if norm is None else float(norm)
    slots[NORM]=torch.tensor(value,dtype=torch.float32)
    slots[CLIP]=torch.tensor(min(1.,recipe.clip_norm/(value+1e-6)),dtype=torch.float32)
    return dict(parameters=parameters,optimizer_state=slots,next_step=completed,model_state={})


class Observation:
    """Copy audit evidence outside normal timing; do not alter live gradients."""
    def __init__(self,model,optimizer,ema,recipe,index,tokens,targets):
        self.model,self.optimizer,self.ema,self.recipe=model,optimizer,ema,recipe
        self.payload=dict(before=persistent(model,optimizer,ema,recipe,index),
            recipe=asdict(recipe),update_index=index,tokens=tokens.cpu(),targets=targets.cpu())
        self.routes={};self.hooks=[]
        for layer_index,layer in enumerate(model.model.layers):
            def hook(module,inputs,output,index=layer_index):
                with torch.no_grad():
                    probabilities=output[1].detach().float().softmax(-1)
                    selected=probabilities.topk(module.top_k,dim=-1,largest=True,sorted=True).indices
                    active=torch.nn.functional.one_hot(selected,module.num_experts).sum((0,1))>0
                    self.routes[f"probabilities.{index}"]=probabilities.cpu()
                    self.routes[f"selected.{index}"]=selected.cpu()
                    self.routes[f"active.{index}"]=active.float().cpu()
            self.hooks.append(layer.block_sparse_moe.register_forward_hook(hook))

    @torch.no_grad()
    def capture_gradients(self):
        self.payload["gradients"]={name:(p.grad.detach().to("cpu",copy=True) if p.grad is not None
                                         else torch.zeros_like(self.payload["before"]["parameters"][name]))
                                   for name,p in self.model.named_parameters()}

    def close(self):
        for hook in self.hooks:hook.remove()
        self.hooks=[]

    def save(self,path,result):
        self.close()
        after=persistent(self.model,self.optimizer,self.ema,self.recipe,self.payload["update_index"]+1,norm=result["norm"])
        after["model_state"]=self.routes
        self.payload.update(after=after,loss=result["loss"].cpu(),
            fixed_routes=torch.stack([self.routes[f"selected.{i}"] for i in range(self.model.config.num_hidden_layers)]),
            routing_observation="recomputed from the actual returned gate logits using identical FP32 softmax/topk; primitive equality independently qualified")
        path=Path(path);torch.save(self.payload,path)
        receipt=dict(bytes=path.stat().st_size,sha256=digest(path),completed_update=self.payload["update_index"]+1)
        path.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt
