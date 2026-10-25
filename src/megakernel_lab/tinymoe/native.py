"""Single-GPU owner for normally compiled TinyMoE phase programs.

The phase artifacts share parameter, optimizer, model-state and immutable
buffer bindings by canonical role/name, never by assumed input ordering.
"""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch

from .data import digest
from .model import Recipe
from .optimizer import NORM, CLIP, stage_schedule
from .pins import public_identity,EXPECTED_CONFIG,TRANSFORMERS_VERSION


def artifact_directory(path):
    path=Path(path)
    if (path/"artifact.json").exists(): return path
    if (path/"artifact"/"artifact.json").exists(): return path/"artifact"
    raise FileNotFoundError("no prepared ModuleUpdate artifact at " + str(path))


def artifact_identity(directory):
    directory=artifact_directory(directory)
    receipt=json.loads((directory/"artifact.json").read_text())
    # Keep exact library path as well as its verified hash: arbitrary cache
    # relocation is not an exact native-restart proof.
    metadata=_build_metadata_path(directory)
    return dict(directory=str(directory.resolve()),artifact_sha256=digest(directory/"artifact.json"),
                build_metadata=str(metadata.resolve()),build_metadata_sha256=digest(metadata),
                library=receipt["library"],library_sha256=receipt["library_sha256"],
                source_hash=receipt["source_hash"],files=receipt["files"])


def _build_metadata_path(directory):
    directory=Path(directory)
    candidates=(directory/"tinymoe-build.json",directory.parent/"tinymoe-build.json")
    for path in candidates:
        if path.is_file():return path
    raise ValueError("prepared TinyMoE build metadata is required: "+str(directory))


def phase_contract(directory,phase,recipe):
    """Validate phase semantics and immutable shared bindings on CPU.

    mmap loads tensor descriptors without materializing the 200M parameter and
    optimizer arrays again. Only the small immutable buffers are read for their
    byte identities. The ordinary loader still verifies every artifact file.
    """
    from megakernel_lab.operation_compiler.variants import persistent_keys
    directory=artifact_directory(directory)
    receipt=json.loads((directory/"artifact.json").read_text())
    metadata=json.loads(_build_metadata_path(directory).read_text())
    if (metadata.get("schema")!="latch-tinymoe-build-v1" or metadata.get("phase")!=phase
            or metadata.get("recipe")!=asdict(recipe)):
        raise ValueError("prepared phase/recipe differs from requested execution")
    # explain is a post-build report pointer added to the returned receipt after
    # artifact.json is written; it has no execution or binding semantics.
    bound=metadata.get("artifact",{})
    if ({k:v for k,v in bound.items() if k!="explain"}!=
            {k:v for k,v in receipt.items() if k!="explain"}):
        raise ValueError("TinyMoE metadata belongs to a different artifact receipt")
    if any(metadata.get(k)!=v for k,v in public_identity().items()):
        raise ValueError("prepared TinyMoE public identity differs")
    expected_ema=recipe.name=="supervision_ema" and phase=="plain"
    if metadata.get("ema_active") is not expected_ema or metadata.get("routing")!="unconstrained":
        raise ValueError("prepared phase EMA/routing semantics differ")
    for filename in ("manifest.json","bindings.pt"):
        if receipt["files"].get(filename)!=digest(directory/filename):
            raise ValueError("artifact identity mismatch: "+filename)
    manifest=json.loads((directory/"manifest.json").read_text())
    saved=torch.load(directory/"bindings.pt",weights_only=True,map_location="cpu",mmap=True)["args"]
    immutable={}
    for index,(role,name) in persistent_keys(manifest,receipt["module_api"]).items():
        if role in {"buffer","frozen"}:
            value=saved[index]
            raw=value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            immutable[role+":"+name]=dict(shape=list(value.shape),strides=list(value.stride()),
                dtype=str(value.dtype),sha256=hashlib.sha256(raw).hexdigest())
    common={key:metadata.get(key) for key in (*public_identity(),"recipe","batch_size","sequence_length",
        "routing","routing_diagnostics","objective","attention","expert_policy")}
    return dict(common=common,immutable=immutable)


def checkpoint_contract(state,contract,*,recipe,data_dir,batch_size,sequence_length):
    """Validate data and schedule before importing a trained paired-run state."""
    expected=dict(**public_identity(),recipe=asdict(recipe),backend="latch",
        batch_size=batch_size,sequence_length=sequence_length,
        data_manifest_sha256=digest(Path(data_dir)/"manifest.json"),identity=state["identity"])
    if not isinstance(contract,dict) or any(contract.get(k)!=v for k,v in expected.items()):
        raise ValueError("native checkpoint benchmark data/model/schedule contract differs")


def evaluator_config(model_dir):
    """Read only the hash-pinned public config used by zero-copy evaluation."""
    import transformers
    from transformers import AutoConfig
    from . import variant
    directory=Path(model_dir)
    if transformers.__version__!=TRANSFORMERS_VERSION:
        raise ValueError("evaluator Transformers version differs from pinned semantics")
    if variant.is_variant(directory):
        return variant.config(directory)
    identity=json.loads((directory/"latch_identity.json").read_text())
    if any(identity.get(k)!=v for k,v in public_identity().items()):
        raise ValueError("evaluator public model/tokenizer identity mismatch")
    path=directory/"config.json"
    spec=identity.get("files",{}).get("config.json",{})
    if spec.get("bytes")!=path.stat().st_size or spec.get("sha256")!=digest(path):
        raise ValueError("evaluator config size/hash mismatch")
    config=AutoConfig.from_pretrained(directory,local_files_only=True)
    if any(getattr(config,k,None)!=v for k,v in EXPECTED_CONFIG.items()):
        raise ValueError("evaluator config differs from the public model semantics")
    return config


def launch_identity(programs,device):
    """Bind reduction geometry while permitting another compatible device."""
    return dict(compute_capability=list(torch.cuda.get_device_capability(device)),
        phases={phase:dict(blocks=int(program.runtime.blocks),multiprocessors=int(program.runtime.multiprocessors))
                for phase,program in programs.items()})


class Native:
    def __init__(self, *, plain_dir, auxiliary_dir=None, recipe=None, device="cuda", blocks=None):
        from megakernel_lab.operation_compiler import ModuleUpdate
        from megakernel_lab.operation_compiler.variants import persistent_keys
        self.recipe=recipe or Recipe()
        if self.recipe.name=="supervision_ema" and auxiliary_dir is None:
            raise ValueError("supervision recipe requires both compiled phases")
        self.device=torch.empty(0,device=device).device
        self.directories={"plain":artifact_directory(plain_dir)}
        if auxiliary_dir is not None:
            self.directories["auxiliary"]=artifact_directory(auxiliary_dir)
        contracts={phase:phase_contract(directory,phase,self.recipe) for phase,directory in self.directories.items()}
        if any(value!=contracts["plain"] for value in contracts.values()):
            raise ValueError("compiled phases disagree on model semantics or immutable shared buffer bytes")
        self.arena=ModuleUpdate.allocate_shared_arena(*self.directories.values(),device=self.device)
        self.programs={}
        canonical=None
        for phase,directory in self.directories.items():
            manifest=json.loads((directory/"manifest.json").read_text())
            receipt=json.loads((directory/"artifact.json").read_text())
            keys=persistent_keys(manifest,receipt["module_api"])
            if canonical is not None and set(keys.values())!=set(canonical):
                raise ValueError("compiled phases disagree on persistent role/name keys")
            bindings=None if canonical is None else {index:canonical[key] for index,key in keys.items()}
            program=ModuleUpdate.load(directory,device=self.device,blocks=blocks,bindings=bindings,arena=self.arena)
            if canonical is None:
                canonical={key:program.runtime.args[index] for index,key in keys.items()}
            self.programs[phase]=program
        self.current=self.programs["plain"]
        self.params=self.current.parameters
        self.opt=self.current.optimizer_state
        self.model_state=self.current.state
        for program in self.programs.values():
            for kind in ("parameters","optimizer_state","state"):
                left,right=getattr(program,kind),getattr(self.current,kind)
                if left.keys()!=right.keys() or any(left[k].data_ptr()!=right[k].data_ptr() for k in left):
                    raise ValueError("phase persistent tensors are not shared: "+kind)
        self.next_step=0;self.staged_step=None
        self.coefficients=torch.empty(2,dtype=torch.float32,device=self.device)
        self._identity=self.identity()

    def identity(self):
        from megakernel_lab.operation_compiler import api,runtime
        roots=[Path(__file__).with_name(name) for name in ("native.py","optimizer.py","model.py",
               "train.py","data.py","conventional.py","compiled_model.py","pins.py")]
        roots.extend((Path(api.__file__),Path(runtime.__file__)))
        return dict(artifacts={p:artifact_identity(d) for p,d in self.directories.items()},
                    wrappers={str(p.name):digest(p) for p in roots if p.exists()},torch_version=str(torch.__version__),
                    launch=launch_identity(self.programs,self.device))

    def stage(self,x,y,index):
        if self.staged_step is not None or int(index)!=self.next_step:
            raise ValueError("native absolute update cursor mismatch or unfinished staged update")
        weight=self.recipe.supervision_weight(index)
        self.current=self.programs["auxiliary" if weight>0 else "plain"]
        self.coefficients.copy_(torch.tensor([1/(1+weight),weight/(2*(1+weight))],dtype=torch.float32,device=self.device))
        self.current.stage((x,),(y,self.coefficients))
        stage_schedule(self.opt,self.recipe,index)
        self.staged_step=int(index);self.weight=weight

    def step(self):
        if self.staged_step is None:
            raise RuntimeError("stage before native step")
        phase="auxiliary" if self.weight>0 else "plain"
        if self.current.runtime.blocks!=self._identity["launch"]["phases"][phase]["blocks"]:
            raise RuntimeError("native launch geometry changed; construct a new Native owner")
        self.current.step()
        index=self.staged_step
        self.staged_step=None;self.next_step=index+1
        return dict(loss=self.current.loss,norm=self.opt[NORM],clip=self.opt[CLIP],
                    learning_rate=self.recipe.lr(index),supervision_weight=self.weight)

    def state_dict(self, *, host=True):
        if self.staged_step is not None: raise RuntimeError("cannot snapshot a staged update")
        def snapshot(values):
            return {k:v.detach().to(device="cpu" if host else v.device,copy=True) for k,v in values.items()}
        return dict(schema="latch-tinymoe-native-state-v1",recipe=asdict(self.recipe),identity=self._identity,
                    next_step=self.next_step,parameters=snapshot(self.params),optimizer_state=snapshot(self.opt),
                    model_state=snapshot(self.model_state))

    @torch.no_grad()
    def snapshot(self, *, host=True, include_gradients=True):
        """Observe complete persistent state and actual generated diagnostics.

        Raw gradient outputs refer to the just-completed update; they are copied
        before any subsequent program may reuse the shared temporary arena.
        This is diagnostic work outside the complete-update performance clock.
        """
        result=self.state_dict(host=host)
        target="cpu" if host else self.device
        result["diagnostics"]={"loss":self.current.loss.detach().to(target,copy=True),
            "gradient_norm":self.opt[NORM].detach().to(target,copy=True),
            "clip_coefficient":self.opt[CLIP].detach().to(target,copy=True)}
        if include_gradients:
            result["gradients_before_clip"]={k:v.detach().to(target,copy=True) for k,v in self.current.grads.items()}
        return result

    @torch.no_grad()
    def load_state_dict(self,state):
        if (state["schema"]!="latch-tinymoe-native-state-v1" or state["recipe"]!=asdict(self.recipe)
                or state["identity"]!=self._identity):
            raise ValueError("native checkpoint requires exact recipe/program/library/runtime identity")
        if type(state["next_step"]) is not int or not 0<=state["next_step"]<=self.recipe.horizon:
            raise ValueError("invalid checkpoint cursor")
        for key,values in (("parameters",self.params),("optimizer_state",self.opt),("model_state",self.model_state)):
            incoming=state[key]
            if incoming.keys()!=values.keys(): raise ValueError("native tensor names differ: "+key)
            for name,v in values.items():
                s=incoming[name]
                if s.shape!=v.shape or s.dtype!=v.dtype: raise ValueError("native tensor ABI differs: "+name)
                v.copy_(s)
        self.next_step=state["next_step"];self.staged_step=None

    def save_checkpoint(self,path,contract):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(contract=contract,state=self.state_dict())
        temporary=path.with_suffix(path.suffix+".tmp")
        torch.save(payload,temporary);os.replace(temporary,path)
        receipt=dict(schema="latch-tinymoe-native-checkpoint-v1",bytes=path.stat().st_size,
                     sha256=digest(path),next_step=self.next_step,identity=self._identity)
        path.with_suffix(path.suffix+".json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt

    def load_checkpoint(self,path,contract):
        path=Path(path);receipt=json.loads(path.with_suffix(path.suffix+".json").read_text())
        if receipt["bytes"]!=path.stat().st_size or receipt["sha256"]!=digest(path):
            raise ValueError("native checkpoint size/hash mismatch")
        if receipt["identity"]!=self._identity: raise ValueError("native artifact identity differs")
        payload=torch.load(path,map_location="cpu",weights_only=False)
        if payload["contract"]!=contract: raise ValueError("native data/config contract differs")
        self.load_state_dict(payload["state"])
        if receipt["next_step"]!=self.next_step: raise ValueError("checkpoint completion cursor differs")

    def evaluator(self,model_dir,*,ema=False):
        """Zero-copy evaluation views; no second GPU copy of the 200M weights."""
        from .model import evaluation_view
        if ema and (self.recipe.name!="supervision_ema" or self.next_step<(self.recipe.horizon+1)//2):
            raise ValueError("EMA has not activated")
        config=evaluator_config(model_dir)
        parameters={name:self.opt["ema:"+name] if ema else value for name,value in self.params.items()}
        return evaluation_view(config,parameters,self.device)

    def execution_summary(self):
        result={}
        for phase,program in self.programs.items():
            manifest=program.runtime.manifest
            phases=manifest.get("phases",[])
            result[phase]=dict(phase_count=len(phases),phase_kinds=dict(Counter(p.get("kind","unknown") for p in phases)),
                generated_kernel_count=1,external_operations_in_step=[],
                evidence="ModuleUpdate Runtime.step calls one verified latch_launch/latch_launch_admitted cooperative kernel",
                external_staging="input/target/coefficient copies and two explicit scalar schedule fills before step",
                arena_bytes=self.arena.numel()*self.arena.element_size(),
                execution=manifest.get("execution"),source_hash=program.runtime.program.cache_key,
                launch_geometry=self._identity["launch"]["phases"][phase],
                library=program.runtime.library,manifest_path=str(self.directories[phase]/"manifest.json"))
        return result
