"""CPU bitwise comparison of uninterrupted versus resumed same-backend states."""
import argparse
import json
from pathlib import Path

import torch

from .data import digest


def load(path):
    path=Path(path)
    receipt=json.loads(path.with_suffix(path.suffix+".json").read_text())
    if receipt["sha256"]!=digest(path) or receipt["bytes"]!=path.stat().st_size:
        raise ValueError("checkpoint completion hash/size mismatch: "+str(path))
    return torch.load(path,map_location="cpu",weights_only=False,mmap=True)


def compare(left,right):
    failures=[];tensors=0;elements=0
    def visit(a,b,path):
        nonlocal tensors,elements
        if isinstance(a,torch.Tensor):
            tensors+=1;elements+=a.numel()
            if (not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape
                    or not torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                       b.contiguous().reshape(-1).view(torch.uint8))):
                failures.append(dict(path=path,kind="tensor",shape=list(a.shape),dtype=str(a.dtype)))
        elif isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():
                failures.append(dict(path=path,kind="mapping_keys"));return
            for key,value in a.items():visit(value,b[key],path+"/"+str(key))
        elif isinstance(a,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):
                failures.append(dict(path=path,kind="sequence"));return
            for index,(x,y) in enumerate(zip(a,b)):visit(x,y,path+"/"+str(index))
        elif a!=b:
            failures.append(dict(path=path,kind="metadata",left=repr(a),right=repr(b)))
    visit(left,right,"")
    return dict(passed=not failures,tensor_count=tensors,tensor_elements=elements,failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left",required=True);p.add_argument("--right",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();torch.set_num_threads(1)
    result=dict(schema="latch-tinymoe-exact-restart-v1",left=a.left,right=a.right,**compare(load(a.left),load(a.right)))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
