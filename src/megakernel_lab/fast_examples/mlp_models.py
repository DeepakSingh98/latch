"""Historical USER MODEL reproduction only. Frozen Latch handles all derivatives/codegen."""
import math,json
from pathlib import Path
import torch
from torch import nn
from megakernel_lab.fast_examples.recovered.models import Case,parameters,fixture
spec=json.loads((Path(__file__).parent/'historical_suite.json').read_text())
CASES={}
for name,s in {**spec['mlp_programs']['tune'],**spec['mlp_programs']['heldout']}.items():
 for bf in (False,True):
  CASES[name+('-bf16' if bf else '-fp32')]=dict(kind='conv',family='historical-mlp',case=s['case'],untied=s['untied'],native_bf16=bf,lr=.01,momentum=.9,decay=.001)
# Nearby structural transfer: only depth changes relative to aux-64.
for bf in (False,True):
 c=dict(CASES['aux-64'+('-bf16' if bf else '-fp32')]);c['case']=dict(c['case'],depth=4)
 CASES['aux-64-depth4'+('-bf16' if bf else '-fp32')]=c
# Reserved layout transfers: values/arithmetic are identical, storage strides differ.
for base in ('shared-64-bf16', 'heldout-128-bf16', 'chain-aux-skip-320-fp32', 'shared-aux-odd-200-bf16'):
 CASES[base+'-strided']=dict(CASES[base],layout='strided')
class Historical(nn.Module):
 def __init__(self,c):
  super().__init__();self.c=c;case=Case(name='historical',**c['case']);p=parameters(case)
  if c['untied']:
   del p['shared'];g=torch.Generator().manual_seed(case.seed+500)
   for i in range(case.depth):p[f'shared_{i}']=torch.randn(case.width,case.width,generator=g)*(.6/math.sqrt(case.width))
  self.p=nn.ParameterDict({n:nn.Parameter(v) for n,v in p.items()});self.bf16=c.get('native_bf16',False) or c.get('bf16',False)
 def forward(self,x):
  c=self.c['case'];p=self.p
  def mm(a,b):return (a.bfloat16()@b.bfloat16()).float() if self.bf16 else a@b
  activate=torch.tanh if c['activation']=='tanh' else torch.sigmoid
  h0=activate(mm(x,p['input'])+p['input_bias']);h=h0;aux=None
  for i in range(c['depth']):
   h=h+c['residual']*activate(mm(h,p[f'shared_{i}' if self.c['untied'] else 'shared'])+p['shared_bias'])
   if i==0 and c['auxiliary']:aux=mm(h,p['aux_head'])
  if c['skip']:h=h+.2*h0
  return mm(h,p['head'])+p['head_bias'],aux

def model(c):return Historical(c)
def batch(c,step,device='cpu',cl=False):
 x,y=fixture(Case(name='historical',**c['case']),step=step,device=device)
 if c.get('layout')=='strided':
  x=x.t().contiguous().t()
  y=y.t().contiguous().t()
 return x,y
def loss_fn(c):
 def loss(pred,target):
  main,aux=pred;r=((main-target)**2).sum(1).mean()
  return r+c['case']['auxiliary']*((aux-target)**2).sum(1).mean() if aux is not None else r
 return loss
def objective(m,x,y):return loss_fn(m.c)(m(x),y)
