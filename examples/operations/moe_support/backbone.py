"""Pinned recipe constructors, restricted to the single supported MoE arm. See SOURCE.json."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from .fgm_model import ModelConfig as RoutingConfig, Attention, _init_weight
from .moe import ExpertChoice
MOE_ARMS = ('moe_intermediate_supervision',)
SUPERVISED_ARMS = MOE_ARMS

@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 384
    n_layers: int = 3
    n_heads: int = 6
    seq_len: int = 2048
    vocab_size: int = 50257
    routing_group: int = 32
    seed: int = 1729
    arm: str = 'ordinary'
    d_ff: int | None = None
    init_scale: float = .1
    attention_precision: str = 'bf16'
    adaptive_chunk: int = 128
    inner_lr: float = 3.
    implementation: str = 'torch'
    compile_adaptive: bool = False
    def __post_init__(self):
        if self.d_ff is None: object.__setattr__(self, 'd_ff', 4*self.d_model)
        if self.arm != 'moe_intermediate_supervision': raise ValueError('this example supports the MoE intermediate-supervision recipe only')
        if self.implementation != 'torch': raise ValueError('only conventional torch is qualified; future executors remain separate')
        if min(self.d_model,self.n_layers,self.n_heads,self.seq_len,self.routing_group,self.d_ff,self.adaptive_chunk)<1: raise ValueError('positive dimensions required')
        if self.d_model%self.n_heads or self.d_ff%8: raise ValueError('exact head/expert dimensions required')
        if self.arm in SUPERVISED_ARMS and self.n_layers<3: raise ValueError('two intermediate exits require >=3 layers')
    @property
    def architecture(self): return 'moe' if self.arm in MOE_ARMS else 'dense'
    @property
    def n_experts(self): return 32
    def routing(self):
        return RoutingConfig(vocab_size=self.vocab_size,d_model=self.d_model,n_layers=self.n_layers,n_heads=self.n_heads,seq_len=self.seq_len,d_ff=self.d_ff,seed=self.seed,architecture=self.architecture,granularity=8,expansion=4,routing_group=self.routing_group,init_scale=self.init_scale,attention_precision=self.attention_precision)
    def to_dict(self): return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):
        dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
        z=x.to(dtype)
        return z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*self.weight


class Block(nn.Module):
    def __init__(self,cfg,index):
        super().__init__(); stream=f'blocks.{index}'
        self.attn_norm=RMSNorm(cfg.d_model); self.attn=Attention(cfg,stream+'.attn')
        self.ff_norm=RMSNorm(cfg.d_model)
        self.ff=ExpertChoice(cfg.routing(),stream+'.ff')
    def forward(self,x):
        x=x+self.attn(self.attn_norm(x)); return x+self.ff(self.ff_norm(x))


class LanguageModel(nn.Module):
    def __init__(self,cfg,context):
        super().__init__(); self.config=cfg; self.parallel_context=context; self._gradient_reducer=None
        self.token_embed=nn.Embedding(cfg.vocab_size,cfg.d_model,_weight=torch.empty(cfg.vocab_size,cfg.d_model))
        self.position_embed=nn.Embedding(cfg.seq_len,cfg.d_model,_weight=torch.empty(cfg.seq_len,cfg.d_model))
        _init_weight(self.token_embed.weight,cfg.d_model,cfg,'token_embed.weight'); _init_weight(self.position_embed.weight,1,cfg,'position_embed.weight')
        self.blocks=nn.ModuleList([Block(cfg,i) for i in range(cfg.n_layers)])
        self.final_norm=RMSNorm(cfg.d_model)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.lm_head.weight=self.token_embed.weight
        self.aggregation=None
        self.second_head=None
        if context.world_size != 1:
            raise ValueError('this example supports one complete routing group on one GPU')
