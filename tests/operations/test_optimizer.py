"""Stage 4 optimizer contracts. Run only on the admitted Modal CPU worker."""
from __future__ import annotations

import json
import traceback

import torch

from megakernel_lab.operation_compiler.capture import capture_training, capture_conditional_training
from megakernel_lab.operation_compiler.ir import CaptureError, TrainingIR, UniformConditionalIR
from megakernel_lab.operation_compiler.optim import ADAMW_STEP_LIMIT, AdamWConfig, SGDConfig, initialize_optimizer_state, prepare_optimizer
from megakernel_lab.operation_compiler.optimizer_reference import (adamw_reference_step, adamw_fp64_formula, compare_optimizer_step,
                                  adamw_update_from_gradients)


def _passed(value):
    if isinstance(value, dict):
        return value.get("passed", True) and all(_passed(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_passed(v) for v in value)
    return True


def _fixture():
    gen = torch.Generator().manual_seed(47101)
    p = {"z_weight": torch.randn(4, 4, generator=gen)*0.1,
         "a_bias": torch.randn(4, generator=gen)*0.1}
    data = (torch.randn(3, 4, generator=gen), torch.randn(3, 4, generator=gen))
    def loss(p, x, y):
        return (torch.tanh(x @ p["z_weight"] + p["a_bias"])-y).square().mean()
    return p, data, loss


def _refusal(function):
    try:
        function()
    except CaptureError as exc:
        return {"code": exc.code, "message": str(exc)}
    raise AssertionError("invalid optimizer contract was accepted")


def test_static_options_and_tied_state():
    p, _, _ = _fixture()
    tied = {"weight": p["z_weight"], "alias": p["z_weight"]}
    c = AdamWConfig(param_options={"alias": {"weight_decay": 0.0}})
    state = initialize_optimizer_state(tied, c)
    prepared = prepare_optimizer(c, tied, optimizer_state=state)
    assert prepared.parameter_names == ("weight",)
    assert prepared.state_names == ("exp_avg:weight", "exp_avg_sq:weight", "step:weight")
    assert prepared.options["weight"]["weight_decay"] == 0.0
    refusals = [
        _refusal(lambda: AdamWConfig(amsgrad=True).validate()),
        _refusal(lambda: AdamWConfig(maximize=True).validate()),
        _refusal(lambda: AdamWConfig(betas=(1., .99)).validate()),
        _refusal(lambda: AdamWConfig(eps=0).validate()),
        _refusal(lambda: AdamWConfig(lr=float("nan")).validate()),
        _refusal(lambda: prepare_optimizer(AdamWConfig(param_options={"absent": {"lr": .1}}), p)),
        _refusal(lambda: prepare_optimizer(AdamWConfig(param_options={"weight": {"lr": .1}, "alias": {"lr": .2}}), tied)),
        _refusal(lambda: prepare_optimizer(c, tied, optimizer_state={})),
    ]
    return {"state_names": prepared.state_names, "refusals": refusals}


def test_fresh_restored_trajectory_and_roundtrip():
    p, data, loss = _fixture()
    config = AdamWConfig(lr=.003, betas=(.85,.97), eps=1e-7, weight_decay=.04,
                          param_options={"a_bias": {"weight_decay": 0.0, "lr": .002}})
    state = initialize_optimizer_state(p, config)
    captured = capture_training(loss, p, None, data, config, optimizer_state=state)
    restored = TrainingIR.from_dict(json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True)))
    assert restored.output_names() == captured.ir.output_names()
    assert restored.lifetimes() == captured.ir.lifetimes()
    assert captured.optimizer_state_names == tuple(state)
    assert "momentum" not in captured.ir.outputs
    checks = []
    for step in range(41):
        args = (*p.values(), *state.values(), *data)
        got = captured.unpack(captured.graph_module(*args))
        expected = adamw_reference_step(loss, p, state, data, config=config)
        check = compare_optimizer_step(got, expected, p)
        assert _passed(check), (step, check)
        if step in (0,8,40):
            checks.append({"state":step, "checks":check})
        p, state = expected["params"], expected["optimizer_state"]
    return {"states":checks,"updates":41,"roundtrip_order":True}


def test_none_zero_detached_and_shared():
    p = {"used": torch.full((4,4),.2), "zero": torch.full((4,4),.3),
         "unused": torch.full((4,4),.4), "detached": torch.full((4,4),.5)}
    p["tied"] = p["used"]
    config = AdamWConfig(lr=.02, weight_decay=.2)
    state = initialize_optimizer_state(p, config)
    def loss(p,x):
        return ((x @ p["used"] + x @ p["tied"]).square().mean()
                + (p["zero"]*0).sum() + p["detached"].detach().sum())
    data = (torch.eye(4),)
    captured = capture_training(loss,p,None,data,config,optimizer_state=state)
    got = captured.unpack(captured.graph_module(*captured.example_args))
    expected = adamw_reference_step(loss,p,state,data,config=config)
    canonical = {k:p[k] for k in captured.parameter_names}
    assert _passed(compare_optimizer_step(got,expected,canonical))
    assert captured.ir.metadata["gradient_participation"] == {"used":True,"zero":True,"unused":False,"detached":False}
    for name in ("used","zero"):
        assert got["optimizer_state"][f"step:{name}"].item()==1
    for name in ("unused","detached"):
        assert torch.equal(got["params"][name],p[name])
        assert got["optimizer_state"][f"step:{name}"].item()==0
    assert not torch.equal(got["params"]["zero"],p["zero"])
    return {"participation":captured.ir.metadata["gradient_participation"],"single_tied_update":True}


def test_fp64_formula_and_wrong_version_controls():
    p = {"w": torch.tensor([[.3,-.4],[.2,-.1]])}
    grad = {"w":torch.tensor([[.07,-.02],[.04,.11]])}
    config = AdamWConfig(lr=.03,betas=(.7,.93),eps=.002,weight_decay=.4)
    state = initialize_optimizer_state(p,config)
    state["exp_avg:w"].fill_(.025)
    state["exp_avg_sq:w"].fill_(.007)
    state["step:w"].fill_(8)
    prepared=prepare_optimizer(config,p,optimizer_state=state)
    gotp,goto=prepared.update(tuple(p.values()),tuple(grad.values()),tuple(state.values()))
    refp,refo=adamw_fp64_formula(p,grad,state,config)
    torch.testing.assert_close(gotp[0].double(),refp["w"],atol=8e-8,rtol=2e-7)
    for name,value in zip(prepared.state_names,goto):
        if value.dtype==torch.int64: assert torch.equal(value,refo[name])
        else: torch.testing.assert_close(value.double(),refo[name],atol=5e-9,rtol=2e-7)
    # Independent deliberately wrong arithmetic: old moments, absent bias
    # correction, absent decay, and zeroed restored second moment must fail the
    # unchanged update gate, not just the permissive small-moment absolute floor.
    from megakernel_lab.operation_compiler.numerical_reference import compare
    oldm,oldv=p["w"].double()*0+.025,p["w"].double()*0+.007
    nm,nv=refo["exp_avg:w"],refo["exp_avg_sq:w"]
    b1,b2=config.betas
    decayed=p["w"].double()*(1-config.lr*config.weight_decay)
    candidates={
        "old_moment":decayed-config.lr*(oldm/(1-b1**9))/((oldv/(1-b2**9)).sqrt()+config.eps),
        "no_bias_correction":decayed-config.lr*nm/(nv.sqrt()+config.eps),
        "no_decay":refp["w"]+p["w"].double()*config.lr*config.weight_decay,
        "zero_restored_second_moment":decayed-config.lr*(nm/(1-b1**9))/(((1-b2)*grad["w"].double().square()/(1-b2**9)).sqrt()+config.eps),
    }
    rejected={k:compare(v-p["w"],refp["w"]-p["w"],"updates") for k,v in candidates.items()}
    assert all(not v["passed"] for v in rejected.values())
    return {"fp64_tight_formula":True,"intentional_negative_controls":rejected}


def test_beta_zero_and_near_one():
    p={"w":torch.tensor([.2,-.3])};g={"w":torch.tensor([.1,-.2])}
    reports=[]
    for betas in ((0.,0.),(.99999999,.9999999999)):
        config=AdamWConfig(lr=.001,betas=betas,eps=1e-6)
        state=initialize_optimizer_state(p,config)
        prepared=prepare_optimizer(config,p,optimizer_state=state)
        a,_=prepared.update(tuple(p.values()),tuple(g.values()),tuple(state.values()))
        b,_=adamw_fp64_formula(p,g,state,config)
        torch.testing.assert_close(a[0].double(),b["w"],atol=1e-7,rtol=1e-6)
        assert torch.isfinite(a[0]).all()
        reports.append({"betas":betas,"finite":True})
    return reports


def test_counter_bound_and_restored_moment_admission():
    params={"w":torch.tensor([.2,-.3])};grads={"w":torch.tensor([.1,-.2])}
    config=AdamWConfig(lr=.001,betas=(.9999999,.99999999),eps=1e-6)
    state=initialize_optimizer_state(params,config)
    state["step:w"].fill_(ADAMW_STEP_LIMIT-2)
    state["exp_avg:w"].fill_(.02);state["exp_avg_sq:w"].fill_(.003)
    prepared=prepare_optimizer(config,params,optimizer_state=state)
    gotp,goto=prepared.update(tuple(params.values()),tuple(grads.values()),tuple(state.values()))
    expected=adamw_update_from_gradients(params,grads,state,config)
    torch.testing.assert_close(gotp[0],expected["params"]["w"],atol=8e-8,rtol=2e-7)
    assert goto[-1].item()==ADAMW_STEP_LIMIT-1==expected["optimizer_state"]["step:w"].item()
    refusals=[]
    for step in (-1,ADAMW_STEP_LIMIT,2**63-1):
        invalid={k:v.clone() for k,v in state.items()};invalid["step:w"].fill_(step)
        refusals.append(_refusal(lambda:prepare_optimizer(config,params,optimizer_state=invalid)))
    for key,value in (("exp_avg:w",float("nan")),("exp_avg_sq:w",float("inf")),("exp_avg_sq:w",-.1)):
        invalid={k:v.clone() for k,v in state.items()};invalid[key].fill_(value)
        refusals.append(_refusal(lambda:prepare_optimizer(config,params,optimizer_state=invalid)))
    return {"largest_input_checked":ADAMW_STEP_LIMIT-2,"next_counter_exact":True,"refusals":refusals}


def test_conditional_parameter_participation():
    p={"left":torch.ones(4,4)*.1,"right":torch.ones(4,4)*.2}
    x=torch.eye(4)
    config=AdamWConfig(lr=.01,weight_decay=.3)
    state=initialize_optimizer_state(p,config)
    functions=(lambda p,x:(x@p["left"]).square().mean(),lambda p,x:(x@p["right"]).square().mean())
    cap=capture_conditional_training(*functions,p,None,(x,),torch.tensor(True),config,optimizer_state=state)
    roundtrip=UniformConditionalIR.from_dict(json.loads(json.dumps(cap.ir.to_dict(),sort_keys=True)))
    assert roundtrip.output_names()==cap.ir.output_names()
    counters={"left":0,"right":0}
    for flag in (True,False,False,True,False):
        branch=cap.true_branch if flag else cap.false_branch
        got=cap.unpack(branch.graph_module(*p.values(),*state.values(),x))
        expected=adamw_reference_step(functions[0 if flag else 1],p,state,(x,),config=config)
        assert _passed(compare_optimizer_step(got,expected,p))
        counters["left" if flag else "right"]+=1
        for name,value in counters.items():assert got["optimizer_state"][f"step:{name}"].item()==value
        p,state=got["params"],got["optimizer_state"]
    return {"independent_counters":counters,"conditional_roundtrip":True}


def test_constant_snapshot_and_rebinding():
    p,data,loss=_fixture()
    constant=torch.tensor([.1,.2,.3,.4])
    def shifted(p,x,y):return loss(p,x,y+constant)
    cap=capture_training(shifted,p,None,data,AdamWConfig())
    assert cap.constants and cap.ir.metadata["constant_input_names"]
    before=cap.graph_module(*cap.example_args)
    constant.add_(17)
    after=cap.step_fn(*cap.example_args)
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    assert len([v for v in cap.ir.values.values() if v.role=="input"])==len(data)
    assert all(cap.ir.values[name].role=="constant_input" for name in cap.ir.metadata["constant_input_names"])
    changed=(*cap.example_args[:-1],cap.example_args[-1]+.5)
    assert not torch.equal(cap.step_fn(*changed)[0],before[0])
    return {"frozen_closure":True,"explicit_rebinding":True}


def test_sgd_legacy_adapter():
    p,data,loss=_fixture()
    config=SGDConfig()
    state=initialize_optimizer_state(p,config)
    a=capture_training(loss,p,state,data,config)
    b=capture_training(loss,p,None,data,config,optimizer_state=state)
    ao=a.unpack(a.graph_module(*a.example_args));bo=b.unpack(b.graph_module(*b.example_args))
    assert a.ir.output_names()==b.ir.output_names()
    assert ao["momentum"] is ao["optimizer_state"]
    assert all(torch.equal(ao["params"][k],bo["params"][k]) for k in p)
    return {"legacy_argument_layout":True,"momentum_view_alias":True}


def test_ordinary_cross_entropy_loss():
    gen=torch.Generator().manual_seed(47801)
    params={"head":torch.randn(4,3,generator=gen)*.1}
    data=(torch.randn(5,4,generator=gen),torch.tensor([0,2,1,0,1]))
    def loss(p,x,target):return torch.nn.functional.cross_entropy(x@p["head"],target)
    config=AdamWConfig(eps=1e-5)
    state=initialize_optimizer_state(params,config)
    captured=capture_training(loss,params,None,data,config,optimizer_state=state)
    actual=captured.unpack(captured.graph_module(*captured.example_args))
    expected=adamw_reference_step(loss,params,state,data,config=config)
    assert _passed(compare_optimizer_step(actual,expected,params))
    assert not any("nll_loss" in op.target for op in captured.ir.operations)
    return {"ordinary_functional_cross_entropy":True,"operations":sorted({op.target for op in captured.ir.operations})}


def run():
    torch.set_num_threads(1)
    results={}
    for name,fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: results[name]={"passed":True,"details":fn()}
            except Exception as exc: results[name]={"passed":False,"error":str(exc),"traceback":traceback.format_exc()}
    return {"torch":torch.__version__,"tests":results,"passed":all(v["passed"] for v in results.values())}


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
