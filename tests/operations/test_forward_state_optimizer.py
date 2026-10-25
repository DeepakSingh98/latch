"""Forward-produced participation is explicit and separate from gradient value."""
import torch
import pytest
from megakernel_lab.operation_compiler.user_optimizer import tensor_update, prepare_tensor_rule
from megakernel_lab.operation_compiler import capture_module
from megakernel_lab.operation_compiler.ir import CaptureError


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2., 3.]))

    def forward(self, x, membership, *, state):
        return self.weight * x, {"active": membership.float()}


def rule():
    def init(p, hp):
        return {"count": torch.zeros((), dtype=torch.int64)}
    def update(p, g, s, hp, forward):
        return {"weight": torch.where(forward["active"] > 0, p["weight"] - .1*g["weight"], p["weight"])}, {"count": s["count"] + 1}
    return tensor_update(update, init_state=init, forward_state=True)


def test_forward_state_capture():
    model = Model()
    c = capture_module(model, lambda y: y.sum(), (torch.ones(2), torch.tensor([1.,0.])),
                       state={"active": torch.zeros(2)}, optimizer=rule())
    assert c.ir.metadata["optimizer"]["forward_state"]
    assert c.ir.metadata["module_api"]["stateful_forward"]


def test_forward_participation_preserves_inactive_parameter():
    model = Model()
    p = dict(model.named_parameters())
    prepared = prepare_tensor_rule(rule(), p)
    new, state = prepared.update(tuple(p.values()), (torch.ones(2),), prepared.state_values,
                                 forward_state={"active": torch.tensor([0., 1.])})
    torch.testing.assert_close(new[0], torch.tensor([2.,2.9]))
    assert state[0].item() == 1


def test_forward_state_mutation_is_refused():
    def update(p,g,s,hp,forward):
        forward['active'].zero_()
        return p,s
    prepared=prepare_tensor_rule(tensor_update(update,forward_state=True),{'w':torch.ones(2)})
    with pytest.raises(CaptureError,match='OPTIMIZER_MUTATION'):
        prepared.update((torch.ones(2),),(torch.ones(2),),(),forward_state={'active':torch.ones(2)})


def test_forward_state_rule_refuses_stateless_model():
    with pytest.raises(CaptureError,match='OPTIMIZER'):
        capture_module(torch.nn.Linear(2,1),lambda y:y.sum(),(torch.ones(1,2),),optimizer=rule())


@pytest.mark.parametrize('detach_again',[False,True])
def test_returned_forward_input_is_copied_before_commit(detach_again):
    rule=tensor_update(lambda p,g,s,hp,f:({'w':f['replacement'].detach().detach()
                                         if detach_again else f['replacement']},{}),forward_state=True)
    prepared=prepare_tensor_rule(rule,{'w':torch.ones(2)})
    replacement=torch.tensor([3.,4.])
    updated,_=prepared.update((torch.ones(2),),(torch.ones(2),),(),
                              forward_state={'replacement':replacement})
    assert torch.equal(updated[0],replacement)
    assert updated[0].data_ptr()!=replacement.data_ptr()


def test_detached_forward_alias_is_captured_as_copied_output():
    update=tensor_update(lambda p,g,s,hp,f:({'weight':f['active'].detach().detach()},{}),forward_state=True)
    captured=capture_module(Model(),lambda y:y.sum(),(torch.ones(2),torch.tensor([1.,0.])),
                            state={'active':torch.zeros(2)},optimizer=update)
    result=captured.ir.outputs['params']['weight']
    operation=next(op for op in captured.ir.operations if result in op.outputs)
    assert operation.target=='aten.clone.default'
