from copy import deepcopy

import pytest
import torch

from megakernel_lab.tinymoe.evaluate import checkpoint_contents


def native_checkpoint(completed=200):
    recipe=dict(name="supervision_ema",horizon=400)
    identity=dict(artifacts={"plain":"bound-artifact"})
    raw={"weight":torch.tensor([1.])};average=torch.tensor([2.])
    payload=dict(contract=dict(backend="latch",recipe=recipe,identity=identity),
        state=dict(schema="latch-tinymoe-native-state-v1",recipe=deepcopy(recipe),
                   identity=deepcopy(identity),next_step=completed,parameters=raw,
                   optimizer_state={"ema:weight":average}))
    return payload,dict(identity=deepcopy(identity),next_step=completed)


def test_native_evaluation_selects_ema_only_after_original_boundary():
    before,receipt=native_checkpoint(199)
    assert checkpoint_contents(before,receipt)[3]=={}
    active,receipt=native_checkpoint(200)
    assert checkpoint_contents(active,receipt)[3]["weight"].item()==2.


@pytest.mark.parametrize("mismatch",["state_recipe","state_identity","receipt_identity","receipt_cursor","backend"])
def test_native_evaluation_rejects_internal_checkpoint_mismatch(mismatch):
    payload,receipt=native_checkpoint()
    if mismatch=="state_recipe":payload["state"]["recipe"]["name"]="unchanged"
    elif mismatch=="state_identity":payload["state"]["identity"]["artifacts"]={}
    elif mismatch=="receipt_identity":receipt["identity"]={}
    elif mismatch=="receipt_cursor":receipt["next_step"]=199
    else:payload["contract"]["backend"]="compiled-grouped"
    with pytest.raises(ValueError,match="state/receipt/contract"):
        checkpoint_contents(payload,receipt)


def conventional_checkpoint():
    raw={"weight":torch.tensor([1.])}
    payload=dict(schema="latch-tinymoe-conventional-checkpoint-v1",
        contract=dict(backend="compiled-grouped",recipe=dict(name="supervision_ema",horizon=400)),
        completed_updates=200,model=raw,
        ema=dict(policy=dict(horizon_updates=400),last_step=200,updates=0,
                 average={"weight":torch.tensor([2.])}))
    return payload,dict(completed_updates=200)


@pytest.mark.parametrize("mismatch",[None,"receipt_cursor","ema_cursor","ema_horizon","ema_coverage"])
def test_conventional_evaluation_validates_cursor_and_ema(mismatch):
    payload,receipt=conventional_checkpoint()
    if mismatch is None:
        assert checkpoint_contents(payload,receipt)[3]["weight"].item()==2.
        return
    if mismatch=="receipt_cursor":receipt["completed_updates"]=199
    elif mismatch=="ema_cursor":payload["ema"]["last_step"]=199
    elif mismatch=="ema_horizon":payload["ema"]["policy"]["horizon_updates"]=200
    else:payload["ema"]["average"]={}
    with pytest.raises(ValueError):checkpoint_contents(payload,receipt)
