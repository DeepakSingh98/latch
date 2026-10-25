import torch

from megakernel_lab.tinymoe.compare_checkpoints import compare


def test_restart_comparator_distinguishes_signed_zero_bits():
    left={"parameter":torch.tensor([0.0,-0.0]),"cursor":200}
    right={"parameter":torch.tensor([-0.0,0.0]),"cursor":200}
    assert torch.equal(left["parameter"],right["parameter"])
    assert not compare(left,right)["passed"]
    assert compare(left,{"parameter":left["parameter"].clone(),"cursor":200})["passed"]


def test_restart_comparator_checks_scalar_slots_and_cursor():
    state={"step":torch.tensor(200,dtype=torch.int64),"next_step":200}
    assert compare(state,{"step":torch.tensor(200,dtype=torch.int64),"next_step":200})["passed"]
    assert not compare(state,{"step":torch.tensor(199,dtype=torch.int64),"next_step":200})["passed"]
    assert not compare(state,{"step":torch.tensor(200,dtype=torch.int64),"next_step":199})["passed"]
