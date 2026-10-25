"""CPU mutation tests for phase sharing and trained paired-run imports."""
from dataclasses import asdict
import json

import pytest
import torch

from megakernel_lab.tinymoe.data import digest
from megakernel_lab.tinymoe.model import Recipe
from megakernel_lab.tinymoe.native import (Native,artifact_identity,phase_contract,checkpoint_contract,
                                         evaluator_config,launch_identity)
from megakernel_lab.tinymoe.pins import public_identity,EXPECTED_CONFIG


def artifact(path,phase="plain",buffer=None):
    path.mkdir()
    recipe=Recipe(name="supervision_ema")
    buffer=torch.tensor([0.,-0.]) if buffer is None else buffer
    torch.save({"args":(buffer,),"constants":{}},path/"bindings.pt")
    manifest=dict(inputs=["rope"],input_roles={"rope":{"role":"input","parameter":None}})
    (path/"manifest.json").write_text(json.dumps(manifest))
    (path/"library.so").write_bytes(b"fixture")
    receipt=dict(files={name:digest(path/name) for name in ("manifest.json","bindings.pt")},
        library=str(path/"library.so"),library_sha256=digest(path/"library.so"),source_hash="fixture",
        module_api={"input_indices":{"buffers":{"rope":0}}})
    (path/"artifact.json").write_text(json.dumps(receipt))
    metadata=dict(schema="latch-tinymoe-build-v1",**public_identity(),phase=phase,recipe=asdict(recipe),
        ema_active=phase=="plain",routing="unconstrained",batch_size=1,sequence_length=256,
        routing_diagnostics=True,objective="fixed objective",attention="fixed attention",expert_policy="fixed experts",
        artifact={**receipt,"explain":{"directory":"post-build-report"}})
    (path/"tinymoe-build.json").write_text(json.dumps(metadata))
    return recipe,metadata


def test_equal_immutable_buffers_allow_two_phases(tmp_path):
    recipe,_=artifact(tmp_path/"plain")
    artifact(tmp_path/"auxiliary",phase="auxiliary")
    assert phase_contract(tmp_path/"plain","plain",recipe)==phase_contract(tmp_path/"auxiliary","auxiliary",recipe)


def test_changed_shared_buffer_bits_differ_even_with_same_shape(tmp_path):
    recipe,_=artifact(tmp_path/"plain")
    artifact(tmp_path/"auxiliary",phase="auxiliary",buffer=torch.tensor([-0.,0.]))
    left=phase_contract(tmp_path/"plain","plain",recipe)
    right=phase_contract(tmp_path/"auxiliary","auxiliary",recipe)
    assert left["common"]==right["common"]
    assert left["immutable"]!=right["immutable"]
    with pytest.raises(ValueError,match="immutable shared buffer bytes"):
        Native(plain_dir=tmp_path/"plain",auxiliary_dir=tmp_path/"auxiliary",recipe=recipe,device="cpu")


@pytest.mark.parametrize("mutation",["missing","recipe","phase","receipt","diagnostic_routes"])
def test_metadata_mismatch_is_refused(tmp_path,mutation):
    path=tmp_path/"artifact";recipe,metadata=artifact(path)
    if mutation=="missing":(path/"tinymoe-build.json").unlink()
    else:
        if mutation=="recipe":metadata["recipe"]["weight_decay"]*=2
        if mutation=="phase":metadata["phase"]="auxiliary"
        if mutation=="receipt":metadata["artifact"]["source_hash"]="other program"
        if mutation=="diagnostic_routes":metadata["routing"]="diagnostic supplied membership"
        (path/"tinymoe-build.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):phase_contract(path,"plain",recipe)


def test_modified_binding_file_requires_matching_receipt(tmp_path):
    path=tmp_path/"artifact";recipe,_=artifact(path)
    torch.save({"args":(torch.ones(2),),"constants":{}},path/"bindings.pt")
    with pytest.raises(ValueError,match="artifact identity"):
        phase_contract(path,"plain",recipe)


def test_native_identity_binds_build_metadata_bytes(tmp_path):
    path=tmp_path/"artifact";_,metadata=artifact(path)
    first=artifact_identity(path)
    metadata["capture_seconds"]=3.
    (path/"tinymoe-build.json").write_text(json.dumps(metadata))
    second=artifact_identity(path)
    assert first["artifact_sha256"]==second["artifact_sha256"]
    assert first["build_metadata_sha256"]!=second["build_metadata_sha256"]


@pytest.mark.parametrize("mutation",[None,"corpus","batch","sequence","recipe","public","identity","missing"])
def test_paired_checkpoint_contract_binds_data_shape_schedule(tmp_path,mutation):
    (tmp_path/"manifest.json").write_text('{"corpus":"first"}')
    recipe=Recipe();state={"identity":{"artifact":"fixture"}}
    contract=dict(**public_identity(),recipe=asdict(recipe),backend="latch",batch_size=1,sequence_length=256,
                  data_manifest_sha256=digest(tmp_path/"manifest.json"),identity=state["identity"])
    if mutation=="corpus":(tmp_path/"manifest.json").write_text('{"corpus":"other"}')
    if mutation=="batch":contract["batch_size"]=2
    if mutation=="sequence":contract["sequence_length"]=512
    if mutation=="recipe":contract["recipe"]["horizon"]=800
    if mutation=="public":contract["model_revision"]="other"
    if mutation=="identity":contract["identity"]={"artifact":"other"}
    if mutation=="missing":contract=None
    if mutation is None:
        checkpoint_contract(state,contract,recipe=recipe,data_dir=tmp_path,batch_size=1,sequence_length=256)
    else:
        with pytest.raises(ValueError,match="contract differs"):
            checkpoint_contract(state,contract,recipe=recipe,data_dir=tmp_path,batch_size=1,sequence_length=256)


@pytest.mark.parametrize("mutation",[None,"public","unbound_config","bound_wrong_semantics","transformers"])
def test_evaluator_validates_config_before_model_creation(tmp_path,monkeypatch,mutation):
    import transformers
    from types import SimpleNamespace
    config=dict(EXPECTED_CONFIG)
    path=tmp_path/"config.json";path.write_text(json.dumps(config))
    identity=dict(**public_identity(),files={"config.json":{"bytes":path.stat().st_size,"sha256":digest(path)}})
    if mutation=="public":identity["model_revision"]="other"
    if mutation in {"unbound_config","bound_wrong_semantics"}:
        config["rope_theta"]=10000.
        path.write_text(json.dumps(config))
    if mutation=="bound_wrong_semantics":
        identity["files"]["config.json"]={"bytes":path.stat().st_size,"sha256":digest(path)}
    if mutation=="transformers":monkeypatch.setattr(transformers,"__version__","other")
    (tmp_path/"latch_identity.json").write_text(json.dumps(identity))
    calls=[]
    def read(directory,*,local_files_only):
        calls.append(str(directory));assert local_files_only
        return SimpleNamespace(**json.loads((directory/"config.json").read_text()))
    monkeypatch.setattr(transformers.AutoConfig,"from_pretrained",read)
    if mutation is None:
        assert evaluator_config(tmp_path).rope_theta==EXPECTED_CONFIG["rope_theta"]
        assert len(calls)==1
    else:
        with pytest.raises(ValueError,match="evaluator"):
            evaluator_config(tmp_path)
        assert len(calls)==int(mutation=="bound_wrong_semantics")


def test_launch_identity_binds_geometry_without_device_identity(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda device:(9,0))
    runtime=SimpleNamespace(blocks=132,multiprocessors=132)
    programs={"plain":SimpleNamespace(runtime=runtime)}
    first=launch_identity(programs,"cuda:0")
    assert first==launch_identity(programs,"cuda:1")
    runtime.blocks=66
    assert first!=launch_identity(programs,"cuda:0")
    runtime.blocks=132;runtime.multiprocessors=114
    assert first!=launch_identity(programs,"cuda:0")


def test_changed_launch_blocks_refused_before_update():
    from types import SimpleNamespace
    owner=Native.__new__(Native)
    owner.staged_step=0;owner.weight=0.
    owner.current=SimpleNamespace(runtime=SimpleNamespace(blocks=66))
    owner._identity={"launch":{"phases":{"plain":{"blocks":132}}}}
    with pytest.raises(RuntimeError,match="geometry changed"):
        owner.step()
