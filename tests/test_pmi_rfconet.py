"""Actual network, source fidelity and shared training/evaluation integration."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import urllib.request
import pytest
import torch

model_module = importlib.import_module("comparison.PMI-RFCoNet.model")
trainer = importlib.import_module("comparison.PMI-RFCoNet.train")
evaluator = importlib.import_module("comparison.PMI-RFCoNet.test_hsi_deformed")
PMIRFCoNet = model_module.PMIRFCoNet
torch.set_num_threads(2)


def test_real_backward_keeps_two_by_two_bottleneck_and_reaches_all_stages():
    torch.manual_seed(7)
    model = PMIRFCoNet(7,3,mid_channels=8,num_fc=1).train()
    shapes = []
    handle = model.net.field_1x.reg_field.conv_maxpool3.register_forward_hook(
        lambda _module,_args,out: shapes.append(tuple(out.shape[-2:])))
    lr=torch.rand(2,7,16,16);msi=torch.rand(2,3,64,64)
    out=model(lr,msi)
    torch.nn.functional.l1_loss(out,torch.rand_like(out)).backward()
    handle.remove()
    assert out.shape==(2,7,64,64) and shapes==[(2,2)]
    for name in ("field_1x","field_2x","field_4x","StyleBlock_1x","StyleBlock_2x","StyleBlock_4x"):
        grads=[v.grad for v in getattr(model.net,name).parameters() if v.grad is not None]
        assert grads and all(torch.isfinite(v).all() for v in grads)
        assert any(torch.count_nonzero(v)>0 for v in grads)


def test_both_observations_affect_output_and_forward_does_not_optimize():
    torch.manual_seed(12)
    model=PMIRFCoNet(7,3,mid_channels=8,num_fc=1).eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    lr=torch.rand(1,7,16,16);msi=torch.rand(1,3,64,64)
    with torch.no_grad():
        out=model(lr,msi)
        assert not torch.allclose(out,model(lr*.1,msi))
        assert not torch.allclose(out,model(lr,msi*.1))
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


@pytest.mark.parametrize("lr_hw,hr_hw",[((8,8),(32,32)),((17,16),(68,64)),((16,16),(48,48))])
def test_rejects_unsupported_spatial_shapes(lr_hw,hr_hw):
    model=PMIRFCoNet(7,3,mid_channels=8,num_fc=1)
    with pytest.raises(ValueError):
        model(torch.rand(1,7,*lr_hw),torch.rand(1,3,*hr_hw))


@pytest.mark.skipif(os.environ.get("PMI_TEST_OFFICIAL")!="1",reason="Optional pinned upstream download")
def test_matches_full_released_forward_and_gradients():
    url=("https://raw.githubusercontent.com/Jiahuiqu/PMI-RFCoNet/"
         +model_module.UPSTREAM_COMMIT+"/model.py")
    raw=urllib.request.urlopen(url,timeout=45).read()
    assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==model_module.UPSTREAM_BLOB_SHA
    source={"__name__":"pmi_official_reference"}
    exec(compile(raw.decode(),"official_pmi_model","exec"),source)
    reference=source["SuperResolutionModel"](0,7,7,8,4,1).eval()
    for module in reference.modules():
        if isinstance(module,source["SpatialTransformation"]):
            module.use_gpu=False
    model=PMIRFCoNet(7,4,mid_channels=8,num_fc=1).eval()
    reference.load_state_dict(model.net.state_dict(),strict=True)
    lr=torch.rand(1,7,16,16,requires_grad=True);msi=torch.rand(1,4,64,64)
    lr_ref=lr.detach().clone().requires_grad_(True)
    actual=model(lr,msi);expected=reference(lr_ref,msi)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(lr.grad,lr_ref.grad,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(model.net.to_hsi.weight.grad,reference.to_hsi.weight.grad,rtol=1e-6,atol=1e-7)


def test_root_commands_keep_method_optimizer_and_shared_geometry(monkeypatch,tmp_path):
    import experiment_protocol as protocol
    adapter=protocol.resolve_model("PMIRFCoNet")
    monkeypatch.setattr(protocol,"ROOT",tmp_path)
    stage1=protocol.checkpoint_path(adapter,"PaviaU","registered")
    stage1.parent.mkdir(parents=True);stage1.touch()
    for stage in ("registered","mixed"):
        cmd=protocol.build_train_command(adapter,"PaviaU",stage,"cpu")
        assert cmd[cmd.index("--lr")+1]=="0.001"
        assert cmd[cmd.index("--weight_decay")+1]=="0.0"
        assert cmd[cmd.index("--batch_size")+1]=="1"
        if stage=="mixed":
            assert cmd[cmd.index("--registered_probability")+1]=="0.1"
            assert "--init_checkpoint" in cmd
        test=protocol.build_test_command(adapter,"PaviaU",stage,"cpu",check_files=False)
        if stage=="mixed":
            assert Path(test[1]).name=="test_multiseed_deformation.py"
        else:
            assert test[test.index("--test_mode")+1]=="registered_only"


def fixture_datasets():
    from degradations import build_degradation
    p0=build_degradation("physical",scale_ratio=4,mtf_nyquist=.2,truncate=3.)
    g=torch.Generator().manual_seed(29)
    srf=torch.rand(3,7,generator=g);srf/=srf.sum(1,keepdim=True)
    class Data:
        degradation_operator=p0
        def __init__(self,size):
            self.gt=torch.rand(7,size,size,generator=g)*.6+.1
            self.msi=torch.einsum("mc,chw->mhw",srf,self.gt)
        def __len__(self): return 1
        def __getitem__(self,_):return {"gt":self.gt.clone(),"hr_msi":self.msi.clone()}
    return Data(64),Data(128),Data(128),dict(n_bands=7,n_select_bands=3,srf_weights=srf)


@pytest.fixture
def trained_stages(tmp_path,monkeypatch):
    fixture=fixture_datasets()
    monkeypatch.setattr(trainer,"build_datasets",lambda *a,**k:fixture)
    monkeypatch.setattr(evaluator,"build_datasets",lambda *a,**k:fixture)
    common=["--dataset","PaviaU","--device","cpu","--epochs","1",
            "--validation_interval","1","--validation_cases","1","--max_train_batches","1",
            "--mid_channels","8","--num_fc","1","--lr_milestones","1","2"]
    registered=tmp_path/"registered";mixed=tmp_path/"mixed"
    trainer.main("registered",common+["--checkpoint_dir",str(registered),"--log_dir",str(tmp_path/"log_reg")])
    init=str(registered/"best.pth.tar")
    trainer.main("mixed",common+["--init_checkpoint",init,"--checkpoint_dir",str(mixed),"--log_dir",str(tmp_path/"log_mix")])
    return fixture,common,registered,mixed


def test_actual_stage_training_scheduler_and_exact_resume(trained_stages,tmp_path):
    fixture,common,registered,mixed=trained_stages
    reg=torch.load(registered/"best.pth.tar",weights_only=False)
    state=torch.load(mixed/"last.pth.tar",weights_only=False)
    assert reg["training_stage"]=="registered" and state["training_stage"]=="mixed"
    assert state["scheduler"]["last_epoch"]==1
    assert state["optimizer"]["param_groups"][0]["lr"]==pytest.approx(1e-4)
    direct=tmp_path/"direct";resumed=tmp_path/"resumed"
    full=common.copy();full[full.index("--epochs")+1]="2"
    trainer.main("mixed",full+["--init_checkpoint",str(registered/"best.pth.tar"),
        "--checkpoint_dir",str(direct),"--log_dir",str(tmp_path/"direct_log")])
    trainer.main("mixed",full+["--resume",str(mixed/"last.pth.tar"),
        "--checkpoint_dir",str(resumed),"--log_dir",str(tmp_path/"resume_log")])
    a=torch.load(direct/"last.pth.tar",weights_only=False)
    b=torch.load(resumed/"last.pth.tar",weights_only=False)
    for key,value in a["model"].items():
        torch.testing.assert_close(value,b["model"][key],rtol=0,atol=0)
    assert a["scheduler"]==b["scheduler"]
    model=PMIRFCoNet(7,3,mid_channels=8,num_fc=1)
    args=trainer.parser("mixed",common)
    with pytest.raises(ValueError,match="sensor SRF"):
        trainer.load_checkpoint(mixed/"best.pth.tar",model,args,expected_stage="mixed",fingerprint="wrong")
    with pytest.raises(ValueError,match="Expected registered"):
        trainer.load_checkpoint(mixed/"best.pth.tar",model,args,expected_stage="registered",fingerprint=state["sensor_fingerprint"])


@pytest.mark.parametrize("suite",["registered","fixed","multiseed"])
def test_actual_shared_evaluation_builds_checkpoint_and_scores(trained_stages,tmp_path,suite):
    _,_,registered,mixed=trained_stages
    output=tmp_path/("results_"+suite)
    checkpoint=str((registered if suite=="registered" else mixed)/"best.pth.tar")
    if suite=="registered":
        result=evaluator.main(["--dataset","PaviaU","--device","cpu","--test_mode","registered_only",
            "--checkpoint",checkpoint,"--output_json",str(output/"summary.json")])
        assert result["registered"]["PSNR"]>0 and result["warp"] is None
    else:
        entry=importlib.import_module("test_fixed_deformation" if suite=="fixed" else "test_multiseed_deformation")
        cmd=["--model","PMI-RFCoNet","--dataset","PaviaU","--device","cpu",
             "--checkpoint",checkpoint,"--output_dir",str(output),"--cases","1"]
        cmd+= ["--levels","1"] if suite=="fixed" else ["--geometry_seeds","10001","--max_samples","1"]
        entry.main(cmd)
        summary=json.loads((output/"summary.json").read_text())
        assert "PMI-RFCoNet" in json.dumps(summary)
        assert (output/"summary.csv").exists()

