"""NonRegSRNet pairwise objective, source agreement and integration guards."""
import ast
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from comparison.NonRegSRNet.fit import fit_pair, observation_key
from comparison.NonRegSRNet.model import NonRegSRNet, FitConfig, ARCHITECTURE_ID
from comparison.NonRegSRNet import upstream_core as core


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def observations():
    g=torch.Generator().manual_seed(37)
    lr=torch.rand(1,7,4,4,generator=g)*0.7+0.1
    msi=torch.rand(1,3,16,16,generator=g)*0.7+0.1
    srf=torch.rand(3,7,generator=g);srf=srf/srf.sum(1,keepdim=True)
    return lr,msi,srf


def test_shapes_gradient_and_released_flow_resolution():
    lr,msi,srf=observations()
    model=NonRegSRNet(7,3,srf,num_rho=8)
    sizes=[]
    hook=model.net_Displacefiled.bottleneck.register_forward_hook(
        lambda module,inputs,output:sizes.append(tuple(output.shape[-2:])))
    loss,terms=model.observation_loss(lr,msi,FitConfig(num_rho=8))
    assert set(terms)=={"lr_pixelwise","lr_sumtoone","msi_pixelwise","msi_sumtoone",
                        "recovered_pixelwise","voxmorph","ncc","smooth"}
    loss.backward();hook.remove()
    assert sizes==[(4,4)]  # Release's commented-out pools must remain disabled.
    for module in (model.net_HSI2A,model.net_MSI2A,model.net_A2Img,model.net_Displacefiled,
                   model.net_PSF,model.net_HSI2MSI):
        gradients=[p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert "gt" not in inspect.signature(model.observation_loss).parameters


def test_pair_fit_works_inside_no_grad_and_restores_rng(tmp_path):
    lr,msi,srf=observations()
    config=FitConfig(iterations=3,num_rho=8)
    rng=torch.get_rng_state().clone()
    with torch.no_grad():
        prediction,report=fit_pair(lr,msi,srf,config=config,cache_dir=tmp_path)
    assert torch.equal(torch.get_rng_state(),rng)
    assert prediction.shape==(1,7,16,16) and torch.isfinite(prediction).all()
    assert report["completed_iterations"]==3 and report["gt_used_for_optimization"] is False
    again,second=fit_pair(lr,msi,srf,config=config,cache_dir=tmp_path)
    torch.testing.assert_close(prediction,again,rtol=0,atol=0)
    assert second["resumed_from_iteration"]==3
    assert len(list(tmp_path.glob("*.pth.tar")))==1


def test_lr_hsi_influences_independently_fitted_prediction():
    lr,msi,srf=observations()
    config=FitConfig(iterations=8,num_rho=8)
    a,_=fit_pair(lr,msi,srf,config=config)
    b,_=fit_pair(1-lr,msi,srf,config=config)
    assert float((a-b).abs().max())>1e-7
    assert observation_key(lr,msi,srf,4,config)!=observation_key(1-lr,msi,srf,4,config)


def test_srf_projection_has_finite_zero_row_fallback():
    _,_,srf=observations()
    model=NonRegSRNet(7,3,srf,num_rho=8)
    with torch.no_grad():
        for layer in model.net_HSI2MSI.conv2d_list:
            layer.weight.fill_(-1.)
    model.project_parameters()
    for layer in model.net_HSI2MSI.conv2d_list:
        assert torch.isfinite(layer.weight).all()
        torch.testing.assert_close(layer.weight.sum(),torch.tensor(1.))


def test_registered_and_multiseed_launchers_require_no_pretraining():
    import experiment_protocol as protocol
    adapter=protocol.resolve_model("NonRegSRNet")
    for dataset in protocol.DATASETS:
        command=protocol.build_test_command(adapter,dataset,"registered","cpu")
        assert "--checkpoint" not in command and "--nonreg_iterations" in command
        mixed=protocol.build_test_command(adapter,dataset,"mixed","cpu",max_samples=1,
                                         nonreg_options={"nonreg_iterations":2})
        assert "test_multiseed_deformation.py" in mixed[1]
        assert "--checkpoint" not in mixed
        assert mixed[mixed.index("--nonreg_iterations")+1]=="2"
    with pytest.raises(NotImplementedError,match="observation pair"):
        protocol.build_train_command(adapter,"PaviaU","registered","cpu")
    with pytest.raises(ValueError,match="checkpoint"):
        protocol.build_test_command(adapter,"PaviaU","registered","cpu",checkpoint_override="wrong.pth")


def test_shared_test_builder_registers_pairwise_callback():
    import test_fixed_deformation as fixed
    assert "NonRegSRNet" in fixed.MODEL_BUILDERS
    import test_multiseed_deformation as multi
    args=multi.parse_args(["--model","NonRegSRNet","--dataset","CAVE",
                          "--nonreg_iterations","2","--max_samples","1"])
    assert args.nonreg_iterations==2


@pytest.mark.skipif(os.environ.get("NONREG_TEST_OFFICIAL")!="1",
                    reason="Optional pinned upstream source agreement (network access)")
def test_exact_forward_and_objective_agreement_with_official_source():
    import urllib.request
    url=("https://raw.githubusercontent.com/saber-zero/NonRegSRNet/"
         "84e236db23695120ce0eb2a2f25b78d7fde70c21/"
         "DeformatedHyperImageFusion/model/network.py")
    raw=urllib.request.urlopen(url,timeout=60).read()
    blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    assert blob=="6f23cef4a6371b6d99d58087b2df4e58097a6d73"
    tree=ast.parse(raw.decode())
    names={"Msi2A","HSI2A","A2Img","PSF","ConvolutionHsi2Msi","UNet",
           "SpatialTransformation","SumToOneLoss","cross_correlation_loss","smooothing_loss"}
    selected=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    reference=dict(torch=torch,nn=nn,F=F,np=np,use_gpu=False)
    exec(compile(ast.Module(body=selected,type_ignores=[]),"official_nonreg","exec"),reference)
    lr,msi,srf=observations()
    model=NonRegSRNet(7,3,srf,num_rho=8).eval()
    ref={}
    for key,class_name,args in [
        ("net_HSI2A","HSI2A",(7,8)),("net_MSI2A","Msi2A",(3,8)),
        ("net_A2Img","A2Img",(8,7)),("net_PSF","PSF",(4,)),
        ("net_HSI2MSI","ConvolutionHsi2Msi",(7,3,np.array([[0,6]]*3))),
        ("net_Displacefiled","UNet",(16,2)),
    ]:
        ref[key]=reference[class_name](*args).eval()
        ref[key].load_state_dict(getattr(model,key).state_dict(),strict=True)
    with torch.no_grad():
        la=ref["net_HSI2A"](lr);ha=ref["net_MSI2A"](msi)
        fused=ref["net_A2Img"](ha)
        down=ref["net_PSF"](ha)
        flow=ref["net_Displacefiled"](torch.cat((la,down),1)).permute(0,2,3,1)
        recovered=reference["SpatialTransformation"](False)(la.permute(0,2,3,1),flow).permute(0,3,1,2)
        cfg=FitConfig(num_rho=8)
        expected=(F.l1_loss(ref["net_A2Img"](la),lr,reduction="sum")
            +cfg.beta*((la.sum(1)-1).abs().sum()+(ha.sum(1)-1).abs().sum())
            +cfg.alpha*F.l1_loss(ref["net_HSI2MSI"](fused),msi,reduction="sum")
            +cfg.delta*F.l1_loss(ref["net_A2Img"](recovered),ref["net_PSF"](fused),reduction="sum")
            +cfg.gamma*(1-reference["cross_correlation_loss"](recovered,down,cfg.ncc_window))
            +cfg.theta*reference["smooothing_loss"](flow.permute(0,3,1,2)))
        actual,terms=model.observation_loss(lr,msi,cfg)
        out=model(lr,msi,return_aux=True)
    torch.testing.assert_close(out["fused"],fused,rtol=0,atol=0)
    torch.testing.assert_close(out["recovered_a"],recovered,rtol=0,atol=0)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-5)


@pytest.mark.parametrize("suite",["fixed","multiseed"])
def test_end_to_end_shared_scoring_fits_observations_without_checkpoint(tmp_path,monkeypatch,suite):
    from comparison.NonRegSRNet import test_hsi_deformed as evaluator
    from degradations import build_degradation
    p0=build_degradation("physical",scale_ratio=4,mtf_nyquist=0.2,truncate=3.0)
    generator=torch.Generator().manual_seed(9)
    gt=torch.rand(7,32,32,generator=generator)*0.6+0.1
    srf=torch.rand(3,7,generator=generator);srf=srf/srf.sum(1,keepdim=True)

    class Data:
        degradation_operator=p0
        def __len__(self): return 1
        def __getitem__(self,index):
            return dict(gt=gt.clone(),hr_msi=torch.einsum("mc,chw->mhw",srf,gt))

    fixture=(Data(),Data(),Data(),dict(n_bands=7,n_select_bands=3,srf_weights=srf))
    monkeypatch.setattr(evaluator,"build_datasets",lambda *a,**k:fixture)
    common=["--model","NonRegSRNet","--dataset","CAVE","--device","cpu",
            "--output_dir",str(tmp_path),"--cases","1",
            "--nonreg_iterations","2","--nonreg_num_rho","8"]
    if suite=="fixed":
        import test_fixed_deformation as entry
        entry.main([*common,"--levels","1"])
    else:
        import test_multiseed_deformation as entry
        entry.main([*common,"--geometry_seeds","10001","--max_samples","1"])
    import json
    report=json.loads((tmp_path/"summary.json").read_text())
    assert "NonRegSRNet" in json.dumps(report)
    assert "observation_pair_self_supervision" in json.dumps(report)
    assert list(tmp_path.glob("*.json"))
