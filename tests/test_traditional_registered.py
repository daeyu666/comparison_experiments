"""Validate registered-only routing, author CNMF updates and the HySure objective."""
import hashlib
import importlib
import io
import json
import os
import urllib.request
import zipfile
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits
from comparison.traditional.operators import PhysicalOperator
from comparison.CNMF.model import fuse as cnmf,CNMFConfig
from comparison.HySure.model import fuse as hysure,HySureConfig,solve_coefficients,objective

torch.set_num_threads(2)


def observations(c=12,m=3,size=32):
    rng=np.random.default_rng(19)
    e=rng.uniform(.1,.8,(c,4));a=rng.uniform(.1,1,(4,size,size));a/=a.sum(0)
    gt=(e@a.reshape(4,-1)).reshape(c,size,size)
    r=rng.uniform(.1,1,(m,c));r/=r.sum(1,keepdims=True)
    op=PhysicalOperator()
    return gt,op.forward(gt),(r@gt.reshape(c,-1)).reshape(m,size,size),r,op


def test_operator_matches_shared_torch_pixels_and_has_exact_adjoint():
    from degradations import build_degradation
    rng=np.random.default_rng(71)
    x=rng.random((5,32,48));p=PhysicalOperator()
    ref=build_degradation("physical",scale_ratio=4,mtf_nyquist=.2,truncate=3.)
    actual=p.forward(x)
    expected=ref.degrade(torch.from_numpy(x).float()[None])[0].numpy()
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=1e-7)
    y=rng.normal(size=actual.shape)
    assert np.sum(actual*y)==pytest.approx(np.sum(x*p.adjoint(y)),abs=1e-11)
    assert p.forward(np.ones((1,32,32)))[0,0,0]<.9


@pytest.mark.parametrize("method",["CNMF","HySure"])
def test_root_registered_requires_no_checkpoint_and_rejects_mixed_and_pretraining(method):
    import experiment_protocol as protocol
    adapter=protocol.resolve_model(method)
    command=protocol.build_test_command(adapter,"PaviaU","registered","cpu")
    assert "--checkpoint" not in command and "--geometry_seeds" not in command
    with pytest.raises(ValueError,match="Registered-only"):
        protocol.build_test_command(adapter,"PaviaU","mixed","cpu")
    with pytest.raises(NotImplementedError,match="no dataset pretraining"):
        protocol.build_train_command(adapter,"PaviaU","registered","cpu")
    import test as root
    args=root.parse_args(["--model",method,"--dataset","PaviaU","--mode","registered","--max_samples","1"])
    assert args.max_samples==1


@pytest.mark.parametrize("method",["CNMF","HySure"])
def test_numerical_fusion_finite_deterministic_observation_only_and_no_input_mutation(method):
    _,lr,msi,r,p=observations()
    initial=(lr.copy(),msi.copy())
    fn,cfg=(cnmf,CNMFConfig(10,4)) if method=="CNMF" else (hysure,HySureConfig(iterations=20,rank=4))
    with threadpool_limits(1):
        np.random.seed(7);saved=np.random.get_state()
        a,_=fn(lr,msi,r,p,cfg)
        state=np.random.get_state()
        assert saved[0]==state[0] and np.array_equal(saved[1],state[1])
        b,_=fn(lr,msi,r,p,cfg)
    assert a.shape==(12,32,32) and np.isfinite(a).all()
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(lr,initial[0]);np.testing.assert_array_equal(msi,initial[1])


@pytest.mark.skipif(os.environ.get("CLASSICAL_TEST_OFFICIAL")!="1",reason="Optional pinned author archive")
def test_cnmf_matches_author_release_with_known_shared_sensors():
    from lib2to3.refactor import RefactoringTool,get_fixers_from_package
    raw=urllib.request.urlopen("https://naotoyokoya.com/assets/zip/CNMF_Python.zip",timeout=45).read()
    assert hashlib.sha256(raw).hexdigest()=="29acccea7c2109a9a7bd2ec9fd64f7aa9e0611fb073e043e7afe6e5f58eab55f"
    source=zipfile.ZipFile(io.BytesIO(raw)).read("CNMF_Python/CNMF.py").decode()
    source=str(RefactoringTool(get_fixers_from_package("lib2to3.fixes")).refactor_string(source+"\n","author_CNMF"))
    for old,new in (("w = rows1/rows2","w = rows1//rows2"),("hx = xdata/w","hx = xdata//w"),("hy = ydata/w","hy = ydata//w")):
        source=source.replace(old,new)
    reference={}
    exec(compile(source,"author_CNMF","exec"),reference)
    _,lr,msi,r,p=observations(c=12)
    reference["estR"]=lambda *args:np.column_stack((r,np.zeros(r.shape[0])))
    reference["gaussian_down_sample"]=lambda data,w,mask=0:p.forward(data.transpose(2,0,1)).transpose(1,2,0)
    # Same released rank rule and 200 inner / one outer iteration.
    with threadpool_limits(1):
        np.random.seed(1234)
        expected=reference["CNMF_fusion"](lr.transpose(1,2,0).copy(),msi.transpose(1,2,0).copy()).transpose(2,0,1)
        actual,_=cnmf(lr,msi,r,p,CNMFConfig())
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_hysure_agrees_with_independent_convex_solver():
    cp=pytest.importorskip("cvxpy")
    rng=np.random.default_rng(5);H=W=8;p=2;c=4;m=3;N=H*W
    operator=PhysicalOperator(scale=2,mtf=.4)
    e=np.linalg.qr(rng.normal(size=(c,p)))[0]
    r=rng.random((m,c));r/=r.sum(1,keepdims=True)
    target=rng.normal(size=(p,H,W))
    lr=operator.forward((e@target.reshape(p,-1)).reshape(c,H,W))
    msi=((r@e)@target.reshape(p,-1)).reshape(m,H,W)
    pm=operator.forward(np.eye(N).reshape(N,H,W)).reshape(N,-1).T
    dh=np.zeros((N,N));dv=np.zeros((N,N))
    for i in range(H):
        for j in range(W):
            k=i*W+j;dh[k,k]=1;dh[k,i*W+(j+1)%W]-=1
            dv[k,k]=1;dv[k,((i+1)%H)*W+j]-=1
    z=cp.Variable((p,N))
    cfg=HySureConfig(iterations=2500,rank=p,lambda_phi=.02,cg_rtol=1e-10)
    problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(e@z@pm.T-lr.reshape(c,-1))
        +.5*cfg.lambda_m*cp.sum_squares(r@e@z-msi.reshape(m,-1))
        +cfg.lambda_phi*cp.sum(cp.norm(cp.vstack((z@dh.T,z@dv.T)),axis=0))))
    with threadpool_limits(1):
        optimum=problem.solve(solver="CLARABEL",tol_gap_abs=1e-7,tol_feas=1e-7,tol_gap_rel=1e-7)
        x=solve_coefficients(lr,msi,r,e,operator,cfg)
    value=objective(x,lr,msi,r,e,operator,cfg)
    assert value==pytest.approx(optimum,rel=2e-4,abs=1e-6)


@pytest.mark.parametrize("method",["CNMF","HySure"])
def test_real_registered_evaluator_scores_one_pair_and_writes_no_warp(method,tmp_path,monkeypatch):
    from comparison.traditional import evaluate as evaluator
    from degradations import build_degradation
    gt,_,msi,r,_=observations(c=12,size=32)
    class Data:
        degradation_operator=build_degradation("physical",scale_ratio=4,mtf_nyquist=.2,truncate=3.)
        def __len__(self):return 1
        def __getitem__(self,_):return dict(gt=torch.tensor(gt,dtype=torch.float32),hr_msi=torch.tensor(msi,dtype=torch.float32))
    info=dict(n_bands=12,n_select_bands=3,srf_weights=torch.tensor(r,dtype=torch.float32))
    monkeypatch.setattr(evaluator,"build_datasets",lambda *a,**kw:(Data(),Data(),Data(),info))
    result=evaluator.main(method,["--dataset","PaviaU","--output_dir",str(tmp_path),
        "--cnmf_iterations","5","--cnmf_endmembers","4","--hysure_iterations","5",
        "--hysure_rank","4","--classical_save_predictions"])
    assert result["mode"]=="registered" and result["test_samples"]==1
    assert result["gt_used_for_optimization"] is False
    assert set(result["average_metrics"])=={"Registered"}
    assert result["average_metrics"]["Registered"]["PSNR"]>0
    assert (tmp_path/"summary.csv").is_file() and (tmp_path/"sample_0001.npy").is_file()
    assert "Warp" not in json.loads((tmp_path/"summary.json").read_text())["average_metrics"]

