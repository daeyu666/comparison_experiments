"""CLSNet observation IO, two-stage provenance and exact continuation checks."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from comparison.CLSNet.model import build_clsnet
from comparison.CLSNet.common import (
    checkpoint_payload, deformed_observation, load_checkpoint, make_generator,
    restore_rng, set_seed, verify_checkpoint,
)
from comparison.CLSNet.train import parser
from experiment_protocol import resolve_model, build_train_command, DATASET_VALIDATION_INTERVAL


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('hsi,msi',[(31,3),(103,4),(128,8),(144,8),(145,8),(242,4)])
def test_dataset_channel_shapes(hsi,msi):
    model=build_clsnet(hsi,msi).eval()
    with torch.no_grad():
        pred=model(torch.rand(1,hsi,4,6),torch.rand(1,msi,16,24))
    assert pred.shape==(1,hsi,16,24)
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize('scale,h,w',[(4,20,28),(3,24,24),(8,16,24)])
def test_scale_padding_backward(scale,h,w):
    model=build_clsnet(31,3,scale)
    pred=model(torch.rand(1,31,h//scale,w//scale),torch.rand(1,3,h,w))
    assert pred.shape==(1,31,h,w)
    pred.square().mean().backward()
    assert torch.isfinite(model.conv_in_x.weight.grad).all()
    with pytest.raises(ValueError,match='ratio'):
        model(torch.rand(1,31,2,2),torch.rand(1,3,h,w))


def test_checkpoint_resume_and_rejection(tmp_path):
    args=parser().parse_args(['--device','cpu'])
    args.validation_interval=20
    model=build_clsnet(31,3)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    generator=make_generator(torch.device('cpu'),404)
    set_seed(1001)
    # Exercise optimizer state after a real update.
    model(torch.rand(1,31,4,4),torch.rand(1,3,16,16)).mean().backward()
    optimizer.step()
    state=checkpoint_payload(model,optimizer,3,20.,1,args,generator)
    path=tmp_path/'last.pth.tar'
    torch.save(state,path)
    expected=(torch.rand(3),torch.rand(3,generator=generator))
    restored=build_clsnet(31,3)
    new_optimizer=torch.optim.AdamW(restored.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loaded=load_checkpoint(path,restored,args,expected_stage='registered',optimizer=new_optimizer)
    restore_rng(loaded['rng'],generator)
    torch.testing.assert_close(torch.rand(3),expected[0])
    torch.testing.assert_close(torch.rand(3,generator=generator),expected[1])
    assert loaded['epoch']==3 and loaded['bad_validations']==1
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b)
    assert len(new_optimizer.state)==len(optimizer.state)>0
    with pytest.raises(ValueError,match='stage mismatch'):
        verify_checkpoint(state,model,args,expected_stage='mixed')
    wrong=copy.copy(args); wrong.dataset='CAVE'
    with pytest.raises(ValueError,match='dataset'):
        verify_checkpoint(state,model,wrong,expected_stage='registered')


def test_mixed_identity_and_warp_order(monkeypatch):
    import comparison.CLSNet.common as common
    gt=torch.rand(2,3,16,16)
    args=SimpleNamespace(registered_probability=1.,max_translation=4.,max_rotation_deg=2.,
                         max_local_px=4.,control_grid=5,min_jacobian=.5)
    class P0:
        def degrade(self,x):
            return F.avg_pool2d(x,4)
    generator=make_generator(torch.device('cpu'),10)
    lr,n=deformed_observation(gt,P0(),args,generator,mixed=True)
    torch.testing.assert_close(lr,P0().degrade(gt)); assert n==2
    args.registered_probability=0.
    monkeypatch.setattr(common,'sample_training_geometry_batch',lambda *a,**k: 'geometry')
    seen=[]
    def warp_then_degrade(image,geometry,p0):
        seen.append((image,geometry,p0))
        return p0.degrade(image+1.)
    monkeypatch.setattr(common,'make_deformed_lr_hsi',warp_then_degrade)
    lr,n=deformed_observation(gt,P0(),args,generator,mixed=True)
    torch.testing.assert_close(lr,P0().degrade(gt+1.)); assert n==0
    assert seen[0][0] is gt and seen[0][1]=='geometry'


def test_unified_registry():
    adapter=resolve_model('clsnet')
    for dataset,interval in DATASET_VALIDATION_INTERVAL.items():
        cmd=build_train_command(adapter,dataset,'registered','cpu')
        parsed=parser().parse_args(cmd[2:])
        assert parsed.dataset==dataset and parsed.validation_interval==interval
        assert parsed.epochs==800 and parsed.lr==1e-5 and parsed.batch_size==1
    from test_fixed_deformation import MODEL_BUILDERS
    assert 'CLSNet' in MODEL_BUILDERS


def test_registered_mixed_training_and_final_test(tmp_path,monkeypatch):
    """Real CLSNet updates and checkpoint transitions on a tiny synthetic fixture."""
    import comparison.CLSNet.train as train
    import comparison.CLSNet.test_hsi_deformed as evaluate
    from comparison.CLSNet.model import ARCHITECTURE_ID
    class P0:
        def to(self,device): return self
        def degrade(self,x): return F.avg_pool2d(x,4)
    class Dataset:
        degradation_operator=P0()
        def __len__(self): return 1
        def __getitem__(self,index):
            gen=torch.Generator().manual_seed(100)
            gt=torch.rand(31,16,16,generator=gen)*.7+.1
            return {'gt':gt,'hr_msi':gt[:3]}
    fixture=(Dataset(),Dataset(),Dataset(),{
        'n_bands':31,'n_select_bands':3,'srf_weights':torch.rand(3,31),
    })
    monkeypatch.setattr(train,'build_datasets',lambda *a,**k:fixture)
    for stage in ('registered','mixed'):
        dest=tmp_path/stage
        argv=['train.py','--device','cpu','--dataset','CAVE','--epochs','1',
              '--validation_interval','1','--validation_cases','1',
              '--checkpoint_dir',str(dest),'--log_dir',str(dest/'logs')]
        if stage=='mixed': argv+=['--init_checkpoint',str(tmp_path/'registered'/'best.pth.tar')]
        monkeypatch.setattr(sys,'argv',argv)
        train.main(stage)
        state=torch.load(dest/'last.pth.tar',weights_only=False)
        assert state['architecture_id']==ARCHITECTURE_ID and state['training_stage']==stage
        assert (dest/'best.pth.tar').is_file()
    # Final test enforces formal 128/64/32; use matching images and same checkpoint args.
    class FinalSet(Dataset):
        def __len__(self): return 2
        def __getitem__(self,index):
            item=super().__getitem__(index)
            return {k:F.interpolate(v[None],size=(128,128),mode='nearest')[0] for k,v in item.items()}
    fixture=(Dataset(),Dataset(),FinalSet(),fixture[3])
    monkeypatch.setattr(evaluate,'build_datasets',lambda *a,**k:fixture)
    output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['test.py','--dataset','CAVE','--device','cpu','--cases','2',
        '--checkpoint',str(tmp_path/'mixed'/'best.pth.tar'),'--output_json',str(output)])
    evaluate.main()
    report=json.loads(output.read_text())
    assert report['test_conditions']['warp_observations']==4
    assert set(report['average_metrics'])=={'Registered','Warp'}
    rows=report['per_sample_case']
    for key in ('dx_hr_px','dy_hr_px','rotation_deg','local_max_hr_px'):
        assert [row[key] for row in rows[:2]]==[row[key] for row in rows[2:]]
