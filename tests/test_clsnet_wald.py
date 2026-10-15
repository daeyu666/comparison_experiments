"""Exercise Wald train/reference/native/QNR with synthetic observed-cache fixtures."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def test_wald_training_reference_and_native_qnr(tmp_path,monkeypatch):
    from comparison.CLSNet import train_augsburg2_wald as train
    from comparison.CLSNet import infer_augsburg2_wald as infer
    root=tmp_path/'wald'; root.mkdir()
    protocol='Augsburg2-Wald-center-holdout-v1'
    bbox=[24,36,72,84]
    rng=np.random.default_rng(10)
    def save_json(path,obj): path.write_text(json.dumps(obj))
    save_json(root/'wald_psf.json',dict(scale_ratio=3,terminal_sigma_hr_pixels=1.2))
    save_json(root/'roi.json',dict(protocol_id=protocol,test_bbox_30m=bbox,test_bbox_10m=[3*x for x in bbox]))
    np.save(root/'srf_weights.npy',np.full((4,242),1/242,dtype=np.float32))
    for split,h,w in [('train',24,24),('validation',48,48),('test',48,48),('full',288,288)]:
        dest=root/split; dest.mkdir()
        if split=='full':
            save_json(dest/'meta.json',dict(region='sub_area_2'))
        else:
            np.save(dest/'gt.npy',rng.uniform(.1,.6,(h,w,242)).astype(np.float32))
            save_json(dest/'meta.json',dict(
                msi_source='real_Sentinel_2_Wald_30m',target='30m_EnMAP_like',
                gt_source='observed_30m_HSI_only',scale_ratio=3,protocol_id=protocol,
                test_bbox_30m=bbox,training_area='synthetic fixture',
                forbidden_bbox_30m=[18,30,78,90]))
        np.save(dest/'lr_hsi.npy',rng.uniform(.1,.6,(h//3,w//3,242)).astype(np.float32))
        np.save(dest/'hr_msi.npy',rng.uniform(.1,.6,(h,w,4)).astype(np.float32))
        np.save(dest/'valid_mask.npy',np.ones((h,w),dtype=np.uint8))
    calibration=tmp_path/'radiometry.json'
    save_json(calibration,dict(dataset='Augsburg-2-Wald',uses_EnMAP10_reference=False,
                              spatial_protocol_id=protocol,gain=[1]*4,bias=[0]*4))
    dest=tmp_path/'checkpoints'
    shared=['--device','cpu','--wald_root',str(root),'--radiometry_json',str(calibration),
            '--checkpoint_dir',str(dest),'--log_dir',str(tmp_path/'logs')]
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        monkeypatch.setattr(sys,'argv',['wald.py',*shared,'--epochs','1','--eval_interval','1','--save_interval','1'])
        train.main()
        best=dest/'best.pth.tar'
        checkpoint=torch.load(best,weights_only=False)
        assert checkpoint['protocol']=='Augsburg-2-Wald-CLSNet'
        assert checkpoint['backbone_width']==270 and checkpoint['attention_heads']==9
        assert checkpoint['attention_head_dim']==30 and checkpoint['architecture_profile']=='Augsburg'
        assert checkpoint['split_protocol_id']==protocol
        monkeypatch.setattr(sys,'argv',['wald.py',*shared,'--stage','test'])
        train.main()
        out=tmp_path/'native'
        monkeypatch.setattr(sys,'argv',['infer.py','--center_holdout','--device','cpu',
            '--wald_root',str(root),'--radiometry_json',str(calibration),
            '--checkpoint',str(best),'--save_root',str(out)])
        infer.main()
    finally:
        torch.set_num_threads(threads)
    cube=np.load(out/'Augsburg2_Wald_CLSNet_heldout_HSI.npy')
    assert cube.shape==(144,144,242) and np.isfinite(cube).all()
    quality=json.loads((out/'CLSNet_Wald_heldout_QNR.json').read_text())
    assert all(np.isfinite(quality[key]) for key in ('QNR','Dlambda','Ds'))
    report=json.loads((out/'CLSNet_Wald_heldout_protocol.json').read_text())
    assert report['test_bbox_30m']==bbox and report['full_reference_HSI_10m'] is False
    assert report['backbone_width']==270 and report['attention_heads']==9
