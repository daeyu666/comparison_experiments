"""Registered and mixed CLSNet training under the shared comparison protocol."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from comparison.CLSNet.common import (
    HERE, DATASET_VALIDATION_INTERVAL, add_data_args, add_geometry_args,
    build_shared_cfg, checkpoint_payload, deformed_observation, geometry_kwargs,
    load_checkpoint, make_generator, model_for_info, resolve_device, restore_rng,
    sample_synthetic_geometry, make_deformed_lr_hsi, set_seed,
)
from data_loader import build_datasets
from metrics import MetricAverager, calc_metrics
from comparison.CLSNet.model import UPSTREAM_COMMIT, parameter_count, architecture_metadata


def parser(stage='registered'):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.set_defaults(stage=stage)
    p.add_argument('--train_misalignment_mode', choices=['registered'], default='registered')
    p.add_argument('--epochs', type=int, default=800 if stage=='registered' else 600)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=1001 if stage=='registered' else 10)
    p.add_argument('--validation_interval', type=int, default=None)
    p.add_argument('--validation_cases', type=int, default=5)
    p.add_argument('--early_stop_patience', type=int, default=999999)
    p.add_argument('--early_stop_min_delta', type=float, default=.02)
    p.add_argument('--checkpoint_dir', default='')
    p.add_argument('--log_dir', default='')
    p.add_argument('--resume', default='')
    p.add_argument('--init_checkpoint', default='')
    return p


@torch.no_grad()
def evaluate(model, loader, device, p0, args):
    model.eval()
    reg, warp = MetricAverager(), MetricAverager()
    for batch in loader:
        gt, msi = batch['gt'].to(device), batch['hr_msi'].to(device)
        pred = model(p0.degrade(gt), msi)
        reg.update(calc_metrics(pred, gt, args.scale_ratio))
        if args.stage == 'mixed':
            # Same validation cases and generator reset per patch as EMR-Diff.
            generator = make_generator(device, args.seed + 60000)
            for _ in range(args.validation_cases):
                geometry = sample_synthetic_geometry(
                    *gt.shape[-2:], device=device, dtype=gt.dtype, generator=generator,
                    local_strength_min_fraction=0., local_strength_max_fraction=1.,
                    **geometry_kwargs(args))
                pred = model(make_deformed_lr_hsi(gt, geometry, p0), msi)
                warp.update(calc_metrics(pred, gt, args.scale_ratio))
    return reg.average(), warp.average()


def main(stage='registered'):
    args = parser(stage).parse_args()
    args.validation_interval = args.validation_interval or DATASET_VALIDATION_INTERVAL[args.dataset]
    if min(args.epochs,args.batch_size,args.validation_interval,args.validation_cases) < 1:
        raise ValueError('Training schedule values must be positive')
    if not 0 <= args.registered_probability <= 1:
        raise ValueError('registered_probability must lie in [0,1]')
    if args.stage == 'mixed' and args.degradation_mode != 'physical':
        raise ValueError('Formal mixed protocol uses physical degradation')
    if args.stage == 'registered' and args.init_checkpoint:
        raise ValueError('Registered training starts from scratch; use resume to continue')
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_set, val_set, _unused_test, info = build_datasets(build_shared_cfg(args), include_validation=True)
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,
                              num_workers=args.num_workers,pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_set,batch_size=1,shuffle=False,num_workers=0)
    p0 = train_set.degradation_operator.to(device)
    model = model_for_info(info,args,device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    generator = make_generator(device,args.seed+404)
    start, best, bad = 0, -float('inf'), 0
    if args.resume:
        state = load_checkpoint(args.resume,model,args,expected_stage=args.stage,optimizer=optimizer)
        start,best,bad = state['epoch']+1,state['best_psnr'],state['bad_validations']
        restore_rng(state['rng'],generator)
    elif args.stage == 'mixed':
        args.init_checkpoint = args.init_checkpoint or str(HERE/'checkpoints'/'physical'/args.dataset/'best.pth.tar')
        load_checkpoint(args.init_checkpoint,model,args,expected_stage='registered')
        print(f'Loaded registered model weights; new AdamW state: {args.init_checkpoint}')
    family = args.degradation_mode if stage=='registered' else 'hsi_warp_final'
    ckpts = Path(args.checkpoint_dir or HERE/'checkpoints'/family/args.dataset)
    logs = Path(args.log_dir or HERE/'logs'/family/args.dataset)
    ckpts.mkdir(parents=True,exist_ok=True)
    logs.mkdir(parents=True,exist_ok=True)
    protocol = dict(method='CLSNet',**architecture_metadata(model),upstream_commit=UPSTREAM_COMMIT,
                    hsi_channels=model.hsi_channels,msi_channels=model.msi_channels,
                    parameters=parameter_count(model),optimizer='AdamW',loss='L1',scheduler=None,
                    observation='Y_H=P0(W_phi(X)); HR-MSI=R0(X), GT=X; identity uses P0(X)',
                    validation_rect=info.get('validation_rect'),srf_profile=info.get('srf_profile'),
                    arguments=vars(args))
    (ckpts/'run_protocol.txt').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(protocol,indent=2))
    history=logs/'validation_history.csv'
    for epoch in range(start,args.epochs):
        model.train()
        loss_sum,count,identities = 0.,0,0
        progress=tqdm(
            train_loader, desc=f'CLSNet {stage} {epoch+1}/{args.epochs}',
            dynamic_ncols=True, mininterval=1.0, disable=None,
        )
        for batch in progress:
            gt,msi=batch['gt'].to(device),batch['hr_msi'].to(device)
            with torch.no_grad():
                lr,identity_count=deformed_observation(gt,p0,args,generator,mixed=stage=='mixed')
            optimizer.zero_grad(set_to_none=True)
            pred=model(lr,msi)
            loss=F.l1_loss(pred,gt)  # official CLSNet reconstruction loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'CLSNet nonfinite loss at epoch {epoch+1}')
            loss.backward()
            optimizer.step()
            count+=gt.shape[0]
            identities+=identity_count
            loss_sum+=float(loss.item())*gt.shape[0]
            progress.set_postfix(l1=f'{loss_sum/count:.6f}',identity=f'{identities/count:.3f}',refresh=False)
        print(
            f'CLSNet {stage} {epoch+1}/{args.epochs} '
            f'loss={loss_sum/max(count,1):.6f} identity={identities/max(count,1):.3f}',
            flush=True,
        )
        do_eval=(epoch+1)%args.validation_interval==0 or epoch+1==args.epochs
        if do_eval:
            reg,warp=evaluate(model,val_loader,device,p0,args)
            score=(warp if stage=='mixed' else reg)['PSNR']
            improved=score > best+args.early_stop_min_delta
            if improved:
                best,bad=score,0
            else:
                bad+=1
            row=dict(epoch=epoch+1,train_l1=loss_sum/max(count,1),identity_fraction=identities/max(count,1),
                     **{'reg_'+k:v for k,v in reg.items()},**{'warp_'+k:v for k,v in warp.items()})
            with history.open('a',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(row))
                if f.tell()==0:
                    writer.writeheader()
                writer.writerow(row)
            print(f'VALIDATION {row} best_PSNR={best:.6f}')
            if improved:
                torch.save(checkpoint_payload(model,optimizer,epoch,best,bad,args,generator),ckpts/'best.pth.tar')
        # Save exact model/optimizer/RNG continuation state after every epoch.
        torch.save(checkpoint_payload(model,optimizer,epoch,best,bad,args,generator),ckpts/'last.pth.tar')
        if do_eval and args.early_stop_patience > 0 and bad>=args.early_stop_patience:
            print('Validation early stopping')
            break


if __name__=='__main__':
    main()
