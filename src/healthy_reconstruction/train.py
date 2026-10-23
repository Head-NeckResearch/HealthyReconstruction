"""Participant-exclusive, strict-mask healthy-only training.

The shared reference includes all healthy development records. The validation
participants are for early stopping, not an independently nested reference test.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from . import models
from .data import DEVELOPMENT, load_prepared, load_sample, write_json


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def train_branch(branch,task,records,dataset_root,output_root,settings):
    """Fit one branch; normalize on optimizer participants, validate uncorrupted."""
    selected=[r for r in records if r['role'] in DEVELOPMENT and (branch=='static' or r['task']==task)]
    arrays=[load_sample(dataset_root,r) for r in selected]
    values=np.stack([a['N' if branch=='static' else 'X'] for a in arrays])
    mask=np.stack([a['neutral_mask' if branch=='static' else 'M'] for a in arrays]).astype(bool)
    if not np.isfinite(values[mask]).all() or (mask.reshape(len(mask),-1).sum(1)==0).any():
        raise ValueError('Nonfinite or empty supported training target')
    values=np.where(mask[...,None],values,0.).astype(np.float32)
    train=np.array([i for i,r in enumerate(selected) if r['role']=='optimizer'])
    validation=np.array([i for i,r in enumerate(selected) if r['role']=='validation'])
    if not len(train) or not len(validation):
        raise ValueError('Missing optimizer or validation rows')
    models.set_seed(settings['seed'])
    mean,std=models.compute_norm(values[train],mask[train])
    z=models.normalize(values,mask,mean,std)
    device=torch.device(settings['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose CPU explicitly')
    dataset=TensorDataset(torch.from_numpy(z),torch.from_numpy(mask.astype(np.float32)))
    train_loader=DataLoader(torch.utils.data.Subset(dataset,train),batch_size=settings['batch_size'],shuffle=True)
    val_loader=DataLoader(torch.utils.data.Subset(dataset,validation),batch_size=settings['batch_size'],shuffle=False)
    params={k:settings[k] for k in ('hidden','latent','dropout')}
    params['n_markers']=105
    if branch=='static':
        model=models.StaticNeutralAE(**params).to(device)
        corrupt=models.corrupt_static
    else:
        if settings['lambda_velocity'] or settings['lambda_acceleration']:
            raise ValueError('The article uses no temporal loss regularization')
        params['n_frames']=100
        model=models.DynamicMotionProjector(**params).to(device)
        corrupt=models.corrupt_dynamic
    opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    args=argparse.Namespace(**settings)
    history=[]; best=float('inf'); best_state=None; bad=0; best_epoch=0
    for epoch in range(1,settings['epochs']+1):
        model.train(); totals=[0.,0.]
        for target,support in train_loader:
            target,support=target.to(device),support.to(device)
            noisy,input_mask=corrupt(target,support,args)
            numerator,denominator=models.masked_error_sums(model(noisy,input_mask),target,support)
            loss=numerator/denominator
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            totals[0]+=float(numerator.detach().cpu()); totals[1]+=float(denominator.cpu())
        model.eval(); val_totals=[0.,0.]
        with torch.no_grad():
            for target,support in val_loader:
                target,support=target.to(device),support.to(device)
                numerator,denominator=models.masked_error_sums(model(target,support),target,support)
                val_totals[0]+=float(numerator.cpu()); val_totals[1]+=float(denominator.cpu())
        row={'epoch':epoch,'train_loss':totals[0]/totals[1],'val_loss':val_totals[0]/val_totals[1],
             'train_coordinate_count':int(totals[1]),'validation_coordinate_count':int(val_totals[1])}
        if not np.isfinite(row['val_loss']):
            raise RuntimeError('Nonfinite validation loss')
        history.append(row)
        if row['val_loss']<best-1e-6:
            best,best_epoch,bad=row['val_loss'],epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
        if bad>=settings['patience']:
            break
    if best_state is None:
        raise RuntimeError('No finite best model')
    folder=Path(output_root)/branch
    if task is not None:
        folder/=task
    folder.mkdir(parents=True)
    torch.save({'model_state_dict':best_state,'parameters':params,'mean':mean.tolist(),'std':std.tolist(),
                'branch':branch,'best_epoch':best_epoch,'best_val_loss':best},folder/'model.pt')
    write_json(folder/'history.json',history)
    write_json(folder/'normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'fit_role':'optimizer'})
    return {'branch':branch,'task':task,'best_epoch':best_epoch,'best_validation_loss':best,'epochs_run':len(history),
            'optimizer_rows':len(train),'validation_rows':len(validation),'device':str(device),
            'parameter_count':sum(p.numel() for p in model.parameters())}


def train(dataset_root,output_root,config):
    """Train pooled static and task-specific dynamic networks into a fresh folder."""
    dataset_root,output_root=Path(dataset_root),Path(output_root)
    if output_root.exists():
        raise FileExistsError('Model output already exists; choose a fresh private directory')
    manifest=load_prepared(dataset_root)
    prepared_config=json.loads((dataset_root/'config.json').read_text())
    if config['preprocessing']!=prepared_config['preprocessing']:
        raise ValueError('Training preprocessing settings differ from the prepared dataset')
    records=manifest['records']
    dev=[r for r in records if r['role'] in DEVELOPMENT]
    if not dev:
        raise ValueError('No development data')
    output_root.mkdir(parents=True)
    inputs={r['sample_id']:file_hash(dataset_root/r['file']) for r in dev}
    references={task:file_hash(dataset_root/'references'/f'{task}.npz') for task in manifest['tasks']}
    protocol={'schema_version':1,'config':copy.deepcopy(config),'development_input_hashes':inputs,'reference_hashes':references,
              'normalization':'Optimizer participants only',
              'reference_caveat':'Reference uses all healthy development participants; validation is for early stopping only.',
              'torch_version':str(torch.__version__),'numpy_version':np.__version__,'deterministic_algorithms':True}
    write_json(output_root/'protocol.json',protocol)
    summaries=[train_branch('static',None,records,dataset_root,output_root,config['training']['static'])]
    for task in manifest['tasks']:
        summaries.append(train_branch('dynamic',task,records,dataset_root,output_root,config['training']['dynamic']))
    write_json(output_root/'training_summary.json',summaries)
    return summaries
