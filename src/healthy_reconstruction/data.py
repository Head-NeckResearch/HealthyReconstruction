"""Private runtime input validation and strict neutral-anchored preparation.

Only the documented numerical NPZ arrays and minimal opaque schema are read.
Neither this module nor the release contains any participant records.
"""
from __future__ import annotations
import hashlib
import json
import re
import warnings
from pathlib import Path
import numpy as np
from .geometry import world_to_dental, fixed_rot_xyz, robust_registration, active_window_from_energy
from .preprocessing import anchored_representation, displacement_energy_from_neutral
from .settings import validate_direction_rule

ROLES = ('optimizer','validation','healthy_eval','pathological_eval')
DEVELOPMENT = ('optimizer','validation')
OPAQUE = re.compile(r'[0-9a-f]{32}')


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_direction_rule(config)
    if config['preprocessing']['n_markers'] != 105 or config['preprocessing']['n_frames'] != 100:
        raise ValueError('The study pipeline expects 105 facial markers and 100 phase frames')
    if any(float(x) != 0. for x in config['reconstruction'].values()):
        raise ValueError('The article implementation uses unblended AE outputs only')
    for branch in ('static','dynamic'):
        settings = config['training'][branch]
        if int(settings['epochs']) < 1 or int(settings['batch_size']) < 1 or int(settings['patience']) < 1:
            raise ValueError('Training epochs, batch size, and patience must be positive')
    if config['training']['dynamic']['lambda_velocity'] or config['training']['dynamic']['lambda_acceleration']:
        raise ValueError('The study has no temporal-loss regularization')
    return config


def grouped_participants(participants, seed=13, fraction=.2, required_optimizer=()):
    """Generic participant split; original assignments must be supplied privately."""
    participants=sorted(set(participants))
    required=set(required_optimizer)
    if len(participants)<3 or not required.issubset(participants):
        raise ValueError('Need at least three development participants and valid required IDs')
    eligible=[p for p in participants if p not in required]
    n=max(1,int(round(len(participants)*fraction)))
    if n>=len(participants) or n>len(eligible):
        raise ValueError('Participant split would empty optimizer set')
    validation=sorted(np.random.default_rng(seed).permutation(eligible)[:n].tolist())
    return sorted(set(participants)-set(validation)),validation


def _integer(value, name, minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def read_manifest(path):
    path=Path(path).resolve()
    manifest=json.loads(path.read_text(encoding='utf-8'))
    allowed={'schema_version','coordinate_system','records','template_sample_id'}
    if set(manifest)-allowed or manifest.get('schema_version')!=1:
        raise ValueError('Unrecognized manifest schema or fields')
    if manifest.get('coordinate_system') not in ('world','dental'):
        raise ValueError('coordinate_system must be world or dental')
    records=manifest.get('records',[])
    if not records:
        raise ValueError('Manifest has no records')
    allowed_record={'sample_id','participant_id','role','task','visit_index','file','neutral_raw_index','crop_start','crop_stop','alignment_seed'}
    seen=set(); participant_roles={}; records=[dict(r) for r in records]
    for r in records:
        if set(r)-allowed_record or not allowed_record.difference({'crop_start','crop_stop','alignment_seed'}).issubset(r):
            raise ValueError('Unrecognized or missing record fields; clinical metadata is not accepted')
        for field in ('sample_id','participant_id'):
            if not isinstance(r[field],str) or not OPAQUE.fullmatch(r[field]):
                raise ValueError('IDs must be opaque 32-character lowercase hexadecimal tokens')
        if r['sample_id'] in seen:
            raise ValueError('Duplicate sample ID')
        seen.add(r['sample_id'])
        if r['role'] not in ROLES or r['task'] not in [f'M{i}' for i in range(1,6)]:
            raise ValueError('Invalid role or task')
        previous=participant_roles.setdefault(r['participant_id'],r['role'])
        if previous!=r['role']:
            raise ValueError('Participant overlap across development or evaluation roles')
        _integer(r['visit_index'],'visit_index',1)
        _integer(r['neutral_raw_index'],'neutral_raw_index')
        if 'alignment_seed' in r:
            _integer(r['alignment_seed'],'alignment_seed')
            if r['alignment_seed'] >= 2**32:
                raise ValueError('alignment_seed must be less than 2**32')
        relative=Path(r['file'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.npz':
            raise ValueError('Input files must be relative NPZ paths within the private manifest folder')
        resolved=(path.parent/relative).resolve()
        if not resolved.is_relative_to(path.parent):
            raise ValueError('Input path escapes private manifest folder')
        r['_input_file']=resolved
        if ('crop_start' in r)!=('crop_stop' in r):
            raise ValueError('Specify both crop bounds or neither')
        if 'crop_start' in r:
            _integer(r['crop_start'],'crop_start'); _integer(r['crop_stop'],'crop_stop',1)
    if not all(role in participant_roles.values() for role in DEVELOPMENT):
        raise ValueError('Both optimizer and validation participants are required')
    if manifest.get('template_sample_id') is not None and manifest['template_sample_id'] not in seen:
        raise ValueError('Unknown template_sample_id')
    return manifest,sorted(records,key=lambda r:r['sample_id'])


def trimmed_mean(values,trim_fraction=.1):
    """Coordinatewise finite quantile trimming used by the study reference."""
    values=np.asarray(values,float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if len(values)<3 or trim_fraction==0:
            return np.nanmean(values,axis=0)
        low,high=np.nanquantile(values,[trim_fraction,1-trim_fraction],axis=0)
        return np.nanmean(np.where((values>=low)&(values<=high),values,np.nan),axis=0)


def prepare(manifest_path,output_root,config):
    """Prepare user-provided private sequences without copying input metadata."""
    output_root=Path(output_root)
    if output_root.exists():
        raise FileExistsError('Preparation output already exists; choose a fresh private directory')
    manifest,records=read_manifest(manifest_path)
    p=config['preprocessing']; loaded=[]; contents={}; excluded=0
    rotation=fixed_rot_xyz(tuple(p['fixed_rot_xyz']))
    for r in records:
        with np.load(r['_input_file'],allow_pickle=False) as z:
            if set(z.files)!={'points'}:
                raise ValueError('Raw NPZ must contain only the numerical points array')
            raw=np.asarray(z['points'],dtype=np.float64)
        count=108 if manifest['coordinate_system']=='world' else 105
        if raw.ndim!=3 or raw.shape[1:]!=(count,3) or len(raw)<2 or np.isinf(raw).any():
            raise ValueError('Invalid raw point shape or infinity')
        # Hash only numeric input, with a stable NaN representation.
        canonical=raw.copy(); canonical[np.isnan(canonical)]=np.nan
        digest=hashlib.sha256(canonical.astype('<f8').tobytes()).hexdigest()
        if digest in contents:
            previous=contents[digest]
            if any(previous[k]!=r[k] for k in ('participant_id','role','task')):
                raise ValueError('Identical content disagrees on participant, role, or task')
            excluded+=1; continue
        contents[digest]=r
        if manifest['coordinate_system']=='world':
            dental,ok=world_to_dental(raw)
            face=dental[:,3:]
        else:
            face=raw.copy()
        face=face@rotation.T
        ni=r['neutral_raw_index']
        if ni>=len(face) or np.isfinite(face[ni]).all(-1).sum()<p['minimum_neutral_correspondences']:
            raise ValueError('Selected neutral is out of bounds or lacks required support')
        loaded.append((r,face,digest))
    development=[item for item in loaded if item[0]['role'] in DEVELOPMENT]
    if not development:
        raise ValueError('No healthy development records')
    requested=manifest.get('template_sample_id')
    candidates=[item for item in development if requested is None or item[0]['sample_id']==requested]
    if not candidates:
        raise ValueError('Registration template must be an included healthy development sample')
    template_item=min(candidates,key=lambda item:(-np.isfinite(item[1][item[0]['neutral_raw_index']]).all(-1).sum(),item[0]['sample_id']))
    template=template_item[1][template_item[0]['neutral_raw_index']].copy()
    prepared=[]
    for r,face,digest in loaded:
        seed=r.get('alignment_seed',(int(digest[:8],16)+p['registration_seed'])%(2**32))
        trials=p['ransac_trials_training' if r['role'] in DEVELOPMENT else 'ransac_trials_evaluation']
        rot,translation=robust_registration(face[r['neutral_raw_index']],template,trials=trials,seed=seed,min_points=p['minimum_neutral_correspondences'])
        aligned=face@rot.T+translation
        if 'crop_start' in r:
            start,stop=r['crop_start'],r['crop_stop']
        else:
            energy=displacement_energy_from_neutral(aligned,aligned[r['neutral_raw_index']])
            support=(np.isfinite(aligned).all(-1)&np.isfinite(aligned[r['neutral_raw_index']]).all(-1)).sum(-1)
            energy[support<p['minimum_energy_marker_support']]=np.nan
            start,stop=active_window_from_energy(energy,p['energy_percentile'],p['minimum_window_length'],p['max_inactive_gap'])
        rep=anchored_representation(aligned,r['neutral_raw_index'],start,stop,p['n_frames'],p['max_interpolation_gap_raw_rows'])
        # Gap-imputed values stay separate and never enter core models or scoring.
        rep['X']=np.where(rep['B_target'][...,None],rep['X'],0.).astype(np.float32)
        prepared.append((r,rep))
    tasks=sorted({r['task'] for r,_ in prepared})
    for task in tasks:
        if any(not any(r['role']==role and r['task']==task for r,_ in prepared) for role in DEVELOPMENT):
            raise ValueError('Every task needs optimizer and validation support')
    output_root.mkdir(parents=True)
    (output_root/'samples').mkdir(); (output_root/'references').mkdir()
    output_records=[]
    for r,rep in prepared:
        name=f"samples/{r['sample_id']}.npz"
        np.savez_compressed(output_root/name,N=rep['N'],X=rep['X'],M=rep['B_target'],neutral_mask=rep['B_neutral'],P=rep['P'],B_imputed=rep['B_imputed'],phase_positions=rep['raw_positions_resampled'])
        output_records.append({k:r[k] for k in ('sample_id','participant_id','role','task','visit_index')})
        output_records[-1]['file']=name
    for task in tasks:
        reps=[rep for r,rep in prepared if r['role'] in DEVELOPMENT and r['task']==task]
        neutral=np.stack([np.where(rep['B_neutral'][...,None],rep['N'],np.nan) for rep in reps])
        motion=np.stack([np.where(rep['B_target'][...,None],rep['X'],np.nan) for rep in reps])
        nr,xr=trimmed_mean(neutral),trimmed_mean(motion)
        np.savez_compressed(output_root/'references'/f'{task}.npz',N_ref=nr,X_ref=xr,neutral_support=np.isfinite(neutral).all(-1).sum(0),motion_support=np.isfinite(motion).all(-1).sum(0))
    write_json(output_root/'manifest.json',{'schema_version':1,'tasks':tasks,'records':output_records,'excluded_exact_duplicates':excluded,'template_sample_id':template_item[0]['sample_id']})
    write_json(output_root/'config.json',config)
    return output_root


def load_prepared(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    records=manifest['records']
    roles={}
    for r in records:
        old=roles.setdefault(r['participant_id'],r['role'])
        if old!=r['role']:
            raise ValueError('Prepared manifest has participant role overlap')
    return manifest


def load_sample(root,record):
    root=Path(root).resolve(); path=(root/record['file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Prepared sample path escapes dataset')
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k] for k in ('N','X','M','neutral_mask','P')}
    if a['N'].shape!=(105,3) or a['X'].shape!=(100,105,3) or a['M'].shape!=(100,105):
        raise ValueError('Prepared sample shape mismatch')
    for name in ('M','neutral_mask'):
        if not np.isin(a[name],[0,1]).all():
            raise ValueError('Masks must be binary')
    if not np.isfinite(a['X'][a['M'].astype(bool)]).all() or (a['M'].astype(bool)&~np.isfinite(a['N']).all(-1)[None]).any():
        raise ValueError('Invalid target/neutral support')
    return a
