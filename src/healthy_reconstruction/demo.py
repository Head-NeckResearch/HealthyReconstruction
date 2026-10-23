"""Generate mathematical point lattices and exercise the complete public pipeline.

No real reference, recording, metadata, or learned parameter is used. Synthetic
arrays and resulting weights are created only when this module is executed.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
from .data import prepare, write_json
from .settings import default_config


def synthetic_inputs(output_root,task_count=5):
    """Create explicitly artificial curved-grid movements in a fresh directory."""
    output_root=Path(output_root)
    if output_root.exists():
        raise FileExistsError('Synthetic input directory already exists')
    output_root.mkdir(parents=True)
    rng=np.random.default_rng(20261023)
    column,row=np.meshgrid(np.linspace(-1.,1.,15),np.linspace(-1.,1.,7))
    grid=np.column_stack((column.ravel()*30,row.ravel()*35,5*np.cos(column.ravel()*2)*np.cos(row.ravel())))
    phases=np.linspace(0,1,120)
    roles=['optimizer']*3+['validation']+['healthy_eval']*2+['pathological_eval']*2
    records=[]
    for person,role in enumerate(roles):
        participant=hashlib.sha256(f'artificial-person-{person}'.encode()).hexdigest()[:32]
        neutral=grid+rng.normal(0,.25,grid.shape)
        for task_index in range(1,task_count+1):
            token=hashlib.sha256(f'artificial-take-{person}-{task_index}'.encode()).hexdigest()[:32]
            direction=np.array([.15*task_index,.3,1.])
            amplitude=2.+task_index*.6
            movement=(np.sin(np.pi*phases)**2)[:,None,None]*direction[None,None,:]*amplitude
            points=neutral[None]+np.broadcast_to(movement,(len(phases),105,3))
            points=points+rng.normal(0,.01,points.shape)
            if role=='pathological_eval':
                points[:,column.ravel()<0]=neutral[None,column.ravel()<0]+.4*(points[:,column.ravel()<0]-neutral[None,column.ravel()<0])
            # An artificial short gap and one missing resting marker exercise masks.
            points[25:27,8]=np.nan
            if person%2:
                points[:,17]=np.nan
            np.savez_compressed(output_root/f'{token}.npz',points=points)
            records.append({'sample_id':token,'participant_id':participant,'role':role,'task':f'M{task_index}',
                            'visit_index':1,'file':f'{token}.npz','neutral_raw_index':0,'crop_start':0,'crop_stop':120})
    manifest=output_root/'manifest.json'
    write_json(manifest,{'schema_version':1,'coordinate_system':'dental','records':records})
    return manifest


def run_demo(output_root,epochs=2,task_count=5):
    """Smoke-test data generation, preparation, both networks, and all baselines."""
    import torch
    from .train import train
    from .inference import predict
    if epochs<1 or not 1<=task_count<=5:
        raise ValueError('epochs>=1 and 1<=task_count<=5 required')
    output_root=Path(output_root)
    if output_root.exists():
        raise FileExistsError('Demo output already exists; choose a fresh directory')
    output_root.mkdir(parents=True)
    torch.set_num_threads(1)
    config=default_config()
    config['preprocessing']['ransac_trials_training']=12
    config['preprocessing']['ransac_trials_evaluation']=12
    for branch in ('static','dynamic'):
        config['training'][branch].update(epochs=epochs,patience=epochs,hidden=16,latent=4,device='cpu')
    manifest=synthetic_inputs(output_root/'private_inputs',task_count)
    dataset=prepare(manifest,output_root/'prepared',config)
    training=train(dataset,output_root/'models',config)
    scores=predict(dataset,output_root/'models',output_root/'results',config)
    summary={'synthetic_only':True,'tasks':task_count,'external_recordings_scored':len(scores),
             'methods_per_recording':len(scores[0]['scores']),'all_scores_finite':bool(all(np.isfinite(list(r['scores'].values())).all() for r in scores)),
             'models_trained':len(training),'scientific_validation':False}
    write_json(output_root/'smoke_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--tasks',type=int,default=5)
    args=parser.parse_args()
    import json
    print(json.dumps(run_demo(args.output,args.epochs,args.tasks),indent=2))


if __name__=='__main__':
    main()
