"""Command-line access to the anonymized public implementation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    demo=commands.add_parser('demo',help='Run only generated mathematical point lattices')
    demo.add_argument('--output',type=Path,required=True)
    demo.add_argument('--epochs',type=int,default=2)
    demo.add_argument('--tasks',type=int,default=5)
    prep=commands.add_parser('prepare',help='Prepare private numeric NPZ sequences')
    prep.add_argument('--manifest',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    train=commands.add_parser('train',help='Fit healthy-only static and task-specific networks')
    train.add_argument('--dataset',type=Path,required=True)
    train.add_argument('--output',type=Path,required=True)
    predict=commands.add_parser('predict',help='Score external records with frozen healthy references')
    predict.add_argument('--dataset',type=Path,required=True)
    predict.add_argument('--models',type=Path,required=True)
    predict.add_argument('--output',type=Path,required=True)
    predict.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    analyze=commands.add_parser('analyze',help='Aggregate private runtime scores and compute robustness')
    analyze.add_argument('--scores',type=Path,required=True)
    analyze.add_argument('--output',type=Path,required=True)
    analyze.add_argument('--iterations',type=int,default=5000)
    analyze.add_argument('--seed',type=int,default=20261023)
    for command in (prep,train,predict):
        command.add_argument('--config',type=Path,help='Public hyperparameter JSON; defaults to built-in study settings')
    args=parser.parse_args(argv)
    if args.command=='demo':
        from .demo import run_demo
        print(json.dumps(run_demo(args.output,args.epochs,args.tasks),indent=2)); return 0
    if args.command=='analyze':
        from .analysis import analyze_file
        analyze_file(args.scores,args.output,iterations=args.iterations,seed=args.seed)
        print('analyze: completed'); return 0
    from .data import load_config
    from .settings import default_config
    config=load_config(args.config) if args.config else default_config()
    if args.command=='prepare':
        from .data import prepare
        prepare(args.manifest,args.output,config)
    elif args.command=='train':
        from .train import train as run_training
        run_training(args.dataset,args.output,config)
    elif args.command=='predict':
        from .inference import predict as run_prediction
        run_prediction(args.dataset,args.models,args.output,config,args.device)
    print(f'{args.command}: completed')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
