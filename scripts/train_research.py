import argparse
from mindscape.core.config import load_config
from mindscape.data.generation import load_dataset
from mindscape.training.claims import CONDITIONS,train_claims


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--config',default='configs/experiments/research_study.toml')
    parser.add_argument('--condition',choices=CONDITIONS,required=True)
    parser.add_argument('--budget',type=int,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    config={k:v for k,v in load_config(args.config).items() if k not in ('budgets','seeds','thresholds')}
    config.update(condition=args.condition,budget=args.budget,seed=args.seed)
    splits,manifest=load_dataset(args.dataset)
    _,meta=train_claims(splits,manifest,config,args.output)
    print(meta['checkpoint'])


if __name__=='__main__':main()
