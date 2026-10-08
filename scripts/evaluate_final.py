"""Offline final-checkpoint evaluation or structured episode inspection."""
import argparse,json
from pathlib import Path
from mindscape.models.final_backend import FrozenFeatures,FinalBackend
from mindscape.models.final_policy import FinalPolicy
from mindscape.evaluation.runner import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True);p.add_argument('--cache',default='work/final_features');p.add_argument('--output',required=True);p.add_argument('--split',choices=['validation','test','ood_test'],default='test');args=p.parse_args()
 meta=json.loads((Path(args.checkpoint).parent/'metadata.json').read_text());encoder=FrozenFeatures(args.model,args.cache);backend=FinalBackend.load(args.checkpoint,encoder);model=FinalPolicy(backend,meta['config']['condition'],meta,dream=False,no_state=meta['config']['variant']=='no_explicit_state')
 folder,metrics=run(model,args.dataset,args.output,args.split,meta['regime'],meta['config']['budget']);print(folder);print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
