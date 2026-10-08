import argparse
from mindscape.evaluation.runner import run
from mindscape.models.study import StudyModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',default='results')
    parser.add_argument('--split',choices=['test','ood_test','validation'],default='test')
    parser.add_argument('--constrained',action='store_true')
    parser.add_argument('--no-dream',action='store_true')
    args=parser.parse_args()
    model=StudyModel.load(args.checkpoint)
    model.constrained=args.constrained
    model.dream=not args.no_dream
    folder,metrics=run(model,args.dataset,args.output,args.split,model.training_metadata['regime'])
    print(folder);print(metrics)


if __name__=='__main__':main()
