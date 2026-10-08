"""Goal-oriented bounded hypothetical rollout demonstration, outside primary arithmetic scoring."""
import argparse
import json
from mindscape.reasoning.simulator import DreamEngine,FunctionalTransitionModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--depth',type=int,default=3)
    parser.add_argument('--goal',type=int,default=3)
    args=parser.parse_args()
    engine=DreamEngine(FunctionalTransitionModel(lambda state,action:state+action),
        lambda state:[1,-1],lambda state,action,next_state:-abs(args.goal-next_state),args.depth,2)
    paths=engine.rollouts(0)
    print(json.dumps({'real_state':0,'goal':args.goal,'selected_action':engine.select(0),
        'rollouts':[{'state':s.value,'hypothetical':s.hypothetical,'actions':path,'score':score} for s,path,score in paths]},indent=2))


if __name__=='__main__':main()
