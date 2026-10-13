"""Fresh fixed integration examples, not performance holdouts or training examples."""
import numpy as np
from .contract import Request

def mathematics():
    return [Request('mathematics','integer',{'operands':[-137,29,4],'operations':['add','mul']}),Request('mathematics','fractions',{'a':[2,3],'b':[5,6],'operation':'add'}),Request('mathematics','matrix',{'a':[[1,2],[3,4]],'b':[[2,0],[0,2]]}),Request('mathematics','calculus',{'coefficients':[1,2,3],'lower':0,'upper':2})]

def coding(repair=False):
    repository={'solution.py':'def total(values):\n    return sum(v for v in values if v >= 7)\n'}
    visible={'entry':'solution.total','cases':[{'args':[[3,7,8]],'expected':15,'expected_type':'int','unchanged':True}]}
    if not repair:return Request('coding','execute',{'repository':repository,'visible_tests':visible}),None
    repository={'solution.py':repository['solution.py'].replace('>= 7','> 7')}
    contract={'entry':'solution.total','description':'Sum all integers at least seven, including seven, without mutating the input.','inputs':['list of integers'],'outputs':'int','invariants':['Input unchanged']}
    private={'entry':'solution.total','cases':[{'args':[[7,-2,9,7]],'expected':23,'expected_type':'int','unchanged':True},{'args':[[]],'expected':0,'expected_type':'int','unchanged':True}]}
    return Request('coding','repair',{'repository':repository,'contract':contract,'visible_tests':visible}),{'private_tests':private}

def physics():
    from mindscape.v3.physics_world.schema import Scene,ObjectSpec,Action,observe
    from mindscape.v3.physics_world.environment import Environment
    scene=Scene('motion','integration',0,2026101304,'freeflight',(ObjectSpec((0.,4.),(1.,0.),1.,.25,.2,.5),),9.,0.,False,False,0.,8)
    env=Environment(scene);observations=[observe(env.state(),scene,0)];prior=np.zeros((3,3,4),np.float32)
    for t,a in enumerate(prior):observations.append(observe(env.step(Action.from_vector(a)),scene,t+1))
    contexts=np.repeat(observations[-1]['context'][None],8,axis=0);contexts[:,3]+=np.arange(8)/30
    req=Request('physics','forecast',{'history':np.stack([o['values'] for o in observations]).tolist(),'masks':np.stack([o['mask'] for o in observations]).tolist(),'contexts':contexts.tolist()},{'schedule':np.zeros((8,3,4)).tolist()})
    return req,{'scene':scene.record(),'prior_actions':prior.tolist()}

def probes():
    return {'mathematics':{'values':[17,8],'operation':'add','destination':0,'source':1,'amount':2},'coding':{'x':3,'constants':[4],'operations':['add'],'target_leaf':0,'increment':2,'entry':'kernel_probe'},'physics':{'positions':[[0.,3.],[2.,3.]],'velocities':[[.5,0.],[-.5,0.]],'masses':[1.,1.2],'destination':0,'impulse':[1.,0.],'gravity':0.,'scene_seed':2026101304}}
