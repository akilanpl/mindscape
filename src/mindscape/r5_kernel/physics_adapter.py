"""Public-only predictions. Private simulator access occurs after prediction in verifier context."""
import numpy as np
from .contract import plain

def predict(public,action,mode,resources):
    from mindscape.r5_3_physics.public import request,validate
    if set(public)!={'history','masks','contexts'} or set(action)!={'schedule'}:raise ValueError('Physics public fields')
    req=request(public['history'],public['masks'],action['schedule'],public['contexts']);validate(req)
    if mode=='validated':
        from mindscape.r5_3_physics.baselines import forecast
        value=forecast(req,'analytical_contacts')
        provenance={'method':'analytical_contacts','neural':False,'uncertainty':'uncalibrated fixed-noise baseline; R5.3 coverage failure','reference':'R5.3 recovered strongest mean-error baseline in both families'}
    elif mode=='experimental_neural':
        from mindscape.r5_3_physics.predict import forecast
        model,provenance=resources.physics();value=forecast(model,req,seed=11,particles=32);value.pop('samples',None)
    else:raise ValueError('Unsupported physics mode')
    return plain(value),provenance

def verify_execution(public,action,context,prediction):
    if context is None:return {'kind':'not_executed'},{'success':None,'method':'no private executor supplied; forecast remains unverified'}
    if set(context)!={'scene','prior_actions'}:raise ValueError('Private physics verifier context')
    from mindscape.v3.physics_world.schema import Scene,Action,observe
    from mindscape.v3.physics_world.environment import Environment
    scene=Scene.from_record(context['scene']);prior=context['prior_actions']
    if len(prior)!=3:raise ValueError('Exactly three prior actions for four public observations')
    env=Environment(scene);observations=[observe(env.state(),scene,0)]
    for t,a in enumerate(prior):observations.append(observe(env.step(Action.from_vector(a)),scene,t+1))
    for key,field in [('history','values'),('masks','mask')]:
        if not np.array_equal(np.asarray(public[key],dtype=np.float32),np.stack([o[field] for o in observations])):raise ValueError('Public/private observation binding mismatch')
    expected=np.repeat(observations[-1]['context'][None],len(action['schedule']),axis=0);expected[:,3]+=np.arange(len(expected))/30
    if not np.allclose(np.asarray(public['contexts']),expected,atol=1e-7,rtol=1e-6):raise ValueError('Public context binding mismatch')
    states=np.stack([env.step(Action.from_vector(a)) for a in action['schedule']])
    other=Environment(scene)
    for a in prior:other.step(Action.from_vector(a))
    replay=np.stack([other.step(Action.from_vector(a)) for a in action['schedule']])
    if not np.array_equal(states,replay):raise ValueError('Independent physics replay changed')
    active=states[0,:,9]>0;pred=np.asarray(prediction['states'],dtype=np.float64);mse=float(((pred[:,active]-states[:,active,:4])**2).mean())
    return {'kind':'actual_execution','engine':'Pymunk 7.2.0','states':states.tolist()},{'success':bool(np.array_equal(pred[:,active],states[:,active,:4])),'execution_verified':True,'method':'independent Pymunk action replay','prediction_mse':mse,'meaning':'success means exact prediction; execution integrity is separate'}

def execute(request,resources,verification):
    if request.operation!='forecast':raise ValueError('Unsupported physics operation')
    value,provenance=predict(request.observation,request.action,request.mode,resources)
    execution,evaluation=verify_execution(request.observation,request.action,verification,value)
    return dict(prediction={'kind':'neural_prediction' if request.mode=='experimental_neural' else 'procedural_prediction','value':value},execution=execution,verification=evaluation,provenance=provenance)
