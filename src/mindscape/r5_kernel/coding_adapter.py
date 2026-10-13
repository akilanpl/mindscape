"""Delegates unmodified coding generation, ranking, memory and WASI execution."""
from .evidence import asset

def suite(value):
    if type(value) is not dict or set(value)!={'entry','cases'} or not isinstance(value['entry'],str) or not 1<=len(value['cases'])<=32:raise ValueError('Bounded coding suite required')
    for case in value['cases']:
        if set(case)-{'args','expected','expected_type','unchanged'} or not {'args','expected','expected_type'}<=set(case):raise ValueError('Coding suite fields')
    return value

def execute(request,resources,verification):
    if request.mode!='validated' or request.action:raise ValueError('Coding specialist behavior only; no alternate model mode')
    from mindscape.v3.coding_learning.execution import Executor,safe_source
    runtime,_=asset('wasi');asset('wasi_compiled')
    if request.operation=='execute':
        if set(request.observation)!={'repository','visible_tests'}:raise ValueError('Coding execute public fields')
        safe_source(request.observation['repository']);suite(request.observation['visible_tests'])
        outcome=Executor(runtime.parent).run(request.observation['repository'],request.observation['visible_tests'])
        return dict(prediction={'kind':'submitted_program','value':request.observation['repository']},execution={'kind':'actual_execution','value':outcome},verification={'success':outcome['all_passed'],'execution_verified':outcome['actual_vm_call'],'method':'WASI execution against caller-visible suite; not unseen-task evidence'},provenance={'method':'WASI','neural':False,'sandbox':'unchanged mindscape.v3.coding_learning.execution.Executor'})
    if request.operation!='repair' or set(request.observation)!={'repository','contract','visible_tests'}:raise ValueError('Coding repair public fields')
    if verification is None or set(verification)!={'private_tests'}:raise ValueError('Repair requires separate private verifier context')
    suite(request.observation['visible_tests']);suite(verification['private_tests'])
    from mindscape.v3.integration.runtime import Request as ExistingRequest
    original=ExistingRequest('coding',request.observation)
    adapter=resources.coding(original,verification['private_tests'])
    # Existing direct specialist includes neural proposal/ranking, actual WASI and post-proposal private scoring.
    result=adapter.direct()
    return dict(prediction={'kind':'neural_prediction','value':result.prediction},execution={'kind':'actual_execution','value':result.actual},verification={'success':result.evaluation['success'],'execution_verified':result.evaluation['valid'],'method':result.evaluation['evaluator'],'reason':result.evaluation['reason']},provenance={'method':'preserved_Qwen_WASI','neural':True,'shared_parameters':False,**adapter.provenance})
