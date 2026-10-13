"""Bounded exact tool math by default; never silently replaces a neural result."""
from fractions import Fraction
from .contract import plain

def symbolic(operation,inputs):
    from mindscape.r5_2_math.capability import Request,solve
    request=Request(operation,inputs)
    if operation!='integer':return solve(request)
    value=inputs['operands'][0];trace=[]
    for op,b in zip(inputs['operations'],inputs['operands'][1:],strict=True):
        before=value;value=value+b if op=='add' else value-b if op=='sub' else value*b
        trace.append({'before':str(before),'operation':op,'operand':str(b),'after':str(value)})
    reference=Fraction(inputs['operands'][0])
    for op,b in zip(inputs['operations'],inputs['operands'][1:],strict=True):
        reference={'add':lambda:reference+Fraction(b),'sub':lambda:reference-Fraction(b),'mul':lambda:reference*Fraction(b)}[op]()
    return {'method':'symbolic_tool','tool':'Python arbitrary-precision integers; independent Fraction replay','family':operation,'result':str(value),'trace':trace,'verification':{'success':reference==value}}

def execute(request,resources,verification):
    if request.action:raise ValueError('Math action must be empty; operations are explicit in observation')
    if request.mode=='validated':
        result=symbolic(request.operation,request.observation)
        return dict(prediction={'kind':'symbolic_result','value':result['result']},execution={'kind':'tool_result','value':result['result'],'steps':result['trace']},verification={'success':result['verification']['success'],'method':result['tool']},provenance={'method':'symbolic_tool','neural':False,'tool':result['tool']})
    if request.mode!='experimental_neural' or request.operation!='integer':raise ValueError('Unsupported math mode/operation')
    from mindscape.r5_2_math.arithmetic import run
    from mindscape.r5_2_math.verification import verify
    from mindscape.r5_2_math.capability import Request
    Request('integer',request.observation)
    predictor,provenance=resources.math()
    result=run(request.observation,predictor) # reference unavailable until proposal completes
    frozen=plain(result)
    score=verify(request.observation,result)
    return dict(prediction={'kind':'neural_prediction','value':frozen},execution={'kind':'actual_execution','value':frozen,'semantics':'write-only arithmetic trace; no correction'},verification={'success':score['exact'],'method':'independent integer/local arithmetic','details':score},provenance=provenance)
