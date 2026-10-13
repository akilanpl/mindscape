"""Explicit routing plus common trace; routing is never labeled learned transfer."""
from dataclasses import asdict
from .contract import Request,Response,digest,plain
from .resources import Resources
from .evidence import BASE,REGISTRY,sha

class Kernel:
    def __init__(self,resources=None):self.resources=resources or Resources()
    def close(self):self.resources.close()
    def direct(self,request,verification=None):
        # Fresh copy prevents caller mutation from changing dispatch mid-flight.
        request=Request(**asdict(request))
        if request.mode=='experimental_shared':
            from .shared import execute
        elif request.domain=='mathematics':
            from .math_adapter import execute
        elif request.domain=='coding':
            from .coding_adapter import execute
        else:
            from .physics_adapter import execute
        return plain(execute(request,self.resources,verification))
    def run(self,request,verification=None):
        request=Request(**asdict(request));before=digest(request.observation)
        result=self.direct(request,verification)
        if before!=digest(request.observation):raise ValueError('Public observation mutated')
        state={'domain':request.domain,'operation':request.operation,'mode':request.mode,'version':request.version,'representation':'domain-qualified public observation','value':request.observation,'observation_hash':before}
        stages=[('observation',request.observation),('state',state),('action',request.action),('prediction',result['prediction']),('execution',result['execution']),('verification',result['verification'])]
        trace=tuple({'stage':stage,'value_hash':digest(value)} for stage,value in stages)
        provenance={**result['provenance'],'runtime_source_sha256':digest({p.name:sha(p) for p in (BASE/'src/mindscape/r5_kernel').glob('*.py')}),'asset_registry_sha256':sha(REGISTRY),'routing':'explicit_request','learned_transfer':False,'optimizer_steps':0,'persistent_memory_updated':False,'shared_parameters':result['provenance'].get('shared_parameters',False)}
        return Response(request.identity,request.domain,state,request.action,result['prediction'],result['execution'],result['verification'],provenance,trace)
    def handle(self,record,verification=None):
        try:return {'status':'OK','response':self.run(Request(**record),verification).record()}
        except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,RuntimeError,OSError,SyntaxError,OverflowError) as error:
            return {'status':'REJECTED','error_type':type(error).__name__,'reason':str(error),'fallback_used':False}
