"""Subprocess boundary for deterministic tools; no user code is exec'd."""
import json,sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(4,5))
    if sys.platform!='darwin':resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
except (ImportError,ValueError):pass
from mindscape_delivery.solvers import solve
try:
    request=json.loads(sys.stdin.buffer.read(32001))
    result=solve(request)
    text=json.dumps(result,allow_nan=False)
    if len(text)>100000:raise ValueError('Tool output limit')
    print(text)
except Exception as exc:
    print(json.dumps({'supported':False,'answer':'Unable to verify this problem within the supported bounds.','method':'verified tool rejected request','error':type(exc).__name__,'steps':[],'verification':{'passed':False}}))
