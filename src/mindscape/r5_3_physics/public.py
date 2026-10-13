"""Versioned request validation; no private scene or future-state fields."""
import numpy as np
VERSION='mindscape.physics.transition/5.3.1'

def validate(request):
    if set(request)!={'version','history','masks','actions','contexts'} or request['version']!=VERSION:
        raise ValueError('Public contract field/version mismatch')
    h,m,a,c=(np.asarray(request[k],dtype=np.float32) for k in ('history','masks','actions','contexts'))
    if h.shape!=(4,3,10) or m.shape!=h.shape or a.ndim!=3 or a.shape[1:]!=(3,4) or c.shape!=(len(a),4) or not 1<=len(a)<=64:
        raise ValueError('Public shape/horizon')
    if not all(np.isfinite(v).all() for v in (h,m,a,c)) or abs(a).max()>20 or not np.isin(m,[0,1,2]).all():
        raise ValueError('Public values')
    if np.any(h[m==0]!=0) or not np.allclose(c[:,1],1/30): raise ValueError('Masked private field or timestep')
    return h,m,a,c

def request(history,masks,actions,contexts):
    return dict(version=VERSION,history=history,masks=masks,actions=actions,contexts=contexts)
