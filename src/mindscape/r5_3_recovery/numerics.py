"""A priori IEEE error bounds, not tolerance fitted to observed discrepancies."""
import numpy as np
from mindscape.r5_3_physics.uncertainty import interval_score

def native_bounds(lower,upper):
    lower=np.asarray(lower,dtype=np.float64);upper=np.asarray(upper,dtype=np.float64)
    for value in (lower,upper):
        if not np.isfinite(value).all() or not np.array_equal(value,value.astype(np.float32).astype(np.float64)):
            raise ValueError('Learned bounds not losslessly float32 representable')
    return lower.astype(np.float32),upper.astype(np.float32)

def score_tolerance(truth,lower,upper):
    """Float32 width subtraction: half-ULP average + conservative float64 reduction error."""
    lo,hi=native_bounds(lower,upper)
    width=hi-lo
    half_ulp=float(np.mean(np.abs(np.spacing(width).astype(np.float64)))/2)
    component=np.abs(interval_score(np.asarray(truth,dtype=np.float64),np.asarray(lower,dtype=np.float64),np.asarray(upper,dtype=np.float64)))
    eps=np.finfo(np.float64).eps
    atol=half_ulp+64*eps*max(1,float(np.max(component)))
    return dict(atol=atol,rtol=64*eps,float32_width_half_ulp_mean=half_ulp)

def acceptable(saved,recomputed,tolerance):
    return abs(saved-recomputed)<=tolerance['atol']+tolerance['rtol']*abs(saved)
