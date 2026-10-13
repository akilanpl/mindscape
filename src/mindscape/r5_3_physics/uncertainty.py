"""Full covariance propagation and consistent terminal squared-distance risk."""
import numpy as np

def propagate(covariance, jacobian, noise_map, noise_covariance):
    p=jacobian@covariance@jacobian.T+noise_map@noise_covariance@noise_map.T
    return (p+p.T)/2

def kinematic_matrices(dt):
    f=np.eye(4); f[:2,2:]=dt*np.eye(2)
    b=np.concatenate((dt*np.eye(2),np.eye(2)),axis=0)
    return f,b

def terminal_risk(mean,covariance,target):
    return float(np.sum((np.asarray(mean)[:2]-target)**2)+np.trace(covariance[:2,:2]))

def interval_score(truth,lower,upper,alpha=.05):
    return upper-lower+2/alpha*(lower-truth)*(truth<lower)+2/alpha*(truth-upper)*(truth>upper)
