"""Public approximate contact/spring baseline; no simulator imports."""
import numpy as np
from mindscape.v3.physics_world.baselines import predict as free_flight
from .public import validate,VERSION
from .uncertainty import propagate,kinematic_matrices
def predict(history, masks, action, context, kind):
    current = history[:, -1]
    mass = np.where(masks[:, -1, :, 4] > 0, current[:, :, 4], 1.0)
    mass = np.maximum(mass, 0.05)
    modified = action.copy()
    # Fixed known rest length1.5, known spring coefficient. Approximate free-flight integration.
    for batch in range(len(current)):
        if context[batch, 2] and current[batch, 1, 9]:
            delta = current[batch, 1, :2] - current[batch, 0, :2]
            distance = np.linalg.norm(delta)
            force = context[batch, 2] * (distance - 1.5) * delta / max(distance, 1e-6)
            modified[batch, 0, :2] += force
            modified[batch, 1, :2] -= force
    result, contact = free_flight(history, masks, modified, context, "analytical_available")
    for batch in range(len(current)):
        for i in range(3):
            if not current[batch, i, 9]:
                continue
            radius = current[batch, i, 5]
            restitution = current[batch, i, 7] if masks[batch, -1, i, 7] else 0.5
            friction = current[batch, i, 6] if masks[batch, -1, i, 6] else 0.2
            if result[batch, i, 1] < radius + 0.02:
                result[batch, i, 1] = radius + 0.02
                if result[batch, i, 3] < 0:
                    result[batch, i, 3] *= -restitution
                reduction = friction * abs(context[batch, 0]) / 30
                result[batch, i, 2] = np.sign(result[batch, i, 2]) * max(
                    0, abs(result[batch, i, 2]) - reduction
                )
                contact[batch, i] = 1
            for j in range(i + 1, 3):
                if not current[batch, j, 9]:
                    continue
                delta = result[batch, j, :2] - result[batch, i, :2]
                distance = np.linalg.norm(delta)
                normal = delta / max(distance, 1e-6)
                closing = np.dot(result[batch, j, 2:4] - result[batch, i, 2:4], normal)
                if distance < current[batch, i, 5] + current[batch, j, 5] and closing < 0:
                    e2 = current[batch, j, 7] if masks[batch, -1, j, 7] else 0.5
                    impulse = (
                        -(1 + restitution * e2)
                        * closing
                        / (1 / mass[batch, i] + 1 / mass[batch, j])
                    )
                    result[batch, i, 2:4] -= impulse / mass[batch, i] * normal
                    result[batch, j, 2:4] += impulse / mass[batch, j] * normal
                    contact[batch, [i, j]] = 1
    return result, contact


def forecast(request,kind):
    h,m,a,c=validate(request); h=h[None].copy(); m=m[None].copy(); results=[]; contacts=[]; covariance=[]
    f,b=kinematic_matrices(1/30); p=np.eye(4)*.005**2
    for action,context in zip(a,c,strict=True):
        state,contact=(predict(h,m,action[None],context[None],kind) if kind=='analytical_contacts' else free_flight(h,m,action[None],context[None],kind))
        p=propagate(p,f,b,np.eye(2)*.01**2)
        full=np.zeros((12,12))
        for i in range(3): full[i*4:(i+1)*4,i*4:(i+1)*4]=p
        covariance.append(full); results.append(state[0]);contacts.append(contact[0])
        current=h[:,-1].copy();current[:,:,:4]=state;current[:,:,8]=contact
        h=np.concatenate((h[:,1:],current[:,None]),axis=1)
        newmask=m[:,-1].copy();newmask[:,:,:4]=2*current[:,:,9:10];m=np.concatenate((m[:,1:],newmask[:,None]),axis=1)
    means=np.stack(results);cov=np.stack(covariance);width=1.96*np.sqrt(np.diagonal(cov,axis1=1,axis2=2).reshape(-1,3,4))
    return dict(version=VERSION,states=means,covariance=cov,lower=means-width,upper=means+width,contacts=np.stack(contacts),private_queries=0)
