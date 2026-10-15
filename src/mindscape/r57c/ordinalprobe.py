"""Single fixed seed17 comparison: three2000update full-mixed fits, no search."""
import torch
from .ordinal_model import make_model,domain_loss
from .math_audit import score


def probe(train,dev):
    results=[]
    for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
        model=make_model(condition,17);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        before=score(model,dev)
        for step in range(2000):
            row=train[step%128];output=model.predict(row['problem'])
            loss=domain_loss(output,row,torch,'cpu');optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        results.append({'condition':condition,'optimizer_updates':2000,'initial':before,'final':score(model,dev),
                        'ordinal_weights':model.ordinal_weights.detach().tolist(),
                        'parameter_count':sum(p.numel() for p in model.parameters())})
    return {'optimizer_updates_total':6000,'device':'cpu','seed':17,'development_only':True,
        'prior':'ordinal-decimal-output-geometry','shared_math_core_gradient':False,'results':results}
