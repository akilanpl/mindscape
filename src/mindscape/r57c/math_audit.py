"""Bounded dev-only CPU comparisons; exact update counters, no final data access."""
import json
from pathlib import Path
import torch
from mindscape.r56.compact_gpu import domain_loss
from .math_model import make_model


def score(model,rows):
    records=[]
    with torch.no_grad():
        for row in rows:
            if row['problem']['domain']!='math':continue
            out=model.predict(row['problem']);digit=int(out['math_digit_logits'].argmax());carry=int(out['math_carry_logits'].argmax())
            records.append({'identity':row.get('identity'), 'digit':digit,'carry':carry,
                'correct':digit==row['labels']['math_digit'] and carry==row['labels']['math_carry']})
    return {'exact':sum(r['correct'] for r in records)/len(records),'records':records}


def probe(train,dev,updates=1000,priors=('mlp','lattice'),conditions=('graph-concept-on','graph-concept-off','matched-gru')):
    if updates*len(priors)*len(conditions)>12000:raise ValueError('CPU budget exceeds12000')
    results=[]
    for prior in priors:
        for condition in conditions:
            model=make_model(condition,17,prior);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            before=score(model,dev);losses=[]
            for step in range(updates):
                row=train[step%len(train)];out=model.predict(row['problem'])
                loss=domain_loss(out,row,torch,'cpu');optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
            results.append({'prior':prior,'condition':condition,'optimizer_updates':updates,'initial':before,
                'final':score(model,dev),'weights':model.scalar_arithmetic.weights.detach().tolist(),
                'first_loss':losses[0],'last_loss':losses[-1]})
    return {'device':'cpu','optimizer_updates_total':updates*len(priors)*len(conditions),'development_only':True,'results':results}
