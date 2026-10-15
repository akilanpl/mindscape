"""Read-only paired mechanism diagnostics; exploratory, never a fitted claim."""
from __future__ import annotations
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import torch
from mindscape.r56.neural import R56Model
from mindscape.r56.native_domains import pack_problem
from mindscape.r56.benchmarks import matched_baseline,controlled_graph,model_accounting,observed_tokens
from mindscape.r56.learning import load_checkpoint


def permute_graph(graph,order):
    order=torch.tensor(order,dtype=torch.long,device=graph.node_features.device)
    if sorted(order.tolist())!=list(range(len(graph.node_features))):raise ValueError('Not a node bijection')
    inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device)
    return replace(graph,node_features=graph.node_features[order],node_types=graph.node_types[order],edges=inverse[graph.edges])


def perturb_graph(graph,kind):
    if kind=='permuted':return permute_graph(graph,list(reversed(range(len(graph.node_features)))))
    if kind=='disconnected':return replace(graph,edges=graph.edges[:,:0],relation_types=graph.relation_types[:0])
    if kind=='reversed':return replace(graph,edges=graph.edges.flip(0))
    if kind=='relation-shuffled':return replace(graph,relation_types=graph.relation_types.flip(0))
    raise ValueError('Unknown graph perturbation')


def predict(model,graph):
    with torch.no_grad():return model(graph,memory=None,stateless=False)['physics_delta'].cpu().tolist()


def concept_probe(model,problem):
    graph=pack_problem(problem);results={}
    for name,ids in [('on',graph.concept_ids),('zero-vector',graph.concept_ids[:0]),('id0',torch.tensor([0])),
                     ('id1',torch.tensor([1])),('fixed-random-id',torch.tensor([19])),('duplicate-id',torch.tensor([2,2]))]:
        local=replace(graph,concept_ids=ids)
        model.zero_grad(set_to_none=True)
        target=torch.tensor([f*problem['dt']/problem['mass'] for f in problem['force']])
        out=model(local,memory=None,stateless=False);loss=(out['physics_delta']-target).square().mean();loss.backward()
        results[name]={'prediction':out['physics_delta'].detach().tolist(),'mse':float(loss.detach()),
                      'concept_norm':float(model.concepts(ids).mean(0).norm().detach()) if ids.numel() else 0.,
                      'action_norm':float(torch.tanh(model.action(graph.action)).norm().detach()),
                      'domain_norm':float(model.domains.weight[graph.domain].norm().detach()),
                      'gradient_norms':{key:float(sum((p.grad.square().sum() for p in module.parameters() if p.grad is not None),torch.tensor(0.)).sqrt()) for key,module in
                        [('concepts',model.concepts),('features',model.features),('message',model.message),('memory',model.working_memory),('physics',model.physics_effect)]}}
    # Same checkpoint/common weights, normalize the supplied embedding magnitude.
    normalized=copy.deepcopy(model)
    with torch.no_grad():normalized.concepts.weight.div_(normalized.concepts.weight.norm(dim=1,keepdim=True).clamp(min=1e-8))
    results['unit-norm-context']={'prediction':predict(normalized,graph)}
    return results


def audit_checkpoint(model,problems,condition):
    records=[]
    for problem in problems:
        graph=controlled_graph(pack_problem(problem),condition);base=predict(model,graph)
        expected=[f*problem['dt']/problem['mass'] for f in problem['force']]
        record={'problem':problem,'prediction':base,'reference':expected,
                'mse':sum((a-b)**2 for a,b in zip(base,expected))/2,
                'graph_probes':{kind:predict(model,perturb_graph(graph,kind)) for kind in ('permuted','disconnected','reversed','relation-shuffled')},'finite_differences':{}}
        for variable in ('force_x','force_y','mass','dt'):
            delta=.01; changed=copy.deepcopy(problem)
            if variable.startswith('force'):changed['force'][0 if variable.endswith('x') else 1]+=delta
            else:changed[variable]+=delta
            observed=predict(model,controlled_graph(pack_problem(changed),condition))
            numerical=[(a-b)/delta for a,b in zip(observed,base)]
            if variable=='force_x':reference=[problem['dt']/problem['mass'],0]
            elif variable=='force_y':reference=[0,problem['dt']/problem['mass']]
            elif variable=='mass':reference=[-f*problem['dt']/problem['mass']**2 for f in problem['force']]
            else:reference=[f/problem['mass'] for f in problem['force']]
            record['finite_differences'][variable]={'observed':numerical,'reference':reference}
        changed=copy.deepcopy(problem);changed['force']=[-f for f in problem['force']]
        record['force_sign_reversal']=predict(model,controlled_graph(pack_problem(changed),condition))
        records.append(record)
    return records


def run_audit(baseline_root,output):
    baseline_root=Path(baseline_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    records=[];hashes={};concepts={};accounting={}
    problems=[{'domain':'physics','mass':mass,'force':[2.,-3.],'dt':dt,'velocity':[0.,0.],'concept_ids':[2]}
              for mass in (1.,2.,4.) for dt in (.25,1.)]
    for seed in (17,29,43):
        for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
            directory=baseline_root/'seed17-2000-updates-v1' if seed==17 and condition=='graph-concept-on' else baseline_root/'remaining-eight-v1'/f'{seed}-{condition}'
            checkpoint=directory/'checkpoint.json';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();hashes[str(checkpoint)]=sha
            torch.manual_seed(seed);initial=matched_baseline('gru',44962,seed) if condition=='matched-gru' else R56Model()
            final=copy.deepcopy(initial);load_checkpoint(final,checkpoint,sha);initial.eval();final.eval()
            accounting[condition]=model_accounting(final)
            records.append({'seed':seed,'condition':condition,'initial':audit_checkpoint(initial,problems,condition),
                            'final':audit_checkpoint(final,problems,condition),'checkpoint_sha256':sha})
            if condition=='graph-concept-on':concepts[str(seed)]={'initial':concept_probe(initial,problems[0]),'final':concept_probe(final,problems[0])}
    report={'scope':'EXPLORATORY_READ_ONLY_PAIRED_DIAGNOSTICS','optimizer_updates':0,'records':records,
            'concept_interventions_shared_weights':concepts,'accounting':accounting,'checkpoint_hashes':hashes,
            'causal_status':'PARTIAL: implementation/invariance and route sensitivities; training gap mechanism not established'}
    (output/'paired-predictions.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run_audit(args.baseline,args.output)
