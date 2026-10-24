"""Read-only CPU dataflow diagnostics. No optimizer is constructed or invoked."""
import argparse,copy,hashlib,json,math,statistics
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from research.r6 import common
from research.r6.architectures.model import make_model,predict
from research.r6.datasets import build,coding,mathematics
from research.r6.training.worker import BalancedSampler,_loss

def entropy(values):
    counts=Counter(values);n=sum(counts.values())
    return -sum((v/n)*math.log2(v/n) for v in counts.values())
def digest_state(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def decode_context(model,context,steps=common.MAX_OUTPUT):
    state=context.unsqueeze(0);token=torch.full((len(context),1),common.BOS,dtype=torch.long);seq=[]
    for _ in range(steps):
        decoded,state=model.decoder(model.output_embedding(token),state)
        token=model.text_head(decoded).argmax(-1);seq.append(token[:,0].tolist())
    result=[]
    for i in range(len(context)):
        chars=[]
        for step in seq:
            x=step[i]
            if x==common.EOS:break
            chars.append('-' if x==common.MINUS else str(x-4) if x>=4 else '?')
        result.append(''.join(chars))
    return result

def measure(model,rows):
    batch=common.encode_batch(rows,'cpu');model.eval()
    with torch.no_grad():
        ctx=model.encode(batch['tokens'],batch['lengths'],batch['features'],batch['domains'])
        contexts={'actual':ctx,'reversed':ctx.flip(0),'domain_mean':ctx.mean(0,keepdim=True).expand_as(ctx),'zero':torch.zeros_like(ctx)}
        losses={};tokens={};greedy={};full_logits={}
        for k,c in contexts.items():
            dec,_=model.decoder(model.output_embedding(batch['target_input']),c.unsqueeze(0));logits=model.text_head(dec)
            full_logits[k]=logits
            losses[k]=float(F.cross_entropy(logits.reshape(-1,14),batch['target_output'].flatten(),ignore_index=0))
            mask=batch['target_output']!=0
            tokens[k]=float((logits.argmax(-1)[mask]==batch['target_output'][mask]).float().mean())
            greedy[k]=decode_context(model,c)
        distances=torch.pdist(ctx)
        input_change_effect=(full_logits['actual']-full_logits['reversed']).abs().amax((0,2)).tolist()
        return {'context_unique_exact':len({tuple(v) for v in ctx.tolist()}),'context_pairwise_distance_min':float(distances.min()),'context_pairwise_distance_median':float(distances.median()),'context_feature_std_mean':float(ctx.std(0).mean()),'context_tanh_abs_gt_095_fraction':float((ctx.abs()>.95).float().mean()),'teacher_forced_cross_entropy':losses,'teacher_forced_token_accuracy':tokens,'context_reversal_logit_max_delta_by_answer_position':input_change_effect,'greedy_diversity':{k:dict(Counter(v)) for k,v in greedy.items()},'greedy_changed_on_reversal':sum(a!=b for a,b in zip(greedy['actual'],greedy['reversed']))/len(rows),'greedy_changed_on_mean_replacement':sum(a!=b for a,b in zip(greedy['actual'],greedy['domain_mean']))/len(rows)}

def gradients(model,rows):
    before=digest_state(model)
    for p in model.parameters():p.grad=None
    batch=common.encode_batch(rows,'cpu');captured=[]
    def hook(module,args,output):output.retain_grad();captured.append(output)
    handle=model.input_embedding.register_forward_hook(hook)
    out=model(batch['tokens'],batch['lengths'],batch['features'],batch['domains'],batch['target_input'])
    ce=F.cross_entropy(out['text_logits'].reshape(-1,14),batch['target_output'].flatten(),ignore_index=0)
    manual=-F.log_softmax(out['text_logits'],-1).gather(-1,batch['target_output'].unsqueeze(-1)).squeeze(-1)
    manual=manual[batch['target_output']!=0].mean()
    original=_loss(torch,out,batch)[1]
    ce.backward();handle.remove()
    norms={name:{'norm':float(p.grad.norm()) if p.grad is not None else 0.,'finite':bool(p.grad.isfinite().all()) if p.grad is not None else None,'connected':p.grad is not None} for name,p in model.named_parameters()}
    g=captured[0].grad.norm(dim=-1);by_position=[]
    for i,row in enumerate(rows):
        n=len(row['input']); digit=[j for j,c in enumerate(row['input']) if c.isdigit()];early=g[i,:max(1,n//3)].mean();late=g[i,2*n//3:n].mean()
        by_position.append({'record_id':row['record_id'],'bytes':n,'early_third_gradient_mean':float(early),'late_third_gradient_mean':float(late),'early_to_late_ratio':float(early/(late+1e-30)),'digit_gradient_mean':float(g[i,digit].mean()) if digit else None})
    assert before==digest_state(model)
    for p in model.parameters():p.grad=None
    return {'loss':float(ce.detach()),'manual_loss_abs_delta':float((ce-manual).detach().abs()),'original_worker_loss_abs_delta':float((ce-original).detach().abs()),'parameter_gradients':norms,'per_record_input_position_gradients':by_position,'weights_and_buffers_unchanged':True,'optimizer_updates':0}

def probe_rows(domain):
    if domain=='mathematics':
        texts=['Calculate (12) + (7).','Calculate (21) + (7).','Calculate (12) - (7).','Calculate (7) + (12).','Calculate  (12) + (7).']
        answers=['19','28','5','19','19']
        groups=['base','operand','operator','commute','whitespace']
    else:
        texts=['def f(n):\n x=0\n for i in range(n):\n  x=x+i\n return x\nr=f(4)',
               'def f(n):\n x=0\n for i in range(n):\n  x=x+i\n return x\nr=f(5)',
               'def f(n):\n x=0\n for i in range(n):\n  x=x+n\n return x\nr=f(4)',
               'def f(n):\n x=0\n for i in range(n):\n  x=x+i\n return x\nr=f(0)',
               'def h(k):\n y=0\n for j in range(k):\n  y=y+j\n return y\ns=h(4)',
               'def f(n):\n  x=0\n  for i in range(n):\n    x=x+i\n  return x\nr=f(4)',
               'def f(n):\n x=1\n if n>2:\n  x=3\n return x\nr=f(4)',
               'def f(n):\n x=1\n if n<2:\n  x=3\n return x\nr=f(4)']
        answers=[str(coding.interpret(t)) for t in texts];groups=['base','loop_bound','variable_binding','zero_loop','rename','whitespace','branch_gt','branch_lt']
    return [{'record_id':'r61-probe-'+domain+'-'+g,'domain':domain,'input':t,'features':[],'target':{'answer':a},'probe':g} for t,a,g in zip(texts,answers,groups)]

def run(r6,output):
    r6=Path(r6);output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    files=sorted(p for p in r6.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    preservation={str(p.relative_to(r6)):common.sha(p) for p in files}
    common.write(output/'r6-preservation-before.json',preservation)
    splits={s:common.read(r6/'dataset-v1'/f) for s,f in [('train','train.json'),('dev','dev.json'),('final_iid','protected/final_iid.json'),('final_ood','protected/final_ood.json')]}
    allrows=sum(splits.values(),[]);textrows=[r for r in allrows if r['domain']!='physics']
    for row in allrows:build.verify_row(row)
    pairs=[]
    for i,(s,a) in enumerate(splits.items()):
        for t,b in list(splits.items())[i+1:]:
            pairs.append({'splits':[s,t],'exact_input_overlap':len({r['input'] for r in a}&{r['input'] for r in b}),'canonical_overlap':len({r['canonical_key'] for r in a}&{r['canonical_key'] for r in b})})
    distributions={}
    for domain in ('mathematics','coding'):
        ds={}
        majority=Counter(r['target']['answer'] for r in splits['train'] if r['domain']==domain).most_common(1)[0][0]
        for split,rows in splits.items():
            rows=[r for r in rows if r['domain']==domain];a=[r['target']['answer'] for r in rows]
            ids=[common.encode_answer(x) for x in a]
            for answer,seq in zip(a,ids):
                chars=''.join('-' if x==3 else str(x-4) for x in seq[:-1]);assert chars==answer
            ds[split]={'count':len(rows),'distinct_answers':len(set(a)),'entropy_bits':entropy(a),'answer_counts':dict(Counter(a)),'training_majority_answer':majority,'training_majority_accuracy':a.count(majority)/len(a),'zero_constant_accuracy':a.count('0')/len(a),'hindsight_best_constant_accuracy_not_deployable':max(Counter(a).values())/len(a),'target_token_counts':dict(Counter(x for seq in ids for x in seq)),'max_input_bytes':max(len(r['input']) for r in rows),'max_answer_tokens':max(map(len,ids)),'exact_input_unique':len({r['input'] for r in rows})}
        distributions[domain]=ds
    checks=[]
    def check(name,status,evidence):checks.append({'name':name,'status':status,'evidence':evidence})
    check('target_correctness_diversity_codec','PASS',{'independently_verified_records':len(allrows),'text_roundtrips':len(textrows),'no_truncation':True,'vocabulary':'PAD0 BOS1 EOS2 minus3 digits4..13'})
    check('train_test_exact_canonical_overlap','PASS' if not any(p['exact_input_overlap'] or p['canonical_overlap'] for p in pairs) else 'FAIL',pairs)
    sampled=BalancedSampler(splits['train'],61017)
    byid={r['record_id']:r for r in splits['train']}
    batches=[sampled.next() for _ in range(70)]
    assert all(r==byid[r['record_id']] for batch in batches for r in batch)
    check('input_target_pairing_balanced_sampler','PASS',{'batches_checked':len(batches),'examples_checked':sum(map(len,batches)),'domains_per_batch':dict(Counter(r['domain'] for r in batches[0]))})
    models={};prediction_records=[]
    for fit in ('p1-raw','p1-scaled'):
        receipt=common.read(r6/fit/'fit/receipt.json');initial=make_model(receipt['config']);common.load_checkpoint(r6/fit/'fit/initial.json',initial)
        model=make_model(receipt['config']);common.load_checkpoint(r6/fit/'fit/best-dev.json',model)
        before=digest_state(model);entry={'selected_step':receipt['best_development_step'],'executed_steps':receipt['optimizer_updates'],'checkpoint_sha256':common.sha(r6/fit/'fit/best-dev.json'),'domains':{},'weight_changes':{name:{'initial_norm':float(value.detach().norm()),'selected_norm':float(dict(model.named_parameters())[name].detach().norm()),'update_norm':float((dict(model.named_parameters())[name]-value).detach().norm())} for name,value in initial.named_parameters()}}
        losses=[json.loads(x) for x in (r6/fit/'fit/loss.jsonl').read_text().splitlines()]
        entry['optimization_history']={'count':len(losses),'lr_values':sorted({x['lr'] for x in losses}),'text_gradient_norm_min':min(x['text_gradient_norm'] for x in losses),'text_gradient_norm_max':max(x['text_gradient_norm'] for x in losses),'first_50_text_ce_mean':statistics.mean(x['text_ce'] for x in losses[:50]),'last_50_text_ce_mean':statistics.mean(x['text_ce'] for x in losses[-50:]),'all_loss_gradients_finite':all(math.isfinite(x[k]) for x in losses for k in ('loss','text_ce','gradient_norm_before_clip'))}
        for domain in ('mathematics','coding'):
            rows=[r for r in splits['train'] if r['domain']==domain];sample=rows[:32]
            diag=measure(model,sample);diag['gradients']=gradients(model,sample[:16]);diag['splits']={}
            for split,rr in splits.items():
                rr=[r for r in rr if r['domain']==domain];pred=predict(model,rr);init=predict(initial,rr)
                diag['splits'][split]={'count':len(rr),'selected_correct':sum(p['answer']==r['target']['answer'] for r,p in zip(rr,pred)),'initial_correct':sum(p['answer']==r['target']['answer'] for r,p in zip(rr,init)),'selected_output_counts':dict(Counter(p['raw_text'] for p in pred)),'initial_output_counts':dict(Counter(p['raw_text'] for p in init))}
                prediction_records.extend({'fit':fit,'split':split,'record_id':r['record_id'],'domain':domain,'target':r['target']['answer'],'selected':p,'initial':i} for r,p,i in zip(rr,pred,init))
            probes=probe_rows(domain);pp=predict(model,probes);bb=common.encode_batch(probes,'cpu')
            with torch.no_grad():cc=model.encode(bb['tokens'],bb['lengths'],bb['features'],bb['domains'])
            diag['probes']=[{'kind':r['probe'],'input':r['input'],'reference':r['target']['answer'],'prediction':p,'context_distance_from_base':float((cc[i]-cc[0]).norm())} for i,(r,p) in enumerate(zip(probes,pp))]
            other=make_model(receipt['config']);common.load_checkpoint(r6/fit/'fit/best-dev.json',other)
            assert predict(other,sample)==predict(model,sample)
            # Independent greedy implementation must agree, including canonical validity checks downstream.
            b=common.encode_batch(sample,'cpu')
            with torch.no_grad():c=model.encode(b['tokens'],b['lengths'],b['features'],b['domains']);raw=decode_context(model,c)
            assert raw==[p['raw_text'] for p in predict(model,sample)]
            diag['reload_exact_predictions']=True;diag['independent_decoder_exact']=True
            entry['domains'][domain]=diag
        assert before==digest_state(model);entry['model_state_unchanged']=True;models[fit]=entry
    # Compare newly recomputed published final predictions to immutable original evaluator records.
    original_match={}
    for fit in ('p1-raw','p1-scaled'):
        f=r6/'independent-evaluation/downloads/r6-evaluation-output'/('r6-'+fit+'-best-predictions.json')
        saved=common.read(f)
        fresh={(x['record_id'],x['split']):x for x in prediction_records if x['fit']==fit and x['split'].startswith('final_')}
        compared=[x for x in saved if x['domain']!='physics']
        assert all(fresh[(x['record_id'],x['split'])]['selected']==x['prediction'] for x in compared)
        original_match[fit]={'path':str(f),'sha256':common.sha(f),'exact_recomputed_prediction_matches':len(compared),'record_type':type(saved).__name__,'first_entry_keys':list(saved[0]) if isinstance(saved,list) else list(saved)}
    for domain in ('mathematics','coding'):
        ds=[x['domains'][domain] for x in models.values()]
        check(domain+'_input_distinguishability','PASS' if all(x['context_unique_exact']==32 for x in ds) else 'FAIL',{'unique_contexts_in_32_training_examples':[x['context_unique_exact'] for x in ds],'limitation':'Distinct states do not establish retained semantics.'})
        relevant=('input_embedding','sequence','domain_embedding','fusion','output_embedding','decoder','text_head')
        gg=[v for x in ds for n,v in x['gradients']['parameter_gradients'].items() if n.startswith(relevant)]
        check(domain+'_gradient_reachability_and_loss','PASS' if all(x['finite'] and x['norm']>0 for x in gg) and all(x['gradients']['manual_loss_abs_delta']<1e-5 for x in ds) else 'FAIL',{'relevant_parameter_tensors_checked':len(gg),'per_layer_details':'models/*/domains/'+domain+'/gradients','no_optimizer_constructed':True})
        check(domain+'_semantic_output_sensitivity','FAIL',{'probes':'models/*/domains/'+domain+'/probes','note':'See answer-level changes and exact references; nonzero latent/logit changes are insufficient.'})
        check(domain+'_semantics_preserving_invariance','INCONCLUSIVE',{'note':'Unchanged wrong predictions are not semantic robustness; inspect probes.'})
    check('checkpoint_reload_and_independent_decoder','PASS',{'all_four_fit_domain_checks':True,'original_cloud_cpu_prediction_matches':sum(x['exact_recomputed_prediction_matches'] for x in original_match.values())})
    check('learning_collapse_recovered','INCONCLUSIVE',{'reason':'No recovery fitting is authorized. Zero-update diagnostics cannot establish learnability.'})
    for rel,h in preservation.items():assert common.sha(r6/rel)==h
    common.write(output/'predictions.json',prediction_records)
    report={'schema':'mindscape-r61-zero-update-diagnostics-v1','device':'cpu','optimizer_updates':0,'optimizer_constructed':False,'historical_r6_files_unchanged':len(preservation),'diagnostic_scope':'Previously disclosed R6 train/dev/final cases, not a new protected final benchmark. Forward/backward CPU diagnostics permitted; no fitting.','checks':checks,'target_distributions':distributions,'models':models,'original_evaluator_prediction_provenance':original_match}
    common.write(output/'diagnostics.json',report)
    print(json.dumps({'status':'ZERO_UPDATE_DIAGNOSTICS_COMPLETED','checks':[{k:c[k] for k in ('name','status')} for c in checks]}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--r6',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.r6,a.output)
