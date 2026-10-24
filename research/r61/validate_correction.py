"""Zero-update tests of the intervention; never estimates trained recovery."""
import argparse,copy,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from research.r6 import common
from research.r6.architectures.model import make_model as baseline,predict
from research.r61.model import make_model
from research.r61.diagnose import gradients,digest_state

def run(r6,output):
    torch.set_num_threads(2);r6=Path(r6);rows=common.read(r6/'dataset-v1/train.json');result={'optimizer_updates':0,'scope':'Same frozen train cases and checkpoint tensors; diagnostic changes only, not recovery training','fits':{}}
    for fit in ('p1-raw','p1-scaled'):
        config=common.read(r6/fit/'fit/receipt.json')['config'];models={'baseline':baseline(config),'pooled':make_model({**config,'sequence_pooling':'masked-mean'})};entry={}
        for model in models.values():common.load_checkpoint(r6/fit/'fit/best-dev.json',model)
        assert all(torch.equal(v,models['pooled'].state_dict()[k]) for k,v in models['baseline'].state_dict().items())
        for mode,model in models.items():
            entry[mode]={}
            for domain in ('mathematics','coding'):
                sample=[r for r in rows if r['domain']==domain][:16];before=digest_state(model);g=gradients(model,sample)
                preds=predict(model,sample);altered=copy.deepcopy(sample)
                for r in altered:r['target']={'unavailable':True};r['structured_input']={'ignored':True}
                assert predict(model,sample)==predict(model,altered)
                assert digest_state(model)==before
                entry[mode][domain]={'median_early_to_late_input_gradient':statistics.median(x['early_to_late_ratio'] for x in g['per_record_input_position_gradients']),'gradient_details':g,'predictions':preds,'correct':sum(p['answer']==r['target']['answer'] for r,p in zip(sample,preds)),'target_and_structured_fields_blind_inference':True}
        result['fits'][fit]=entry
    # Shared encoder objective interaction at actual R6 selected weights. A
    # single-batch cosine diagnoses local interference; it is not a historical
    # causal explanation of the whole training trajectory.
    result['shared_encoder_gradient_cosines']={}
    for fit in ('p1-raw','p1-scaled'):
        config=common.read(r6/fit/'fit/receipt.json')['config'];model=baseline(config);common.load_checkpoint(r6/fit/'fit/best-dev.json',model)
        vectors={};norms={}
        for domain in ('mathematics','coding','physics'):
            for p in model.parameters():p.grad=None
            sample=[r for r in rows if r['domain']==domain][:16];b=common.encode_batch(sample,'cpu');o=model(b['tokens'],b['lengths'],b['features'],b['domains'],b['target_input'])
            loss=F.mse_loss(o['physics'],(b['physics_target']-model.target_mean)/model.target_scale) if domain=='physics' else F.cross_entropy(o['text_logits'].reshape(-1,14),b['target_output'].flatten(),ignore_index=0)
            loss.backward();v=torch.cat([p.grad.flatten() for n,p in model.named_parameters() if n.startswith(('input_embedding','sequence','fusion'))]);vectors[domain]=v;norms[domain]=float(v.norm())
        result['shared_encoder_gradient_cosines'][fit]={'norms':norms,'math_vs_coding':float(F.cosine_similarity(vectors['mathematics'],vectors['coding'],dim=0)),'math_vs_physics':float(F.cosine_similarity(vectors['mathematics'],vectors['physics'],dim=0)),'coding_vs_physics':float(F.cosine_similarity(vectors['coding'],vectors['physics'],dim=0)),'scope':'First16 training cases/domain; selected checkpoint only; not a training-time trajectory.'}
    common.write(output,result)
    print(json.dumps({f:{m:{d:x['median_early_to_late_input_gradient'] for d,x in domains.items()} for m,domains in v.items()} for f,v in result['fits'].items()}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--r6',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.r6,a.output)
