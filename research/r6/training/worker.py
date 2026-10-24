"""Bounded CUDA-only R6 fitting; every optimizer update is counted and exported.

There is deliberately no automatic resume/retry. A failed reservation consumes its
actual updates and requires controller review before any further cloud fitting.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def require_cloud():
    """No torch/optimizer import occurs before basic cloud identity guards."""
    if platform.system() != 'Linux' or not Path('/kaggle/working').is_dir():
        raise RuntimeError('KAGGLE_LINUX_CUDA_REQUIRED; no optimizer constructed')
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') != 'Batch':
        raise RuntimeError('KAGGLE_BATCH_EXECUTION_REQUIRED; no optimizer constructed')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError('CUDA_REQUIRED; CPU fallback forbidden')
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    return torch


def validate_config(config):
    if not isinstance(config.get('fit_id'),str) or not config['fit_id']:
        raise ValueError('Missing immutable fit identity')
    if type(config.get('updates')) is not int or not 1 <= config['updates'] <= 600:
        raise ValueError('Pilot must contain 1..600 authorized updates')
    if type(config.get('seed')) is not int:
        raise ValueError('Recorded integer initialization/sampling seed required')
    if type(config.get('dev_interval',100)) is not int or not 1 <= config.get('dev_interval',100) <= config['updates']:
        raise ValueError('Invalid bounded development interval')
    if config.get('batch_per_domain') != 8:
        raise ValueError('Frozen balanced pilot requires eight rows per domain')
    if config.get('optimizer','adam') not in ('adam','adamw'):
        raise ValueError('Unsupported optimizer')
    if config.get('schedule','constant') not in ('constant','cosine'):
        raise ValueError('Unsupported learning-rate schedule')
    if not 0 < config.get('lr',0) <= .01 or not 0 < config.get('clip_norm',0) <= 10:
        raise ValueError('Invalid bounded learning rate or clipping')
    if not 0 < config.get('max_seconds',0) <= 3600:
        raise ValueError('Pilot time limit must be at most one hour')
    for name in ('physics_target_mean','physics_target_scale'):
        if len(config.get(name,[])) != 4 or any(not math.isfinite(v) for v in config[name]):
            raise ValueError('Four finite target normalization values required')
    if any(v <= 0 for v in config['physics_target_scale']):
        raise ValueError('Target scales must be positive')


def validate_package(package):
    package=Path(package).resolve()
    config=json.loads((package/'config.json').read_text());validate_config(config)
    manifest=json.loads((package/'manifest.json').read_text())
    inventory=manifest['files_sha256']
    for required in ('config.json','train.json','dev.json','normalization.json'):
        if required not in inventory:
            raise ValueError('Missing hashed package input: '+required)
    for name,digest in inventory.items():
        target=(package/name).resolve()
        if not target.is_relative_to(package) or not target.is_file() or sha(target)!=digest:
            raise ValueError('Package hash/path mismatch: '+name)
    train=json.loads((package/'train.json').read_text());dev=json.loads((package/'dev.json').read_text())
    for split,rows in [('train',train),('dev',dev)]:
        if not rows or any(r['split']!=split for r in rows):
            raise ValueError('Only exact train/dev records accepted')
        if len({r['record_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate record IDs')
        if set(r['domain'] for r in rows) != {'mathematics','coding','physics'}:
            raise ValueError('All three domains required')
    if {r['canonical_key'] for r in train} & {r['canonical_key'] for r in dev}:
        raise ValueError('Train/development context overlap')
    return config,manifest,train,dev


class BalancedSampler:
    """Deterministic per-domain shuffle cycles; no final cases or global RNG."""
    def __init__(self,rows,seed,batch_per_domain=8):
        self.rows={d:[r for r in rows if r['domain']==d] for d in ('mathematics','coding','physics')}
        if any(not v for v in self.rows.values()):raise ValueError('Missing domain')
        self.rng={d:random.Random(f'{seed}:{d}') for d in self.rows}
        self.order={d:[] for d in self.rows};self.position={d:0 for d in self.rows}
        self.batch_per_domain=batch_per_domain
    def next(self):
        batch=[]
        for domain,rows in self.rows.items():
            for _ in range(self.batch_per_domain):
                if self.position[domain]>=len(self.order[domain]):
                    self.order[domain]=list(range(len(rows)));self.rng[domain].shuffle(self.order[domain]);self.position[domain]=0
                batch.append(rows[self.order[domain][self.position[domain]]]);self.position[domain]+=1
        return batch


def _loss(torch,outputs,batch):
    import torch.nn.functional as F
    physics_mask=batch['physics_mask'].bool();text_mask=batch['text_mask'].bool()
    text=F.cross_entropy(outputs['text_logits'][text_mask].reshape(-1,outputs['text_logits'].shape[-1]),
                         batch['target_output'][text_mask].reshape(-1),ignore_index=0) if text_mask.any() else outputs['text_logits'].sum()*0
    physics=F.mse_loss(outputs['physics'][physics_mask],batch['physics_target'][physics_mask]) if physics_mask.any() else outputs['physics'].sum()*0
    return text+physics,text,physics


def prediction_metrics(rows,predictions):
    result={}
    for domain in ('mathematics','coding','physics'):
        pairs=[(r,p) for r,p in zip(rows,predictions) if r['domain']==domain]
        if domain=='physics':
            errors=[[p['state'][i]-r['target']['state'][i] for i in range(4)] for r,p in pairs]
            result[domain]={'count':len(pairs),'mse':sum(e*e for x in errors for e in x)/(4*len(pairs)),
                            'mae':sum(abs(e) for x in errors for e in x)/(4*len(pairs)),
                            'mse_per_dimension':[sum(x[i]**2 for x in errors)/len(pairs) for i in range(4)],
                            'mae_per_dimension':[sum(abs(x[i]) for x in errors)/len(pairs) for i in range(4)]}
        else:
            correct=sum(p.get('answer')==r['target']['answer'] for r,p in pairs)
            result[domain]={'count':len(pairs),'exact_correct':correct,'exact_accuracy':correct/len(pairs),
                            'decoding':'autoregressive native integer output'}
    return result


def gpu_sample():
    try:
        process=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
                               capture_output=True,text=True,timeout=5)
        return {'scope':'instantaneous sample at development boundary; not continuous utilization',
                'returncode':process.returncode,'csv':process.stdout.strip(),'stderr':process.stderr.strip()}
    except Exception as exc:return {'scope':'sample unavailable','error':type(exc).__name__}


def run(package_root,output):
    torch=require_cloud()
    config,manifest,train,dev=validate_package(package_root)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from research.r6.architectures.model import make_model, predict
    from research.r6 import common
    started=time.perf_counter();updates=0;model=None;optimizer=None
    receipt={'fit_id':config['fit_id'],'status':'STARTED','optimizer_updates':0,'resume_policy':'NO_AUTO_RESUME',
             'config':config,'package_manifest_sha256':sha(Path(package_root)/'manifest.json'),
             'training_sha256':sha(Path(package_root)/'train.json'),'development_sha256':sha(Path(package_root)/'dev.json'),
             'hardware':{'device':'cuda:0','gpu_count':torch.cuda.device_count(),'gpu_name':torch.cuda.get_device_name(0),
                         'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
                         'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
                         'gpu_memory_is_unified':False,'cudnn':torch.backends.cudnn.version(),
                         'cpu_count':os.cpu_count(),'disk_free_bytes':shutil.disk_usage(output).free,
                         'ram_total_bytes':int(Path('/proc/meminfo').read_text().split('MemTotal:')[1].split()[0])*1024},'precision':'FP32','tf32':False,'dev_history':[]}
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);random.seed(config['seed'])
    exposures=Counter();sampler=BalancedSampler(train,config['seed'],8)
    normalization=json.loads((Path(package_root)/'normalization.json').read_text())
    if normalization.get('computed_from')!='train-only' or normalization.get('train_sha256')!=sha(Path(package_root)/'train.json'):
        raise ValueError('Training-only baseline normalization identity mismatch')
    majority=normalization['training_majority_answers'];physics_mean=normalization['physics_mean_state']
    receipt['development_training_only_baseline']=prediction_metrics(dev,[{'state':physics_mean} if r['domain']=='physics' else {'answer':majority[r['domain']]} for r in dev])
    try:
        model=make_model(config).to('cuda:0');model.train()
        if any(p.dtype!=torch.float32 or p.device.type!='cuda' for p in model.parameters()):
            raise RuntimeError('FP32 CUDA parameter placement required')
        receipt['parameter_count']=sum(p.numel() for p in model.parameters())
        initial_path=output/'initial.json';common.save_checkpoint(initial_path,model,{'config':config,'optimizer_updates':0})
        receipt['initial_sha256']=sha(initial_path)
        opt_type=torch.optim.Adam if config.get('optimizer','adam')=='adam' else torch.optim.AdamW
        optimizer=opt_type(model.parameters(),lr=config['lr'],weight_decay=config.get('weight_decay',0.))
        actual_steps=[0]
        optimizer.register_step_post_hook(lambda *_:actual_steps.__setitem__(0,actual_steps[0]+1))
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['updates']) if config.get('schedule','constant')=='cosine' else None
        mean=torch.tensor(config['physics_target_mean'],device='cuda:0');scale=torch.tensor(config['physics_target_scale'],device='cuda:0')
        def batch_for(rows):
            batch=common.encode_batch(rows,'cuda:0')
            # Encoding API returns raw target state; config alone defines the fitted target geometry.
            batch=dict(batch);batch['physics_target']=(batch['physics_target']-mean)/scale
            return batch
        def measure(rows):
            model.eval();per_domain={}
            with torch.no_grad():
                for domain in ('mathematics','coding','physics'):
                    domain_rows=[r for r in rows if r['domain']==domain]
                    sums=[0.,0.,0.,0.];count=0
                    for i in range(0,len(domain_rows),24):
                        subset=domain_rows[i:i+24];b=batch_for(subset);o=model(b['tokens'],b['lengths'],b['features'],b['domains'],b['target_input'])
                        total,tx,ph=_loss(torch,o,b)
                        mse=float((((o['physics']*scale+mean)-(b['physics_target']*scale+mean))[b['physics_mask'].bool()]).square().mean()) if domain=='physics' else 0.
                        n=len(subset);count+=n
                        for j,v in enumerate((float(total),float(tx),float(ph),mse)):sums[j]+=v*n
                    per_domain[domain]={'count':count,'objective':sums[0]/count,'text_ce':sums[1]/count,'scaled_physics_mse':sums[2]/count,'raw_physics_mse':sums[3]/count if domain=='physics' else None}
            native_metrics=prediction_metrics(rows,predict(model,rows))
            model.train()
            return {'objective':sum(v['text_ce'] for v in per_domain.values())/2+per_domain['physics']['scaled_physics_mse'],
                    'domains':per_domain,'native_autoregressive_metrics':native_metrics}
        initial_dev=measure(dev);receipt['initial_development']=initial_dev
        receipt['initial_training']=measure(train)
        best=initial_dev['objective'];receipt['best_development_step']=0
        best_path=initial_path
        torch.cuda.reset_peak_memory_stats(0)
        log=(output/'loss.jsonl').open('x')
        for step in range(1,config['updates']+1):
            if time.perf_counter()-started>config['max_seconds']:raise TimeoutError('Bounded pilot wall time exhausted')
            rows=sampler.next();b=batch_for(rows);optimizer.zero_grad(set_to_none=True)
            o=model(b['tokens'],b['lengths'],b['features'],b['domains'],b['target_input'])
            loss,text_loss,physics_loss=_loss(torch,o,b)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            domain_diagnostics={}
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient: '+name)
                    exposures[name]+=1
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
            for label,patterns in [('physics',('physics_head',)),('text',('text_head','decoder')),('features',('property_','message','node_update','numeric_recurrent'))]:
                norms=[p.grad.detach().square().sum() for name,p in model.named_parameters() if any(pattern in name for pattern in patterns) and p.grad is not None]
                domain_diagnostics[label+'_gradient_norm']=float(torch.stack(norms).sum().sqrt()) if norms else None
            optimizer.step();updates=actual_steps[0]
            if updates!=step:raise RuntimeError('Optimizer post-hook update mismatch')
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite updated parameter')
            if schedule is not None:schedule.step()
            entry={'update':updates,'loss':float(loss.detach()),'text_ce':float(text_loss.detach()),'scaled_physics_mse':float(physics_loss.detach()),
                   'gradient_norm_before_clip':float(grad_norm),'input_feature_rms':float(b['features'].square().mean().sqrt()),
                   'physics_output_rms':float(o['physics'].detach().square().mean().sqrt()),'lr':optimizer.param_groups[0]['lr'],
                   **domain_diagnostics,'activations':dict(model.last_activation_diagnostics)}
            log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            if step%config.get('dev_interval',100)==0 or step==config['updates']:
                dev_metrics=measure(dev);receipt['dev_history'].append({'update':step,'gpu_sample':gpu_sample(),**dev_metrics})
                common.save_checkpoint(output/f'step-{step:04d}.json',model,{'config':config,'optimizer_updates':step,'development':dev_metrics})
                if dev_metrics['objective']<best:
                    best=dev_metrics['objective'];receipt['best_development_step']=step
                    best_path=output/f'step-{step:04d}.json'
        log.close()
        receipt['final_development']=dev_metrics
        loss_records=[json.loads(line) for line in (output/'loss.jsonl').read_text().splitlines()]
        receipt['loss_summary']={'first_50_mean':sum(r['loss'] for r in loss_records[:50])/len(loss_records[:50]),
                                 'last_50_mean':sum(r['loss'] for r in loss_records[-50:])/len(loss_records[-50:])}
        for name,p in model.named_parameters():
            actual=int(optimizer.state[p].get('step',0)) if p in optimizer.state else 0
            if actual!=exposures[name]:raise RuntimeError('Per-parameter Adam/gradient exposure mismatch: '+name)
        receipt['final_training']=measure(train)
        common.save_checkpoint(output/'final.json',model,{'config':config,'optimizer_updates':updates})
        shutil.copyfile(best_path,output/'best-dev.json')
        # Training-only state for failure diagnosis; never mounted into the evaluator.
        torch.save({'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'optimizer_updates':updates,'fit_id':config['fit_id'],'auto_resume_allowed':False},output/'optimizer-state.pt')
        receipt.update(status='COMPLETED',optimizer_updates=updates,checkpoint_sha256=sha(output/'final.json'),
                       selected_checkpoint_sha256=sha(output/'best-dev.json'),best_development_objective=best,
                       gradient_exposures=dict(exposures),per_parameter_adam_steps_verified=True)
        if updates!=config['updates']:raise RuntimeError('Exact bounded update count mismatch')
        # Loader verification is performed below through the shared safe JSON API.
        restored=make_model(config).to('cuda:0')
        reload_metadata=common.load_checkpoint(output/'final.json',restored)
        if reload_metadata['optimizer_updates']!=updates or any(not torch.equal(model.state_dict()[k],v) for k,v in restored.state_dict().items()):
            raise RuntimeError('Safe checkpoint reload mismatch')
        receipt['checkpoint_reload_verified']=True
    except Exception as exc:
        receipt.update(status='FAILED',optimizer_updates=updates,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if model is not None:
            common.save_checkpoint(output/'partial.json',model,{'config':config,'optimizer_updates':updates,'failed':True})
        if optimizer is not None:
            torch.save({'optimizer':optimizer.state_dict(),'optimizer_updates':updates,'auto_resume_allowed':False},output/'partial-optimizer-state.pt')
        raise
    finally:
        torch.cuda.synchronize(0)
        elapsed=time.perf_counter()-started
        receipt.update(elapsed_seconds=elapsed,updates_per_second=updates/elapsed,
                       examples_processed=updates*24,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
        receipt['output_hashes']={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='receipt.json'}
        write(output/'receipt.json',receipt)
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.package,args.output)
