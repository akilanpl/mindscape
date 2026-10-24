"""Prepared but unexecuted paired tiny-set pilot. CUDA and human budget gates.

Run only after explicit user approval. A local approval record is an operational
controller gate, not independent proof of human identity or a security boundary.
"""
import argparse,hashlib,json,os,platform,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def require_cloud():
    if platform.system()!='Linux' or not Path('/kaggle/working').is_dir() or os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!='Batch':raise RuntimeError('KAGGLE_CUDA_BATCH_REQUIRED; no optimizer constructed')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED; no optimizer constructed')
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch

def validate_approval(config_path,approval_path):
    cfg=read(config_path)
    if approval_path is None or not Path(approval_path).is_file():raise RuntimeError('EXPLICIT_USER_BUDGET_APPROVAL_REQUIRED; no optimizer constructed')
    approval=read(approval_path)
    if approval.get('approved') is not True or approval.get('config_sha256')!=sha(config_path) or not approval.get('user_approval_text'):raise RuntimeError('APPROVAL_NOT_BOUND_TO_CONFIG')
    if (cfg['updates_per_fit_max']!=1200 or cfg['total_optimizer_update_ceiling']!=2400 or cfg['wall_seconds_per_fit_max']!=300 or cfg['total_cuda_fit_seconds_ceiling']!=600 or cfg['seed']!=61117):raise ValueError('OUTSIDE_PREPARED_BUDGET')
    return cfg

def run(config_path,data_root,source_root,output,approval_path=None):
    cfg=validate_approval(config_path,approval_path)
    torch=require_cloud()  # The development Mac cannot pass this boundary.
    import torch.nn.functional as F
    from research.r6 import common
    from research.r6.architectures.model import predict
    from research.r61.model import make_model
    source_root=Path(source_root);data_root=Path(data_root)
    for rel,digest in cfg['trainer_code_sha256'].items():
        if sha(source_root/rel)!=digest:raise ValueError('TRAINER_SOURCE_HASH_CHANGED')
    for name,digest in [('train.json',cfg['train_sha256']),('dev.json',cfg['dev_sha256'])]:
        if sha(data_root/name)!=digest:raise ValueError('DATA_HASH_CHANGED')
    train=read(data_root/'train.json');dev=read(data_root/'dev.json')
    if len(train)!=32 or any(r['domain'] not in ('mathematics','coding') for r in train+dev):raise ValueError('TINY_TEXT_ONLY_REQUIRED')
    if {r['canonical_key'] for r in train}&{r['canonical_key'] for r in dev}:raise ValueError('TRAIN_DEV_OVERLAP')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);campaign={'status':'STARTED','optimizer_updates':0,'fits':[],'config_sha256':sha(config_path),'approval_record_sha256':sha(approval_path)};write(output/'campaign.json',campaign)
    total=0
    def metrics(model,rows):
        preds=predict(model,rows);return {d:{'count':len(rr:=[(r,p) for r,p in zip(rows,preds) if r['domain']==d]),'exact_accuracy':sum(r['target']['answer']==p['answer'] for r,p in rr)/len(rr),'distinct_valid_outputs':len({p['answer'] for _,p in rr if p['answer'] is not None})} for d in ('mathematics','coding')}
    for fit_id,mode in zip(cfg['fit_ids'],('last','masked-mean')):
        directory=output/fit_id;directory.mkdir(exist_ok=False);started=time.perf_counter();steps=0
        receipt={'fit_id':fit_id,'status':'STARTED','optimizer_updates':0,'pooling':mode,'device':'cuda:0','gpu':torch.cuda.get_device_name(0),'gpu_count':torch.cuda.device_count(),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__,'cuda':torch.version.cuda,'config_sha256':sha(config_path),'train_sha256':sha(data_root/'train.json'),'dev_sha256':sha(data_root/'dev.json')}
        try:
            torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            model=make_model({'width':48,'architecture':'native-relational','seed':cfg['seed'],'sequence_pooling':mode}).to('cuda:0')
            receipt['parameter_count']=sum(p.numel() for p in model.parameters());receipt['initial_sha256']=common.save_checkpoint(directory/'initial.json',model,{'fit_id':fit_id,'step':0,'pooling':mode})
            optimizer=torch.optim.Adam(model.parameters(),lr=.001);actual=[0];hook=optimizer.register_step_post_hook(lambda *args:actual.__setitem__(0,actual[0]+1))
            batch=common.encode_batch(train,'cuda:0');initial_state={k:v.detach().clone() for k,v in model.named_parameters()}
            receipt['initial_train']=metrics(model,train);logs=[]
            with (directory/'loss.jsonl').open('w') as stream:
                for step in range(1,1201):
                    if time.perf_counter()-started>=300:raise TimeoutError('FIT_TIME_CEILING; partial run cannot be silently retried')
                    model.train();optimizer.zero_grad(set_to_none=True)
                    result=model(batch['tokens'],batch['lengths'],batch['features'],batch['domains'],batch['target_input'])
                    loss=F.cross_entropy(result['text_logits'].reshape(-1,14),batch['target_output'].flatten(),ignore_index=0)
                    if not bool(loss.isfinite()):raise ValueError('NONFINITE_LOSS')
                    loss.backward()
                    grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
                    if any(not bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None):raise ValueError('NONFINITE_GRADIENT')
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();steps=actual[0];total+=1
                    if steps!=step or total>2400:raise RuntimeError('OPTIMIZER_COUNT_VIOLATION')
                    if any(not bool(p.isfinite().all()) for p in model.parameters()):raise ValueError('NONFINITE_WEIGHT')
                    stream.write(json.dumps({'update':steps,'text_ce':float(loss.detach()),'gradient_norm':float(norm),'per_layer_gradient_norms':grads})+'\n');stream.flush()
                    if step%100==0:logs.append({'step':step,'train':metrics(model,train),'dev':metrics(model,dev)})
            hook.remove();receipt.update(status='COMPLETED',optimizer_updates=steps,history=logs,final_train=metrics(model,train),final_dev=metrics(model,dev),parameter_update_norms={n:float((p.detach()-initial_state[n]).norm()) for n,p in model.named_parameters()})
            receipt['final_sha256']=common.save_checkpoint(directory/'final.json',model,{'fit_id':fit_id,'step':steps,'pooling':mode})
            receipt['elapsed_seconds']=time.perf_counter()-started
            if receipt['elapsed_seconds']>300:raise TimeoutError('FIT_POSTCHECK_TIME_CEILING')
            fresh=make_model(model.config).to('cuda:0');common.load_checkpoint(directory/'final.json',fresh)
            if predict(fresh,dev)!=predict(model,dev):raise RuntimeError('RELOAD_MISMATCH')
            receipt['reload_predictions_exact']=True
        except Exception as exc:
            receipt.update(status='FAILED_OR_PARTIAL',optimizer_updates=steps,elapsed_seconds=time.perf_counter()-started,error_type=type(exc).__name__)
            write(directory/'receipt.json',receipt);campaign.update(status='STOPPED_NO_AUTO_RETRY',optimizer_updates=total);campaign['fits'].append(receipt);write(output/'campaign.json',campaign);raise
        write(directory/'receipt.json',receipt);campaign['fits'].append(receipt);campaign['optimizer_updates']=total;write(output/'campaign.json',campaign)
    campaign['status']='COMPLETED';write(output/'campaign.json',campaign)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('config','data-root','source-root','output'):p.add_argument('--'+x,required=True)
    p.add_argument('--approval-record');a=p.parse_args();run(a.config,a.data_root,a.source_root,a.output,a.approval_record)
