"""JSON request CLI and bounded CPU/native integration smoke examples."""
import argparse,json,time,sys
from dataclasses import asdict
from pathlib import Path
from .runtime import Kernel
from .contract import Request,digest
from .examples import mathematics,coding,physics,probes
from .evidence import history_check,registry,sha,BASE

def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def comparable(result):
    # Preserve actual outcomes; remove only WASI nonce/time and model timing metadata.
    if isinstance(result,list):return [comparable(v) for v in result]
    if isinstance(result,dict):return {k:comparable(v) for k,v in result.items() if k not in ('duration','stdout','timing','timings','seconds')}
    return result

def smoke(native=False,output=None):
    root=Path(output or ('results/r5_kernel/native-smoke' if native else 'experiments/r5_kernel/cpu-smoke'))
    root.mkdir(parents=True,exist_ok=False);kernel=Kernel();receipts=[]
    initial=history_check()
    try:
        cases=[(r,None) for r in mathematics()]+[coding(),physics()]
        for req,context in cases:
            direct=kernel.direct(req,context);response=kernel.run(req,context)
            for key in ('prediction','execution','verification'):
                if comparable(direct[key])!=comparable(getattr(response,key)):raise ValueError('Kernel/direct parity mismatch: '+req.domain)
            save(root/f'{len(receipts)}-{req.domain}.json',response.record())
            receipts.append({'domain':req.domain,'operation':req.operation,'mode':req.mode,'parity':True,'verification':response.verification})
        if native:
            from .engineering import run
            from .shared import encode
            save(root/'shared-engineering.json',run(kernel.resources))
            for domain,public in probes().items():
                graph=encode(domain,public);req=Request(domain,'transition',{k:graph[k] for k in ('entities','observed','relations')},{'encoding':graph['action']},'experimental_shared')
                response=kernel.run(req);save(root/f'shared-{domain}.json',response.record())
            # Fixed fresh examples, not reruns of scientific holdouts or new benchmarks.
            q=mathematics()[0];req=Request(q.domain,q.operation,q.observation,mode='experimental_neural')
            response=kernel.run(req);save(root/'experimental-math.json',response.record())
            q,context=physics();req=Request(q.domain,q.operation,q.observation,q.action,'experimental_neural')
            response=kernel.run(req,context);save(root/'experimental-physics.json',response.record())
            kernel.resources.close() # Release small models before loading Qwen.
            req,context=coding(repair=True);direct=kernel.direct(req,context);response=kernel.run(req,context)
            if direct['prediction']['value']['repository']!=response.prediction['value']['repository'] or direct['verification']['success']!=response.verification['success']:raise ValueError('Qwen/WASI direct parity')
            save(root/'coding-repair.json',response.record());receipts.append({'domain':'coding','operation':'repair','parity':True,'verification':response.verification})
        final=history_check()
        if initial!=final:raise ValueError('Historical artifacts changed')
        summary={'engineering':'PASS','scope':'native integration' if native else 'CPU tool integration','native_neural_checks':'PASS' if native else 'NOT_RUN','optimizer_steps':0,'scientific_transfer':'NO_GO','routing_is_learning':False,'examples':receipts,'history':final,'neural_load_order':kernel.resources.loads}
        save(root/'summary.json',summary)
    except Exception as e:
        save(root/'failure.json',{'status':'FAIL','type':type(e).__name__,'message':str(e),'optimizer_steps':0});raise
    finally:kernel.close()
    save(root/'MANIFEST.json',{p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name!='MANIFEST.json'})
    print(json.dumps({'output':str(root.resolve()),'engineering':'PASS','native':native,'training':False}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request');parser.add_argument('--verifier');parser.add_argument('--smoke',choices=['cpu','native']);parser.add_argument('--output');parser.add_argument('--capabilities',action='store_true');args=parser.parse_args()
    if args.capabilities:print(json.dumps({'selection':registry()['selection'],'scientific_status':registry()['scientific_status']},indent=2));return
    if args.smoke:smoke(args.smoke=='native',args.output);return
    if not args.request:parser.error('--request, --smoke, or --capabilities required')
    kernel=Kernel()
    try:
        result=kernel.handle(json.loads(Path(args.request).read_text()),json.loads(Path(args.verifier).read_text()) if args.verifier else None)
        if args.output:save(Path(args.output),result)
        else:print(json.dumps(result,indent=2,allow_nan=False))
    finally:kernel.close()
if __name__=='__main__':main()
