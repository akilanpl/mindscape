"""Verify completed native smoke evidence without neural re-forward or optimization."""
import argparse,json
from pathlib import Path
from mindscape.r5_kernel.evidence import sha,history_check
from mindscape.r5_kernel.contract import Request,digest

def verify(root):
    root=Path(root);manifest=json.loads((root/'MANIFEST.json').read_text());summary=json.loads((root/'summary.json').read_text())
    for file,value in manifest.items():
        if sha(root/file)!=value:raise ValueError('Smoke artifact changed: '+file)
    if summary['optimizer_steps']!=0 or summary['scientific_transfer']!='NO_GO':raise ValueError('Unexpected training/transfer receipt')
    for path in root.glob('*.json'):
        value=json.loads(path.read_text())
        if value.get('version')!='mindscape.kernel/1':continue
        request=Request(value['domain'],value['state']['operation'],value['state']['value'],value['action'],value['state']['mode'])
        if request.identity!=value['request_id']:raise ValueError('Request identity mismatch')
        if digest(value['state']['value'])!=value['state']['observation_hash']:raise ValueError('Observation hash mismatch')
        for stage,key in [('state','state'),('action','action'),('prediction','prediction'),('execution','execution'),('verification','verification')]:
            saved=next(e['value_hash'] for e in value['trace'] if e['stage']==stage)
            if digest(value[key])!=saved:raise ValueError('Trace hash mismatch')
        if value['provenance']['learned_transfer'] or value['provenance']['optimizer_steps']!=0:raise ValueError('False transfer/training provenance')
    if summary['native_neural_checks']=='PASS':
        engineering=json.loads((root/'shared-engineering.json').read_text())
        if not engineering['weights_unchanged'] or engineering['optimizer_steps']!=0:raise ValueError('Engineering mutated model')
        shared=[json.loads((root/f'shared-{d}.json').read_text()) for d in ('mathematics','coding','physics')]
        if len({tuple(v['provenance']['core_parameter_ids']) for v in shared})!=1 or len({v['provenance']['checkpoint_sha256'] for v in shared})!=1:raise ValueError('Core was not shared across domains')
    return {'status':'PASS','files':len(manifest),'history':history_check(),'neural_re_forward':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();print(json.dumps(verify(args.root),indent=2))
