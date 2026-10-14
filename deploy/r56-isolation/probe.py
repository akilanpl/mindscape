"""Adversarial boundary probe; contains no evaluator answers and no training code."""
import json
import os
from pathlib import Path
import hashlib
import sys


def denied(path):
    try:
        with open(path,'rb') as stream:stream.read(1)
        return False
    except OSError:return True


def main():
    if len(sys.argv)>1 and sys.argv[1]=='evaluator':
        result={}
        for name,path in [('target','/sealed-evaluation/targets.json'),('checkpoint','/checkpoints/checkpoint.json')]:
            try:
                raw=Path(path).read_bytes()
                result[name+'_readable']=bool(raw)
                result[name+'_sha256']=hashlib.sha256(raw).hexdigest()
            except OSError:result[name+'_readable']=False
        print(json.dumps(result,sort_keys=True));return
    targets=['/sealed-evaluation/targets.json','/evaluation/targets.json']
    try:
        with open('/training/train.json','rb') as stream: readable=bool(stream.read(1))
    except OSError:readable=False
    result={'training_readable':readable,'heldout_denied':all(denied(p) for p in targets),
            'checkpoint_denied':denied('/checkpoints/checkpoint.json'),
            'traversal_denied':denied('/training/../sealed-evaluation/targets.json'),
            'environment_clean':not any(any(token in key.upper() for token in ('TOKEN','SECRET','API_KEY','EVAL')) for key in os.environ),
            'nonroot':os.getuid()!=0}
    link=Path('/tmp/eval-link')
    try:
        link.symlink_to('/sealed-evaluation/targets.json')
        result['symlink_denied']=denied(link)
    except OSError:result['symlink_denied']=False
    try:
        with open('/training/train.json','ab') as stream:stream.write(b'!')
        result['readonly_training']=False
    except OSError:result['readonly_training']=True
    inherited=[]
    for fd in range(3,64):
        try:os.fstat(fd);inherited.append(fd)
        except OSError:pass
    result['inherited_descriptors_clean']=not inherited
    # Linux network-namespace evidence, not a timeout to an unreachable address.
    try:
        interfaces=[p.name for p in Path('/sys/class/net').iterdir()]
        routes=Path('/proc/net/route').read_text().splitlines()[1:]
        nonloopback_routes=[r for r in routes if r.split() and r.split()[0]!='lo']
        result['no_network']=set(interfaces)<= {'lo'} and not nonloopback_routes
        result['network_evidence']={'interfaces':interfaces,'nonloopback_route_count':len(nonloopback_routes)}
    except OSError:result['no_network']=False
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
