"""Nontraining supplemental diagnostics and deployment capability evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
import torch
from mindscape.r56.identifiability import (ConceptProbe,DelayedCueProbe,concept_curriculum,
    delayed_cue_curriculum,pack_concept,pack_memory,supplemental_protocol,
    conditional_information_audit,delayed_cue_variants)
from mindscape.r56.benchmarks import model_accounting
from mindscape.r56.scope_studies import freeze_supplement


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    def forbidden(*a,**k):raise RuntimeError('Optimizer forbidden in remediation diagnostic')
    for name in ('Adam','AdamW','SGD','RMSprop','Adagrad'):setattr(torch.optim,name,forbidden)
    torch.manual_seed(17);concept=ConceptProbe();memory=DelayedCueProbe()
    concept_rows=concept_curriculum();episodes=delayed_cue_curriculum();diagnostics=[]
    for control in ('embedding','context-vector','omitted','independent-corruption'):
        before={k:v.clone() for k,v in concept.state_dict().items()};times=[];active=[]
        for index in range(4):
            row=concept_rows[index];concept.zero_grad(set_to_none=True)
            observed=pack_concept(row['observations'],omitted=control=='omitted',
                context_id=row['corrupted_context_id'] if control=='independent-corruption' else None)
            began=time.perf_counter();result=concept(observed,representation=control if control in ('embedding','context-vector') else 'context-vector')
            torch.nn.functional.cross_entropy(result[None],torch.tensor([row['labels']['class']])).backward()
            times.append(time.perf_counter()-began)
            active.append(sum(p.numel() for p in concept.parameters() if p.grad is not None and p.grad.abs().sum()>0))
        assert all(torch.equal(v,concept.state_dict()[k]) for k,v in before.items())
        diagnostics.append({'study':'concept','control':control,**model_accounting(concept),'nonzero_gradient_parameter_counts':active,
            'forward_backward_seconds':times,'optimizer_updates':0,'weights_unchanged':True})
    for control in ('retain','reset','shuffled-past'):
        before={k:v.clone() for k,v in memory.state_dict().items()};times=[];active=[]
        for ep in episodes[:4]:
            memory.zero_grad(set_to_none=True);prior=None;began=time.perf_counter()
            for obs in ep['observations']:
                out=memory.step(pack_memory(obs),prior,control=control,
                    shuffled_past=torch.zeros(memory.width) if control=='shuffled-past' else None)
                prior=out['memory']
            torch.nn.functional.cross_entropy(out['logits'][None],torch.tensor([ep['labels']['query_class']])).backward()
            times.append(time.perf_counter()-began)
            active.append(sum(p.numel() for p in memory.parameters() if p.grad is not None and p.grad.abs().sum()>0))
        assert all(torch.equal(v,memory.state_dict()[k]) for k,v in before.items())
        diagnostics.append({'study':'memory','control':control,**model_accounting(memory),
            'nonzero_gradient_parameter_counts':active,'forward_backward_seconds':times,
            'optimizer_updates':0,'weights_unchanged':True,
            'limits':'Shuffled diagnostic uses zero independent donor, not learned independent-episode donor distribution.'})
    train=concept_curriculum(split='train');held=concept_curriculum(split='heldout')
    def state_set(rows):return {tuple(r['observations']['state']) for r in rows}
    supplement={'protocol':supplemental_protocol(),'concept_train':train,'concept_heldout':held,
                'memory_episodes':episodes,'delayed_variants':{'train':delayed_cue_variants('train'),'heldout':delayed_cue_variants('heldout')},
                'concept_information':conditional_information_audit(train),'primary_grid_changed':False,
                'state_split_overlap':len(state_set(train)&state_set(held))}
    (output/'supplement.json').write_text(json.dumps(supplement,indent=2))
    isolation={'docker':shutil.which('docker'),'podman':shutil.which('podman'),
               'host_os':platform.system(),'demonstrated':False}
    if shutil.which('sandbox-exec'):
        probe=subprocess.run(['/usr/bin/sandbox-exec','-p','(version 1)(deny default)(allow process-exec)(allow file-read*)','/usr/bin/true'],capture_output=True,text=True,close_fds=True)
        isolation.update(sandbox_exec_exit=probe.returncode,sandbox_exec_stderr=probe.stderr.strip())
    (output/'receipts.json').write_text(json.dumps({'torch':torch.__version__,'python':platform.python_version(),
        'threads':1,'diagnostics':diagnostics,'isolation':isolation,'decision':'NO_GO',
        'note':'Random forward/backward diagnostics only; no fitting and no compute-equality assertion.'},indent=2))
    freeze_supplement(output/'semantics-proposal')
    locks={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    (output/'artifact-locks.json').write_text(json.dumps(locks,indent=2))
    print(json.dumps({'decision':'NO_GO','optimizer_updates':0,'supplemental_study_fits_authorized':0,'isolation':isolation}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
