"""Concrete native execution candidate, with custody before any protected work."""
from pathlib import Path
from ..benchmark import digest
from .adapter import fit_native, predict_task, SPEC
from .custody import require_access, qualify
from .evaluation import CONDITIONS, SEEDS, baseline_row, training_baselines, seal_predictions


def execute_native(*,tasks,traces,historical_ids,certificate,access,evaluation_access,binding,trusted_keys,
                   checkpoints:Path,seal_path:Path):
    """Only future externally approved campaign. No generation or label reads.

    Procedural references are released only after the four-condition prediction
    seal, in the independent scorer, and never enter this execution function.
    Validation remains untouched; final checkpoint is selected by update count.
    """
    require_access(access,'fit',binding)
    require_access(evaluation_access,'evaluate',binding)
    qualify(tasks,historical_ids,certificate,binding,trusted_keys)
    train=tuple(t for t in tasks if t.split=='train')
    ood=tuple(t for t in tasks if t.split=='ood')
    memory,table=training_baselines(train,traces)
    if digest(traces)!=binding['train_demonstration_hash']:
        raise ValueError('baseline demonstration substitution')
    # Reserve all output names before first fit. Existing evidence cannot be
    # overwritten, and partial failure is not an implicit retry permission.
    for condition in ('recurrent','feedforward'):
        for seed in SEEDS:
            if (checkpoints/f'{condition}-s{seed}.pt').exists():
                raise FileExistsError('prior checkpoint; recovery needs new review')
    if seal_path.exists(): raise FileExistsError('prior seal')
    rows=[]; receipts=[]
    try:
        for condition in ('recurrent','feedforward'):
            for seed in SEEDS:
                model,receipt=fit_native(tasks=train,traces=traces,kind=condition,seed=seed,
                    bindings=binding,authorization=access,checkpoint=checkpoints/f'{condition}-s{seed}.pt')
                receipts.append(receipt)
                for task in ood:
                    rows.append({'condition':condition,'seed':seed,**predict_task(task,model,'mps',access=evaluation_access,bindings=binding)})
        for condition in ('memorization','transition_lookup'):
            for seed in SEEDS:
                rows.extend(baseline_row(t,condition,seed,memory,table) for t in ood)
        seal=seal_predictions(ood,rows,binding,seal_path)
    except BaseException as exc:
        import json
        failure={'status':'FAILED_CAMPAIGN_NO_RETRY','binding':binding,'completed_fits':receipts,
                 'partial_raw_predictions':rows,'failure':{'type':type(exc).__name__,'message':str(exc)}}
        with seal_path.with_suffix('.failure.json').open('x') as stream:
            json.dump(failure,stream,sort_keys=True,allow_nan=False)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        raise
    return {'seal':seal,'fits':receipts,'scientific_status':'FROZEN_PREDICTIONS_SEALED_UNVERIFIED'}
