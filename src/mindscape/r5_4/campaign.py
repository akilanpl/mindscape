"""Explicitly authorized six-fit MPS campaign. No module-level execution."""
import json
import os
import subprocess
from pathlib import Path
import torch
from .data import ROOT, canonical, digest, qualify, reference, operand, HELD_PAIRS
from .freeze import verify_lock, PROTOCOL
from .model import Transition, adapter
from .engineering import check
from .evaluate import predict_rows, local_diagnostics


def exclusive_json(path, value):
    with path.open('x') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def run(destination, authorized=False):
    if not authorized:
        raise RuntimeError('Explicit human training authorization required')
    verify_lock()
    if qualify()['status'] != 'DATA_READY':
        raise RuntimeError('DATA_NO_GO')
    tasks = json.loads((ROOT / 'train.json').read_text())
    readiness = check(tasks)
    if readiness['status'] != 'ENGINEERING_READY':
        raise RuntimeError(f'ENGINEERING_NO_GO: {readiness}')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)  # never overwrite or resume/retry a partial campaign
    metadata = dict(protocol_sha256=digest(ROOT / 'protocol.json'), lock_sha256=digest(ROOT / 'LOCK.json'),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    engineering=readiness, fits=[], status='RUNNING')
    exclusive_json(destination / 'start.json', metadata)
    evaluation = json.loads((ROOT / 'development.json').read_text()) + json.loads((ROOT / 'ood.json').read_text())
    free, diagnostics, predictors, checkpoints = [], [], [], []
    try:
        from .training import fit
        for kind in PROTOCOL['models']:
            for seed in PROTOCOL['seeds']:
                fit_id = f'r54-{kind}-neural-seed-{seed}-u800'
                torch.manual_seed(seed)
                model = Transition(kind).to('mps')
                optimizer, progress, exposure = fit(model, tasks, seed, destination / f'{fit_id}.progress.jsonl')
                if any((a, b) in HELD_PAIRS for a, b, c in exposure):
                    raise RuntimeError('Held context exposure')
                path = destination / f'{fit_id}.pt'
                payload = dict(model_id=fit_id, kind=kind, seed=seed, updates=800,
                    parameters=sum(p.numel() for p in model.parameters()), model=model.state_dict(),
                    optimizer=optimizer.state_dict(), exposure=exposure,
                    tensor_metadata={k: dict(dtype=str(v.dtype), shape=list(v.shape), device=str(v.device))
                                     for k, v in model.state_dict().items()})
                torch.save(payload, path)
                checkpoints.append(dict(path=path.name, sha256=digest(path), model_id=fit_id, seed=seed, kind=kind))
                model.eval()
                predictors.append((adapter(model), fit_id, 'neural', seed))
                free.extend(predict_rows(evaluation, predictors[-1][0], fit_id, 'neural', seed))
        seen = {tuple(row[:3]): tuple(row[3:]) for task in tasks for row in reference(operand(task))}
        memory = lambda a, b, c, h: (*seen.get((a, b, c), (0, 0)), None)
        predictors.append((memory, 'r54-local-memory', 'local_memory', 0))
        for kind in PROTOCOL['models']:
            zero = Transition(kind).to('mps')
            with torch.no_grad():
                for p in zero.parameters():
                    p.zero_()
            predictors.append((adapter(zero), f'r54-{kind}-zero', 'zero_weight', 0))
        for predictor, model_id, condition, seed in predictors[6:]:
            free.extend(predict_rows(evaluation, predictor, model_id, condition, seed))
        # Whole-problem memory is necessarily a miss because commuted identities do not overlap.
        whole = {tuple(sorted((t['left'], t['right']))): str(int(t['left']) + int(t['right'])) for t in tasks}
        for task in evaluation:
            free.append(dict(task_id=task['task_id'], model_id='r54-whole-memory', condition='whole_memory', seed=0,
                split=task['split'], family=task['family'], horizon=max(len(task['left']), len(task['right'])),
                answer=whole.get(tuple(sorted((task['left'], task['right']))), '0'), trace=[],
                tensor_metadata={'parameter_dtype': 'not_applicable', 'recorded_classes_dtype': 'json_integer',
                                 'trace_shape': [0, 6]}))
        exclusive_json(destination / 'predictions.json', free)
        exclusive_json(destination / 'PREDICTIONS_SEALED.json', dict(
            prediction_sha256=digest(destination / 'predictions.json'), checkpoints=checkpoints,
            outcomes_opened=False, records=len(free)))
        # Only now may diagnostic code use correct incoming carries or scorer open outcomes.
        for predictor, model_id, condition, seed in predictors:
            diagnostics.extend(local_diagnostics(evaluation, predictor, model_id, condition, seed))
        exclusive_json(destination / 'diagnostics.json', diagnostics)
        from .audit import audit
        report = audit(destination, require_complete=False)
        exclusive_json(destination / 'metrics.json', report)
        exclusive_json(destination / 'COMPLETE.json', dict(status='COMPLETE', files={
            p.name: digest(p) for p in destination.iterdir() if p.is_file()}))
        return report
    except Exception as error:
        exclusive_json(destination / 'FAILED.json', dict(status='FAILED_CAMPAIGN', error=repr(error),
                                                       retraining_authorized=False))
        raise
