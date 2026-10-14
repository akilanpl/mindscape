"""Controlled callback workflow; numerical training choices require approved bindings.

This is orchestration preparation, not a complete native runner. Callbacks must
be independently reviewed, source-frozen implementations of the approved loss,
optimizer, label permutation, checkpoint/prediction sealing and scoring rules.
"""
from pathlib import Path
import json

from .readiness import CONDITIONS, DOMAINS, SEEDS, PROPOSAL_FILE_SHA256, digest, validate_manifest, validate_slot

REQUIRED_BINDINGS = {'model', 'source', 'runtime', 'data', 'scorer', 'loss', 'optimizer',
                     'permutation', 'bootstrap', 'schedule', 'callbacks'}


def validate_execution_spec(spec, manifest):
    validate_manifest(manifest)
    if spec.get('proposal_file_sha256') != PROPOSAL_FILE_SHA256:
        raise ValueError('Frozen proposal binding required')
    if spec.get('version') != 'mindscape.alpha-transfer.execution/1':
        raise ValueError('Execution specification version')
    if set(spec.get('bindings', {})) != REQUIRED_BINDINGS:
        raise ValueError('Full approved numerical/runtime freeze required')
    for value in spec['bindings'].values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid approved binding')
    if spec['bindings']['data'] != digest(manifest):
        raise ValueError('Manifest binding')
    approvals = spec.get('approvals', {})
    if set(approvals) != {'data_qualification', 'training'}:
        raise ValueError('Separate human approval references required')
    if any(not isinstance(v, str) or not v.strip() for v in approvals.values()):
        raise ValueError('Empty approval reference')
    if approvals['data_qualification'] == approvals['training']:
        raise ValueError('Distinct stage authorization references required')
    qualification = spec.get('qualification', {})
    if qualification.get('status') != 'DATA_READY' or qualification.get('manifest_sha256') != digest(manifest):
        raise ValueError('Successful bound qualification required')
    if not isinstance(qualification.get('receipt_sha256'), str) or len(qualification['receipt_sha256']) != 64 or any(c not in '0123456789abcdef' for c in qualification['receipt_sha256']):
        raise ValueError('Qualification receipt binding required')
    schedules = spec.get('schedules', {})
    if set(schedules) != {str(s) for s in SEEDS}:
        raise ValueError('Frozen seed schedules required')
    registry = {r['identity']: r['domain'] for r in manifest if r['split'] == 'train'}
    for slots in schedules.values():
        if len(slots) != 240:
            raise ValueError('240 slots required')
        counts = {d: 0 for d in DOMAINS}
        for index, slot in enumerate(slots):
            if set(slot) != {'index', 'domain', 'batch_ids'} or type(slot['index']) is not int or slot['index'] != index:
                raise ValueError('Frozen ordered slot schema')
            if slot['domain'] not in counts or len(slot['batch_ids']) != 16:
                raise ValueError('Domain/batch allocation')
            if any(registry.get(i) != slot['domain'] for i in slot['batch_ids']):
                raise ValueError('Nontraining or wrong-domain supervision')
            counts[slot['domain']] += 1
        if counts != {'coding': 120, 'mathematics': 60, 'physics': 60}:
            raise ValueError('Matched exposure budget')
    if spec['bindings']['schedule'] != digest(schedules):
        raise ValueError('Schedule binding')
    return {'spec_sha256': digest(spec), 'manifest_sha256': digest(manifest)}


def run_controlled(spec, manifest, callbacks, output):
    """Fail closed before initialization; no retries, resume, or automatic scoring.

    Trusted approval references must be checked against human authorization by
    the caller. This function cannot authenticate the human behind a string.
    verify_bindings must independently resolve every hash before any model is
    made. step must use route_core and return all seven group receipts.
    """
    binding = validate_execution_spec(spec, manifest)
    if callbacks.verify_bindings(spec, manifest) is not True:
        raise ValueError('Independent execution bindings rejected')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {**binding, 'status': 'STARTED', 'fits': [], 'scoring_started': False}
    ledger = output / 'controller.json'

    def save():
        ledger.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')

    save()
    try:
        # Compare full initial tensors BEFORE any fit receives an update.
        contexts = {}
        for seed in SEEDS:
            initial = []
            for condition in CONDITIONS:
                context, state_hash = callbacks.initialize(seed, condition, spec)
                contexts[seed, condition] = context
                initial.append(state_hash)
            if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in initial) or len(set(initial)) != 1:
                raise ValueError('Initial states differ before training')
        seals = []
        for seed in SEEDS:
            for condition in CONDITIONS:
                context = contexts[seed, condition]
                fit = {'seed': seed, 'condition': condition, 'slots_completed': 0}
                receipt['fits'].append(fit)
                for slot in spec['schedules'][str(seed)]:
                    groups = callbacks.step(context, condition, slot, spec)
                    validate_slot(groups, condition, slot['domain'])
                    callbacks.persist_slot(seed, condition, slot, groups, output)
                    fit['slots_completed'] += 1
                    save()
                seal = callbacks.seal_final(context, seed, condition, 240, spec, output)
                if seal.get('sealed') is not True or type(seal.get('update')) is not int or seal.get('update') != 240:
                    raise ValueError('Final checkpoint/prediction seal missing')
                seals.append(seal)
        if len(seals) != 9 or callbacks.verify_all_seals(seals, spec) is not True:
            raise ValueError('Nine independent prediction/checkpoint seals required')
        # Independent executors may open composition outcomes only here.
        receipt['scoring_started'] = True
        save()
        receipt['audit'] = callbacks.independent_evaluate(seals, spec, manifest, output)
        receipt['status'] = 'COMPLETE_PENDING_INDEPENDENT_VERDICT'
        save()
        return receipt
    except BaseException:
        receipt['status'] = 'FAILED_CAMPAIGN'
        save()
        raise
