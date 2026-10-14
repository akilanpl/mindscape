"""Fail-closed receipt validation; never generates tasks or invokes an optimizer.

The frozen proposal still requires approval and a full loss/optimizer/data freeze.
These functions describe and inspect evidence, and cannot authorize a campaign.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math

DOMAINS = ('mathematics', 'coding', 'physics')
CONDITIONS = ('domain_only', 'shared', 'randomized_auxiliary')
SEEDS = (11, 23, 47)
PROPOSAL_FILE_SHA256 = 'a49a05def8cf588e82e0aa62fd32f988266fac3917cc06b8e3c61ca8b05adc64'
PROPOSAL_CONTENT_SHA256 = 'f25a42eae773d0d11e89e77088ae8fc85a6de9135670659da6ebc1fce9b11eca'
GATES = {
    'coding_relative_changed_mse_gain_over_both_controls': .1,
    'all_three_seed_gains_positive': True,
    'paired_scene_or_task_bootstrap95_lower_gain_positive': True,
    'minimum_action_validity_accuracy': .9,
    'maximum_whole_graph_regression_fraction': .05,
    'maximum_other_domain_changed_mse_regression_fraction': .05,
}


@contextmanager
def route_core(model, condition, domain):
    """Future runner helper: freeze tensors, never detach the core computation.

    Calling this does not execute a forward, backward or optimizer step. The
    caller must first zero ALL gradients with set_to_none=True and must persist
    per-slot receipts. Stale core gradients are cleared even on auxiliary slots.
    """
    if condition not in CONDITIONS or domain not in DOMAINS:
        raise ValueError('Unknown condition/domain')
    parameters = list(model.core.parameters())
    original = [p.requires_grad for p in parameters]
    enabled = condition != 'domain_only' or domain == 'coding'
    try:
        for p in parameters:
            p.grad = None
            p.requires_grad_(enabled)
        yield
    finally:
        for p, flag in zip(parameters, original, strict=True):
            p.requires_grad_(flag)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_proposal(proposal):
    if digest(proposal) != PROPOSAL_CONTENT_SHA256:
        raise ValueError('Full frozen proposal content differs')
    expected = {'seeds': list(SEEDS), 'target': 'coding', 'fits': 9,
                'target_updates_per_fit': 120, 'auxiliary_updates_per_fit': 120,
                'auxiliary_updates_each_domain': 60, 'batch_size': 16,
                'target_presentations_per_fit': 1920, 'total_update_slots': 2160,
                'maximum_total_fits': 9}
    if proposal['budget'] != expected or proposal['gates'] != GATES:
        raise ValueError('Frozen budget or gates differ')
    if proposal['status'] != 'FROZEN_PROPOSAL_ONLY_REQUIRES_HUMAN_AUTHORIZATION':
        raise ValueError('Proposal status changed')
    if set(proposal['conditions']) != set(CONDITIONS):
        raise ValueError('Conditions differ')
    return {'proposal_sha256': digest(proposal), 'scientific_status': 'UNRUN',
            'execution_authorized': False}


def prepare(proposal_path, repository):
    """Read-only preparation, including exact bytes and independent executor source."""
    from pathlib import Path
    raw = Path(proposal_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROPOSAL_FILE_SHA256:
        raise ValueError('Exact committed proposal bytes differ')
    allocation = plan(json.loads(raw))
    paths = ('src/mindscape/v5/execute.py', 'src/mindscape/r5_1/metrics.py',
             'src/mindscape/v3/coding_learning/execution.py')
    bindings = {p: hashlib.sha256((Path(repository) / p).read_bytes()).hexdigest()
                for p in paths}
    return {**allocation, 'executor_source_bindings': bindings,
            'independent_evaluators': {'mathematics': 'bounded_register_interpreter',
                                       'coding': 'actual_WASI_and_AST_replay',
                                       'physics': 'pinned_Pymunk'},
            'missing_execution_freeze': ['data_manifest', 'loss', 'optimizer',
                                        'permutation', 'bootstrap', 'runtime'],
            'tasks_executed': 0}


def plan(proposal):
    """Reviewable proposed allocation; exact batch order needs a later approved freeze."""
    receipt = validate_proposal(proposal)
    # Auxiliary slots first, then target; same allocation in all three conditions.
    slots = [DOMAINS[0] if i % 2 == 0 else DOMAINS[2] for i in range(120)]
    slots += ['coding'] * 120
    return {**receipt, 'allocation_status': 'PROPOSED_REQUIRES_EXECUTION_FREEZE',
            'fits': [{'seed': seed, 'condition': condition, 'domains': slots[:],
                      'core_enabled': [condition != 'domain_only' or d == 'coding'
                                       for d in slots]}
                     for seed in SEEDS for condition in CONDITIONS]}


def validate_manifest(rows, historical_ids=()):
    """Inspect a future approved manifest, without executing or opening labels."""
    identities = set()
    counts = Counter()
    historical = set(historical_ids)
    for row in rows:
        if set(row) != {'identity', 'domain', 'split', 'public_sha256'}:
            raise ValueError('Public manifest only; unexpected fields or labels')
        identity = row['identity']
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError('Duplicate or invalid identity')
        if identity in historical:
            raise ValueError('Historical identity reused')
        identities.add(identity)
        if row['domain'] not in DOMAINS or row['split'] not in ('train', 'qualification', 'composition'):
            raise ValueError('Unknown domain or split')
        sha = row['public_sha256']
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid public graph digest')
        counts[row['domain'], row['split']] += 1
    for d in DOMAINS:
        for split, n in (('train', 24), ('qualification', 12), ('composition', 24)):
            if counts[d, split] != n:
                raise ValueError('Frozen task denominators differ')
    # Distinct IDs alone do not establish content disjointness.
    if len({r['public_sha256'] for r in rows}) != len(rows):
        raise ValueError('Repeated public content across identities')
    return {'tasks': len(rows), 'manifest_sha256': digest(rows)}


def validate_slot(receipt, condition, domain):
    """Require all seven parameter groups, including inactive domains and Adam state."""
    if condition not in CONDITIONS or domain not in DOMAINS:
        raise ValueError('Unknown condition/domain')
    groups = {'core'} | {f'{kind}.{d}' for kind in ('encoder', 'head') for d in DOMAINS}
    if set(receipt) != groups:
        raise ValueError('Incomplete parameter-group receipt')
    for group, r in receipt.items():
        if set(r) != {'gradient', 'parameters_changed', 'optimizer_changed'}:
            raise ValueError('Incomplete gradient/optimizer receipt')
        active = group == 'core' or group.endswith('.' + domain)
        frozen = group == 'core' and condition == 'domain_only' and domain != 'coding'
        if not active or frozen:
            if r['gradient'] is not None or r['parameters_changed'] is not False or r['optimizer_changed'] is not False:
                raise ValueError('Inactive/frozen group changed, including optimizer moments')
        else:
            norm = r['gradient']
            if isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(norm) or norm <= 0:
                raise ValueError('Missing finite nonzero active gradient')
            if r['parameters_changed'] is not True or r['optimizer_changed'] is not True:
                raise ValueError('Active group update absent')
    return True


def validate_campaign(records, train_ids, manifest):
    """Validate future receipts; this is integrity only, never scientific PASS."""
    expected = {(s, c) for s in SEEDS for c in CONDITIONS}
    if len(records) != 9 or {(r['seed'], r['condition']) for r in records} != expected:
        raise ValueError('Nine unique frozen fits required')
    # Mapping binds each approved training identity to its domain.
    allowed = dict(train_ids)
    validate_manifest(manifest)
    expected_registry = {r['identity']: r['domain'] for r in manifest if r['split'] == 'train'}
    if allowed != expected_registry:
        raise ValueError('Training registry differs from frozen manifest')
    for seed in SEEDS:
        matched = [r for r in records if r['seed'] == seed]
        if any(r['manifest_sha256'] != digest(manifest) for r in matched):
            raise ValueError('Campaign manifest digest mismatch')
        for key in ('initial_state_sha256', 'optimizer_spec_sha256', 'loss_spec_sha256',
                    'source_sha256', 'manifest_sha256'):
            for r in matched:
                value = r[key]
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Invalid freeze/state digest')
            if len({r[key] for r in matched}) != 1:
                raise ValueError('Unmatched initialization or execution freeze')
        reference = matched[0]['slots']
        for r in matched:
            slots = r['slots']
            if len(slots) != 240 or Counter(x['domain'] for x in slots) != Counter({'coding': 120, 'mathematics': 60, 'physics': 60}):
                raise ValueError('Matched update allocation violated')
            for index, slot in enumerate(slots):
                if type(slot['index']) is not int or slot['index'] != index or len(slot['batch_ids']) != 16:
                    raise ValueError('Slot order or batch budget differs')
                if not set(slot['batch_ids']) <= set(allowed):
                    raise ValueError('Nontraining identity entered optimization')
                if any(allowed[i] != slot['domain'] for i in slot['batch_ids']):
                    raise ValueError('Training identity assigned to wrong domain')
                if (slot['domain'], slot['batch_ids']) != (reference[index]['domain'], reference[index]['batch_ids']):
                    raise ValueError('Conditions received different minibatches/order')
                validate_slot(slot['groups'], r['condition'], slot['domain'])
            if type(r['final_checkpoint_update']) is not int or r['final_checkpoint_update'] != 240:
                raise ValueError('Final-checkpoint boundary violated')
    return {'status': 'RECEIPT_INTEGRITY_ONLY', 'fits': 9, 'update_slots': 2160,
            'scientific_pass': False}
