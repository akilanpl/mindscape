"""Synthetic control evidence only; zero optimizer steps and no scientific tasks."""
import copy
import json
from pathlib import Path
import unittest

from mindscape.alpha_transfer.readiness import (
    CONDITIONS, DOMAINS, SEEDS, digest, plan, prepare, route_core, validate_campaign, validate_manifest, validate_slot,
)

ROOT = Path(__file__).resolve().parents[1]


def groups(condition, domain):
    result = {}
    for g in ['core'] + [f'{kind}.{d}' for kind in ('encoder', 'head') for d in DOMAINS]:
        active = (g == 'core' and (condition != 'domain_only' or domain == 'coding')) or g.endswith('.' + domain)
        result[g] = {'gradient': 1.0 if active else None,
                     'parameters_changed': active, 'optimizer_changed': active}
    return result


def synthetic_campaign():
    records = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            domains = ['mathematics', 'physics'] * 60 + ['coding'] * 120
            records.append({'seed': seed, 'condition': condition,
                            **{k: f'{seed:064x}' for k in ('initial_state_sha256', 'optimizer_spec_sha256',
                               'loss_spec_sha256', 'source_sha256', 'manifest_sha256')},
                            'final_checkpoint_update': 240,
                            'slots': [{'index': i, 'domain': d, 'batch_ids': [f'{d}-train-0'] * 16,
                                       'groups': groups(condition, d)} for i, d in enumerate(domains)]})
    return records


def synthetic_manifest():
    return [{'identity': f'{d}-{split}-{i}', 'domain': d, 'split': split,
                 'public_sha256': f'{index:064x}'}
                for index, (d, split, i) in enumerate(
                    (d, split, i) for d in DOMAINS
                    for split, n in [('train', 24), ('qualification', 12), ('composition', 24)]
                    for i in range(n))]


class ReadinessTests(unittest.TestCase):
    def test_routing_restores_flags_and_clears_stale_core_gradients(self):
        class Parameter:
            requires_grad = True
            grad = 'stale'
            def requires_grad_(self, flag):
                self.requires_grad = flag
        class Core:
            def __init__(self):
                self.p = Parameter()
            def parameters(self):
                return [self.p]
        class Model:
            core = Core()
        m = Model()
        with self.assertRaises(RuntimeError):
            with route_core(m, 'domain_only', 'physics'):
                self.assertFalse(m.core.p.requires_grad)
                self.assertIsNone(m.core.p.grad)
                raise RuntimeError('synthetic failure')
        self.assertTrue(m.core.p.requires_grad)
        with route_core(m, 'shared', 'physics'):
            self.assertTrue(m.core.p.requires_grad)

    def test_frozen_plan(self):
        proposal = json.loads((ROOT / 'configs/r5_kernel/shared-study-proposal.json').read_text())
        p = plan(proposal)
        self.assertEqual(len(p['fits']), 9)
        self.assertFalse(p['execution_authorized'])
        self.assertEqual(prepare(ROOT / 'configs/r5_kernel/shared-study-proposal.json', ROOT)['tasks_executed'], 0)
        self.assertEqual(p['scientific_status'], 'UNRUN')
        proposal['gates']['coding_relative_changed_mse_gain_over_both_controls'] = .09
        with self.assertRaises(ValueError):
            plan(proposal)

    def test_routing_requires_no_core_moments_on_auxiliary(self):
        r = groups('domain_only', 'physics')
        self.assertTrue(validate_slot(r, 'domain_only', 'physics'))
        for key, value in [('gradient', 0.0), ('parameters_changed', True), ('optimizer_changed', True), ('optimizer_changed', 0)]:
            bad = copy.deepcopy(r)
            bad['core'][key] = value
            with self.assertRaises(ValueError):
                validate_slot(bad, 'domain_only', 'physics')

    def test_inactive_and_active_groups(self):
        r = groups('shared', 'coding')
        r['head.physics']['optimizer_changed'] = True
        with self.assertRaises(ValueError):
            validate_slot(r, 'shared', 'coding')
        r = groups('shared', 'coding')
        r['encoder.coding']['gradient'] = float('nan')
        with self.assertRaises(ValueError):
            validate_slot(r, 'shared', 'coding')

    def test_complete_matched_receipts_and_confounds(self):
        r = synthetic_campaign()
        manifest = synthetic_manifest()
        ids = {r['identity']: r['domain'] for r in manifest if r['split'] == 'train'}
        for record in r:
            record['manifest_sha256'] = digest(manifest)
        self.assertEqual(validate_campaign(r, ids, manifest)['update_slots'], 2160)
        with self.assertRaises(ValueError):
            validate_campaign(r, {'arbitrary': 'coding'}, manifest)
        for mutate in (
            lambda x: x[0]['slots'][0]['batch_ids'].__setitem__(0, 'heldout'),
            lambda x: x[0]['slots'][0]['batch_ids'].__setitem__(0, 'coding-train-0'),
            lambda x: x[0].__setitem__('final_checkpoint_update', 240.0),
            lambda x: x[0]['slots'][0].__setitem__('index', 0.0),
            lambda x: x[0].__setitem__('initial_state_sha256', 'different'),
            lambda x: x[0].__setitem__('final_checkpoint_update', 120),
            lambda x: x[0]['slots'].pop(),
            lambda x: x.__setitem__(0, copy.deepcopy(x[1])),
        ):
            bad = copy.deepcopy(r)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_campaign(bad, ids, manifest)

    def test_public_manifest_only_disjoint_identity_and_content(self):
        rows = synthetic_manifest()
        self.assertEqual(validate_manifest(rows)['tasks'], 180)
        for mutate in (
            lambda x: x[0].__setitem__('target', {'valid': True}),
            lambda x: x[0].__setitem__('public_sha256', x[1]['public_sha256']),
            lambda x: x[0].__setitem__('identity', x[1]['identity']),
            lambda x: x.pop(),
        ):
            bad = copy.deepcopy(rows)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_manifest(bad)
        with self.assertRaises(ValueError):
            validate_manifest(rows, [rows[0]['identity']])


if __name__ == '__main__':
    unittest.main()
