"""Synthetic callback events only: no model, optimizer, executor or labels."""
import copy
from pathlib import Path
import tempfile
import unittest

from mindscape.alpha_transfer.controller import REQUIRED_BINDINGS, run_controlled
from mindscape.alpha_transfer.readiness import PROPOSAL_FILE_SHA256, SEEDS, digest
from test_readiness import groups, synthetic_manifest


def specification(manifest):
    schedules = {str(s): [{'index': i, 'domain': d, 'batch_ids': [f'{d}-train-0'] * 16}
                         for i, d in enumerate(['mathematics', 'physics'] * 60 + ['coding'] * 120)]
                 for s in SEEDS}
    bindings = {k: 'a' * 64 for k in REQUIRED_BINDINGS}
    bindings.update(data=digest(manifest), schedule=digest(schedules))
    return {'version': 'mindscape.alpha-transfer.execution/1', 'bindings': bindings,
            'proposal_file_sha256': PROPOSAL_FILE_SHA256,
            'approvals': {'data_qualification': 'synthetic-data-reference',
                          'training': 'synthetic-training-reference'},
            'qualification': {'status': 'DATA_READY', 'manifest_sha256': digest(manifest),
                              'receipt_sha256': 'b' * 64}, 'schedules': schedules}


class FakeCallbacks:
    def __init__(self):
        self.initializations = 0
        self.slots = 0
        self.seals = 0
    def verify_bindings(self, spec, manifest):
        return True
    def initialize(self, seed, condition, spec):
        self.initializations += 1
        return object(), f'{seed:064x}'
    def step(self, context, condition, slot, spec):
        assert self.initializations == 9
        self.slots += 1
        return groups(condition, slot['domain'])
    def persist_slot(self, *args):
        pass
    def seal_final(self, *args):
        self.seals += 1
        return {'sealed': True, 'update': 240}
    def verify_all_seals(self, seals, spec):
        return self.seals == 9
    def independent_evaluate(self, *args):
        assert self.seals == 9
        assert self.slots == 2160
        return {'status': 'SYNTHETIC_ONLY', 'optimizer_steps': 0}


class ControllerTests(unittest.TestCase):
    def test_all_prediction_seals_precede_scoring_and_no_retry(self):
        manifest = synthetic_manifest()
        spec = specification(manifest)
        callbacks = FakeCallbacks()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'campaign'
            result = run_controlled(spec, manifest, callbacks, output)
            self.assertEqual(result['status'], 'COMPLETE_PENDING_INDEPENDENT_VERDICT')
            self.assertEqual(result['audit']['optimizer_steps'], 0)
            with self.assertRaises(FileExistsError):
                run_controlled(spec, manifest, callbacks, output)
            self.assertEqual(callbacks.initializations, 9)

    def test_all_gates_precede_initialization(self):
        manifest = synthetic_manifest()
        spec = specification(manifest)
        for mutate in (
            lambda x: x['approvals'].pop('training'),
            lambda x: x['qualification'].__setitem__('status', 'DATA_NO_GO'),
            lambda x: x['bindings'].pop('loss'),
            lambda x: x['qualification'].__setitem__('receipt_sha256', 'z' * 64),
            lambda x: x.__setitem__('proposal_file_sha256', 'a' * 64),
            lambda x: x['schedules']['11'][0]['batch_ids'].__setitem__(0, 'heldout'),
        ):
            bad = copy.deepcopy(spec)
            mutate(bad)
            callbacks = FakeCallbacks()
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    run_controlled(bad, manifest, callbacks, Path(temp) / 'campaign')
            self.assertEqual(callbacks.initializations, 0)

    def test_initial_mismatch_and_failed_slot_preserved(self):
        manifest = synthetic_manifest()
        spec = specification(manifest)
        callbacks = FakeCallbacks()
        callbacks.initialize = lambda seed, condition, spec: (object(), digest(condition))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'campaign'
            with self.assertRaises(ValueError):
                run_controlled(spec, manifest, callbacks, output)
            self.assertIn('FAILED_CAMPAIGN', (output / 'controller.json').read_text())
            self.assertEqual(callbacks.slots, 0)


if __name__ == '__main__':
    unittest.main()
