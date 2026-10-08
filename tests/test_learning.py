import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Install optional learning dependencies")
class LearningTests(unittest.TestCase):
    def setUp(self):
        from mindscape.data.generation import generate
        self.config = {"environment": "integer_multiplication", "seed": 42, "budgets": [5],
            "splits": {"train": {"count": 10, "structures": ["2x1"]},
                       "validation": {"count": 4, "structures": ["2x1"]},
                       "test": {"count": 4, "structures": ["2x1"]}}}
        self.splits = generate(self.config)

    def test_backend_dimensions_and_interface(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.encoding import INPUT_SIZE
        backend = NumpyMLP([4], hidden=8, seed=1)
        self.assertEqual(backend.logits(np.zeros((3, INPUT_SIZE))).shape, (3, 4))
        self.assertEqual(backend.generate(np.zeros((3, INPUT_SIZE))).shape, (3, 1))
        self.assertEqual(backend.parameter_count, INPUT_SIZE * 8 + 8 + 8 * 4 + 4)

    def test_gradient_numerical_check(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.encoding import INPUT_SIZE
        backend = NumpyMLP([3, 2], hidden=3, seed=1)
        x = np.random.default_rng(2).normal(size=(4, INPUT_SIZE))
        labels = np.array([[0, 0], [1, 1], [2, 0], [1, 0]])
        _, gradients = backend.loss_and_gradients(x, labels)
        for key, index in [("w1", (0, 0)), ("b1", (0,)), ("w2", (0, 0)), ("b2", (0,))]:
            original = backend.weights[key][index]
            backend.weights[key][index] = original + 1e-5
            plus, _ = backend.loss_and_gradients(x, labels)
            backend.weights[key][index] = original - 1e-5
            minus, _ = backend.loss_and_gradients(x, labels)
            backend.weights[key][index] = original
            self.assertAlmostEqual(gradients[key][index], (plus - minus) / 2e-5, places=6)

    def test_optimizer_learns_and_reproduces(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.encoding import INPUT_SIZE
        from mindscape.training.optimizer import fit
        x = np.zeros((4, INPUT_SIZE)); x[:, 0] = [-1, -1, 1, 1]
        y = np.array([[0], [0], [1], [1]])
        a, b = NumpyMLP([2], 8, 3), NumpyMLP([2], 8, 3)
        initial, _ = a.loss_and_gradients(x, y)
        for backend in [a, b]:
            fit(backend, x, y, x, y, steps=100, learning_rate=.02, validation_every=20)
        final, _ = a.loss_and_gradients(x, y)
        self.assertLess(final, initial)
        np.testing.assert_array_equal(a.generate(x), y)
        for key in a.weights:
            np.testing.assert_array_equal(a.weights[key], b.weights[key])

    def test_baseline_adapter_ignores_trace(self):
        from mindscape.training.adapters import arrays
        original = arrays(self.splits['train'], 'baseline')
        poisoned = copy.deepcopy(self.splits['train'])
        for example in poisoned:
            example.target_trajectory[:] = [{}]
        modified = arrays(poisoned, 'baseline')
        for a, b in zip(original, modified):
            np.testing.assert_array_equal(a, b)

    def test_policy_adapter_and_feature_isolation(self):
        from mindscape.training.adapters import arrays
        from mindscape.models.encoding import features
        x, y = arrays(self.splits['train'], 'mindscape')
        self.assertEqual(len(x), sum(len(e.target_trajectory) for e in self.splits['train']))
        self.assertEqual(set(y.flatten()), {0, 1, 2, 3})
        e = self.splits['train'][0]
        s = copy.deepcopy(e.initial_state)
        a = features(e.observation['operands'], s)
        s['phase'] = 'finish'
        np.testing.assert_array_equal(a, features(e.observation['operands'], s))

    def test_serialization_roundtrip(self):
        from mindscape.core.serialization import decode
        from mindscape.core.schema import State, Goal, Action
        from mindscape.models.encoding import serialize_state, FORMAT_VERSION
        e = self.splits['train'][0]
        state = decode(State, e.initial_state)
        representation = serialize_state(e.problem, state, decode(Goal, e.goal), [Action('multiply', 0, 0)])
        restored = json.loads(json.dumps(representation))
        self.assertEqual(restored['format_version'], FORMAT_VERSION)
        self.assertEqual(decode(State, restored['state']), state)
        self.assertIsNone(restored['previous_action'])

    def test_masking_and_invalid_unmasked_action(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.learned import MindscapeModel
        backend = NumpyMLP([4], 8)
        backend.weights['w2'][:] = 0
        backend.weights['b2'][:] = [0, 0, 0, 10]
        e = self.splits['test'][0]
        masked = MindscapeModel(backend)
        result = masked.predict(e.view('experiential'))
        self.assertIsNone(result.error_type)
        self.assertEqual(result.answer, e.target_answer)
        self.assertTrue(result.diagnostics['decisions'][0]['masked_correction'])
        raw = MindscapeModel(backend, mask=False).predict(e.view('experiential'))
        self.assertEqual(raw.error_type, 'invalid_action')
        self.assertIsNone(raw.answer)

    def test_working_memory_reset(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.learned import MindscapeModel
        model = MindscapeModel(NumpyMLP([4], 8))
        a, b = self.splits['test'][:2]
        model.predict(a.view('experiential'))
        first_actions = model.memory.action_history
        model.predict(b.view('experiential'))
        self.assertIsNot(model.memory.action_history, first_actions)
        self.assertEqual(len(model.memory.action_history), len(b.target_trajectory))
        self.assertEqual(model.memory.current_state.answer, b.target_answer)
        self.assertEqual(model.memory.current_observation['actual_result']['kind'], 'actual_result')

    def test_episode_trajectory_and_verifier(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.learned import MindscapeModel
        from mindscape.core.serialization import decode
        from mindscape.core.schema import Trajectory
        from mindscape.verification.verifier import verify
        e = self.splits['test'][0]
        p = MindscapeModel(NumpyMLP([4], 8)).predict(e.view('experiential'))
        self.assertTrue(verify(decode(Trajectory, p.trajectory), p.answer).verified)
        self.assertTrue(all(d['transition_valid'] for d in p.diagnostics['decisions']))
        self.assertEqual(p.model_calls, len(p.trajectory['transitions']))

    def test_step_and_time_limits(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.learned import MindscapeModel
        view = self.splits['test'][0].view('experiential')
        self.assertEqual(MindscapeModel(NumpyMLP([4], 8), max_steps=1).predict(view).error_type, 'timeout')
        self.assertEqual(MindscapeModel(NumpyMLP([4], 8), timeout_seconds=-1).predict(view).error_type, 'timeout')

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        from mindscape.models.numpy_backend import NumpyMLP
        from mindscape.models.learned import BaselineModel, LearnedModel
        from mindscape.models.encoding import ANSWER_DIGITS
        model = BaselineModel(NumpyMLP([10] * ANSWER_DIGITS + [2], 8))
        view = self.splits['test'][0].view('answer_only')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint'
            model.save(path)
            restored = LearnedModel.load(path)
            self.assertEqual(model.predict(view), restored.predict(view))
            with self.assertRaises(FileExistsError):
                model.save(path)

    def test_training_config_validation(self):
        from mindscape.training.run import validate_config
        good = dict(kind='baseline', seed=0, budget=5, hidden=8, steps=5, batch_size=4, learning_rate=.01, validation_every=5)
        validate_config(good)
        for key, value in [('kind', 'experiential'), ('budget', 0), ('seed', -1), ('steps', 0), ('learning_rate', -1)]:
            with self.assertRaises(ValueError):
                validate_config({**good, key: value})

    def test_training_to_evaluation_gate(self):
        from mindscape.data.generation import save_dataset
        from mindscape.training.run import train
        from mindscape.models.learned import LearnedModel
        from mindscape.evaluation.runner import run
        cfg = dict(kind='baseline', seed=0, budget=5, hidden=8, steps=20, batch_size=8, learning_rate=.01, validation_every=10)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data'
            save_dataset(data, self.splits, self.config)
            for kind, regime in [('baseline', 'answer_only'), ('mindscape', 'trajectory_supervised')]:
                model, metadata = train(data, {**cfg, 'kind': kind}, Path(tmp) / kind)
                restored = LearnedModel.load(Path(tmp) / kind / 'checkpoint')
                folder, metrics = run(restored, data, Path(tmp) / 'eval', regime=regime)
                self.assertEqual(metrics['dataset_size'], 5)
                self.assertGreater(metrics['training_time'], 0)
                self.assertTrue((folder / 'predictions.jsonl').exists())
                self.assertEqual(model.parameter_count, restored.parameter_count)
                if kind == 'mindscape':
                    self.assertEqual(metrics['accuracy'], 1)
                    self.assertEqual(metrics['grounded_rate'], 1)
                with self.assertRaises(ValueError):
                    run(restored, data, Path(tmp) / 'bad', regime=regime, training_size=10)

    def test_episodic_partition_isolation(self):
        from mindscape.memory.episodic import EpisodicMemory
        with tempfile.TemporaryDirectory() as tmp:
            memory = EpisodicMemory(Path(tmp) / 'episodes.sqlite')
            key = memory.store_episode('test', {}, 57, True, {}, partition='test')
            self.assertIsNone(memory.get(key))
            self.assertEqual(memory.retrieve(), [])
            self.assertTrue(memory.get(key, 'test')['success'])
            memory.clear('train')
            self.assertIsNotNone(memory.get(key, 'test'))
            memory.clear('test')
            self.assertIsNone(memory.get(key, 'test'))

    def test_regime_c_is_explicitly_unimplemented(self):
        from mindscape.models.learned import ExperientialLearnerInterface
        with self.assertRaises(NotImplementedError):
            ExperientialLearnerInterface().learn_from_feedback(None, None, None, None)
