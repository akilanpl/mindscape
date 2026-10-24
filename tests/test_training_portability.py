"""No optimizer construction or training is permitted in these local tests."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

WORKER_PATH = Path(__file__).resolve().parents[1] / 'cloud/kaggle/training/worker.py'
spec = importlib.util.spec_from_file_location('kaggle_training_worker', WORKER_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
HISTORICAL_TRAIN = Path('/Users/akilan/Documents/Codex/2026-10-14/files-pasted-by-the-user-operation/outputs/r57-focused-correction-v1/data-v1/train')


@pytest.fixture
def package(tmp_path):
    if not HISTORICAL_TRAIN.is_dir():
        pytest.skip('Historical frozen data unavailable; never regenerate replacement records')
    root = WORKER_PATH.parents[3]
    package = tmp_path / 'package'
    source = package / 'source'
    shutil.copytree(root / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(HISTORICAL_TRAIN, package / 'train')
    manifest = {'source_commit': 'd54ff8f', 'protocol': worker.PROTOCOL,
                'source_files_sha256': {str(p.relative_to(source)): worker.sha(p)
                    for p in (source / 'src').rglob('*.py')},
                'training_files_sha256': worker.TRAINING_HASHES}
    (package / 'manifest.json').write_text(json.dumps(manifest))
    return package


def test_original_training_bytes_validate_for_nine_fits(package):
    for seed in worker.SEEDS:
        for condition in worker.CONDITIONS:
            _, rows, _ = worker.validate_package(package, seed, condition)
            assert len(rows) == 128


def test_changed_data_and_protocol_rejected(package):
    path = package / 'train/17.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='bytes/order'):
        worker.validate_package(package, 17, worker.CONDITIONS[0])
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['protocol']['optimizer_updates'] = 5
    (package / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='protocol'):
        worker.validate_package(package, 17, worker.CONDITIONS[0])


def test_changed_source_rejected(package):
    path = package / 'source/src/mindscape/r57c/ordinal_model.py'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Source identity'):
        worker.validate_package(package, 17, worker.CONDITIONS[0])


def test_local_worker_refuses_fit_and_preserves_failure(package, tmp_path, monkeypatch):
    # Deterministic local rejection, without invoking any optimizer operation.
    monkeypatch.setattr(worker, 'require_cuda', lambda: (_ for _ in ()).throw(
        RuntimeError('KAGGLE_CUDA_REQUIRED; no local fitting permitted')))
    output = tmp_path / 'fit'
    with pytest.raises(RuntimeError, match='KAGGLE_CUDA_REQUIRED'):
        worker.run_fit(package, output, 17, worker.CONDITIONS[0])
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['actual_optimizer_updates'] == 0
    with pytest.raises(FileExistsError):
        worker.run_fit(package, output, 17, worker.CONDITIONS[0])
    assert not (output / 'final.json').exists()


@pytest.mark.parametrize('condition', worker.CONDITIONS)
@pytest.mark.parametrize('seed', worker.SEEDS)
def test_cpu_inference_checkpoint_roundtrip_matches_original(condition, seed, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    # If a regression introduces optimizer construction this test fails before any update.
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('Local optimizer forbidden'))
    from mindscape.r57c.ordinal_model import make_model
    from mindscape.r56.learning import save_checkpoint, load_checkpoint
    if not HISTORICAL_TRAIN.is_dir():
        pytest.skip('Historical frozen data unavailable')
    rows = json.loads((HISTORICAL_TRAIN / f'{seed}.json').read_text())
    examples = [next(r['problem'] for r in rows if r['problem']['domain'] == d)
                for d in ('math', 'coding', 'physics')]
    original = make_model(condition, seed).eval()
    portable = worker.make_portable_model(condition, seed, 'cpu').eval()
    path = tmp_path / 'initial.json'
    digest = save_checkpoint(portable, path, {'optimizer_steps': 0})
    restored = worker.make_portable_model(condition, seed, 'cpu')
    load_checkpoint(restored, path, digest)
    with torch.no_grad():
        for problem in examples:
            expected = original.predict(problem)
            actual = portable.predict(problem)
            reloaded = restored.predict(problem)
            assert actual.keys() == expected.keys() == reloaded.keys()
            for key, value in expected.items():
                if torch.is_tensor(value):
                    assert torch.equal(value, actual[key]), key
                    assert torch.equal(value, reloaded[key]), key


@pytest.mark.parametrize('condition', worker.CONDITIONS)
def test_cuda_inference_agrees_with_cpu_when_available(condition):
    """Inference only. Absence of CUDA is an explicit skip, never a simulated pass."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Real CUDA unavailable; CPU/CUDA numerical equivalence unmeasured')
    if not HISTORICAL_TRAIN.is_dir():
        pytest.skip('Historical training observations unavailable')
    rows = json.loads((HISTORICAL_TRAIN / '17.json').read_text())
    cpu = worker.make_portable_model(condition, 17, 'cpu').eval()
    cuda = worker.make_portable_model(condition, 17, 'cuda:0').eval()
    with torch.no_grad():
        for domain in ('math', 'coding', 'physics'):
            problem = next(r['problem'] for r in rows if r['problem']['domain'] == domain)
            expected, actual = cpu.predict(problem), cuda.predict(problem)
            for key, value in expected.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(actual[key].cpu(), value, atol=1e-5, rtol=1e-4)
