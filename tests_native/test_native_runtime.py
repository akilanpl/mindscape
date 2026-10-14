"""Actual native product integration; no remote inference or optimizer steps."""
import json
import os
from pathlib import Path
import pytest
from mindscape.mind1.orchestrator import Mind1Orchestrator, execute_tool


def test_boot_and_requests_ignore_every_external_inference_setting(monkeypatch):
    for name in ('MIND1_API_KEY','MIND1_PROVIDER_URL','MIND1_MODEL','MIND1_REASONING_EFFORT'):
        monkeypatch.setenv(name, 'invalid-and-must-not-be-read')
    monkeypatch.delenv('MIND1_NATIVE_CHECKPOINT', raising=False)
    import urllib.request
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail('Normal Mind1 request attempted network inference')
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    app = Mind1Orchestrator()
    caps = app.capabilities()
    assert caps['external_inference'] is False
    result = app.chat({'message': 'What is 27 plus 18?'})
    assert result['status'] == 'completed'
    assert result['provenance']['r56_used'] is False
    assert '45' in result['answer']
    assert not result['native']['available']
    assert 'invalid-and-must-not-be-read' not in json.dumps(result)
    unsupported = app.chat({'message': 'Give a proof of the Riemann hypothesis'})
    assert unsupported['status'] == 'unavailable'
    assert unsupported['provenance']['external_inference'] is False


def test_actual_model_prediction_travels_through_orchestrator_worker_and_verifier(monkeypatch):
    monkeypatch.delenv('MIND1_NATIVE_CHECKPOINT', raising=False)
    app = Mind1Orchestrator()
    result = app.chat({'message': 'What is 27 plus 18?', 'native_demo': True})
    assert result['status'] == 'engineering_demo', result
    assert result['provenance']['r56_used'] is True
    native = result['native']
    assert native['checkpoint']['trained'] is False
    assert native['checkpoint']['scientifically_qualified'] is False
    assert native['parameter_count'] > 1000
    assert native['verification']['expected'] == '45'
    assert len(native['prediction']['transitions']) == 2
    assert native['concept_source'].startswith('explicit')
    assert result['tools'] == []
    assert 'ENGINEERING DEMONSTRATION' in result['answer']
    # No separately evaluated reference has replaced the model answer.
    assert json.dumps(native['prediction'], indent=2) in result['answer']


def test_native_demo_flag_must_be_boolean():
    with pytest.raises(ValueError, match='boolean'):
        Mind1Orchestrator().chat({'message': '6+7', 'native_demo': 'true'})


def test_native_production_modules_have_no_cloud_or_pretrained_inference_imports():
    root = Path(__file__).resolve().parents[1]
    for file in ['src/mindscape/mind1/provider.py', 'src/mindscape/mind1/orchestrator.py',
                 'src/mindscape/mind1/native_runtime.py', 'src/mindscape/mind1/language.py',
                 'src/mindscape/r56/neural.py', 'src/mindscape/r56/native_domains.py']:
        text = (root/file).read_text()
        assert 'CompatibleProvider' not in text
        for name in ('import transformers', 'from transformers', 'import openai', 'import groq', 'from_pretrained(', '/chat/completions'):
            assert name not in text, (file, name)


def test_malformed_local_checkpoint_never_loads_pickle_or_executes(tmp_path, monkeypatch):
    file = tmp_path/'model.json'; file.write_text('not a checkpoint')
    monkeypatch.setenv('MIND1_NATIVE_CHECKPOINT',str(file))
    monkeypatch.setenv('MIND1_NATIVE_CHECKPOINT_SHA256', '0'*64)
    result = execute_tool('native_transition', {'problem': {'domain':'math','operation':'add_digit','left':1,'right':2}, 'engineering_demo':True})
    assert not result.get('available', False)
    assert result.get('ok') is False


@pytest.mark.parametrize('metadata', [[], None, {'optimizer_steps': -1}, {'optimizer_steps': True}, {'optimizer_steps': '10'}])
def test_checkpoint_metadata_rejected_before_weights_applied(tmp_path, metadata):
    import hashlib
    import torch
    from mindscape.r56.neural import R56Model
    from mindscape.r56.learning import save_checkpoint, load_checkpoint
    model = R56Model()
    file = tmp_path/'checkpoint.json'
    save_checkpoint(model, file, {})
    payload = json.loads(file.read_text()); payload['metadata'] = metadata
    file.write_text(json.dumps(payload))
    before = {k: v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError):
        load_checkpoint(model, file, hashlib.sha256(file.read_bytes()).hexdigest())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
