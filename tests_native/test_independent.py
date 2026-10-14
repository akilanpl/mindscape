"""Independent engineering review checks; no training or qualification issuance."""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(os.getenv('MIND1_AUDIT_ROOT', str(Path(__file__).resolve().parents[1])))


@pytest.mark.parametrize('message,domain', [
    ('What is 27 plus 18?', 'math'),
    ('rename variable x to y in this function:\n```python\ndef f(x):\n    return x+1\n```','coding'),
    ('mass 2 kg force (4,6) N time 3 s initial velocity (1,-2) m/s','physics'),
])
def test_all_domains_reach_real_bounded_neural_worker_without_external_settings(monkeypatch,message,domain):
    from mindscape.mind1.orchestrator import Mind1Orchestrator
    for name in ('MIND1_API_KEY','MIND1_MODEL','MIND1_PROVIDER_URL','MIND1_REASONING_EFFORT'):
        monkeypatch.setenv(name,'DO-NOT-READ-EXTERNAL-SENTINEL')
    monkeypatch.delenv('MIND1_NATIVE_CHECKPOINT',raising=False)
    result = Mind1Orchestrator().chat({'message':message,'native_demo':True})
    assert result['status'] == 'engineering_demo', result
    assert result['domain'] == domain and result['provenance']['r56_used'] is True
    assert result['provenance']['external_inference'] is False
    native = result['native']
    assert native['available'] and native['parameter_count'] > 1000
    assert native['checkpoint']['trained'] is False and native['checkpoint']['scientifically_qualified'] is False
    assert native['attribution']['transition'] == 'original R5.6 neural prediction'
    assert json.dumps(native['prediction'],indent=2) in result['answer']
    assert 'DO-NOT-READ-EXTERNAL-SENTINEL' not in json.dumps(result)
    assert result['tools'] == []
    assert 'ENGINEERING DEMONSTRATION' in result['answer']


def test_protected_labels_do_not_change_graph_features():
    import torch
    from mindscape.r56.native_domains import pack_problem
    base = {'domain':'coding','source':'def f(x):\n return x+1','old_name':'x',
            'requested_name':'y','candidates':['x','y','z'],'target_name':'y'}
    alternate = deepcopy(base); alternate['target_name'] = 'z'; alternate['labels']={'coding_index':2}
    first, second = pack_problem(base),pack_problem(alternate)
    for field in ('node_features','node_types','edges','relation_types','action','concept_ids'):
        assert torch.equal(getattr(first,field),getattr(second,field)),field


def test_verification_does_not_rewrite_prediction_or_update_parameters(monkeypatch):
    import torch
    from mindscape.r56.neural import R56Model
    import mindscape.r56.native_domains as domains
    def forbidden(*args,**kwargs): raise AssertionError('Inference attempted optimizer/backward')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    monkeypatch.setattr(torch.optim,'SGD',forbidden)
    monkeypatch.setattr(torch.Tensor,'backward',forbidden)
    model = R56Model()
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    observed = []
    def verifier(problem,prediction):
        observed.append(deepcopy(prediction))
        prediction.clear()  # Even a mutating verifier receives only a separate copy.
        return {'passed':False,'expected':'separate reference'}
    monkeypatch.setattr(domains,'verify_prediction',verifier)
    result = model.predict({'domain':'math','operation':'add','left':'27','right':'18'})
    assert result['prediction'] == observed[0] and result['prediction']
    assert result['verification']['passed'] is False
    for key,value in model.state_dict().items(): assert torch.equal(value,before[key]),key


def test_training_source_calls_gate_before_model_and_optimizer():
    tree = ast.parse((ROOT/'src/mindscape/r56/learning.py').read_text())
    train = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'train')
    gate = min(n.lineno for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'validate_scientific_gate')
    neural_lines = [n.lineno for n in ast.walk(train) if
        (isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)) or
        (isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='R56Model') or
        (isinstance(n,ast.Attribute) and n.attr=='Adam')]
    assert neural_lines and all(line > gate for line in neural_lines)
    for file in ('native_runtime.py','orchestrator.py'):
        tree = ast.parse((ROOT/'src/mindscape/mind1'/file).read_text())
        assert not any(isinstance(n,ast.Attribute) and n.attr in {'backward','zero_grad','step'} for n in ast.walk(tree))


def test_original_wasi_implementation_is_byte_identical_to_precorrection():
    for name in ('src/mindscape/coding/sandbox.py','src/mindscape/coding/wasi_worker.py'):
        frozen = subprocess.check_output(['git','-C',str(ROOT),'show','3cefed2:'+name])
        assert (ROOT/name).read_bytes() == frozen


def test_json_checkpoint_roundtrip_and_finite_tensor_guard(tmp_path):
    import torch
    from mindscape.r56.learning import save_checkpoint,load_checkpoint
    model = torch.nn.Linear(2,2)
    path = tmp_path/'safe.json'
    identity = save_checkpoint(model,path,{'optimizer_steps':0})
    payload = json.loads(path.read_text())
    assert payload['scientifically_qualified'] is False
    assert load_checkpoint(torch.nn.Linear(2,2),path,identity)['optimizer_steps'] == 0
    payload['tensors']['weight']['values'][0][0] = float('nan')
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError,TypeError)):
        load_checkpoint(torch.nn.Linear(2,2),path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_checkpoint_size_checked_before_json_or_tensor_allocation(tmp_path):
    import torch
    from mindscape.r56.learning import load_checkpoint
    path = tmp_path/'too-large.json'
    with path.open('wb') as stream:
        stream.seek(64_000_000); stream.write(b' ')
    with pytest.raises(ValueError,match='bound'):
        load_checkpoint(torch.nn.Linear(1,1),path,'0'*64)


def test_language_encoder_cannot_claim_qualified_product_interpretation():
    from mindscape.r56.language_model import FrozenVocabulary,build_language_encoder
    vocabulary = FrozenVocabulary.from_training_forms(['add 1 to 2'])
    model = build_language_encoder(vocabulary)
    assert model.qualified is False and 'UNTRAINED' in model.scientific_status
    with pytest.raises(RuntimeError,match='unavailable'): model.interpret('add 1 to 2')
