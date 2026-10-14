"""Random construction/backward and safe replay only: no optimizer steps."""
import json
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
from mindscape.r56.language_model import ARGUMENT_SLOTS,INTENTS,build_language_encoder,tokenize
from mindscape.r56.language_learning import (build_language_dataset,training_vocabulary,batch_examples,
    language_loss,prepare_language_package,train_language,replay_language_checkpoint,labelled_form)
from mindscape.r56.learning import save_checkpoint


def test_disjoint_language_partitions_and_train_only_vocabulary():
    data=build_language_dataset(); vocab=training_vocabulary(data)
    assert set(data)=={'train','paraphrase','structure','vocabulary'}
    surfaces=[{' '.join(tokenize(r['text'])) for r in rows} for rows in data.values()]
    for i,a in enumerate(surfaces):
        for b in surfaces[i+1:]: assert not a&b
    assert 'neuron' not in vocab.tokens and 'activation' not in vocab.tokens
    assert 'inertial' not in vocab.tokens and 'please' not in vocab.tokens
    assert vocab.encode('neuron activation')['unknown_tokens']==2
    assert {r['intent'] for r in data['train']}==set(INTENTS)
    assert {r['template_id'] for r in data['train']}.isdisjoint(r['template_id'] for r in data['structure'])


def test_span_supervision_tracks_arguments_and_separates_labels():
    data=build_language_dataset(); vocab=training_vocabulary(data)
    row=next(r for r in data['train'] if r['intent']=='rename')
    tokens=('<bos>',)+tokenize(row['text'])
    start,end=row['spans']['requested_name']
    assert ' '.join(tokens[start:end+1])=='value'
    batch=batch_examples([row],vocab)
    assert set(batch['inputs'])=={'token_ids','mask'}
    j=ARGUMENT_SLOTS.index('requested_name')
    assert batch['labels']['start'][0,j].item()==start
    assert batch['labels']['start'][0,ARGUMENT_SLOTS.index('mass')].item()==0
    bad={**row,'spans':{'requested_name':[900,901]}}
    with pytest.raises(ValueError): batch_examples([bad],vocab)
    with pytest.raises(ValueError): labelled_form('{target_name}',{'target_name':'secret'},'rename',split='train',template_id='bad',structure_id='bad')


def test_actual_language_loss_backpropagates_intent_and_span_heads():
    torch.manual_seed(17); data=build_language_dataset(); vocab=training_vocabulary(data)
    model=build_language_encoder(vocab,embedding_size=16,hidden_size=16)
    batch=batch_examples(data['train'],vocab)
    losses=language_loss(model(**batch['inputs']),batch['labels'])
    assert torch.isfinite(losses['loss']) and losses['loss'].item()>0
    losses['loss'].backward()
    for name in ('embedding','encoder','intent_head','start_head','end_head'):
        parameters=list(getattr(model,name).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert sum(p.grad.abs().sum().item() for p in parameters)>0
    with pytest.raises(RuntimeError): model.interpret('add 1 and 2')


def test_language_gate_blocks_before_model_or_optimizer(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[1]
    package=tmp_path/'prepared'; manifest=prepare_language_package(root,package)
    assert manifest['stage']=='language-intent-spans'
    assert not (tmp_path/'checkpoint.json').exists()
    def forbidden(*args,**kwargs): raise AssertionError('Model/optimizer constructed before external gate')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    import mindscape.r56.language_learning as module
    monkeypatch.setattr(module,'build_language_encoder',forbidden)
    with pytest.raises(PermissionError): train_language(root,package,{}, {}, {},tmp_path/'checkpoint.json')
    assert not (tmp_path/'checkpoint.json').exists()


def test_random_checkpoint_read_only_replay_and_tamper_rejection(tmp_path):
    data=build_language_dataset(); vocab=training_vocabulary(data)
    model=build_language_encoder(vocab,embedding_size=16,hidden_size=16)
    metadata={'stage':'language-intent-spans','vocabulary':vocab.manifest(),'embedding_size':16,'hidden_size':16,'trained':False,'optimizer_steps':0}
    path=tmp_path/'random.json'; digest=save_checkpoint(model,path,metadata)
    before=path.read_bytes()
    first=replay_language_checkpoint(path,digest,data['train'][:2])
    second=replay_language_checkpoint(path,digest,data['train'][:2])
    assert first==second and first['qualified'] is False
    assert path.read_bytes()==before
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError): replay_language_checkpoint(path,digest,data['train'][:2])


def test_preparation_binds_evaluation_and_issues_only_unsigned_templates(tmp_path):
    from mindscape.r56.language_learning import _digest
    root=Path(__file__).resolve().parents[1]
    package=tmp_path/'prepared'; manifest=prepare_language_package(root,package)
    datasets=json.loads((package/'datasets.json').read_text())
    assert manifest['deterministic_algorithms'] is True
    assert manifest['split_hashes']=={name:_digest(rows) for name,rows in datasets.items()}
    assert manifest['evaluation']['selection'].startswith('fixed final checkpoint')
    for role in ('authorization','qualification'):
        artifact=json.loads((package/f'{role}.UNISSUED.json').read_text())
        assert artifact['issued'] is False and artifact['signature_hex']==''
        assert artifact['freeze_hash']==manifest['freeze_hash']
    frozen=dict(manifest); claimed=frozen.pop('freeze_hash')
    assert _digest(frozen)==claimed


def test_finite_gradient_validation_without_optimizer():
    from mindscape.r56.language_learning import validate_finite_gradients
    data=build_language_dataset(); vocab=training_vocabulary(data)
    model=build_language_encoder(vocab,embedding_size=8,hidden_size=8)
    with pytest.raises(ValueError): validate_finite_gradients(model)
    batch=batch_examples(data['train'],vocab)
    language_loss(model(**batch['inputs']),batch['labels'])['loss'].backward()
    validate_finite_gradients(model)
    next(model.parameters()).grad.flatten()[0]=float('nan')
    with pytest.raises(ValueError): validate_finite_gradients(model)
