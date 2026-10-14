import pytest
from mindscape.mind1.language import interpret
from mindscape.r56.language_model import FrozenVocabulary, validate_split, build_language_encoder, ARGUMENT_SLOTS, INTENTS


@pytest.mark.parametrize('text', ['27 plus 18', 'What is 27 + 18?', 'Add 27 and 18', 'please add 27 to 18'])
def test_addition_is_operands_only(text):
    r = interpret(text)
    assert r['supported'] and r['kind'] == 'native'
    assert r['problem'] == {'domain':'math', 'operation':'add', 'left':'27', 'right':'18'}
    assert 'answer' not in r and 'result' not in r['problem']
    assert r['interpretation'] == 'deterministic-grammar'


def test_digit_subtraction_typed():
    r = interpret('What is 7 minus 9?')
    assert r['problem'] == {'domain':'math','operation':'borrow_digit','left':7,'right':9,'borrow_in':0}


@pytest.mark.parametrize('text', ['Prove the Riemann hypothesis', 'Explain quantum gravity', 'do the same again', '27 and 18', '100 minus 23', '27 plus 18 then multiply by 3'])
def test_unsupported_abstains(text):
    r = interpret(text, [{'role':'user','content':'27 plus 18'}])
    assert not r['supported'] and r['kind'] == 'unsupported'
    assert 'answer' not in r


def test_vector_force_problem_only():
    r = interpret('mass 2 kg; force (4, 6) N; time 3 s; initial velocity (1, -2) m/s')
    assert r['kind'] == 'native'
    assert r['problem'] == {'domain':'physics','operation':'force','mass':2.0,'force':[4.0,6.0], 'dt':3.0,'velocity':[1.0,-2.0]}
    assert 'final_velocity' not in r['problem']


def test_scalar_force_embeds_x_axis():
    r = interpret('An object has mass of 2 kg and force of 6 N for 3 seconds with initial velocity of 1 m/s')
    assert r['problem']['force'] == [6,0]
    assert r['problem']['velocity'] == [1,0]


@pytest.mark.parametrize('text', [
    'mass 0 kg force 2 N time 3 s velocity 0 m/s',
    'mass 2 pounds force 2 N time 3 s velocity 0 m/s',
    'mass 2 kg force 2 N time 3 minutes velocity 0 m/s',
    'mass 2 kg force 2 N time 3 s velocity 0 m/s and drag 4',
    'mass 2 kg force 2 N time 3 s velocity 0 m/s and temperature 4',
    'mass 2 kg force 2 N time 3 s',
])
def test_physics_ambiguity_and_invalid_assumptions_abstain(text):
    assert not interpret(text)['supported']


def test_rename_separates_verifier_target():
    r = interpret('Rename variable x to y in this function:\n```python\ndef f(x):\n    return x + 1\n```')
    p = r['problem']
    assert r['kind'] == 'native' and p['old_name'] == 'x' and p['target_name'] == 'y'
    assert p['requested_name'] == 'y'
    assert p['source'].startswith('def f') and 'y' in p['candidates'] and len(p['candidates']) <= 16
    assert 'renamed_source' not in p


@pytest.mark.parametrize('source', ['def f(x):\n return eval(x)', 'def f(x):\n import os\n return x', 'def f(x):\n return x.y', 'def f(x, y):\n return x+y'])
def test_rename_rejects_effects_or_capture(source):
    assert not interpret('rename x to y\n```python\n'+source+'\n```')['supported']


@pytest.mark.parametrize('text,name', [
    ('integrate x**2','math_calculate'), ('differentiate sin(x) with respect to x','math_calculate'),
    ('inspect code\n```python\nprint(1)\n```','coding_analyze'),
    ('execute\n```python\nprint(1)\n```','coding_execute'),
    ('simulate projectile {"speed": 5}','physics_simulate'),
])
def test_tools_marked_external(text,name):
    r = interpret(text)
    assert r['supported'] and r['kind'] == 'tool' and r['tool']['name'] == name


def test_vocabulary_never_adapts_to_heldout():
    train = ['add 2 to 3', 'mass 2 force 3 time 4 velocity 1']
    heldout = ['combine seven with nine']
    report = validate_split(train,heldout)
    vocab = FrozenVocabulary.from_training_forms(train,max_length=32)
    before = vocab.manifest()
    assert vocab.encode(heldout[0])['unknown_tokens'] > 0
    assert vocab.manifest() == before
    assert 'combine' not in vocab.tokens and report['qualification'] == 'NOT_QUALIFIED'
    with pytest.raises(ValueError): validate_split(train,['ADD 2 TO 3'])


def test_truncation_is_explicit():
    vocab = FrozenVocabulary.from_training_forms(['add 1 2'],max_length=8)
    assert vocab.encode('1 '*20)['truncated']


def test_original_encoder_trainable_but_unavailable():
    torch = pytest.importorskip('torch')
    vocab = FrozenVocabulary.from_training_forms(['add 2 to 3', 'mass force velocity time'],max_length=16)
    model = build_language_encoder(vocab,embedding_size=8,hidden_size=8)
    encoded = vocab.encode('add 2 to 3')
    ids = torch.tensor([encoded['ids']],dtype=torch.long)
    mask = torch.tensor([encoded['mask']],dtype=torch.bool)
    result = model(ids,mask)
    assert result['intent_logits'].shape == (1,len(INTENTS))
    assert result['argument_start_logits'].shape == (1,len(ARGUMENT_SLOTS),16)
    result['intent_logits'].sum().backward()
    assert model.embedding.weight.grad is not None
    assert not model.qualified and 'target_name' not in ARGUMENT_SLOTS
    with pytest.raises(RuntimeError,match='unavailable'): model.interpret('add 2 to 3')
    with pytest.raises(ValueError): model(ids,torch.zeros_like(mask))
