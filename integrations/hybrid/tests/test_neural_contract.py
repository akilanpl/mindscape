"""Contract checks use stub output; actual model smoke evidence is recorded separately."""
from mindscape_delivery.neural import NeuralBackend

def stub(text):
    backend=NeuralBackend()
    backend.generate=lambda *args,**kwargs:text
    return backend

def test_json_parser_rejects_prose():
    assert stub('The answer is 2').structured_parse('question')['valid'] is False

def test_json_parser_rejects_unknown_subject():
    assert stub('{"subject":"astronomy","task":"guess"}').structured_parse('question')['valid'] is False

def test_json_parser_requires_task():
    assert stub('{"subject":"math"}').structured_parse('question')['valid'] is False

def test_json_parser_preserves_solver_fields():
    value=stub('{"subject":"math","task":"differentiate","expression":"x**2","variable":"x"}').structured_parse('differentiate x**2')
    assert value['valid'] is True
    assert value['expression']=='x**2'
    assert 'unverified' in value['method']

def test_json_parser_fenced_object():
    assert stub('```json\n{"subject":"unsupported","task":"unsupported"}\n```').structured_parse('question')['valid'] is True
