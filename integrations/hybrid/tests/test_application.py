import json
import pytest
from mindscape_delivery.parsing import supplied_parse,ground_neural
from mindscape_delivery.assistant import Assistant

@pytest.mark.parametrize('question,task',[
 ('Differentiate x^3 + sin(x) with respect to x','differentiate'),
 ('Find the derivative of 2x^2','differentiate'),
 ('Integrate x^2 from 0 to 3','integrate'),
 ('Solve the equation x^2 - 5*x + 6 = 0 for x','solve_equation'),
 ('Find the determinant of [[1,2],[3,4]]','determinant'),
 ('Find the inverse of [[1,2],[3,4]]','matrix'),
 ('Find current through a 4 ohm resistor with 12 V across it','ohm'),
 ('Find voltage across a 3 ohm resistor carrying 2 A','ohm'),
 ('Find resistance with voltage 12 V and current 2 A','ohm'),
 ('Find the equivalent resistance of 3 ohm and 6 ohm in parallel','resistance_parallel'),
 ('Coulomb force between charges 2 uC and 3 uC separated by 0.1 m','coulomb'),
 ('Find electric field of 2 nC at 0.1 m','electric_field'),
 ('Lens equation, f=20 cm and u=-30 cm','lens'),
 ('Mirror equation, f=-20 cm and u=-30 cm','mirror'),
 ('Predict the output:\n```python\ns=0\nfor i in range(4):\n s+=i\nprint(s)\n```','trace'),
 ('SELECT name FROM marks WHERE score>80 tables: {"marks":[{"name":"A","score":90}]}','select'),
])
def test_supported_grammar_routes(question,task):
    assert supplied_parse(question)['task']==task
    result=Assistant().answer(question,mode='tools')
    assert result['status']=='verified',result
    assert result['provenance']['pretrained_used'] is False
    assert result['answer'].startswith('Method:')

@pytest.mark.parametrize('question',[
 'Tell me everything about quantum physics',
 'Find image distance of a lens with object distance 30 cm',
 'Calculate resistance without any circuit measurements',
 'SELECT * FROM secrets',
])
def test_unsupported_is_explicit(question):
    result=Assistant().answer(question,mode='tools')
    assert result['status']=='unsupported'
    assert result['provenance']['tool_verified'] is False

@pytest.mark.parametrize('question',[
 'Predict the output:\n```python\nimport os\nprint(os.listdir("/"))\n```',
 'Predict the output:\n```python\nwhile True:\n pass\n```',
])
def test_unsafe_programs_rejected_in_subprocess(question):
    assert Assistant().answer(question,mode='tools')['status']=='unsupported'

def test_neural_cannot_rewrite_code_or_invent_physics():
    assert ground_neural('Predict print(1)',{'valid':True,'subject':'coding','task':'trace','code':'print(999)'}) is None
    assert ground_neural('Find a force',{'valid':True,'subject':'physics','task':'coulomb','values':{}}) is None

def test_native_interface_does_not_attribute_tools_to_r6():
    result=Assistant().answer('Differentiate x^2',mode='native')
    assert result['status']=='unsupported'
    assert result['provenance']['native_neural_only'] is True

def test_neural_cannot_change_math_task_variable_or_bounds():
    q='Find derivative of x**2 with respect to x'
    assert ground_neural(q,{'valid':True,'subject':'math','task':'integrate','expression':'x**2','variable':'x'}) is None
    assert ground_neural(q,{'valid':True,'subject':'math','task':'differentiate','expression':'x**2','variable':'y'}) is None
    assert ground_neural('Integrate x**2',{'valid':True,'subject':'math','task':'integrate','expression':'x**2','variable':'x','lower':'1','upper':'2'}) is None
