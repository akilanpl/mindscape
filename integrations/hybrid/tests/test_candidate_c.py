import math
import pytest
from mindscape_delivery.solvers import solve
from mindscape_delivery.parsing import supplied_parse

def q(v,u):return {'value':v,'unit':u}
@pytest.mark.parametrize('expression,a,b',[('1/x',-1,1),('x/x',-1,1),('log(x)-log(x)',-1,1),('sqrt(x)',-1,1)])
def test_definite_domain_survives_simplification(expression,a,b):
 assert not solve(dict(subject='math',task='integrate',expression=expression,lower=a,upper=b))['supported']
def test_trig_integral():
 r=solve(dict(subject='math',task='integrate',expression='sin(x)',lower=0,upper='pi'));assert r['answer']=='2'
def test_task_is_not_ignored():
 assert not solve(dict(subject='coding',task='invented'))['supported']
def test_exception_subset():
 r=solve(dict(subject='coding',task='trace',code='try:\n x=1/0\nexcept ZeroDivisionError:\n x=7\nfinally:\n x+=1\nprint(x)'));assert r['answer']=='8'
@pytest.mark.parametrize('code',['import os\nprint(1)','print(open("secret"))','try:\n print(1/0)\nexcept Exception:\n print(5)','print((1).__class__)'])
def test_unsafe_stays_rejected(code):assert not solve(dict(subject='coding',task='trace',code=code))['supported']
def test_inline_csv():
 assert solve(dict(subject='coding',task='csv_summary',text='a,b\nx,10\ny,20\n',column='b',operation='mean'))['answer']==15
@pytest.mark.parametrize('text',['a,a\n1,2\n','a\nNaN\n','a\n1,2\n'])
def test_csv_invalid(text):assert not solve(dict(subject='coding',task='csv_summary',text=text,column='a'))['supported']
def test_probability_invalid():
 assert not solve(dict(subject='math',task='probability',operation='conditional',intersection='3/4',given='1/2'))['supported']
def test_ode():
 r=solve(dict(subject='math',task='linear_ode',coefficient='2',forcing='4',initial_x='0',initial_y='3'));assert r['supported'] and r['verification']['passed']
def test_singular_network():
 r=solve(dict(subject='physics',task='dc_network',fixed_voltages={'g':q(0,'V')},resistors=[dict(a='a',b='b',resistance=q(2,'ohm'))],unknown='a'));assert not r['supported']
def test_field_singular():
 r=solve(dict(subject='physics',task='field_vector',point=[q(0,'m'),q(0,'m')],charges=[dict(q=q(1,'uC'),position=[q(0,'m'),q(0,'m')])]))
 assert not r['supported']
def test_physics_bad_units():
 assert not solve(dict(subject='physics',task='induction',turns=4,flux_initial=q(1,'V'),flux_final=q(2,'Wb'),duration=q(1,'s')))['supported']
def test_below_threshold():
 assert not solve(dict(subject='physics',task='photoelectric',frequency=q(1e14,'Hz'),work_function=q(5,'eV')))['supported']
def test_tables_before_query():
 p=supplied_parse('Evaluate SQL. Tables: {"a":[{"v":2},{"v":4}]}\nSQL: SELECT SUM(v) FROM a;')
 assert p and solve(p)['answer']['rows']==[[6]]
def test_force_phrase():assert supplied_parse('Find electrostatic force between 1 uC and 2 uC separated by 1 m')['task']=='coulomb'

def test_bayes_worked_equality_preserves_fraction_grouping():
 r=solve(dict(subject='math',task='probability',operation='bayes',priors=['1/5','3/10','1/2'],likelihoods=['1/20','1/10','1/25'],index=1))
 assert r['answer']=='1/2'
 assert '(3/10)*(1/10)/(3/50) = 1/2' in r['steps'][-1]
