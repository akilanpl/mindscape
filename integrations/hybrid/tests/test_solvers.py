import math
import pytest
from mindscape_delivery.solvers import solve

@pytest.mark.parametrize('spec,expected',[
 ({'subject':'math','task':'differentiate','expression':'x**3+2*x','variable':'x'},'3*x**2 + 2'),
 ({'subject':'math','task':'integrate','expression':'2*x'},'x**2 + C'),
 ({'subject':'math','task':'integrate','expression':'x**2','lower':0,'upper':3},'9'),
 ({'subject':'math','task':'determinant','matrix':[[1,2],[3,4]]},'-2'),
 ({'subject':'math','task':'solve_equation','expression':'x**2-4=0'},['-2','2']),
])
def test_exact_math(spec,expected):
 r=solve(spec); assert r['supported']; assert r['answer']==expected; assert r['verification']['passed']

def test_inverse():
 r=solve({'subject':'math','task':'matrix','operation':'inverse','matrix':[[1,2],[3,4]]})
 assert r['answer']==[['-2','1'],['3/2','-1/2']]; assert r['verification']['passed']

@pytest.mark.parametrize('expression',["__import__('os').system('id')",'x.__class__','[x for x in range(4)]','x**1000000','1/0','sin(x,2)'])
def test_math_rejects_unsafe(expression):
 assert not solve({'subject':'math','task':'differentiate','expression':expression})['supported']

def test_singular_definite_integral_rejected():
 assert not solve({'subject':'math','task':'integrate','expression':'1/x','lower':-1,'upper':1})['supported']


def val(v,u):return {'value':v,'unit':u}
def phys(task,**values):return {'subject':'physics','task':task,'values':values}
def test_coulomb():
 r=solve(phys('coulomb',q1=val(1e-6,'C'),q2=val(2e-6,'C'),r=val(.1,'m')))
 assert r['supported']; assert math.isclose(r['answer'],1.79751035846)

def test_ohm():
 r=solve(phys('ohm',voltage=val(12,'V'),resistance=val(4,'ohm')))
 assert r['answer']==3 and r['units']=='A'

@pytest.mark.parametrize('task,expected',[('resistance_series',6),('resistance_parallel',1.5)])
def test_resistors(task,expected):
 r=solve(phys(task,resistors=[val(3,'ohm'),val(3,'ohm')]))
 assert r['answer']==expected

@pytest.mark.parametrize('task,expected',[('lens',.6),('mirror',.12)])
def test_optics(task,expected):
 r=solve(phys(task,f=val(.2,'m'),u=val(-.3,'m')))
 assert math.isclose(r['answer']['image_distance'],expected) and r['verification']['passed']

@pytest.mark.parametrize('spec',[phys('ohm',voltage=12,resistance=4),phys('ohm',voltage=val(12,'V'),resistance=val(0,'ohm')),phys('coulomb',q1=val(1,'C'),q2=val(1,'C'),r=val(1,'cm'))])
def test_physics_units_and_domains(spec):assert not solve(spec)['supported']

def trace(code):return solve({'subject':'coding','task':'trace','code':code})
def test_python_loop_and_branch():
 r=trace('s=0\nfor i in range(6):\n if i%2==0:\n  s+=i\nprint(s)')
 assert r['answer']=='6' and r['variables']['s']==6

def test_python_while():
 assert trace('x=3\nwhile x>0:\n print(x)\n x-=1')['answer']=='3\n2\n1'

def test_python_shortcircuit():
 assert trace('x=False and 1/0\nprint(x)')['answer']=='False'

@pytest.mark.parametrize('code',["import os\nprint(os.getcwd())","print(open('/etc/passwd').read())",'while True:\n pass','print(2**1000000)','print("x"*1000000)','for i in range(1000000):\n print(i)','print(__import__("os"))'])
def test_python_unsafe_or_unbounded(code):assert not trace(code)['supported']

TABLE={'marks':[{'name':'Ann','score':90},{'name':'Bob','score':60},{'name':'Cam','score':80}]}
def sql(q):return solve({'subject':'sql','query':q,'tables':TABLE})
def test_sql_select():
 r=sql('SELECT name FROM marks WHERE score >= 80 ORDER BY score DESC')
 assert r['answer']=={'columns':['name'],'rows':[['Ann'],['Cam']]}

def test_sql_aggregate():assert sql('SELECT avg(score) AS mean FROM marks')['answer']['rows']==[[230/3]]

@pytest.mark.parametrize('query',['DROP TABLE marks','SELECT * FROM marks; DROP TABLE marks','PRAGMA database_list',"SELECT load_extension('/tmp/evil')",'ATTACH DATABASE "/tmp/x" AS x'])
def test_sql_no_writes(query):assert not sql(query)['supported']

@pytest.mark.parametrize('query',['SELECT zeroblob(1000000000)','SELECT randomblob(1000000000)'])
def test_sql_large_allocation_functions_denied(query):assert not sql(query)['supported']


def test_nonfinite_symbolic_input_rejected():
 assert not solve({'subject':'math','task':'differentiate','expression':'tan(pi/2)'})['supported']

def test_nonfinite_physics_output_rejected():
 assert not solve(phys('electric_field',q=val(1e12,'C'),r=val(1e-149,'m')))['supported']

@pytest.mark.parametrize('expression,answer',[('x-x=0','all real numbers'),('0=0','all real numbers'),('3=5',[]),('x**2+1=0',[])])
def test_constant_and_real_equations(expression,answer):
 r=solve({'subject':'math','task':'solve_equation','expression':expression})
 assert r['supported'] and r['answer']==answer and r['verification']['passed']

@pytest.mark.parametrize('expression',['x/x=1','(x**2-1)/(x-1)=2','x**-1=1'])
def test_polynomial_solver_does_not_erase_domain_holes(expression):
 assert not solve({'subject':'math','task':'solve_equation','expression':expression})['supported']

@pytest.mark.parametrize('task,f,u',[('lens',.2,-.2),('mirror',-.2,-.2)])
def test_optical_image_at_infinity(task,f,u):
 r=solve(phys(task,f=val(f,'m'),u=val(u,'m')))
 assert not r['supported'] and 'infinity' in r['reason']

def test_transpose_rectangular():
 r=solve({'subject':'math','task':'matrix','operation':'transpose','matrix':[[1,2,3],[4,5,6]]})
 assert r['answer']==[['1','4'],['2','5'],['3','6']]

def test_python_functions_return_and_local_bindings():
 r=trace('x=10\ndef f(x,y=2):\n z=x*y\n return z\nprint(f(3),f(4,5),x)')
 assert r['answer']=='6 20 10' and 'z' not in r['variables']

def test_python_function_global_lookup_is_lexical_not_callers_local():
 r=trace('x=10\ndef f():\n return x\ndef g():\n x=99\n return f()\nprint(g())')
 assert r['answer']=='10'

def test_python_function_unbound_local_rejected():
 r=trace('x=10\ndef f():\n print(x)\n x=2\nf()')
 assert not r['supported'] and 'before assignment' in r['reason']

def test_python_bounded_recursive_factorial():
 r=trace('def fact(n):\n if n<=1:\n  return 1\n return n*fact(n-1)\nprint(fact(5))')
 assert r['answer']=='120'

def test_python_recursion_rejected_at_depth_bound():
 r=trace('def f(n):\n return f(n+1)\nprint(f(0))')
 assert not r['supported'] and 'depth' in r['reason']

def test_python_lists_mutation_and_index_assignment():
 r=trace('a=[1,2]\na.append(3)\na[0]=4\na[1]+=5\nx=a.pop()\nprint(a,len(a),x)')
 assert r['answer']=='[4, 7] 2 3'

def test_python_tuple_semantics():
 assert trace('a=(1,2)\nprint(a,a[1])')['answer']=='(1, 2) 2'

def test_python_list_cycles_rejected():
 r=trace('a=[]\na.append(a)\nprint(a)')
 assert not r['supported'] and 'Cyclic' in r['reason']

def test_python_trace_snapshots_are_not_mutated_later():
 r=trace('a=[1]\na.append(2)\nprint(a)')
 assert r['trace'][0]['variables']['a']==[1]

@pytest.mark.parametrize('code',['def f():\n return 1\nf=2\nprint(f())','def f():\n return 1\ndef g(f):\n return f()\nprint(g(2))','@bad\ndef f():\n return 1','def f():\n def g():\n  return 1\n return g()\nprint(f())'])
def test_python_rejects_unsupported_function_namespace_features(code):
 assert not trace(code)['supported']

def test_parallel_resistance_underflow_rejected():
 assert not solve(phys('resistance_parallel',resistors=[val(1e-320,'ohm')]))['supported']


def test_matrix_add_development():
 r=solve({'subject':'math','task':'matrix','operation':'add','matrix':[[2,-1,0],[3,4,5]],'matrix_b':[[6,7,8],[-3,2,1]]})
 assert r['supported'] and r['answer']==[['8','6','8'],['0','6','6']] and r['verification']['passed']

def test_matrix_multiply_rectangular_development():
 r=solve({'subject':'math','task':'matrix','operation':'multiply','matrix':[[2,0,-1],[1,3,2]],'matrix_b':[[1,2],[4,0],[2,-2]]})
 assert r['supported'] and r['answer']==[['0','6'],['17','-2']] and r['verification']['passed']

@pytest.mark.parametrize('operation,a,b',[('add',[[1,2]],[[1],[2]]),('multiply',[[1,2]],[[1,2]]),('add',[[1]],[[1,2],[3]]),('multiply',[[1]],None)])
def test_matrix_dimension_mismatch_refused(operation,a,b):
 assert not solve({'subject':'math','task':'matrix','operation':operation,'matrix':a,'matrix_b':b})['supported']

@pytest.mark.parametrize('task,extra,expected',[
 ('coulomb',{'q1':val(3e-6,'C'),'q2':val(-4e-6,'C'),'r':val(.3,'m')},1.2),
 ('electric_field',{'q':val(-5e-6,'C'),'r':val(.2,'m')},-1125000),
])
def test_given_electrostatic_constant_is_used(task,extra,expected):
 r=solve(phys(task,k=9e9,**extra))
 assert r['supported'] and math.isclose(r['answer'],expected) and any('supplied k' in step for step in r['steps'])

@pytest.mark.parametrize('unit',['N*m^2/C^2','N m²/C²','N·m²/C²'])
def test_given_electrostatic_constant_units(unit):
 r=solve(phys('electric_field',q=val(2e-6,'C'),r=val(.1,'m'),k=val(9e9,unit)))
 assert r['supported'] and math.isclose(r['answer'],1800000)

@pytest.mark.parametrize('k',[None,0,-1,float('nan'),float('inf'),True,'9e9',val(9e9,'N/C'),{'value':9e9}, {'value':9e9,'unit':'N*m^2/C^2','extra':1}])
def test_malformed_supplied_constant_is_never_silently_defaulted(k):
 assert not solve(phys('coulomb',q1=val(1e-6,'C'),q2=val(2e-6,'C'),r=val(.2,'m'),k=k))['supported']

def test_default_vacuum_constant_is_disclosed():
 r=solve(phys('electric_field',q=val(1e-6,'C'),r=val(.5,'m')))
 assert r['supported'] and any('Vacuum' in step and 'default' in step for step in r['steps'])

def test_top_level_electrostatic_constant():
 spec=phys('electric_field',q=val(2e-6,'C'),r=val(.3,'m'));spec['k']=val(9e9,'N*m^2/C^2')
 assert math.isclose(solve(spec)['answer'],200000)

def test_conflicting_electrostatic_constants_refused():
 spec=phys('electric_field',q=val(2e-6,'C'),r=val(.3,'m'),k=9e9);spec['k']=8e9
 assert not solve(spec)['supported']

def test_single_loop_kvl_development():
 r=solve(phys('kirchhoff_loop',emf=val(15,'V'),resistors=[val(2,'ohm'),val(3,'ohm')]))
 assert r['supported'] and r['answer']==3 and r['verification']['passed'] and r['units']=='A'

def test_single_loop_opposing_batteries_development():
 r=solve(phys('kirchhoff_loop',emfs=[val(10,'V'),val(-4,'V')],resistors=[val(2,'ohm'),val(1,'ohm')]))
 assert r['supported'] and r['answer']==2 and r['verification']['passed']

@pytest.mark.parametrize('spec',[phys('kirchhoff_loop',emf=val(15,'A'),resistors=[val(2,'ohm')]),phys('kirchhoff_loop',emf=val(15,'V'),resistors=[val(-2,'ohm')]),phys('kirchhoff_loop',emfs=[val(10,'V'),val(-4,'A')],resistors=[val(2,'ohm')])])
def test_kvl_unit_and_domain_failures(spec):
 assert not solve(spec)['supported']
