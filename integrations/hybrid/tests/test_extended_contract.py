import pytest
from mindscape_delivery.parsing import supplied_parse
from mindscape_delivery.assistant import Assistant

def test_exact_unfenced_python_keeps_program():
    code='def total(n):\n s=0\n for j in range(n):\n  s+=j\n return s\nprint(total(6))'
    q='What does the following Python program print?\n'+code
    p=supplied_parse(q)
    assert p['code']==code
    assert Assistant().answer(q,mode='tools')['solver_result']['answer']=='15'

def test_forbidden_import_is_not_stripped_from_unfenced_program():
    q='Predict the output:\nimport os\nprint(4)'
    assert supplied_parse(q)['code'].startswith('import os')
    assert Assistant().answer(q,mode='tools')['status']=='unsupported'

def test_literal_tuple_sql_table():
    q="Table products(name, price) rows: ('pen', 5), ('book', 20), ('ruler', 10). SELECT name FROM products WHERE price >= 10 ORDER BY price;"
    p=supplied_parse(q)
    assert p['tables']['products']==[{'name':'pen','price':5},{'name':'book','price':20},{'name':'ruler','price':10}]
    r=Assistant().answer(q,mode='tools')
    assert r['status']=='verified',r
    assert r['solver_result']['answer']['rows']==[['ruler'],['book']]

def test_literal_json_row_sql_table():
    q='Table scores: [{"name":"M","score":7},{"name":"N","score":9}]\nSELECT AVG(score) AS average FROM scores;'
    r=Assistant().answer(q,mode='tools')
    assert r['solver_result']['answer']['rows']==[[8.0]]

def test_explicit_k_reaches_checked_solver():
    q='Find Coulomb force of charges 4 uC and 5 uC separated by 0.2 m; k=8.8e9 N*m^2/C^2'
    p=supplied_parse(q)
    assert p['values']['k']['value']==8.8e9
    r=Assistant().answer(q,mode='tools')
    assert r['status']=='verified',r
    assert abs(r['solver_result']['answer']-4.4)<1e-10

def test_matrix_input_pairs():
    q='Multiply matrices [[1,2,3]] and [[1],[2],[3]]'
    p=supplied_parse(q);assert p['operation']=='multiply'
    assert p['matrix_b']==[[1],[2],[3]]

def test_user_given_constant_bad_unit_is_not_rewritten():
    q='Find electric field of 2 uC at 0.4 m; k=9e9 N/C'
    assert supplied_parse(q) is None
    assert Assistant().answer(q,mode='tools')['status']=='unsupported'

def test_single_loop_current_with_series_resistors():
    q='Use Kirchhoff for current in a loop with 18 V and 3 ohm and 6 ohm in series'
    r=Assistant().answer(q,mode='tools')
    assert r['status']=='verified',r
    assert r['solver_result']['answer']==2.0

@pytest.mark.parametrize('declaration',['k=9×10⁹','k=9\\times10^9'])
def test_unsupported_constant_not_truncated(declaration):
    assert supplied_parse('Find electric field of 2 uC at 0.4 m; '+declaration) is None

def test_explicit_constant_alias_and_power():
    p=supplied_parse('Find electric field of 2 uC at 0.4 m; K: 8.8 x 10^9 N*m^2/C^2')
    assert p['values']['k']['value']==8.8e9

def test_worked_program_explanation_is_actual_trace():
    r=Assistant().answer('Predict output:\nx=3\nx+=7\nprint(x)',mode='tools')
    assert r['solver_result']['answer']=='10'
    assert 'Trace line' in r['answer']
    assert '"x": 10' in r['answer']
