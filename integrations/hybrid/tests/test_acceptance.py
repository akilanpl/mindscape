import pytest
from mindscape_delivery.acceptance import answer_matches,safe_symbolic,safe_unsupported,unit_matches

def c(kind,expected):return {'answer_kind':kind,'expected':expected}
def test_answer_checks_are_conservative():
 assert answer_matches(c('numeric',2.5),'2.5 A')
 assert not answer_matches(c('numeric',2.5),'I think 99; maybe 2.5 A')
 assert answer_matches(c('symbolic','2*x'),'x+x')
 assert answer_matches(c('antiderivative','x**2'),'x**2+C')
 assert not answer_matches(c('antiderivative','x**2'),'x+C')
 assert answer_matches(c('stdout','3\n4'),'3\n4\n')
def test_symbolic_model_text_cannot_execute():
 with pytest.raises(ValueError):safe_symbolic('__import__("os").system("x")')
 with pytest.raises(ValueError):safe_symbolic('x**100000')
def test_safe_decline_and_units():
 assert safe_unsupported({'status':'unsupported','answer':'Cannot solve this problem.'})
 assert not safe_unsupported({'status':'ok','answer':'42'})
 assert unit_matches({'unit':'ohm'},{'solver_result':{'units':'Ω'}})

def test_unit_conversion_is_physical_not_string_equality():
 from mindscape_delivery.acceptance import response_answer_correct
 case={'answer_kind':'numeric','expected':50,'unit':'cm'}
 response={'solver_result':{'supported':True,'answer':.5,'units':'m'}}
 assert unit_matches(case,response)
 assert response_answer_correct(case,response)
 assert not unit_matches({'unit':'N'},{'solver_result':{'units':'m'}})

def test_declared_solver_output_shapes():
 from mindscape_delivery.acceptance import response_answer_correct,candidate_answer
 assert response_answer_correct(c('numeric',9),{'solver_result':{'supported':True,'answer':['9']}})
 assert candidate_answer({'solver_result':{'supported':True,'answer':{'columns':['n'],'rows':[[3]]}}})==[[3]]
 assert response_answer_correct({'answer_kind':'numeric','expected':20,'unit':'cm'},{'solver_result':{'supported':True,'answer':{'image_distance':.2,'magnification':1},'units':{'image_distance':'m'}}})
 assert answer_matches(c('matrix',[[1,2],[3,4]]),[['1','2'],['3','4']])
