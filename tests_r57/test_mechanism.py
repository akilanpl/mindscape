import pytest
import torch
from mindscape.r56.neural import R56Model
from mindscape.r56.native_domains import pack_problem
from mindscape.r56.benchmarks import matched_baseline
from mindscape.r57.mechanism import permute_graph,perturb_graph,predict,concept_probe


def problem():return {'domain':'physics','mass':2.,'force':[2.,-3.],'dt':.5,'velocity':[0.,0.],'concept_ids':[2]}


def test_valid_node_relabeling_preserves_graph_prediction():
 torch.manual_seed(17);model=R56Model().eval();g=pack_problem(problem())
 assert predict(model,g)==pytest.approx(predict(model,permute_graph(g,[2,0,1])),abs=2e-6)
 assert sorted(permute_graph(g,[2,0,1]).relation_types.tolist())==sorted(g.relation_types.tolist())
 with pytest.raises(ValueError):permute_graph(g,[0,0,1])


def test_gru_serialization_is_not_permutation_invariant():
 model=matched_baseline('gru',44962,17).eval();g=pack_problem(problem())
 assert max(abs(a-b)for a,b in zip(predict(model,g),predict(model,permute_graph(g,[2,1,0]))))>1e-5


def test_concept_duplicate_mean_and_gradient_no_optimizer(monkeypatch):
 def forbidden(*a,**k):raise AssertionError('No optimizer in read-only probes')
 monkeypatch.setattr(torch.optim,'Adam',forbidden)
 model=R56Model();before={k:v.clone()for k,v in model.state_dict().items()}
 probe=concept_probe(model,problem())
 assert probe['on']['prediction']==pytest.approx(probe['duplicate-id']['prediction'])
 assert probe['on']['gradient_norms']['concepts']>0
 assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_disconnection_is_intervention_not_valid_invariance():
 graph=pack_problem(problem());changed=perturb_graph(graph,'disconnected')
 assert changed.edges.shape==(2,0) and changed.relation_types.numel()==0
 assert torch.equal(graph.node_features,changed.node_features)
