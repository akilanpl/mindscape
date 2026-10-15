from mindscape.r57.data import identity, split_rows, counts, _math_candidates


def test_normalization_and_disjoint_balance():
    a={'domain':'math','operation':'add_digit','left':2,'right':3,'carry_in':1,'concept_ids':[0]}
    b={**a,'incoming':1,'concept_ids':[9]}
    assert identity(a)==identity(b)
    dev=split_rows(17,'dev',set())
    final=split_rows(17,'final',{r['identity'] for r in dev})
    assert len(dev)==240 and len(final)==240
    assert not {r['identity'] for r in dev}&{r['identity'] for r in final}
    assert counts(dev)['coding']['positions']=={'0':20,'1':20,'2':20,'3':20}
    assert len({identity(p) for p in _math_candidates()})==400
