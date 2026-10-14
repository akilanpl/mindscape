def test_cpu_original_forward_backward_equivalence():
    from scripts.native.aggregation_equivalence import measure
    r=measure()
    assert len(r['records'])==24
    assert r['weights_unchanged'] and r['optimizer_updates']==0
