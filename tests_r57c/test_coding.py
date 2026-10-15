from collections import Counter
from mindscape.r57c.coding_audit import diagnostic_rows,baselines

def test_diagnostic_balances_request_identity_and_position():
    rows=diagnostic_rows();assert len(rows)==160 and len({r['identity'] for r in rows})==160
    assert Counter(r['labels']['coding_index'] for r in rows)=={0:40,1:40,2:40,3:40}
    assert sum(r['problem']['requested_name']==min(r['problem']['candidates']) for r in rows)==40
    b=baselines(rows);assert b['exact-request']==1
    assert all(v==.25 for k,v in b.items() if k!='exact-request')
