from pathlib import Path
import pytest
from mindscape.r57c.scoreboard import precheck


def test_all_receipts_required_before_any_final_data_read(tmp_path):
    calls=[]
    def incomplete(path):
        calls.append(path)
        if len(calls)==4:raise RuntimeError('Incomplete fit')
        return {'optimizer_updates':2000}
    # No freeze/final files exist. Must stop on fit receipt before trying to open them.
    with pytest.raises(RuntimeError,match='Incomplete'):precheck(tmp_path,incomplete)
    assert len(calls)==4
