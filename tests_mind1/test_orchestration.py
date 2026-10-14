import json
import threading
import pytest
from mindscape.mind1.orchestrator import Orchestrator, validate, execute_tool


def test_real_symbolic_subprocess():
    result = execute_tool("math_calculate", {"expression": "sin(x)**2 + cos(x)**2", "operation": "simplify"})
    assert result["ok"]
    assert "1" in json.dumps(result["result"])


def test_no_qualified_native_honestly_declines_open_problem():
    result = Orchestrator().chat({"message": "Prove the Riemann hypothesis", "mode": "research"})
    assert result["status"] == "unavailable"
    assert "No solution is claimed" in result["answer"]
    assert not result["tools"]


def test_cancel_before_tool():
    cancel = threading.Event(); cancel.set()
    with pytest.raises(InterruptedError):
        execute_tool("math_calculate", {"expression": "6+7"}, cancel=cancel)


@pytest.mark.parametrize("data", [{"mode": []}, {"domain": {}}, {"history": [{"role": [], "content": "x"}]}])
def test_malformed_types_are_validation_errors(data):
    with pytest.raises(ValueError):
        validate({"message": "test", **data})




def test_disabled_retrieval_is_not_a_success(monkeypatch):
    monkeypatch.delenv("MIND1_RETRIEVAL", raising=False)
    result = execute_tool("research_retrieve", {"query": "quantum gravity"})
    assert result["ok"] is False
    assert result["result"]["status"] == "disabled"


def test_wasi_through_product_worker_when_configured():
    import os
    if not os.getenv("MIND1_WASI_RUNTIME"):
        pytest.skip("Trusted WASI runtime not configured")
    result = execute_tool("coding_execute", {"source": "print(sum(range(10)))"})
    assert result["ok"], result
    assert result["result"]["stdout"].strip() == "45"
