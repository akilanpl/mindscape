import json
import threading
import pytest
from mindscape.mind1.orchestrator import Orchestrator, validate, execute_tool
from mindscape.mind1.provider import CompatibleProvider, ProviderConfig, ProviderError


class ContractProvider:
    """Deterministic provider contract fixture; never a product backend."""
    available = True
    def __init__(self):
        self.requests = []
    def public(self):
        return {"model": "contract-fixture", "attribution": "test only"}
    def complete(self, messages, tools=None):
        self.requests.append(messages.copy())
        if len(self.requests) == 1:
            return {"content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "math_calculate", "arguments": '{"expression":"6+7"}'}}]}, {"model": "contract-fixture"}
        assert messages[-1]["role"] == "tool"
        return {"content": "The tool computed 13."}, {"model": "contract-fixture"}


def test_provider_tool_loop_preserves_context_and_evidence():
    provider = ContractProvider()
    result = Orchestrator(provider).chat({"message": "Add six and seven", "history": [{"role": "user", "content": "Please keep answers short"}]})
    assert result["status"] == "completed"
    assert len(result["tools"]) == 1 and result["tools"][0]["output"]["ok"]
    assert result["provenance"]["r56_used"] is False
    assert any(x["content"] == "Please keep answers short" for x in provider.requests[0])
    assert [x["kind"] for x in result["trace"]] == ["classification", "provider", "tool", "provider"]


def test_real_symbolic_subprocess():
    result = execute_tool("math_calculate", {"expression": "sin(x)**2 + cos(x)**2", "operation": "simplify"})
    assert result["ok"]
    assert "1" in json.dumps(result["result"])


def test_missing_provider_honestly_declines_open_problem():
    p = CompatibleProvider(ProviderConfig(api_key="", model=""))
    result = Orchestrator(p).chat({"message": "Prove the Riemann hypothesis", "mode": "research"})
    assert result["status"] == "unavailable"
    assert "not been solved" in result["answer"]
    assert not result["tools"]


def test_cancel_before_tool():
    cancel = threading.Event(); cancel.set()
    with pytest.raises(InterruptedError):
        execute_tool("math_calculate", {"expression": "6+7"}, cancel=cancel)


@pytest.mark.parametrize("data", [{"mode": []}, {"domain": {}}, {"history": [{"role": [], "content": "x"}]}])
def test_malformed_types_are_validation_errors(data):
    with pytest.raises(ValueError):
        validate({"message": "test", **data})


def test_groq_payload_hides_reasoning_and_attributes_model(monkeypatch):
    p = CompatibleProvider(ProviderConfig(api_key="test-key"))
    seen = []
    def contract(path, payload):
        seen.append(payload)
        return {"choices": [{"message": {"content": "test"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "model": "openai/gpt-oss-120b"}
    monkeypatch.setattr(p, "_request", contract)
    _, receipt = p.complete([{"role": "user", "content": "test"}])
    assert seen[0]["reasoning_effort"] == "high"
    assert seen[0]["include_reasoning"] is False
    assert seen[0]["parallel_tool_calls"] is False
    assert receipt["model"] == "openai/gpt-oss-120b"
    assert "test-key" not in json.dumps(p.public())
