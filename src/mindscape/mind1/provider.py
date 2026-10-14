"""Explicitly configured OpenAI-compatible provider; no implicit model selection."""
from dataclasses import dataclass
import json
import os
import time
from urllib import request, error, parse


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = ""
    model: str = "openai/gpt-oss-120b"
    reasoning_effort: str = "high"
    context_chars: int = 48000
    timeout: float = 30
    output_tokens: int = 4096
    input_price_million: float | None = None
    output_price_million: float | None = None

    @classmethod
    def from_env(cls):
        return cls(
            base_url=os.getenv("MIND1_PROVIDER_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            api_key=os.getenv("MIND1_API_KEY", ""), model=os.getenv("MIND1_MODEL", "openai/gpt-oss-120b"),
            reasoning_effort=os.getenv("MIND1_REASONING_EFFORT", "high"),
            context_chars=int(os.getenv("MIND1_CONTEXT_CHARS", "48000")),
            timeout=float(os.getenv("MIND1_PROVIDER_TIMEOUT", "30")),
            output_tokens=int(os.getenv("MIND1_OUTPUT_TOKENS", "4096")),
            input_price_million=_price("MIND1_INPUT_PRICE_MILLION"),
            output_price_million=_price("MIND1_OUTPUT_PRICE_MILLION"),
        )


def _price(name):
    value = os.getenv(name)
    return float(value) if value else None


class CompatibleProvider:
    def __init__(self, config=None):
        self.config = config or ProviderConfig.from_env()
        c = self.config
        u = parse.urlsplit(c.base_url)
        if not u.hostname:
            raise ValueError("Provider hostname required")
        if c.reasoning_effort not in {"low", "medium", "high", ""}:
            raise ValueError("Invalid reasoning effort")
        if u.username or u.password or u.query or u.fragment:
            raise ValueError("Provider URL must not contain credentials, query or fragment")
        if u.scheme != "https" and not (u.scheme == "http" and u.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("Use HTTPS or an explicitly configured loopback provider")
        if not (1000 <= c.context_chars <= 200000 and 1 <= c.timeout <= 120 and 64 <= c.output_tokens <= 8192):
            raise ValueError("Provider limits outside supported bounds")

    @property
    def available(self):
        return bool(self.config.model) and (bool(self.config.api_key) or parse.urlsplit(self.config.base_url).hostname in {"localhost", "127.0.0.1", "::1"})

    def public(self):
        return {"configured": self.available, "model": self.config.model or None,
                "endpoint_host": parse.urlsplit(self.config.base_url).hostname,
                "context_chars": self.config.context_chars,
                "context_note": "Operator-defined character ceiling; not a discovered token window",
                "reasoning_effort": self.config.reasoning_effort, "streaming": False, "attribution": "External foundation model, not R5.6"}

    def _request(self, path, payload=None):
        headers = {"Content-Type": "application/json", "User-Agent": "Mindscape-Mind1/0.1"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        req = request.Request(self.config.base_url + path,
                              data=json.dumps(payload).encode() if payload is not None else None,
                              headers=headers)
        try:
            # No redirects: credentials must never follow a redirected endpoint.
            class NoRedirect(request.HTTPRedirectHandler):
                def redirect_request(self, *args, **kwargs):
                    return None
            with request.build_opener(NoRedirect).open(req, timeout=self.config.timeout) as response:
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ProviderError("Provider response exceeds size limit")
                return json.loads(raw)
        except error.HTTPError as exc:
            label = {401: "authentication failed", 403: "access denied", 429: "quota or rate limit", 404: "model or endpoint unavailable"}.get(exc.code, "request rejected")
            raise ProviderError(f"Provider {label} (HTTP {exc.code})") from None
        except (error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise ProviderError("Provider transport or JSON failure") from None

    def discover(self):
        if not self.available:
            raise ProviderError("Provider credential missing")
        result = self._request("/models")
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise ProviderError("Malformed model discovery response")
        return {"models": [x["id"] for x in result.get("data", []) if isinstance(x, dict) and isinstance(x.get("id"), str)],
                "note": "Visibility does not establish tool support, context size, price or task quality"}

    def complete(self, messages, tools=None):
        if not self.available:
            raise ProviderError("No reasoning model configured")
        size = len(json.dumps(messages)) + len(json.dumps(tools or []))
        if size > self.config.context_chars:
            raise ProviderError("Context ceiling exceeded; shorten the conversation")
        payload = {"model": self.config.model, "messages": messages,
                   "max_completion_tokens": self.config.output_tokens}
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if parse.urlsplit(self.config.base_url).hostname == "api.groq.com":
            payload["include_reasoning"] = False
            payload["parallel_tool_calls"] = False
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        began = time.monotonic()
        result = self._request("/chat/completions", payload)
        try:
            message = result["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError()
        except (KeyError, IndexError, TypeError):
            raise ProviderError("Malformed provider completion") from None
        if result.get("choices", [{}])[0].get("finish_reason") == "length":
            raise ProviderError("Provider output limit reached; answer incomplete")
        usage = result.get("usage") or {}
        if not isinstance(usage, dict) or any(not isinstance(usage.get(k, 0), (int, float)) or usage.get(k, 0) < 0 for k in ("prompt_tokens", "completion_tokens")):
            raise ProviderError("Malformed provider usage")
        cost = None
        c = self.config
        if c.input_price_million is not None and c.output_price_million is not None:
            cost = (usage.get("prompt_tokens", 0)*c.input_price_million + usage.get("completion_tokens", 0)*c.output_price_million)/1e6
        return message, {"model": result.get("model", c.model), "usage": usage,
                         "latency_seconds": round(time.monotonic()-began, 3),
                         "estimated_cost_usd": cost, "price_source": "operator configuration" if cost is not None else "unavailable"}
