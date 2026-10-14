# Mind 1 local review candidate

Mind 1 is a separate product application. Existing `website/`, frozen protocols,
negative results and historical evidence are preserved. Baseline: reviewed R5.4
integration `8c141d8`; this does not imply R5.4 learning is complete.

## Reproduce

Python 3.11 or later required. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mind1.txt
export PYTHONPATH="$PWD/src"
export MIND1_PROVIDER_URL=https://api.groq.com/openai/v1
export MIND1_MODEL=openai/gpt-oss-120b
export MIND1_REASONING_EFFORT=high
# Supply MIND1_API_KEY through your local environment/secret manager.
# Do not put credentials in frontend code, command arguments or committed files.
python -m mindscape.mind1 --port 8765
```

Open http://127.0.0.1:8765. Default Groq configuration matches the commands above.
A missing key disables hosted inference. Local arithmetic, polynomial integration,
differentiation, solving and code inspection remain available. Examples:
`What is 6 + 7?`, `Integrate x**2`, `Solve x**2-4=0`,
and `Inspect` followed by a fenced Python block.

Groq supports `high` effort for GPT-OSS and local function calling, per
[reasoning documentation](https://console.groq.com/docs/reasoning) and
[tool documentation](https://console.groq.com/docs/tool-use/overview), consulted
2026-10-14. Account access, quota and model availability must still be measured
with the real credential. All model assistance is attributed to GPT-OSS 120B
hosted by Groq, not R5.6. Alternate HTTPS OpenAI-compatible endpoints and models
are environment-configurable; use an empty MIND1_REASONING_EFFORT for providers
that do not support it. HTTP is accepted only for loopback model servers.

## Optional tools

Set `MIND1_WASI_RUNTIME` to a **trusted** Python WASI runtime containing
`python.wasm` and its `lib` directory. Use a fresh copy and let installed
Wasmtime 49 compile `python.cwasm`; do not reuse a cache from another runtime
version. Execution runs the existing WASI capability probe before guest code;
failed probes disable execution. No unrestricted host-source fallback exists.
The runtime is not bundled in source archives.

Set `MIND1_RETRIEVAL=1` for optional arXiv metadata retrieval. Only public-topic
queries should be sent; public-topic filtering is a model instruction, not an
enforced privacy classifier. Keep retrieval disabled for private conversations; retrieved metadata does not verify mathematical claims.

```bash
python -m pytest tests_mind1 tests_r56 -q
# With a real configured key; makes paid/quota-limited requests:
python -m pytest tests_mind1/test_live_provider.py -v -rs
```

See COVERAGE.md, SECURITY_REVIEW.md, TOOLS.md and ../r56/RESEARCH.md for boundaries.
No claim of universal mathematics, software or physics correctness is made.

## API and limits

POST `/api/chat` accepts message, history (user/assistant text only), domain,
mode and optional opaque request_id. GET `/api/capabilities` exposes configured
capabilities; GET `/api/models` performs actual model discovery. POST
`/api/cancel` accepts request_id. Reasoning modes bound tool rounds/time:
Quick 1/20s, Standard 3/60s, Deep 6/100s, Research 8/120s. Provider socket timeout
is separately bounded; in-flight remote inference can finish after cancellation.
Tool worker wall budget is 15s and CPU budget 12s. Four concurrent requests,
30 requests/minute, 96KB request body, 16K message characters, 40 history messages.
Operator-defined context character ceilings are not discovered token windows.
Unknown model pricing is reported unavailable; optional per-million prices may
be configured with MIND1_INPUT_PRICE_MILLION and MIND1_OUTPUT_PRICE_MILLION.

This is a loopback-only application, without public accounts or uploads.
Conversations stay in browser localStorage until cleared by the user; export
before clearing site data. Server does not persist conversation contents.
The browser profile is the local user boundary; shared browser profiles share
history. API requests use Host/Origin checks, strict JSON, CSP and size limits.
The server sends conversation context to the operator-configured provider when
inference is used. Secrets and private research assets are not sent to tool workers.

## Current limitations

Streaming, account authentication, formal Lean proofs, SageMath, additional
sandboxed language runtimes, multi-file writeback, public deployment and broad
expert-reviewed research-level correctness are not qualified. MathML implements
a documented subset and leaves unsupported notation readable. Provider integration
has deterministic contract tests; those fixtures are not a reasoning backend.
Live acceptance requires credential-backed inference and expert assessment.
