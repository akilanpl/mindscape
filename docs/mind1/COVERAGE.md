# Tiered live provider qualification

The live suite contains twelve public synthetic prompts: foundation,
undergraduate, graduate, and research tiers in mathematics, programming, and
physics. Each passes through the product Orchestrator with the actual configured
provider and tools. It is separate from deterministic tool and mocked transport
tests. A missing credential, wrong endpoint/model, or missing requested high
effort produces an explicit skip, never a fabricated answer.

| Domain | Foundation | Undergraduate | Graduate | Research |
|---|---|---|---|---|
| Mathematics | 6 + 7 = 13 | 2×2 determinant = −2 | compact metric completeness | Riemann hypothesis uncertainty |
| Programming | Python sum(range(5)) = 10 | merge sort invariant | partition/linearizability tradeoff | P versus NP uncertainty |
| Physics | net force = 6 N | oscillator angular frequency = 3 rad/s | stationary action derivation | quantum gravity uncertainty |

Five numeric ground truths have independently elementary calculations. Exact
numeric agreement is checked only when a single JSON answer parses. This checks
the stated result, not every sentence of the explanation, units or assumptions.
Python output here is mathematical prediction, not claimed sandbox execution.
All remaining cases require human expert review. Keyword observations about
uncertainty are reviewer cues only, not a correctness score. Research cases
receive no automatic scientific correctness score; plausible prose is not proof.

Run in Python >=3.11 with pytest and the application dependencies:

```sh
export MIND1_PROVIDER_URL=https://api.groq.com/openai/v1
export MIND1_MODEL=openai/gpt-oss-120b
export MIND1_REASONING_EFFORT=high
# Supply MIND1_API_KEY securely in the process environment; do not paste it in logs.
python -m pytest tests_mind1/test_live_provider.py -q -s --junitxml=work/live-provider.xml
```

The suite records key presence as a boolean only; it never prints credentials.
Provider receipts attribute external model, usage, latency and estimated cost.
Cost stays unavailable when operator pricing is absent, rather than invented.
Public synthetic prompts and answers are logged for review. No private research
labels, repository contents or hidden verifier contexts are sent. Twelve tests
can entail multiple billable completions/tool rounds within product budgets.

A passing transport/status check is not a benchmark of general mastery. Graduate
and research completion means an answer was delivered, with expert review still
required. These results cannot demonstrate R5.6 learning.

Current execution evidence: `api_key_present=false`; all twelve live cases
skipped explicitly because `MIND1_API_KEY` was absent. No live provider quality,
latency, cost, or answer correctness is claimed from that execution.
