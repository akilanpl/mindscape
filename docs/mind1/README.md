# Mind 1 — native architecture correction

Current implementation: `development/mind1-native-r56`, branched from exact PR27
head `3cefed2a2cc4e6a5867602822fcd35bc8d19bc96`. PR27 remains archival and unmerged.

Mind 1 now uses only its bounded native interpreter, Mindscape R5.6 neural engine
and independently attributed tools. Hosted/pretrained language-model inference
is removed. The product ignores all former external inference environment variables.

**R5.6 EXPERIMENTAL — SCIENTIFIC GENERALIZATION UNVERIFIED**
No qualified trained checkpoint exists. Default requests can use supported verified
tools; a native prediction requires explicitly opting into an untrained engineering
demo. Unsupported questions receive an honest availability response, not template
PhD answers. Native grammar and deterministic scheduling are not learned intelligence.

## Setup

Requires Python 3.11 or later, and a supported CPU PyTorch installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mind1.txt
export PYTHONPATH="$PWD/src"
python -m mindscape.mind1 --port 8765
```

Open http://127.0.0.1:8765. No inference account, key, external endpoint or model
is needed. PyTorch is used with CPU construction/inference for engineering tests;
no pretrained weights are downloaded. Missing PyTorch disables native demos;
symbolic and execution tools still disclose their own availability.

Try `What is 27 plus 18?`. With the demo checkbox off, the exact arithmetic tool
is clearly attributed. With **Engineering demo: untrained native model** selected,
the system interprets operands, retrieves supplied concept IDs, runs real random
R5.6 weights and displays their original prediction plus independent verification.
The reference result never repairs the prediction. Seed17 and a weight hash are
recorded; a coincidentally correct demo is not evidence of learned competence.

Other supported native interpretations include single-digit borrow, a rename of a
variable in a bounded pure Python function, and a constant-force velocity update
with explicit mass, initial velocity, force and duration. Tool examples include
`Integrate x**2`, `Inspect this code:` followed by a fenced Python block, `Run`
with fenced Python when WASI is installed, and
`Simulate oscillator {"mass":1,"spring":1,"duration":1}`.

## Checkpoints and science

An optional local operator checkpoint uses `MIND1_NATIVE_CHECKPOINT` and
`MIND1_NATIVE_CHECKPOINT_SHA256`. Loading checks hash, JSON tensor schemas and
finite values; it never unpickles executable objects. A checkpoint does not become
scientifically qualified by containing a trained flag. Current runtime permits
unqualified checkpoint inspection only in explicit demo mode.

The original graph model has a typed state encoder, trainable concept embeddings,
shared relation/message core, action conditioning, bounded GRU state and separate
prediction heads. Native-language StageB is an original compact GRU intent/span
encoder. Both learning pipelines have actual losses, gradients, guarded optimizers,
source/data locks and safe checkpoint replay. Neither has been scientifically
trained or qualified in this correction. See `docs/native/` for architecture,
benchmark preparation, authorization and independent review.

Run permitted engineering checks:

```bash
python -m pytest tests_native tests_mind1 tests_r56 -q
python scripts/native/ci_preservation.py --output work/native-preservation.json
```

Tests include random forward/backward and checkpoint round trips, with no optimizer
updates. Former live-provider tests are replaced by native-path/no-network gates;
other tool and HTTP security regressions remain. Legacy CI defects are separately
documented and are not silently waived by the native suite.

## Security and retained product

Loopback-only HTTP API, Host/Origin checks, strict size/JSON limits, CSP, rate and
concurrency limits, bounded process execution, cancellation and safe DOM rendering
remain. API `/api/chat` accepts message/history/domain/mode plus a boolean
`native_demo`; `/api/capabilities` returns native status and the capability registry.
`/api/models` returns native availability and makes no network request.
Conversations stay in browser localStorage; history is preserved but pronoun-based
contextual language understanding is not qualified. Shared browser profiles share
local history. Export conversations before clearing site storage.

WASI execution still requires a trusted `MIND1_WASI_RUNTIME` with python.wasm and
lib; compile a fresh cache for installed Wasmtime49. Probe failure disables execution,
and user source never runs under unrestricted host Python. Native inference denies
network socket connections and tool workers exclude external inference credentials.
Optional arXiv retrieval remains a separately attributed tool and is not a reasoning
fallback; the native chat router does not automatically send questions to it.

No public accounts, unrestricted advanced-language reasoning, formal proofs, learned
compositional generalization or cross-domain transfer is claimed. The existing
research website, historical studies and evidence remain separately preserved.
