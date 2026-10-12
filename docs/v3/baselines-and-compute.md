# Baseline matrix and compute

All comparisons must declare backbone hash, canonical training task identities,
updates, supervised labels, verifier queries and tokens. `TrainingBudget` rejects
negative/duplicate budgets; `require_fair` requires exact equality. Differences must
be reported as confounds instead of attributed to architecture. Matching total
information does not by itself isolate representation or learning algorithm.

| Condition | Backbone and information | Executed status |
|---|---|---|
| Random weights | Shared 1,287-parameter MLP, observations only, no training | 225 smoke executions |
| Zero weights | Same architecture, all weights zero, same observations | 225 smoke executions |
| Answer-only | Shared-backbone budget contract | Declared interface; untrained |
| Structured | Shared-backbone budget contract | Declared interface; untrained |
| Episodic/retrieval | Same predictor budget; train-only memory interface | Memory integration tested; benefit unrun |
| GPT-Neo | Approved local architecture/file hashes, safetensors | Optional adapter; no weights loaded |
| Pythia | Approved local GPT-NeoX architecture/file hashes, safetensors | Optional adapter; no weights loaded |
| Other early open weights | Requires explicit approval and adapter compatibility | Not approved/implemented |
| Deterministic reference | Evaluator-only expected values | Ceiling/control, never a learned baseline |

Frozen V1 models are not converted or overwritten. Their reported trained numerical
claim results and coding results remain historical context, not entries in a V3
competitive comparison. Historical external models differ in capacity, tokenizer,
pretraining exposure and compute; they cannot be treated as matched-backbone controls.

The tiny model performs roughly 2,300 policy multiply/add operations plus up to 102
transition-head multiply/add operations plus a recomputed 7-to-16 hidden layer per action,
excluding activations, hashing,
serialization and Python overhead. Weight payload is 10,296 bytes at float64; Python
object/JSON overhead is larger. A 450-case run therefore needs approximately one
million elementary arithmetic operations. This is an analytical estimate, not a
hardware FLOP measurement. The recorded run's episode CPU sum is 0.176970 seconds
and wall-latency sum is 0.176977 seconds. Setup, serialization, file writes, tests,
WASI compilation and model initialization are excluded. These sums do not measure
end-to-end throughput, energy, peak memory or accelerator utilization.

Historical model weight memory can be estimated as parameter count times dtype bytes:
125M fp32 about 0.50 GB, 1.3B fp32 about 5.2 GB, before activations, KV cache,
tokenizer and framework overhead. The current historical candidate scorer performs
65 sequence forwards per decision; token compute depends on tokenizer and prompt.
No actual historical-model latency, memory maximum or training cost was measured.
No paid calls or model downloads occurred. Domain training needs a new explicit
budget, protocol, baseline implementation and checkpoint/optimizer lifecycle.
