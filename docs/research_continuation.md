# Research completion continuation

The completion request supersedes the emergency run's operational two-hour cutoff. The original frozen scientific protocol and historical emergency package remain unchanged.

`complete_research.py` resumes C first, then B's missing lockbox cases, remaining learning groups, and a dedicated 16-case latency study. It uses the already selected batch size 16 and float16 MPS path with one resident model/tokenizer. Dedicated latency measures sequential requests, whose actual batch size is 1, separately from batched evaluation throughput. Filesystem, tokenization, and WASI execution remain CPU operations.

A kernel-held file lock prevents simultaneous continuation launches. SIGINT/SIGTERM request a checkpointed stop after the active group; completed cases are fsynced individually. Restart the same command after an operational failure. Existing keys are skipped, scientific hashes are checked, and missing cases never receive invented scores. Failures are saved in `results/coding/research_continuation_v1/failure.json`; completion is recorded only after exact key-set checks.

Native execution is necessary: this tool execution environment cannot allocate MPS tensors, while the user's native Terminal has verified MPS availability and performed the earlier actual MPS evaluation. Native Terminal computer control was rejected for safety reasons. No CPU fallback is used.

Run from any native Terminal directory:

```sh
env -u PYTORCH_ENABLE_MPS_FALLBACK /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/scripts/coding/complete_research.py > /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/research_completion.log 2>&1
```

After neural completion, independent replay, analysis, full tests, fairness checks, report reconciliation, immutable packaging, clean restoration, and hash verification are still required before the completion tag. The freeze gate requires 400 lockbox cases, 1920 learning cases, replay agreement, dedicated latency, and passing tests. A completion tag has not been created.

Verified prelaunch state: 296/400 lockbox cases (A100, structured100, B96, C0), 1122/1920 learning cases, unchanged frozen sources and byte-identical preserved CPU rows, no duplicate keys. Functional tests: 129 tests plus 6 subtests pass; scoped lint passes; fairness audit: 10158 assertions pass. Foundation pretraining exposure remains unknown; these checks do not establish universal contamination freedom.
