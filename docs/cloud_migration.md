# Mindscape Cloud migration checkpoint

This is a continuation of the frozen study. The local native runner was stopped through its checkpoint-boundary SIGTERM handler; all 400 locked cases are now present (A100, structured100, B100, C100). All 400 independently replayed state/tool/terminal records agree. The 1122 original CPU learning cases are byte-preserved; 798 remain. Dedicated latency has not yet run. The project remains partial and has no research-completion tag.

The private repository is `akilanpl/mindscape`; migration branch: `mindscape-cloud-migration-2026-10-08`. Starting local commit: `d034a9b`. The Git checkpoint includes all raw research result directories, historical immutable packages, datasets, checkpoint weights/metadata, WASI source runtime, public benchmark inputs, research logs, source/scripts/configuration/docs/tests/demo and experiment receipts. Required generated evidence is retained in versioned `research_checkpoint/2026-10-08` archive parts, each at most 32 MiB, with SHA-256 per part and per restored file. Foundation model weights remain exact pinned external downloads with shipped hashes. Virtual environments, credentials, generation caches, machine auth logs, Python/OS caches and disposable compiled-runtime caches are excluded; immutable historical package bytes are preserved.

Restore into a clean destination with the existing package verifier:

```sh
python scripts/coding/migration_checkpoint.py restore --destination work/migration_restore_verified
```

The unpacker verifies every archive part and the snapshot manifest, then calls `restore_completion.py` for complete inventory/hash validation and restoration. Restored artifacts belong under their original relative paths; source/code also exist in Git. Cloud must checkout the exact pushed checkpoint, verify it, restore the artifact archive, and route only restored results/datasets/runtime/public-input/logs into the checkout. Do not overwrite Git source files from an older restore.

## Cloud continuation

Inspect actual hardware/framework/RAM/disk; never assume a GPU. Start the persistent 28800-second budget when actual Cloud continuation begins, including setup; never reset it on resume. Preserve the budget file when restoring artifacts. Bootstrap the exact pinned 1.5B backbone only; do not retrain or regenerate datasets. Validate frozen source, dataset, memory and all adapter hashes before inference.

`cloud_runtime.py` is an execution-only adapter around the unchanged frozen resident model/controller. CUDA uses float16 after actual tensor allocation; CPU uses the previously supported float32 reference path, explicitly recorded as an arithmetic/hardware stratum. It fixes inherited MPS-only timing labels to report the actual backend. A regression test performs real CPU neural forwards and compares greedy single/batched outputs. No scientific policy file, prompt, task, score, seed, checkpoint or training data changes.

`cloud_continue.py` profiles real initial requests on already exposed diagnostic tasks at batch1/4/8/16 where feasible, chooses throughput without locked-outcome tuning, and resumes only missing learning keys. It keeps the model/tokenizer resident, preserves per-case fragments durably, merges completed rows even after operational exceptions, and reserves an hour for latency/audit/report/package work. It logs CPU/CUDA resources and actual batches. Operational failures leave incomplete cases unscored; they never receive fabricated outcomes. Dedicated latency uses 16 fresh sequential cases and actual backend labels. CPU/tokenizer/WASI/Git operations remain CPU utilities.

```sh
PYTHONPATH=src python scripts/coding/cloud_continue.py
```

After neural work: independently validate final locked evidence, rerun fairness, complete tests/lint/compile/build, recompute reports/plots, preserve final raw results and verify clean package restoration. If all 1920 learning cases and every mandatory release criterion pass, create `mindscape-research-complete-v1` only after the final commit is pushed and independently checked. Otherwise publish a precise partial checkpoint with unrun counts, no completion tag. Original MPS evidence and historical negative studies remain intact.

Actual locked task success: A IID8/50, OOD12/50; structured IID50/50, OOD22/50; B IID45/50, OOD24/50; C IID49/50, OOD18/50. Successful replay is not successful repair. Different supervision, interaction compute, hardware and arithmetic prevent architecture-only attribution. Semantic hallucination, injected-fault recovery, zero-budget and component intervention metrics are unmeasured; supported natural self-correction is reported separately. No GPT/Gemini superiority is established.
