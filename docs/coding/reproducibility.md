# Reproduction and immutable evidence

Use Python3.12 and the measured versions in `experiments/coding_completion_v2/requirements-measured.txt`. Preserved training/evaluation uses CPU float32; the authorized continuation uses actual MPS float16 with independent requests batched up to 16. No paid services are used. Initial model/runtime downloads require network access; evaluated programs have no network dependency or capability. The bootstrap script pins HF revisions and validates the WASI archive checksum.

Run from the repository root, in a **fresh checkout/output directory**. Existing frozen datasets and final snapshots intentionally refuse overwrite. Preserve existing results before reproducing; never delete historical artifacts. The research scripts use versioned root paths, and exact generation-cache keys include model revision and adapter-weight SHA. Cache hits preserve actual prior responses but have zero newly consumed model calls/tokens/time. Fresh latency disables this cache.

```sh
python3.12 -m venv work/final-venv
work/final-venv/bin/python -m pip install -r experiments/coding_completion_v2/requirements-measured.txt
export PYTHONPATH=src
work/final-venv/bin/python scripts/coding/bootstrap.py
work/final-venv/bin/python scripts/coding/reconstruct_datasets.py
work/final-venv/bin/python scripts/coding/freeze_dataset.py
work/final-venv/bin/python scripts/coding/validate_dataset.py
work/final-venv/bin/python scripts/coding/freeze_lockbox.py
```

The model argument is the downloaded snapshot directory with the exact1.5B revision. Set a task-specific shell variable to that directory (do not overwrite HOME or CODEX_HOME). The following sequence makes the shared dependency artifacts concrete:

```sh
mindscape_model=work/coding/hf/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a
work/final-venv/bin/python scripts/coding/study.py --model "$mindscape_model" --train 5000 --test 20 --budgets 10,25,50,100,250,500,1000,2500,5000 --seeds 11,23,37,53,71 --output results/coding/completion_collection_v1 --locked-from results/coding/final_dataset_v1/dataset.json --collect-only
```

On the original completion run, accepted records were recovered from a prior collector, not regenerated. For a fresh reproduction, copy the accepted compact training-only memory to `results/coding/final_retrieval_v1/teacher_memory.json` because the gradient comparison reads that historical-compatible path; the full tuple archive remains `results/coding/teacher_traces_v1`. This is data routing, not result fabrication. Actual collection produces and verifies these files before training. No held-out correct source or private feedback belongs in learner inputs.

```sh
work/final-venv/bin/python scripts/coding/gradient_pipeline.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/completion_gradient.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/lockbox_eval.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/latency_probe.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/verify_final_traces.py
work/final-venv/bin/python scripts/coding/completion_analysis.py
work/final-venv/bin/python scripts/coding/build_demo.py
work/final-venv/bin/python -m pytest -q
work/final-venv/bin/python -m ruff check src/mindscape/coding src/mindscape/sql scripts/coding tests/test_coding* tests/test_sql_vertical.py tests/test_lora_checkpoint.py
```

Public HumanEval is separately downloaded at its recorded official revision with `download_humaneval.py` and run using `humaneval.py --model SNAPSHOT --output NEW_VERSIONED_ROOT`. Use the exact recorded greedy512-token, one-completion, no-feedback configuration. Never tune on the final lockbox or public results. `fairness_audit.py` and `training_token_audit.py` provide automated local audits and exact token recounting; the final report distinguishes these from unknown foundation-model contamination.

The recorded demo is `demo/coding/index.html`, generated exclusively from completed final episodes. Open it directly or serve locally with `python -m http.server --bind 127.0.0.1 --directory demo/coding 8765`. It shows actual typed actions and visible execution, omits private expected values and raw model reasoning, and labels playback. Its timing is not the fresh latency benchmark.

`freeze_emergency.py` applies the strict 400-lockbox/1920-learning/dedicated-latency publication gate and copies completed data, actual tuples, adapters, raw results, protocols, scripts, docs and demo into `results/final/coding_research_v2`, emits SHA-256 for every frozen file and refuses overwrite. Foundation weights remain in the pinned local HF cache; their public revision identifiers and bootstrap commands are recorded. The historical `mindscape-final-study-v1` release remains intact. Different hardware/library kernels can produce numerical or generation differences; bitwise cross-platform equivalence is not claimed.

The full historical source tree has201 pre-existing Ruff style findings, independently confirmed against commit3e2538f. This completion introduces none; the scoped coding/SQL checks pass. The complete functional test suite remains the acceptance gate.

### Restore the immutable package

To verify every frozen SHA-256 and reconstruct the original repository layout in a new empty directory:

```sh
work/final-venv/bin/python scripts/coding/restore_completion.py --snapshot results/final/coding_research_v2 --destination work/restored_research_v2
```

The restore command refuses a nonempty destination. Shared configurations, scripts, documentation, experiments, tests, source code and actual result/checkpoint files are included. It does not create or reset Git history. Foundation weights remain a separately pinned download. Use the release tag for the complete historical Git repository.

### Exact completed-study evidence

The authoritative analysis is `scripts/coding/emergency_report.py`, with the original scientific configuration in `results/coding/emergency_mps_v1/locked_protocol.json`. `complete_research.py` resumes only missing frozen keys on native MPS; it does not reprofile or tune on locked outcomes. `finalize_research.py --wait` performs independent replay, local fairness audits, complete tests, report generation, strict publication gating, freezing and clean restoration. Run it only in the study checkout; it may create the completion tag only after all gates pass.

The commands above include historical data/training entry points, not an instruction to rerun a saved completed study. Restore the final package and recompute metrics directly for exact evidence reproduction. A newly generated study must use a fresh directory, record its actual device/precision, and retain separate CPU/MPS strata; cross-kernel bitwise generation identity is not promised. The original 1122 CPU rows are byte-preserved, while the remaining cases use the measured MPS configuration.

Zero-budget, injected-recovery, SQL model evaluation, and component-ablation entry points exist but were not executed in this completion scope. They are optional new experiments and cannot supply claims for the saved release. The completion scope is exactly four conditions × 100 locked tasks, four conditions × three seeds × four budgets × 40 learning tasks, and 16 dedicated sequential latency cases. Sequential latency uses actual batch1; batch16 evaluation throughput is reported separately.
