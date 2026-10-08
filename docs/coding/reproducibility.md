# Reproduction and immutable evidence

Use Python3.12 and the measured versions in `experiments/coding_completion_v2/requirements-measured.txt`. The evaluated machine uses a CPU float32 pipeline, with no paid services. Initial model/runtime downloads require network access; evaluated programs have no network dependency or capability. The bootstrap script pins HF revisions and validates the WASI archive checksum.

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
work/final-venv/bin/python scripts/coding/zero_budget.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/stress_study.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/latency_probe.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/sql_study.py --model "$mindscape_model"
work/final-venv/bin/python scripts/coding/verify_final_traces.py
work/final-venv/bin/python scripts/coding/completion_analysis.py
work/final-venv/bin/python scripts/coding/build_demo.py
work/final-venv/bin/python -m pytest -q
work/final-venv/bin/python -m ruff check src/mindscape/coding src/mindscape/sql scripts/coding tests/test_coding* tests/test_sql_vertical.py tests/test_lora_checkpoint.py
```

Public HumanEval is separately downloaded at its recorded official revision with `download_humaneval.py` and run using `humaneval.py --model SNAPSHOT --output NEW_VERSIONED_ROOT`. Use the exact recorded greedy512-token, one-completion, no-feedback configuration. Never tune on the final lockbox or public results. `fairness_audit.py` and `training_token_audit.py` provide automated local audits and exact token recounting; the final report distinguishes these from unknown foundation-model contamination.

The recorded demo is `demo/coding/index.html`, generated exclusively from completed final episodes. Open it directly or serve locally with `python -m http.server --bind 127.0.0.1 --directory demo/coding 8765`. It shows actual typed actions and visible execution, omits private expected values and raw model reasoning, and labels playback. Its timing is not the fresh latency benchmark.

`freeze_completion.py` copies completed data, actual tuples, adapters, raw results, protocols, scripts, docs and demo into `results/final/coding_research_v2`, emits SHA-256 for every frozen file and refuses overwrite. Foundation weights remain in the pinned local HF cache; their public revision identifiers and bootstrap commands are recorded. The historical `mindscape-final-study-v1` release remains intact. Different hardware/library kernels can produce numerical or generation differences; bitwise cross-platform equivalence is not claimed.

The full historical source tree has201 pre-existing Ruff style findings, independently confirmed against commit3e2538f. This completion introduces none; the scoped coding/SQL checks pass. The complete functional test suite remains the acceptance gate.
