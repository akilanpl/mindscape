# Reproduce the final study

Python3.11+ and a local machine with sufficient memory for a135M frozen model. Tested Python3.12.14, NumPy2.3.5, PyTorch2.14.1, Transformers4.57.6 on Apple M5/16GB, CPU. No paid service or API key. Initial package/model downloads need internet; subsequent generation, adaptation and evaluation work offline. Exact installed optional package versions are in `experiments/final_audit/local_requirements.txt`. ReportLab renders scientific SVG charts.

Run from repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[learning,plots,pretrained]'
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/download_final_model.py
```

The downloader prints the pinned snapshot path. Use that path below. Local weights are Apache2.0, from https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct . Backbone revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The head is trained locally. Cache folders are ignored by Git; do not mistake bundled results for bundled pretrained weights.

```sh
PYTHONPATH=src python scripts/generate_data.py \
  --config configs/experiments/final_dataset.toml \
  --output datasets/generated/final_reproduction
PYTHONPATH=src python scripts/run_final_study.py \
  --dataset datasets/generated/final_reproduction \
  --model work/hf/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --output results/final/reproduction_v1
PYTHONPATH=src python scripts/analyze_final_study.py results/final/reproduction_v1
PYTHONPATH=src python scripts/audit_final_artifacts.py results/final/reproduction_v1 \
  --dataset datasets/generated/final_reproduction
PYTHONPATH=src python scripts/final_references.py \
  --dataset datasets/generated/final_reproduction --output results/final/references_reproduction
PYTHONPATH=src python scripts/build_final_demo.py results/final/reproduction_v1 \
  --dataset datasets/generated/final_reproduction --output demo/reproduction.html
```

All output roots must be new. The frozen study fails rather than overwriting an existing result. The full command adapts84 primary models and6 ablation models, reloads every checkpoint and evaluates each condition on IID/OOD. It may take substantial CPU time; the transformer feature cache amortizes repeated target-free state encodings. A small smoke study can use a new TOML with budgets `[10]`, seeds `[0]`, steps1200 and attempts_per_problem64; currently the full runner's ablation stage expects n50, so use budgets `[10,50]` when running that stage. Main final analyses require the declared complete three-seed/seven-budget matrix.

Evaluate one final checkpoint on a fresh output root:

```sh
PYTHONPATH=src python scripts/evaluate_final.py \
  --model work/hf/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --checkpoint results/final/reproduction_v1/full_trajectory_n1000_s0/checkpoint \
  --dataset datasets/generated/final_reproduction \
  --split ood_test --output results/rechecks/ood_v1
```

Inspect `predictions.jsonl` in each UUID evaluation folder. Fields include predicted/target answer, complete predicted trajectory, correctness, goal, groundedness and diagnostics. Correct-action/transition judgments are posthoc evaluation annotations; model inputs exclude them. Configs contain exact training IDs, regime and model metadata. Saved heads use NPZ without pickle. Ten summary SVG figures and per-run outcome plots are generated from measured runs. `statistics.json`, `difficulty_failures.json`, `failure_taxonomy.json`, `data_efficiency.json` and `claims.json` are machine-readable.

Reproduce the historical diagnostics with `scripts/audit_final.py` on a new copy or rename its new output directory: it intentionally refuses overwrites. Preserve original results rather than deleting them. Dream demonstration: `PYTHONPATH=src python scripts/run_dream_demo.py`. View the static evidence UI by opening `demo/index.html` or serving `demo/` with `python -m http.server 8000 --directory demo`.

Free external compute is optional, not used in this study. No external scheduler/service is required. Reproductions have new timestamps and measured runtime; exact predictions should match with the pinned checkpoint, seed, CPU implementation and package versions, while other hardware/kernel versions may differ numerically.

Additional checks/references:

```sh
PYTHONPATH=src python scripts/final_sanity.py results/final/reproduction_v1 \
  --model work/hf/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --dataset datasets/generated/final_reproduction
PYTHONPATH=src python scripts/analyze_holdout_coverage.py results/final/reproduction_v1 \
  --dataset datasets/generated/final_reproduction
PYTHONPATH=src python scripts/final_historical_mlp.py \
  --dataset datasets/generated/final_reproduction --output results/final/historical_mlp_reproduction
```

The last reference repeats the historical MLP at1000 problems, three seeds, on the fresh final dataset. It does not equalize its architecture, losses or update budget with the final frozen-backbone adaptation, so their difference does not isolate pretraining benefits.

Finalize measured resource reporting and digest the complete validated study:

```sh
PYTHONPATH=src python scripts/summarize_final_costs.py results/final/reproduction_v1
PYTHONPATH=src python scripts/freeze_final_artifacts.py results/final/reproduction_v1
```

The SHA256 freeze manifest covers raw predictions, checkpoints, metadata, statistics and plots. Generate all analyses and audits before freezing. Do not modify that folder afterward; use a new output root for a reproduction or correction.

Before final freezing, preserve separate hypothetical posthoc judgments without rewriting raw predictions:

```sh
PYTHONPATH=src python scripts/annotate_final_hypotheticals.py results/final/reproduction_v1
```
