# Native commands and status

Run from normal macOS Terminal, not a Codex subprocess. The launcher never clears sandbox indicators or changes system security settings. Use the pinned intended interpreter; dependencies are isolated under `work/physics-deps/site-packages`. Pymunk7.2.0 is pinned; the original research environment and protected artifacts are unchanged.

```sh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v3_physics/native_diagnostic.command
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v3_physics/native_run.command --mode pilot
```

The pilot first performs float32/float16/bfloat16 × batches16/32/64/128 hardware microbenchmarks. Only stable float32 is eligible for selection; reduced precision remains exploratory. It then tests real world-model optimizer/checkpoint continuation, trains both environments/variants on two trajectories, runs batched rollouts, planning controls and independent replay. Every launch creates a fresh non-overwriting `results/v3_physics/native-pilot-*` directory. Neural tensors and weights remain on MPS; CPU is used only for simulation, preprocessing, statistics and serialization. Logs and command durations survive failures.

After the actual pilot passes and its receipts are independently reviewed:

```sh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v3_physics/native_run.command --mode full --pilot-receipt /Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/results/v3_physics/native-pilot-1791811351750339000/pilot-pass.json
# If interrupted, retain the original directory and source, then resume:
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v3_physics/native_run.command --mode full --pilot-receipt /absolute/path/to/pilot-pass.json --resume /absolute/path/to/native-full-...
```

Full scheduling is serial:90 dynamics fits (primary/matched across five budgets, plus five final-budget ablations), six model-free policy fits,18,000 neural prediction rollouts,1,200 nonneural prediction rollouts and2,160 planning episodes. The primary alone contributes exactly6,000 rollouts. Extra model-free teachers execute nine alternatives per approved training scene and have separate information/compute receipts. These are planned counts until actual acceptance coverage verifies them.

CPU verification after a completed native run:

```sh
cd /Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3
export PYTHONPATH=src:work/physics-deps/site-packages
physics_python=/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python
$physics_python scripts/v3_physics/verify.py --dataset results/v3_physics/dataset-full-v1 --run /absolute/path/to/native-full-.../study --output work/physics-verification.json
$physics_python scripts/v3_physics/acceptance.py --dataset results/v3_physics/dataset-full-v1 --run /absolute/path/to/native-full-.../study --output work/physics-acceptance.json
```

Corrected native pilot `1791811351750339000` passed engineering qualification: MPS float32, batch128, 40 forecasts and 44 selected-action replays independently verified. All 20 genuine checkpoints load with weights_only=True; both families have exact parameters, optimizer state, memory, step12 and provenance after interrupted/resumed versus uninterrupted native training. The full study `native-full-1791811860380119000` completed:6,000 primary forecasts,19,200 total forecasts and2,160 planning episodes independently verified. Its scientific improvement target failed. Do not reuse the earlier Phase0 “hardware unavailable” conclusion as a statement about native Terminal, and do not treat native diagnostic success as scientific model success. Preserved old receipts remain historical evidence.
