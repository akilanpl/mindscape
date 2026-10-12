# Native pilot and reproducibility

From normal macOS Terminal:

```zsh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v3_1_physics/native_pilot.command
```

The pinned interpreter and isolated Pymunk dependency are reused. Actual MPS allocation is mandatory, fallback is rejected, source/dataset/protocol hashes guard identity, and logs are captured in a new results/v3_1_physics/native-pilot-* directory. Multi-step native interrupted6+6 versus uninterrupted12 updates must produce identical model and Adam state before pilot fitting proceeds. Model checkpoints retain generation hashes and progress; unsafe pickle loading is never enabled.

If interrupted, retain the original source, frozen data and run directory:

```zsh
/Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3/scripts/v3_1_physics/native_pilot.command --resume /absolute/path/to/original/native-pilot-directory
```

CPU tests and independent verification:

```zsh
cd /Users/akilan/Documents/Codex/2026-10-12/you-are-gpt-6-1-sol/work/mindscape-v3
export PYTHONPATH=src:work/physics-deps/site-packages
physics_python=/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python
$physics_python -m pytest tests_physics_v3_1 -q
$physics_python scripts/v3_1_physics/verify.py /absolute/path/to/completed/native-pilot-directory
```

There is no full-study launch mode. Sealed evaluation remains unrun unless a future preregistration authorizes it. Observed results and GO/NO-GO must come from verified receipts, not launcher completion alone.
