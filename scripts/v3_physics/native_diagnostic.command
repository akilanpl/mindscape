#!/bin/zsh
set -eu
research_root="${0:A:h:h:h}"
research_python=/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python
cd "$research_root"
export PYTHONPATH="$research_root/src"
exec "$research_python" scripts/v3_physics/native_diagnostic.py --output-root "$research_root/results/v3_physics/native-diagnostics"
