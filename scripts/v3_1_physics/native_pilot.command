#!/bin/zsh
set -eu
research_root="${0:A:h:h:h}"
research_python=/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python
cd "$research_root"
export PYTHONPATH="$research_root/src:$research_root/work/physics-deps/site-packages"
exec "$research_python" scripts/v3_1_physics/launch.py "$@"
