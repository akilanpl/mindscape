#!/bin/zsh
set -eu
research_root="${0:A:h:h:h}"
cd "$research_root"
export PYTHONPATH="$research_root/src:$research_root/work/physics-deps/site-packages"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python -u -m mindscape.v4 "$@"
