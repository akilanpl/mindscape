#!/bin/zsh
set -eu
research_root="${0:A:h:h:h}"
cd "$research_root"
export PYTHONPATH="$research_root/src:$research_root/work/physics-deps/site-packages"
exec /Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python -m mindscape.v3.integration "$@"
