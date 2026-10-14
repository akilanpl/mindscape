#!/bin/zsh
set -eu
research_root="${0:A:h:h:h}"
python_path="${MIND1_NATIVE_PYTHON:-/Users/akilan/Documents/Codex/2026-10-08/files-pasted-by-the-user-mindscape/outputs/mindscape/work/final-venv/bin/python}"
if [[ $# -ne 1 ]]; then
  print -u2 'Usage: zsh native_mps_preflight.command /absolute/new/receipt.json'
  exit 2
fi
[[ -x "$python_path" ]] || { print -u2 'Set MIND1_NATIVE_PYTHON to the verified native interpreter'; exit 2; }
if [[ "${PYTORCH_ENABLE_MPS_FALLBACK:-0}" != 0 ]]; then
  print -u2 'Disable MPS CPU fallback before preflight'
  exit 2
fi
export PYTORCH_ENABLE_MPS_FALLBACK=0
exec "$python_path" "$research_root/scripts/native/mps_preflight.py" --output "$1"
