#!/bin/sh
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
if [ ! -f models/Qwen3-0.6B/model.safetensors ]; then
  HF_HOME="$APP_DIR/.download-cache" HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/download_model.py --destination models/Qwen3-0.6B --receipt model-receipt.json
fi
printf '\nSetup complete. Run ./start.sh and open http://127.0.0.1:8765\n'
