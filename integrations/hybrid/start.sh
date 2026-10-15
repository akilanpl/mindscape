#!/bin/sh
set -eu
APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$APP_DIR"
if [ -n "${MINDSCAPE_PYTHON:-}" ]; then
  PYTHON_BIN="$MINDSCAPE_PYTHON"
elif [ -x "$APP_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$APP_DIR/.venv/bin/python"
else
  echo "Run ./setup.sh first, or set MINDSCAPE_PYTHON to your installed environment."
  exit 1
fi
export PYTHONPATH="$APP_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MINDSCAPE_MODEL_PATH="${MINDSCAPE_MODEL_PATH:-$APP_DIR/models/Qwen3-0.6B}"
if [ "${MINDSCAPE_TOOLS_ONLY:-0}" = 1 ]; then
  exec "$PYTHON_BIN" -m mindscape_delivery.server --port "${MINDSCAPE_PORT:-8765}" --tools-only
fi
exec "$PYTHON_BIN" -m mindscape_delivery.server --port "${MINDSCAPE_PORT:-8765}" --device "${MINDSCAPE_DEVICE:-cpu}"
