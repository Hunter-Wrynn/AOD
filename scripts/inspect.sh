#!/usr/bin/env bash
# Inspect a VLM's HF config without loading the weights.
#
# Usage:
#   bash scripts/inspect.sh <model>
#
# <model>: qwen2_5vl | llava | internvl3   (or a raw HF repo id)

set -euo pipefail

MODEL_ALIAS="${1:-qwen2_5vl}"
PY="${PY:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$MODEL_ALIAS" in
  qwen2_5vl|qwen)     MODEL_ID="qwen2_5_vl" ;;
  llava)              MODEL_ID="llava" ;;
  internvl3|internvl) MODEL_ID="internvl" ;;
  *)                  MODEL_ID="$MODEL_ALIAS" ;;
esac

shift || true
"$PY" "$REPO_ROOT/cli/inspect_vlm_config.py" --model_id "$MODEL_ID" "$@"
