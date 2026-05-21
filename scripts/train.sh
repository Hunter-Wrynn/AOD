#!/usr/bin/env bash
# Train an AOD direction from extracted hidden states.
#
# Usage:
#   bash scripts/train.sh <layers_dir> [extra args...]
#
# Example:
#   bash scripts/train.sh output/layers/qwen2_5vl_pope --layers 24
#
# Defaults match the paper appendix (seed=42, 5 epochs, batch=256, lr=1e-3,
# probe hidden=512, --grl_lambda 1.0). Override with extra flags.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  awk '/^#/{print; next} {exit}' "$0"
  exit 1
fi

LAYERS_DIR="$1"; shift
PY="${PY:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Output directory mirrors the layers source directory by default.
BASENAME="$(basename "$LAYERS_DIR")"
OUTPUT_DIR="output/aod_ckpt/${BASENAME}"

"$PY" "$REPO_ROOT/cli/train_aod_layers.py" \
  --layers_dir "$LAYERS_DIR" \
  --layers 24 \
  --output_dir "$OUTPUT_DIR" "$@"
