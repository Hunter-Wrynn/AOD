#!/usr/bin/env bash
set -euo pipefail

HF_HOME="${HF_HOME:-/data/xyc/mhx/ACL2026/.cache/huggingface}"
PY="${PY:-/home/xiaoyicheng/miniconda3/envs/vilma/bin/python}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME
cd "$REPO_ROOT"

# Default run config (edit as needed)
DEFAULT_ARGS=(
  --jsonl_path data/POPE/coco_pope_popular.json
  --image_root data/POPE/val2014
  --output_dir output/layers/qwen2_5vl_layers_pope
  --layers 1,4,8,12,16,20,24,28
)

# You can still pass extra args after the script; if the same flag appears twice,
# argparse will take the last occurrence.
"$PY" "$REPO_ROOT/cli/extract_qwen2_5vl_layers_pope.py" "${DEFAULT_ARGS[@]}" "$@"
