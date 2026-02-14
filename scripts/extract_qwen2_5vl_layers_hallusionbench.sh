#!/usr/bin/env bash
set -euo pipefail

HF_HOME="${HF_HOME:-/data/xyc/mhx/ACL2026/.cache/huggingface}"
PY="${PY:-/home/xiaoyicheng/miniconda3/envs/vilma/bin/python}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME
cd "$REPO_ROOT"

DEFAULT_ARGS=(
  --hb_json_path data/hallusion_bench/HallusionBench.json
  --image_root data/hallusion_bench/hallusion_bench
  --output_dir output/layers/qwen2_5vl_layers_hallusionbench
  --layers 1,4,8,12,16,20,24,28
)

"$PY" "$REPO_ROOT/cli/extract_qwen2_5vl_layers_hallusionbench.py" "${DEFAULT_ARGS[@]}" "$@"

