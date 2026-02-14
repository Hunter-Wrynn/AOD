#!/usr/bin/env bash
set -euo pipefail

HF_HOME_DEFAULT="/data/xyc/mhx/ACL2026/.cache/huggingface"
PY_DEFAULT="/home/xiaoyicheng/miniconda3/envs/vilma/bin/python"

HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
PY="${PY:-$PY_DEFAULT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME
"$PY" "$REPO_ROOT/cli/inspect_qwen2_5vl_config.py" "$@"

