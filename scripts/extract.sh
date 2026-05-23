#!/usr/bin/env bash
# Extract hidden states for AOD direction training.
#
# Usage:
#   bash scripts/extract.sh <model> <benchmark> [extra args...]
#
# <model>:     qwen2_5vl | llava | internvl3
# <benchmark>: pope | pope_random | pope_adversarial | hallusionbench | amber
#
# Examples:
#   bash scripts/extract.sh qwen2_5vl pope             --layers 24
#   bash scripts/extract.sh qwen2_5vl pope_random      --layers 24
#   bash scripts/extract.sh qwen2_5vl pope_adversarial --layers 24
#   bash scripts/extract.sh llava     hallusionbench   --layers 24
#   bash scripts/extract.sh internvl3 amber            --layers 24 --typology existence

set -euo pipefail

if [[ $# -lt 2 ]]; then
  awk '/^#/{print; next} {exit}' "$0"
  exit 1
fi

MODEL_ALIAS="$1"; BENCH="$2"; shift 2
PY="${PY:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Map model alias → --model_id value (the family alias accepted by aod.vlm.loader).
case "$MODEL_ALIAS" in
  qwen2_5vl|qwen) MODEL_ID="qwen2_5_vl"; MODEL_DIR="qwen2_5vl" ;;
  llava)          MODEL_ID="llava";      MODEL_DIR="llava" ;;
  internvl3|internvl) MODEL_ID="internvl"; MODEL_DIR="internvl3" ;;
  *) echo "Unknown model alias: $MODEL_ALIAS (expected qwen2_5vl|llava|internvl3)" >&2; exit 2 ;;
esac

case "$BENCH" in
  pope|pope_popular)
    "$PY" "$REPO_ROOT/cli/extract_vlm_layers_pope.py" \
      --model_id "$MODEL_ID" \
      --jsonl_path data/POPE/coco_pope_popular.json \
      --image_root data/POPE/val2014 \
      --output_dir "output/layers/${MODEL_DIR}_pope" \
      --layers 24 "$@"
    ;;
  pope_random)
    "$PY" "$REPO_ROOT/cli/extract_vlm_layers_pope.py" \
      --model_id "$MODEL_ID" \
      --jsonl_path data/POPE/coco_pope_random.json \
      --image_root data/POPE/val2014 \
      --output_dir "output/layers/${MODEL_DIR}_pope_random" \
      --layers 24 "$@"
    ;;
  pope_adversarial)
    "$PY" "$REPO_ROOT/cli/extract_vlm_layers_pope.py" \
      --model_id "$MODEL_ID" \
      --jsonl_path data/POPE/coco_pope_adversarial.json \
      --image_root data/POPE/val2014 \
      --output_dir "output/layers/${MODEL_DIR}_pope_adversarial" \
      --layers 24 "$@"
    ;;
  hallusionbench|hb)
    "$PY" "$REPO_ROOT/cli/extract_vlm_layers_hallusionbench.py" \
      --model_id "$MODEL_ID" \
      --hb_json_path data/hallusion_bench/HallusionBench.json \
      --image_root data/hallusion_bench/hallusion_bench \
      --output_dir "output/layers/${MODEL_DIR}_hallusionbench" \
      --layers 24 "$@"
    ;;
  amber)
    "$PY" "$REPO_ROOT/cli/extract_vlm_layers_amber.py" \
      --model_id "$MODEL_ID" \
      --query_path data/AMBER/data/query/query_all.json \
      --annotations_path data/AMBER/data/annotations.json \
      --image_root data/AMBER/image \
      --output_dir "output/layers/${MODEL_DIR}_amber" \
      --layers 24 "$@"
    ;;
  *)
    echo "Unknown benchmark: $BENCH (expected pope|pope_random|pope_adversarial|hallusionbench|amber)" >&2; exit 2 ;;
esac
