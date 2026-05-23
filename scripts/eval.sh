#!/usr/bin/env bash
# Evaluate an AOD-trained direction (or the base model) on any paper benchmark.
#
# Usage:
#   bash scripts/eval.sh <model> <benchmark> [extra args...]
#
# <model>:     qwen2_5vl | llava | internvl3
# <benchmark>: pope | pope_random | pope_adversarial             (binary Yes/No)
#              hallusionbench | amber                            (binary Yes/No)
#              realworldqa | mmstar | mmmu                       (multi-choice)
#              chair | ocrbench                                  (generative)
#              layers                                            (diagnostic on extracted hidden states)
#
# Hallucination benchmarks (pope/hallusionbench/amber/chair) use the
# in-domain trained direction. Utility benchmarks (realworldqa/mmstar/mmmu/
# ocrbench) reuse the POPE-trained direction per the paper protocol.
#
# Required: --ckpt <path>  (except for base mode and the `layers` diagnostic)
# Common flags: --mode {base|direct|cd}  --aod_alpha 1.0  --beta 0.5  --apc_alpha 0.1
#
# Examples:
#   bash scripts/eval.sh qwen2_5vl pope         --mode cd --ckpt output/aod_ckpt/qwen2_5vl_pope/aod_pope_layer_24.pt
#   bash scripts/eval.sh llava     mmmu         --mode cd --ckpt output/aod_ckpt/llava_pope/aod_pope_layer_24.pt
#   bash scripts/eval.sh internvl3 chair        --mode cd --ckpt output/aod_ckpt/internvl3_pope/aod_pope_layer_24.pt
#   bash scripts/eval.sh qwen2_5vl layers       --ckpt output/aod_ckpt/qwen2_5vl_pope/aod_pope_layer_24.pt --layers_dir output/layers/qwen2_5vl_pope

set -euo pipefail

if [[ $# -lt 2 ]]; then
  awk '/^#/{print; next} {exit}' "$0"
  exit 1
fi

MODEL_ALIAS="$1"; BENCH="$2"; shift 2
PY="${PY:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$MODEL_ALIAS" in
  qwen2_5vl|qwen) MODEL_ID="qwen2_5_vl"; MODEL_DIR="qwen2_5vl" ;;
  llava)          MODEL_ID="llava";      MODEL_DIR="llava" ;;
  internvl3|internvl) MODEL_ID="internvl"; MODEL_DIR="internvl3" ;;
  *) echo "Unknown model alias: $MODEL_ALIAS (expected qwen2_5vl|llava|internvl3)" >&2; exit 2 ;;
esac

case "$BENCH" in
  pope|pope_popular)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_binary.py" \
      --model_id "$MODEL_ID" \
      --dataset_format pope \
      --data_path data/POPE/coco_pope_popular.json \
      --image_root data/POPE/val2014 "$@"
    ;;
  pope_random)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_binary.py" \
      --model_id "$MODEL_ID" \
      --dataset_format pope \
      --data_path data/POPE/coco_pope_random.json \
      --image_root data/POPE/val2014 "$@"
    ;;
  pope_adversarial)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_binary.py" \
      --model_id "$MODEL_ID" \
      --dataset_format pope \
      --data_path data/POPE/coco_pope_adversarial.json \
      --image_root data/POPE/val2014 "$@"
    ;;
  hallusionbench|hb)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_binary.py" \
      --model_id "$MODEL_ID" \
      --dataset_format hallusionbench \
      --data_path data/hallusion_bench/HallusionBench.json \
      --image_root data/hallusion_bench/hallusion_bench "$@"
    ;;
  amber)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_binary.py" \
      --model_id "$MODEL_ID" \
      --dataset_format amber \
      --data_path data/AMBER/data/query/query_all.json \
      --amber_annotations_path data/AMBER/data/annotations.json \
      --image_root data/AMBER/image "$@"
    ;;
  realworldqa|rwqa)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_mc.py" \
      --model_id "$MODEL_ID" \
      --dataset_format realworldqa \
      --data_path data/RealWorldQA/test.jsonl \
      --image_root data/RealWorldQA/images "$@"
    ;;
  mmstar)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_mc.py" \
      --model_id "$MODEL_ID" \
      --dataset_format mmstar \
      --data_path data/MMStar/test.jsonl \
      --image_root data/MMStar/images \
      --category_field category "$@"
    ;;
  mmmu)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_mc.py" \
      --model_id "$MODEL_ID" \
      --dataset_format mmmu \
      --data_path data/MMMU/validation.jsonl \
      --image_root data/MMMU/images \
      --category_field subject "$@"
    ;;
  chair)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_generative.py" \
      --model_id "$MODEL_ID" \
      --dataset_format chair_coco \
      --data_path data/coco/annotations/captions_val2014.json \
      --image_root data/coco/val2014 \
      --coco_instances_path data/coco/annotations/instances_val2014.json \
      --output_path "output/generations/${MODEL_DIR}_chair.jsonl" \
      --max_new_tokens 128 "$@"
    ;;
  ocrbench)
    "$PY" "$REPO_ROOT/cli/eval_vlm_aod_generative.py" \
      --model_id "$MODEL_ID" \
      --dataset_format ocrbench \
      --data_path data/OCRBench_v2/ocrbench_v2.jsonl \
      --image_root data/OCRBench_v2/images \
      --output_path "output/generations/${MODEL_DIR}_ocrbench.jsonl" \
      --max_new_tokens 64 "$@"
    ;;
  layers)
    "$PY" "$REPO_ROOT/cli/eval_aod_layers.py" "$@"
    ;;
  *)
    echo "Unknown benchmark: $BENCH" >&2
    echo "Expected: pope|pope_random|pope_adversarial|hallusionbench|amber|realworldqa|mmstar|mmmu|chair|ocrbench|layers" >&2
    exit 2
    ;;
esac
