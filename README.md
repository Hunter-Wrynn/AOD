Code base for AOD.

## Layer extraction (Qwen2.5-VL-7B)

- POPE (JSONL): `scripts/extract_qwen2_5vl_layers_pope.sh`
- HallusionBench (JSON): `scripts/extract_qwen2_5vl_layers_hallusionbench.sh`

## AOD training / eval (HallusionBench, extracted layers)

This repo supports training AOD on the extracted hidden states (e.g. `output/layers/qwen2_5vl_layers_hallusionbench/layer_24_dataset.json`)
and evaluating with an alpha sweep.

- Train (saves checkpoints per layer to `output/aod_ckpt/hallusionbench/`):
  - `python cli/train_aod_hallusionbench.py --layers_dir output/layers/qwen2_5vl_layers_hallusionbench --layers 1,4,8,12,16,20,24,28`

- Eval one checkpoint (alpha sweep):
  - `python cli/eval_aod_hallusionbench.py --ckpt output/aod_ckpt/hallusionbench/aod_hallusionbench_layer_24.pt`
