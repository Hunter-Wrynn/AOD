from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

from aod.vlm.loader import (
    DEFAULT_MODEL_IDS,
    build_yes_no_inputs,
    get_last_nonpad_index,
    get_yes_no_token_ids,
    load_vlm,
    resolve_default_model_id,
)
from aod.vlm.intervention import first_parameter_device, move_tensor_inputs


def parse_layers(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return out


def iter_hallusionbench(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("HallusionBench.json must be a list of samples.")
    yield from data


def resolve_image_path(image_root: str, filename: str | None) -> str | None:
    if not filename:
        return None
    rel = filename[2:] if filename.startswith("./") else filename
    candidate = os.path.join(image_root, rel)
    return candidate if os.path.exists(candidate) else None


def normalize_gt_answer(gt: Any) -> str | None:
    if gt is None:
        return None
    gt_s = str(gt).strip().lower()
    if gt_s in {"0", "1"}:
        return gt_s
    if gt_s in {"no", "false"}:
        return "0"
    if gt_s in {"yes", "true"}:
        return "1"
    return None


def extract_layers_on_hallusionbench(
    model_id: str,
    hb_json_path: str,
    image_root: str,
    output_dir: str,
    layers: Sequence[int],
    dtype: str = "bf16",
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    storage: Dict[int, List[Dict[str, Any]]] = {int(l): [] for l in layers}

    loaded = load_vlm(
        model_id,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    yes_ids, no_ids = get_yes_no_token_ids(loaded.tokenizer)
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Failed to derive Yes/No token ids for {model_id}.")
    input_device = first_parameter_device(loaded.model)
    failed = 0

    for item_idx, item in tqdm(enumerate(iter_hallusionbench(hb_json_path))):
        question = item.get("question")
        if not question:
            continue

        gt = normalize_gt_answer(item.get("gt_answer"))
        if gt is None:
            continue

        visual_input = str(item.get("visual_input", "")).strip()
        filename = item.get("filename")

        image_path = resolve_image_path(image_root=image_root, filename=filename) if visual_input == "1" else None
        image: Image.Image | None
        if visual_input == "1" and image_path is None:
            continue
        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
        else:
            image = None

        try:
            inputs = build_yes_no_inputs(loaded.processor, loaded.family, question, image)
            inputs = move_tensor_inputs(inputs, input_device)
            with torch.inference_mode():
                outputs = loaded.model(**inputs, output_hidden_states=True, return_dict=True)

            attention_mask: torch.Tensor = inputs.get(
                "attention_mask", torch.ones_like(inputs["input_ids"])
            )
            token_index = get_last_nonpad_index(attention_mask[0])

            logits = outputs.logits[0, token_index, :]
            score_yes = max(float(logits[i].item()) for i in yes_ids)
            score_no = max(float(logits[i].item()) for i in no_ids)
            model_prediction = "1" if score_yes >= score_no else "0"

            is_hallucination = model_prediction != gt

            meta: Dict[str, Any] = {
                "id": int(item_idx),
                "question_id": item.get("question_id"),
                "figure_id": item.get("figure_id"),
                "set_id": item.get("set_id"),
                "category": item.get("category"),
                "subcategory": item.get("subcategory"),
                "visual_input": visual_input,
                "filename": filename,
                "question": question,
                "gt_answer": gt,
                "model_prediction": model_prediction,
                "is_consistent": model_prediction == gt,
                "is_hallucination": is_hallucination,
            }

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states; check transformers version and flags.")

            for l in layers:
                l = int(l)
                if l < 0 or l >= len(hidden_states):
                    continue
                vec = hidden_states[l][0, token_index, :].detach().float().cpu().tolist()
                rec = dict(meta)
                rec[f"layer_{l}_hidden"] = vec
                storage[l].append(rec)
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"[WARN] failed sample id={item_idx}: {type(exc).__name__}: {exc}")
            continue

    total_records = sum(len(v) for v in storage.values())
    if total_records == 0:
        raise RuntimeError(
            f"No records extracted from {hb_json_path}; failed_samples={failed}. "
            "Check image paths, processor prompt format, and model/device compatibility."
        )

    for l in layers:
        l = int(l)
        out_path = os.path.join(output_dir, f"layer_{l}_dataset.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(storage[l], f)
        print(f"[Saved] layer={l} records={len(storage[l])} -> {out_path}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_IDS["qwen2_5_vl"],
        help=(
            "HF repo id, or one of the family aliases: "
            + ", ".join(f"{k}={v}" for k, v in DEFAULT_MODEL_IDS.items())
        ),
    )
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", ""))
    ap.add_argument("--hb_json_path", default="data/hallusion_bench/HallusionBench.json")
    ap.add_argument("--image_root", default="data/hallusion_bench/hallusion_bench")
    ap.add_argument("--output_dir", default="output/layers/hallusionbench")
    ap.add_argument("--layers", default="24", help="e.g. '24' or '1,4,12,24,28' or '1-28'")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    ap.add_argument("--dry_run", action="store_true", help="Only print model config and exit.")
    args = ap.parse_args(argv)

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    model_id = resolve_default_model_id(args.model_id)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    text_cfg = getattr(cfg, "text_config", None)
    num_hidden_layers = getattr(cfg, "num_hidden_layers", None) or getattr(text_cfg, "num_hidden_layers", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(text_cfg, "hidden_size", None)
    print(f"[Model] {model_id}")
    print(
        f"[Config] num_hidden_layers={num_hidden_layers} "
        f"hidden_size={hidden_size} model_type={getattr(cfg,'model_type',None)}"
    )
    if args.dry_run:
        return 0

    layers = parse_layers(args.layers)
    if not layers:
        raise ValueError("Empty --layers.")

    extract_layers_on_hallusionbench(
        model_id=model_id,
        hb_json_path=args.hb_json_path,
        image_root=args.image_root,
        output_dir=args.output_dir,
        layers=layers,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
