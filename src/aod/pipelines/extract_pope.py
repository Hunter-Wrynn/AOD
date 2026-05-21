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


def jsonl_iter(path: str) -> Iterable[Dict[str, Any]]:
    """POPE distribution files use one JSON object per line, often with a `.json` suffix."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_layers_on_pope_jsonl(
    model_id: str,
    jsonl_path: str,
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

    for item_idx, item in tqdm(enumerate(jsonl_iter(jsonl_path))):
        image_rel = item.get("image")
        question = item.get("text")
        gt = str(item.get("label", "")).strip().lower()
        if not image_rel or not question or gt not in {"yes", "no"}:
            continue

        image_path = image_rel if os.path.isabs(image_rel) else os.path.join(image_root, image_rel)
        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

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
            base_pred = "yes" if score_yes >= score_no else "no"

            is_hallucination = base_pred != gt
            is_object_hallucination = (gt == "no" and base_pred == "yes")

            meta = {
                "id": int(item_idx),
                "image": image_rel,
                "question": question,
                "base_pred": base_pred,
                "gt": gt,
                "gt_answer": "1" if gt == "yes" else "0",
                "model_prediction": "1" if base_pred == "yes" else "0",
                "is_consistent": base_pred == gt,
                "is_hallucination": is_hallucination,
                "is_object_hallucination": is_object_hallucination,
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
            f"No records extracted from {jsonl_path}; failed_samples={failed}. "
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
    ap.add_argument("--jsonl_path", required=True, help="POPE-style JSONL file (one JSON object per line).")
    ap.add_argument("--image_root", required=True, help="Root dir containing images referenced by JSONL")
    ap.add_argument("--output_dir", default="output_cache_pope")
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

    extract_layers_on_pope_jsonl(
        model_id=model_id,
        jsonl_path=args.jsonl_path,
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
