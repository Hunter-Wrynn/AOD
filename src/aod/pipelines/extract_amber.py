from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

from aod.data.amber import amber_to_records, load_amber_discriminative
from aod.vlm.loader import (
    DEFAULT_MODEL_IDS,
    build_yes_no_inputs,
    get_last_nonpad_index,
    get_yes_no_token_ids,
    load_vlm,
    resolve_default_model_id,
)


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


def resolve_image_path(image_root: str, filename: str | None) -> str | None:
    if not filename:
        return None
    rel = filename[2:] if filename.startswith("./") else filename
    candidate = rel if os.path.isabs(rel) else os.path.join(image_root, rel)
    return candidate if os.path.exists(candidate) else None


def extract_layers_on_amber(
    model_id: str,
    query_path: str,
    annotations_path: str,
    image_root: str,
    output_dir: str,
    layers: Sequence[int],
    *,
    typology: str | None = None,
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

    amber_records = load_amber_discriminative(query_path, annotations_path, typology=typology)
    flat = amber_to_records(amber_records)

    for item_idx, item in tqdm(enumerate(flat)):
        question = item["question"]
        gt = item["gt_answer"]  # "0" | "1"
        image_path = resolve_image_path(image_root, item["image"])
        if not question or image_path is None:
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        try:
            inputs = build_yes_no_inputs(loaded.processor, loaded.family, question, image)
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

            meta: Dict[str, Any] = {
                "id": int(item["id"]),
                "image": item["image"],
                "amber_type": item.get("amber_type"),
                "question": question,
                "gt_answer": gt,
                "model_prediction": model_prediction,
                "is_consistent": model_prediction == gt,
                "is_hallucination": model_prediction != gt,
            }

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states.")

            for l in layers:
                l = int(l)
                if l < 0 or l >= len(hidden_states):
                    continue
                vec = hidden_states[l][0, token_index, :].detach().float().cpu().tolist()
                rec = dict(meta)
                rec[f"layer_{l}_hidden"] = vec
                storage[l].append(rec)
        except Exception:
            continue

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
    ap.add_argument("--query_path", default="data/AMBER/data/query/query_all.json")
    ap.add_argument("--annotations_path", default="data/AMBER/data/annotations.json")
    ap.add_argument("--image_root", default="data/AMBER/image")
    ap.add_argument("--output_dir", default="output/layers/amber")
    ap.add_argument(
        "--typology",
        default="",
        choices=["", "existence", "attribute", "relation"],
        help="If set, only extract from this AMBER typology subset.",
    )
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
    print(f"[Model] {model_id}")
    print(
        f"[Config] num_hidden_layers={getattr(cfg,'num_hidden_layers',None)} "
        f"hidden_size={getattr(cfg,'hidden_size',None)} model_type={getattr(cfg,'model_type',None)}"
    )
    if args.dry_run:
        return 0

    layers = parse_layers(args.layers)
    if not layers:
        raise ValueError("Empty --layers.")

    extract_layers_on_amber(
        model_id=model_id,
        query_path=args.query_path,
        annotations_path=args.annotations_path,
        image_root=args.image_root,
        output_dir=args.output_dir,
        layers=layers,
        typology=(args.typology.strip() or None),
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
