from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_last_nonpad_index(attention_mask: torch.Tensor) -> int:
    nz = (attention_mask != 0).nonzero(as_tuple=False)
    return int(nz[-1].item()) if nz.numel() else 0


@dataclass(frozen=True)
class YesNoTokenIds:
    yes_ids: Sequence[int]
    no_ids: Sequence[int]


def get_yes_no_token_ids(tokenizer) -> YesNoTokenIds:
    yes_words = ["Yes", "yes", " Yes", " yes"]
    no_words = ["No", "no", " No", " no"]
    yes_ids: List[int] = []
    no_ids: List[int] = []
    for w in yes_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            yes_ids.append(ids[-1])
    for w in no_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            no_ids.append(ids[-1])
    yes_ids = list(dict.fromkeys(yes_ids))
    no_ids = list(dict.fromkeys(no_ids))
    return YesNoTokenIds(yes_ids=yes_ids, no_ids=no_ids)


def build_inputs(processor, image: Image.Image, question: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{question}\nAnswer with Yes or No."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(images=image, text=text, return_tensors="pt")


def extract_qwen2_5vl_layers_on_pope_jsonl(
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

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("AutoProcessor did not expose a tokenizer; cannot score Yes/No logits.")
    yn = get_yes_no_token_ids(tokenizer)
    if not yn.yes_ids or not yn.no_ids:
        raise RuntimeError("Failed to derive Yes/No token ids from tokenizer.")

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
            inputs = build_inputs(processor, image, question)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            input_ids: torch.Tensor = inputs["input_ids"]
            attention_mask: torch.Tensor = inputs.get("attention_mask", torch.ones_like(input_ids))

            # Match the LLaVA extraction semantics:
            # extract the last prompt token representation (typically `[:, -1, :]`),
            # i.e. the last non-pad index.
            token_index = get_last_nonpad_index(attention_mask[0])

            logits = outputs.logits[0, token_index, :]
            score_yes = max(float(logits[i].item()) for i in yn.yes_ids)
            score_no = max(float(logits[i].item()) for i in yn.no_ids)
            base_pred = "yes" if score_yes >= score_no else "no"

            is_hallucination = base_pred != gt
            is_object_hallucination = (gt == "no" and base_pred == "yes")

            meta = {
                "id": int(item_idx),
                "image": image_rel,
                "question": question,
                "base_pred": base_pred,
                "gt": gt,
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
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--hf_home", default=os.environ.get("HF_HOME", ""))
    ap.add_argument("--jsonl_path", required=True, help="POPE-style JSONL file")
    ap.add_argument("--image_root", required=True, help="Root dir containing images referenced by JSONL")
    ap.add_argument("--output_dir", default="output_cache_qwen2_5vl")
    ap.add_argument("--layers", default="1,4,8,12,16,20,24,28", help="e.g. '1,4,12,24,28' or '1-28'")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    ap.add_argument("--dry_run", action="store_true", help="Only print model config and exit.")
    args = ap.parse_args(argv)

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    print(f"[Model] {args.model_id}")
    print(
        f"[Config] num_hidden_layers={getattr(cfg,'num_hidden_layers',None)} "
        f"hidden_size={getattr(cfg,'hidden_size',None)} model_type={getattr(cfg,'model_type',None)}"
    )
    if args.dry_run:
        return 0

    layers = parse_layers(args.layers)
    if not layers:
        raise ValueError("Empty --layers.")

    extract_qwen2_5vl_layers_on_pope_jsonl(
        model_id=args.model_id,
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
