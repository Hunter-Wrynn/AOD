from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from aod.data.amber import amber_to_records, load_amber_discriminative
from aod.core.aod import load_checkpoint
from aod.core.dataset import normalize_binary_answer
from aod.vlm.loader import (
    DEFAULT_MODEL_IDS,
    build_yes_no_inputs,
    get_yes_no_token_ids,
    load_vlm,
    resolve_default_model_id,
)
from aod.vlm.intervention import (
    AODDecodeConfig,
    aod_next_token_logits,
    first_parameter_device,
    move_tensor_inputs,
)


@dataclass(frozen=True)
class BinarySample:
    sample_id: str
    question: str
    answer01: int
    image_path: str | None
    raw: Dict[str, Any]


def jsonl_iter(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def json_iter(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    yield from data


def resolve_image_path(image_root: str, image_ref: str | None) -> str | None:
    if not image_ref:
        return None
    rel = image_ref[2:] if image_ref.startswith("./") else image_ref
    path = rel if os.path.isabs(rel) else os.path.join(image_root, rel)
    return path if os.path.exists(path) else None


def load_binary_samples(
    path: str,
    image_root: str,
    dataset_format: str,
    *,
    question_field: str = "question",
    answer_field: str = "answer",
    image_field: str = "image",
    amber_annotations_path: str = "",
    amber_typology: str = "",
) -> List[BinarySample]:
    samples: List[BinarySample] = []
    if dataset_format == "amber":
        if not amber_annotations_path:
            raise ValueError("--amber_annotations_path is required for dataset_format=amber.")
        typology = amber_typology.strip() or None
        amber_records = load_amber_discriminative(path, amber_annotations_path, typology=typology)
        for rec in amber_to_records(amber_records):
            image_path = resolve_image_path(image_root, rec["image"])
            if image_path is None:
                continue
            samples.append(
                BinarySample(
                    sample_id=str(rec["id"]),
                    question=str(rec["question"]),
                    answer01=int(rec["gt_answer"]),
                    image_path=image_path,
                    raw=rec,
                )
            )
        return samples

    if dataset_format == "pope":
        iterator = jsonl_iter(path)
        for idx, rec in enumerate(iterator):
            question = rec.get("text") or rec.get("question")
            answer = normalize_binary_answer(rec.get("label") or rec.get("gt") or rec.get("answer"))
            image_path = resolve_image_path(image_root, rec.get("image"))
            if not question or answer is None or image_path is None:
                continue
            samples.append(BinarySample(str(rec.get("id", idx)), str(question), int(answer), image_path, rec))
        return samples

    if dataset_format == "hallusionbench":
        iterator = json_iter(path)
        for idx, rec in enumerate(iterator):
            question = rec.get("question")
            answer = normalize_binary_answer(rec.get("gt_answer") or rec.get("answer"))
            visual_input = str(rec.get("visual_input", "")).strip()
            image_path = resolve_image_path(image_root, rec.get("filename")) if visual_input == "1" else None
            if not question or answer is None:
                continue
            if visual_input == "1" and image_path is None:
                continue
            samples.append(BinarySample(str(rec.get("question_id", idx)), str(question), int(answer), image_path, rec))
        return samples

    if dataset_format in {"generic_json", "generic_jsonl"}:
        iterator = json_iter(path) if dataset_format == "generic_json" else jsonl_iter(path)
        for idx, rec in enumerate(iterator):
            question = rec.get(question_field)
            answer = normalize_binary_answer(rec.get(answer_field))
            image_ref = rec.get(image_field) if image_field else None
            image_path = resolve_image_path(image_root, image_ref) if image_ref else None
            if not question or answer is None:
                continue
            if image_ref and image_path is None:
                continue
            samples.append(BinarySample(str(rec.get("id", idx)), str(question), int(answer), image_path, rec))
        return samples

    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def predict_yes_no(logits: torch.Tensor, yes_ids: Sequence[int], no_ids: Sequence[int]) -> int:
    yes_score = torch.max(logits[:, list(yes_ids)], dim=-1).values
    no_score = torch.max(logits[:, list(no_ids)], dim=-1).values
    return int((yes_score >= no_score).item())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_format",
        choices=["pope", "hallusionbench", "amber", "generic_json", "generic_jsonl"],
        required=True,
    )
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument("--image_field", default="image")
    ap.add_argument(
        "--amber_annotations_path",
        default="",
        help="Required when --dataset_format=amber. Path to AMBER's data/annotations.json.",
    )
    ap.add_argument(
        "--amber_typology",
        default="",
        choices=["", "existence", "attribute", "relation"],
        help="If set, only keep this AMBER typology subset.",
    )
    ap.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_IDS["qwen2_5_vl"],
        help=(
            "HF repo id, or one of the family aliases: "
            + ", ".join(f"{k}={v}" for k, v in DEFAULT_MODEL_IDS.items())
        ),
    )
    ap.add_argument("--ckpt", default="", help="Required for --mode direct or --mode cd.")
    ap.add_argument("--mode", choices=["base", "direct", "cd"], default="cd")
    ap.add_argument("--layer", type=int, default=0, help="Override checkpoint layer when > 0.")
    ap.add_argument("--aod_alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--apc_alpha", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output_path", default="")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    args = ap.parse_args(argv)

    samples = load_binary_samples(
        args.data_path,
        args.image_root,
        args.dataset_format,
        question_field=args.question_field,
        answer_field=args.answer_field,
        image_field=args.image_field,
        amber_annotations_path=args.amber_annotations_path,
        amber_typology=args.amber_typology,
    )
    if args.limit > 0:
        samples = samples[: int(args.limit)]
    if not samples:
        raise ValueError("No usable binary samples found.")

    model_id = resolve_default_model_id(args.model_id)
    loaded = load_vlm(
        model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    yes_ids, no_ids = get_yes_no_token_ids(loaded.tokenizer)
    if not yes_ids or not no_ids:
        raise RuntimeError("Failed to derive Yes/No token ids.")

    cfg = None
    if args.mode != "base":
        if not args.ckpt:
            raise ValueError("--ckpt is required for AOD direct/cd modes.")
        meta, aod_model = load_checkpoint(args.ckpt, map_location="cpu")
        direction = aod_model.v_unit().detach().cpu().view(-1)
        cfg = AODDecodeConfig(
            layer=int(args.layer) if int(args.layer) > 0 else int(meta.layer),
            direction=direction,
            alpha=float(args.aod_alpha),
            beta=float(args.beta),
            apc_alpha=float(args.apc_alpha),
            mode=args.mode,
        )

    input_device = first_parameter_device(loaded.model)
    records: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    for sample in tqdm(samples, desc=f"eval-{args.mode}-{loaded.family}"):
        image = Image.open(sample.image_path).convert("RGB") if sample.image_path is not None else None
        inputs = build_yes_no_inputs(loaded.processor, loaded.family, sample.question, image)
        inputs = move_tensor_inputs(inputs, input_device)
        with torch.inference_mode():
            logits = aod_next_token_logits(loaded.model, inputs, cfg)
        pred01 = predict_yes_no(logits, yes_ids=yes_ids, no_ids=no_ids)
        ok = pred01 == int(sample.answer01)
        correct += int(ok)
        total += 1
        records.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "answer": "yes" if sample.answer01 == 1 else "no",
                "prediction": "yes" if pred01 == 1 else "no",
                "correct": ok,
                "image_path": sample.image_path,
            }
        )

    acc = 100.0 * correct / max(1, total)
    print(
        f"model={model_id} family={loaded.family} dataset={args.dataset_format} "
        f"mode={args.mode} total={total} correct={correct} acc={acc:.2f}%"
    )
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Saved] {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
