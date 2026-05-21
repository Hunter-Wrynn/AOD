"""Generative eval for CHAIR (COCO captioning) and OCRBench-v2.

We use greedy decoding under AOD intervention (`aod.vlm.intervention.greedy_generate_ids`)
to emit a free-form answer per sample, then score downstream:

  * `--dataset_format chair_coco`: write generations to JSONL and (optionally)
    compute CHAIR_S / CHAIR_I in-process via `aod.data.chair`.
  * `--dataset_format ocrbench`: write generations to JSONL. Use the official
    OCRBench-v2 evaluator on the dump for the final score (the format mirrors
    the upstream `predict_*.jsonl` shape).

Per the paper, utility-style benchmarks (which include OCRBench-v2 alongside
RealWorldQA / MMStar / MMMU) reuse the POPE-trained AOD direction.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from aod.core.aod import load_checkpoint
from aod.data.chair import (
    DEFAULT_COCO_SYNONYMS,
    chair_score,
    load_coco_image_objects,
    load_coco_synonyms_json,
)
from aod.vlm.loader import (
    DEFAULT_MODEL_IDS,
    load_vlm,
    resolve_default_model_id,
)
from aod.vlm.intervention import (
    AODDecodeConfig,
    first_parameter_device,
    greedy_generate_ids,
    move_tensor_inputs,
)


@dataclass(frozen=True)
class GenSample:
    sample_id: str
    image_id: Optional[int]
    prompt: str
    image_path: Optional[str]
    reference: Optional[str]
    task: str
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
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            yield from data["data"]
            return
        for v in data.values():
            if isinstance(v, list):
                yield from v
                return
        raise ValueError(f"{path}: expected a JSON list or list-bearing object")
    if not isinstance(data, list):
        raise ValueError(f"Expected a list or list-bearing object in {path}")
    yield from data


def _resolve_path(image_root: str, ref: Any) -> Optional[str]:
    if not ref:
        return None
    if isinstance(ref, list):
        ref = ref[0] if ref else None
        if not ref:
            return None
    ref = str(ref)
    ref = ref[2:] if ref.startswith("./") else ref
    path = ref if os.path.isabs(ref) else os.path.join(image_root, ref)
    return path if os.path.exists(path) else None


def _coco_image_id_from_filename(path_or_name: str) -> Optional[int]:
    """Parse COCO-style filenames like `COCO_val2014_000000123456.jpg` → 123456."""
    name = os.path.basename(path_or_name)
    stem, _ = os.path.splitext(name)
    last = stem.rsplit("_", 1)[-1] if "_" in stem else stem
    if last.isdigit():
        try:
            return int(last)
        except ValueError:
            return None
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def load_chair_coco_samples(
    captions_path: str,
    image_root: str,
    prompt: str,
) -> List[GenSample]:
    """Read either a COCO `captions_val2014.json` or a flat list of records.

    Output records carry an `image_id` (COCO numeric id) so we can join to the
    instance annotations during scoring.
    """
    with open(captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples: List[GenSample] = []
    if isinstance(data, dict) and "images" in data:
        for img in data["images"]:
            file_name = img.get("file_name") or img.get("filename")
            iid = img.get("id")
            if file_name is None or iid is None:
                continue
            image_path = _resolve_path(image_root, file_name)
            if image_path is None:
                continue
            samples.append(
                GenSample(
                    sample_id=str(iid),
                    image_id=int(iid),
                    prompt=prompt,
                    image_path=image_path,
                    reference=None,
                    task="caption",
                    raw={"file_name": file_name},
                )
            )
        return samples
    if not isinstance(data, list):
        raise ValueError(f"Expected COCO captions json or a JSON list in {captions_path}")
    for idx, rec in enumerate(data):
        file_name = rec.get("file_name") or rec.get("image") or rec.get("filename")
        if not file_name:
            continue
        image_path = _resolve_path(image_root, file_name)
        if image_path is None:
            continue
        iid = rec.get("image_id")
        if iid is None:
            iid = _coco_image_id_from_filename(file_name)
        samples.append(
            GenSample(
                sample_id=str(rec.get("id", idx)),
                image_id=int(iid) if iid is not None else None,
                prompt=prompt,
                image_path=image_path,
                reference=None,
                task="caption",
                raw=rec,
            )
        )
    return samples


def load_ocrbench_samples(
    data_path: str,
    image_root: str,
) -> List[GenSample]:
    """OCRBench-v2 ships per-task JSONL; each record has `question`/`answer`/`image`."""
    iterator = jsonl_iter(data_path) if data_path.endswith(".jsonl") else json_iter(data_path)
    samples: List[GenSample] = []
    for idx, rec in enumerate(iterator):
        question = rec.get("question") or rec.get("instruction") or rec.get("prompt")
        if not question:
            continue
        image_ref = (
            rec.get("image")
            or rec.get("image_path")
            or rec.get("filename")
            or rec.get("img_path")
        )
        image_path = _resolve_path(image_root, image_ref)
        if image_ref is not None and image_path is None:
            continue
        reference = rec.get("answer") or rec.get("gt") or rec.get("answers") or rec.get("label")
        if isinstance(reference, list):
            reference = reference[0] if reference else None
        task_name = str(rec.get("type", "") or rec.get("task", "") or rec.get("category", ""))
        samples.append(
            GenSample(
                sample_id=str(rec.get("id", idx)),
                image_id=None,
                prompt=str(question),
                image_path=image_path,
                reference=str(reference) if reference is not None else None,
                task=task_name or "ocrbench",
                raw=rec,
            )
        )
    return samples


def build_generative_inputs(
    processor: Any,
    family: str,
    prompt: str,
    image: Optional[Image.Image],
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except ValueError as exc:
        if "chat template" not in str(exc).lower():
            raise
        image_token = getattr(processor, "image_token", None) or "<image>"
        if image is not None:
            text = f"USER: {image_token}\n{prompt}\nASSISTANT:"
        else:
            text = f"USER: {prompt}\nASSISTANT:"
    if image is None:
        return dict(processor(text=text, return_tensors="pt"))
    return dict(processor(images=image, text=text, return_tensors="pt"))


def _collect_eos_token_ids(tokenizer: Any) -> List[int]:
    candidates: List[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, int):
            candidates.append(eos)
        elif isinstance(eos, (list, tuple)):
            candidates.extend(int(x) for x in eos if x is not None)
    for attr in ("pad_token_id",):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            candidates.append(v)
    return list(dict.fromkeys(candidates))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_format",
        choices=["chair_coco", "ocrbench", "generic_gen_json", "generic_gen_jsonl"],
        required=True,
    )
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument(
        "--chair_prompt",
        default="Please describe this image in detail.",
        help="Prompt to elicit captions for CHAIR scoring (caption-style).",
    )
    ap.add_argument(
        "--coco_instances_path",
        default="",
        help="If set, score CHAIR_S/CHAIR_I in-process against this COCO instances_*.json.",
    )
    ap.add_argument(
        "--coco_synonyms_path",
        default="",
        help="Optional richer COCO synonym map for CHAIR detection.",
    )
    ap.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_IDS["qwen2_5_vl"],
        help=(
            "HF repo id, or one of the family aliases: "
            + ", ".join(f"{k}={v}" for k, v in DEFAULT_MODEL_IDS.items())
        ),
    )
    ap.add_argument(
        "--ckpt",
        default="",
        help="POPE-trained AOD checkpoint. Required for --mode direct or --mode cd.",
    )
    ap.add_argument("--mode", choices=["base", "direct", "cd"], default="cd")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--aod_alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--apc_alpha", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    args = ap.parse_args(argv)

    if args.dataset_format == "chair_coco":
        samples = load_chair_coco_samples(args.data_path, args.image_root, args.chair_prompt)
    elif args.dataset_format == "ocrbench":
        samples = load_ocrbench_samples(args.data_path, args.image_root)
    elif args.dataset_format in {"generic_gen_json", "generic_gen_jsonl"}:
        iterator = (
            jsonl_iter(args.data_path)
            if args.dataset_format == "generic_gen_jsonl"
            else json_iter(args.data_path)
        )
        samples = []
        for idx, rec in enumerate(iterator):
            question = rec.get("question") or rec.get("prompt") or rec.get("instruction")
            if not question:
                continue
            image_ref = rec.get("image") or rec.get("image_path") or rec.get("filename")
            image_path = _resolve_path(args.image_root, image_ref) if image_ref else None
            if image_ref and image_path is None:
                continue
            samples.append(
                GenSample(
                    sample_id=str(rec.get("id", idx)),
                    image_id=None,
                    prompt=str(question),
                    image_path=image_path,
                    reference=str(rec.get("answer") or rec.get("gt") or "") or None,
                    task=str(rec.get("type", "") or rec.get("task", "")),
                    raw=rec,
                )
            )
    else:
        raise ValueError(f"Unsupported dataset_format: {args.dataset_format}")

    if args.limit > 0:
        samples = samples[: int(args.limit)]
    if not samples:
        raise ValueError("No usable generative samples found.")

    model_id = resolve_default_model_id(args.model_id)
    loaded = load_vlm(
        model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    cfg: AODDecodeConfig | None = None
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
    eos_ids = _collect_eos_token_ids(loaded.tokenizer)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_f = open(args.output_path, "w", encoding="utf-8")
    records: List[Dict[str, Any]] = []
    try:
        for sample in tqdm(samples, desc=f"gen-{args.dataset_format}-{args.mode}-{loaded.family}"):
            image = Image.open(sample.image_path).convert("RGB") if sample.image_path is not None else None
            inputs = build_generative_inputs(loaded.processor, loaded.family, sample.prompt, image)
            inputs = move_tensor_inputs(inputs, input_device)
            with torch.inference_mode():
                gen_ids = greedy_generate_ids(
                    loaded.model,
                    inputs,
                    cfg=cfg,
                    max_new_tokens=int(args.max_new_tokens),
                    eos_token_ids=eos_ids,
                )
            text = loaded.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            rec = {
                "id": sample.sample_id,
                "image_id": sample.image_id,
                "image_path": sample.image_path,
                "task": sample.task,
                "prompt": sample.prompt,
                "reference": sample.reference,
                "generation": text,
            }
            records.append(rec)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
    finally:
        out_f.close()
    print(f"[Saved] {len(records)} generations -> {args.output_path}")

    if args.dataset_format == "chair_coco" and args.coco_instances_path:
        image_objects = load_coco_image_objects(args.coco_instances_path)
        synonyms = (
            load_coco_synonyms_json(args.coco_synonyms_path)
            if args.coco_synonyms_path
            else DEFAULT_COCO_SYNONYMS
        )
        pairs: List[tuple] = []
        for rec in records:
            iid = rec.get("image_id")
            if iid is None:
                continue
            pairs.append((int(iid), str(rec["generation"])))
        score = chair_score(pairs, image_objects=image_objects, synonyms=synonyms)
        print(
            f"model={model_id} family={loaded.family} dataset=chair_coco mode={args.mode} "
            f"captions={score.num_captions} mentions={score.num_mentioned_objects} "
            f"CHAIR_s={100.0*score.chair_s:.2f}% CHAIR_i={100.0*score.chair_i:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
