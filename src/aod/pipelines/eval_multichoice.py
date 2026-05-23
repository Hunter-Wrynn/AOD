"""Multiple-choice eval for RealWorldQA, MMStar, MMMU under AOD intervention.

We score each option by the next-token logit of its option letter (A, B, C,
D, ...) under the same AOD direct / cd / base modes as binary eval. The
prompt embeds the options inline as `(A) ... (B) ... \n Answer with the
option's letter from the given choices directly.`

Per the paper, utility benchmarks (RealWorldQA / MMStar / MMMU) use the
hallucination direction extracted from POPE; the loader does not retrain.
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
from aod.vlm.loader import (
    DEFAULT_MODEL_IDS,
    DTYPE_MAP,  # noqa: F401  (touched for compatibility)
    load_vlm,
    resolve_default_model_id,
)
from aod.vlm.intervention import (
    AODDecodeConfig,
    aod_next_token_logits,
    first_parameter_device,
    move_tensor_inputs,
)


MAX_OPTIONS = 10  # supports up to (A)..(J), enough for all paper benchmarks


@dataclass(frozen=True)
class MCSample:
    sample_id: str
    question: str
    options: List[str]
    answer_letter: str  # e.g. "A"
    image_path: Optional[str]
    category: str
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
        raise ValueError(f"{path}: cannot find a list of records in JSON object.")
    if not isinstance(data, list):
        raise ValueError(f"Expected a list or list-bearing object in {path}")
    yield from data


def _resolve_path(image_root: str, ref: str | None) -> str | None:
    if not ref:
        return None
    ref = ref[2:] if ref.startswith("./") else ref
    path = ref if os.path.isabs(ref) else os.path.join(image_root, ref)
    return path if os.path.exists(path) else None


def _parse_options(rec: Dict[str, Any]) -> List[str]:
    # 1) Explicit options list
    options = rec.get("options")
    if isinstance(options, list) and options:
        return [str(o) for o in options]
    # 2) Letter-keyed options (MMMU uses "A","B",... at top level)
    letters = [chr(ord("A") + i) for i in range(MAX_OPTIONS)]
    keyed = [rec[l] for l in letters if l in rec and rec[l] not in (None, "")]
    if len(keyed) >= 2:
        return [str(o) for o in keyed]
    return []


def _normalize_letter(value: Any, num_options: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text[0].isalpha() and text[0].upper() < chr(ord("A") + num_options):
        return text[0].upper()
    # Numeric index (0/1-based) → letter
    try:
        n = int(text)
        if 0 <= n < num_options:
            return chr(ord("A") + n)
        if 1 <= n <= num_options:
            return chr(ord("A") + n - 1)
    except ValueError:
        pass
    return None


def load_mc_samples(
    path: str,
    image_root: str,
    dataset_format: str,
    *,
    question_field: str = "question",
    answer_field: str = "answer",
    image_field: str = "image",
    options_field: str = "options",
    category_field: str = "category",
) -> List[MCSample]:
    samples: List[MCSample] = []
    iterator: Iterable[Dict[str, Any]]
    if dataset_format == "realworldqa":
        # Hugging Face xai-org/RealWorldQA — typical layout: image, question, answer.
        # Questions usually embed multi-choice as "... (A) ... (B) ... (C) ..."
        # We fall back to A/B/C/D as default options unless an explicit list is provided.
        iterator = json_iter(path) if path.endswith(".json") else jsonl_iter(path)
    elif dataset_format == "mmstar":
        iterator = json_iter(path) if path.endswith(".json") else jsonl_iter(path)
    elif dataset_format == "mmmu":
        iterator = json_iter(path) if path.endswith(".json") else jsonl_iter(path)
    elif dataset_format in {"generic_mc_json", "generic_mc_jsonl"}:
        iterator = json_iter(path) if dataset_format == "generic_mc_json" else jsonl_iter(path)
    else:
        raise ValueError(f"Unsupported multi-choice dataset format: {dataset_format}")

    for idx, rec in enumerate(iterator):
        question = rec.get(question_field) or rec.get("question") or rec.get("query")
        if not question:
            continue

        opts: List[str] = []
        explicit = rec.get(options_field)
        if isinstance(explicit, list) and explicit:
            opts = [str(o) for o in explicit]
        else:
            opts = _parse_options(rec)
        if not opts:
            # RealWorldQA often has only 2-4 inline options inside the question. Allow
            # the caller to fall back to "answer is just a letter" via 4 default slots.
            opts = ["A", "B", "C", "D"]

        ans_raw = rec.get(answer_field) or rec.get("answer") or rec.get("gt_answer") or rec.get("label")
        ans_letter = _normalize_letter(ans_raw, num_options=len(opts))
        if ans_letter is None:
            continue

        # Image reference can be a string or a list (MMMU has image_1, image_2 fields).
        image_ref = rec.get(image_field)
        if image_ref is None:
            for k in ("image_1", "image_path", "filename"):
                if k in rec and rec[k]:
                    image_ref = rec[k]
                    break
        if isinstance(image_ref, list):
            image_ref = image_ref[0] if image_ref else None
        image_path = _resolve_path(image_root, image_ref)

        category = str(rec.get(category_field, "") or rec.get("topic", "") or rec.get("subject", ""))

        samples.append(
            MCSample(
                sample_id=str(rec.get("id", idx)),
                question=str(question),
                options=opts[:MAX_OPTIONS],
                answer_letter=ans_letter,
                image_path=image_path,
                category=category,
                raw=rec,
            )
        )
    return samples


def build_mc_inputs(processor: Any, family: str, question: str, options: List[str], image: Optional[Image.Image]) -> Dict[str, Any]:
    """Same chat-template path as build_yes_no_inputs, but with options inlined."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    option_block = "\n".join(f"({l}) {o}" for l, o in zip(letters, options))
    user_text = f"{question}\n{option_block}\nAnswer with the option's letter from the given choices directly."
    content: List[Dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image is None:
        return dict(processor(text=text, return_tensors="pt"))
    return dict(processor(images=image, text=text, return_tensors="pt"))


def get_letter_token_ids(tokenizer: Any, num_options: int) -> List[List[int]]:
    """Return last-token ids per letter, covering both bare and space-prefixed forms."""
    out: List[List[int]] = []
    for i in range(num_options):
        letter = chr(ord("A") + i)
        ids: List[int] = []
        for w in (letter, " " + letter):
            tok = tokenizer.encode(w, add_special_tokens=False)
            if tok:
                ids.append(int(tok[-1]))
        out.append(list(dict.fromkeys(ids)))
    return out


def _score_letters(logits: torch.Tensor, letter_ids: List[List[int]]) -> List[float]:
    scores: List[float] = []
    for ids in letter_ids:
        if not ids:
            scores.append(float("-inf"))
            continue
        scores.append(float(torch.max(logits[:, ids], dim=-1).values.item()))
    return scores


def predict_mc(
    logits: torch.Tensor,
    letter_ids: List[List[int]],
    *,
    fallback_logits: torch.Tensor | None = None,
) -> int:
    scores = _score_letters(logits, letter_ids)
    # Under canonical APC (-inf masking), every candidate letter token can be
    # masked. argmax over all-(-inf) scores deterministically returns 0
    # ("A"), silently biasing the benchmark. Re-score from logits_pos in that
    # case so the model's pre-APC preference among the option letters wins.
    if fallback_logits is not None and not any(s > float("-inf") for s in scores):
        scores = _score_letters(fallback_logits, letter_ids)
    return int(max(range(len(scores)), key=lambda i: scores[i]))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_format",
        choices=["realworldqa", "mmstar", "mmmu", "generic_mc_json", "generic_mc_jsonl"],
        required=True,
    )
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument("--image_field", default="image")
    ap.add_argument("--options_field", default="options")
    ap.add_argument("--category_field", default="category")
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
        help="POPE-trained AOD checkpoint. Required for --mode direct or --mode cd "
             "(utility benchmarks reuse the POPE direction per the paper protocol).",
    )
    ap.add_argument("--mode", choices=["base", "direct", "cd"], default="cd")
    ap.add_argument("--layer", type=int, default=0, help="Override checkpoint layer when > 0.")
    ap.add_argument("--aod_alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--apc_alpha", type=float, default=0.1)
    ap.add_argument(
        "--apc_mode",
        choices=["vcd", "fallback"],
        default="vcd",
        help="APC masking: 'vcd' sets non-plausible token logits to -inf (canonical, default); "
             "'fallback' keeps logits_pos on non-plausible tokens (legacy behavior).",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output_path", default="")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    args = ap.parse_args(argv)

    samples = load_mc_samples(
        args.data_path,
        args.image_root,
        args.dataset_format,
        question_field=args.question_field,
        answer_field=args.answer_field,
        image_field=args.image_field,
        options_field=args.options_field,
        category_field=args.category_field,
    )
    if args.limit > 0:
        samples = samples[: int(args.limit)]
    if not samples:
        raise ValueError("No usable multi-choice samples found.")

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
            apc_mode=args.apc_mode,
        )

    input_device = first_parameter_device(loaded.model)
    records: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    per_cat_total: Dict[str, int] = {}
    per_cat_correct: Dict[str, int] = {}

    for sample in tqdm(samples, desc=f"mc-{args.dataset_format}-{args.mode}-{loaded.family}"):
        image = Image.open(sample.image_path).convert("RGB") if sample.image_path is not None else None
        inputs = build_mc_inputs(loaded.processor, loaded.family, sample.question, sample.options, image)
        inputs = move_tensor_inputs(inputs, input_device)
        letter_ids = get_letter_token_ids(loaded.tokenizer, num_options=len(sample.options))
        with torch.inference_mode():
            logits, logits_fb = aod_next_token_logits(
                loaded.model, inputs, cfg, return_fallback=True
            )
        pred_idx = predict_mc(logits, letter_ids=letter_ids, fallback_logits=logits_fb)
        pred_letter = chr(ord("A") + pred_idx)
        ok = pred_letter == sample.answer_letter
        correct += int(ok)
        total += 1
        if sample.category:
            per_cat_total[sample.category] = per_cat_total.get(sample.category, 0) + 1
            per_cat_correct[sample.category] = per_cat_correct.get(sample.category, 0) + int(ok)
        records.append(
            {
                "id": sample.sample_id,
                "category": sample.category,
                "question": sample.question,
                "options": sample.options,
                "answer": sample.answer_letter,
                "prediction": pred_letter,
                "correct": ok,
                "image_path": sample.image_path,
            }
        )

    acc = 100.0 * correct / max(1, total)
    print(
        f"model={model_id} family={loaded.family} dataset={args.dataset_format} "
        f"mode={args.mode} total={total} correct={correct} acc={acc:.2f}%"
    )
    for cat in sorted(per_cat_total):
        c_total = per_cat_total[cat]
        c_correct = per_cat_correct[cat]
        print(f"  [{cat}] total={c_total} correct={c_correct} acc={100.0 * c_correct / max(1, c_total):.2f}%")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Saved] {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
