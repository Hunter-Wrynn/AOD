from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image


DEFAULT_MODEL_IDS: Dict[str, str] = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "internvl": "OpenGVLab/InternVL3-8B-hf",
}


DTYPE_MAP: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


SUPPORTED_FAMILIES = tuple(DEFAULT_MODEL_IDS.keys())


def detect_family(model_id: str) -> str:
    """Return one of qwen2_5_vl / llava / internvl based on the HF repo id."""
    mid = model_id.lower()
    if "qwen2.5-vl" in mid or "qwen2_5_vl" in mid or "qwen2-vl" in mid:
        return "qwen2_5_vl"
    if "llava" in mid:
        return "llava"
    if "internvl" in mid:
        return "internvl"
    raise ValueError(
        f"Cannot infer model family from model_id={model_id!r}. "
        f"Supported substrings: qwen2.5-vl, llava, internvl."
    )


def resolve_default_model_id(family_or_id: str) -> str:
    """Allow either a family alias (e.g. 'llava') or a full HF repo id."""
    if family_or_id in DEFAULT_MODEL_IDS:
        return DEFAULT_MODEL_IDS[family_or_id]
    return family_or_id


@dataclass(frozen=True)
class LoadedVLM:
    family: str
    model_id: str
    processor: Any
    model: torch.nn.Module
    tokenizer: Any


def _load_model_cls(family: str, model_id: str, *, torch_dtype, device_map, trust_remote_code):
    if family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    if family == "llava":
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    if family == "internvl":
        # Newer transformers releases expose InternVLForConditionalGeneration;
        # otherwise fall back to the generic image-text-to-text auto class.
        try:
            from transformers import InternVLForConditionalGeneration as _Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as _Cls
        return _Cls.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    raise ValueError(f"Unsupported family: {family}")


def load_vlm(
    model_id: str,
    *,
    dtype: str = "bf16",
    device_map: str | dict = "auto",
    trust_remote_code: bool = False,
) -> LoadedVLM:
    """Load processor + model + tokenizer for one of the supported VLM families."""
    from transformers import AutoProcessor

    model_id = resolve_default_model_id(model_id)
    family = detect_family(model_id)
    torch_dtype = DTYPE_MAP[dtype]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = _load_model_cls(
        family,
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
    if hasattr(processor, "patch_size") and getattr(processor, "patch_size", None) is None and vision_cfg is not None:
        processor.patch_size = getattr(vision_cfg, "patch_size", None)
    if (
        hasattr(processor, "vision_feature_select_strategy")
        and getattr(processor, "vision_feature_select_strategy", None) is None
    ):
        processor.vision_feature_select_strategy = getattr(
            getattr(model, "config", None),
            "vision_feature_select_strategy",
            "default",
        )
    if (
        hasattr(processor, "num_additional_image_tokens")
        and getattr(processor, "num_additional_image_tokens", None) is None
        and vision_cfg is not None
    ):
        processor.num_additional_image_tokens = 1 if getattr(vision_cfg, "add_cls_token", False) else 0
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            f"AutoProcessor for {model_id} did not expose a tokenizer; "
            "cannot derive Yes/No token ids."
        )
    return LoadedVLM(family=family, model_id=model_id, processor=processor, model=model, tokenizer=tokenizer)


def build_yes_no_inputs(
    processor: Any,
    family: str,
    question: str,
    image: Optional[Image.Image],
) -> Dict[str, Any]:
    """Build tokenized inputs for a binary yes/no question across all 3 families.

    All three processors here support the OpenAI-style chat-message format with
    `{"type": "image"}` and `{"type": "text", ...}` content blocks, plus an
    `images=` keyword on `__call__`. We pass the PIL image through both channels
    to be safe across processor versions.
    """
    answer_prompt = f"{question}\nAnswer with Yes or No."
    content: List[Dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": answer_prompt})
    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except ValueError as exc:
        if "chat template" not in str(exc).lower():
            raise
        image_token = getattr(processor, "image_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if image_token is None and tokenizer is not None:
            image_token = getattr(tokenizer, "image_token", None)
        image_token = image_token or "<image>"
        if image is not None:
            text = f"USER: {image_token}\n{answer_prompt}\nASSISTANT:"
        else:
            text = f"USER: {answer_prompt}\nASSISTANT:"
    if image is None:
        return dict(processor(text=text, return_tensors="pt"))
    return dict(processor(images=image, text=text, return_tensors="pt"))


def get_yes_no_token_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """Return canonical Yes / No final-token ids from any HF tokenizer."""
    yes_words = ["Yes", "yes", " Yes", " yes"]
    no_words = ["No", "no", " No", " no"]
    yes_ids: List[int] = []
    no_ids: List[int] = []
    for word in yes_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            yes_ids.append(int(ids[-1]))
    for word in no_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            no_ids.append(int(ids[-1]))
    return list(dict.fromkeys(yes_ids)), list(dict.fromkeys(no_ids))


def get_last_nonpad_index(attention_mask: torch.Tensor) -> int:
    nz = (attention_mask != 0).nonzero(as_tuple=False)
    return int(nz[-1].item()) if nz.numel() else 0
