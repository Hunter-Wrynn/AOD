from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from transformers import AutoConfig


@dataclass(frozen=True)
class Qwen2_5VLModelInfo:
    model_id: str
    model_type: Optional[str]
    num_hidden_layers: Optional[int]
    hidden_size: Optional[int]


def load_qwen2_5vl_model_info(model_id: str, trust_remote_code: bool = False) -> Qwen2_5VLModelInfo:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return Qwen2_5VLModelInfo(
        model_id=model_id,
        model_type=getattr(cfg, "model_type", None),
        num_hidden_layers=getattr(cfg, "num_hidden_layers", None),
        hidden_size=getattr(cfg, "hidden_size", None),
    )


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    info = load_qwen2_5vl_model_info(model_id)
    print(f"model_id={info.model_id}")
    print(f"model_type={info.model_type}")
    print(f"num_hidden_layers={info.num_hidden_layers}")
    print(f"hidden_size={info.hidden_size}")


if __name__ == "__main__":
    main()

