from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

from transformers import AutoConfig

from aod.vlm.loader import DEFAULT_MODEL_IDS, resolve_default_model_id


@dataclass(frozen=True)
class VLMModelInfo:
    model_id: str
    model_type: Optional[str]
    num_hidden_layers: Optional[int]
    hidden_size: Optional[int]


def load_vlm_model_info(model_id: str, trust_remote_code: bool = False) -> VLMModelInfo:
    model_id = resolve_default_model_id(model_id)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    text_cfg = getattr(cfg, "text_config", None)
    return VLMModelInfo(
        model_id=model_id,
        model_type=getattr(cfg, "model_type", None),
        num_hidden_layers=getattr(cfg, "num_hidden_layers", None)
        or getattr(text_cfg, "num_hidden_layers", None),
        hidden_size=getattr(cfg, "hidden_size", None) or getattr(text_cfg, "hidden_size", None),
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_id",
        default=os.environ.get("MODEL_ID", DEFAULT_MODEL_IDS["qwen2_5_vl"]),
        help=(
            "HF repo id, or one of the family aliases: "
            + ", ".join(f"{k}={v}" for k, v in DEFAULT_MODEL_IDS.items())
        ),
    )
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    args = ap.parse_args(argv)

    info = load_vlm_model_info(args.model_id, trust_remote_code=args.trust_remote_code)
    print(f"model_id={info.model_id}")
    print(f"model_type={info.model_type}")
    print(f"num_hidden_layers={info.num_hidden_layers}")
    print(f"hidden_size={info.hidden_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
