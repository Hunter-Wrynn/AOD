"""VLM loader, intervention plumbing, and config inspector."""

from aod.vlm.loader import (
    DEFAULT_MODEL_IDS,
    DTYPE_MAP,
    LoadedVLM,
    SUPPORTED_FAMILIES,
    build_yes_no_inputs,
    detect_family,
    get_last_nonpad_index,
    get_yes_no_token_ids,
    load_vlm,
    resolve_default_model_id,
)
from aod.vlm.intervention import (
    AODDecodeConfig,
    aod_next_token_logits,
    first_parameter_device,
    forward_next_logits,
    greedy_generate_ids,
    move_tensor_inputs,
    resolve_decoder_layers,
)

__all__ = [
    "AODDecodeConfig",
    "DEFAULT_MODEL_IDS",
    "DTYPE_MAP",
    "LoadedVLM",
    "SUPPORTED_FAMILIES",
    "aod_next_token_logits",
    "build_yes_no_inputs",
    "detect_family",
    "first_parameter_device",
    "forward_next_logits",
    "get_last_nonpad_index",
    "get_yes_no_token_ids",
    "greedy_generate_ids",
    "load_vlm",
    "move_tensor_inputs",
    "resolve_decoder_layers",
    "resolve_default_model_id",
]
