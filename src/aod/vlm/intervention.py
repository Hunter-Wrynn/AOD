from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch


LAYER_PATH_CANDIDATES = (
    "model.language_model.layers",
    "model.language_model.model.layers",
    "language_model.layers",
    "language_model.model.layers",
    "model.model.layers",
    "model.layers",
)


@dataclass(frozen=True)
class AODDecodeConfig:
    layer: int
    direction: torch.Tensor
    alpha: float = 1.0
    beta: float = 0.5
    apc_alpha: float = 0.1
    mode: str = "cd"
    # APC implementation:
    #   "vcd"      - canonical VCD masking: non-plausible tokens get -inf
    #                (faithful to Adaptive Plausibility Constraint, default).
    #   "fallback" - non-plausible tokens fall back to logits_pos
    #                (original implementation in this repo, kept for back-compat).
    apc_mode: str = "vcd"


def resolve_attr(root: object, dotted_path: str) -> object:
    current = root
    for part in dotted_path.split("."):
        current = getattr(current, part)
    return current


def resolve_decoder_layers(model: object) -> tuple[torch.nn.ModuleList | list[torch.nn.Module], str]:
    for path in LAYER_PATH_CANDIDATES:
        try:
            layers = resolve_attr(model, path)
        except AttributeError:
            continue
        if isinstance(layers, (torch.nn.ModuleList, list)) and len(layers) > 0:
            return layers, path
    raise AttributeError(
        "Could not locate decoder layers. Tried: " + ", ".join(LAYER_PATH_CANDIDATES)
    )


def move_tensor_inputs(inputs: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_output_logits(outputs: object) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Model output has no logits field.")
    return logits[:, -1, :]


def _normalize_direction(direction: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    v = direction.detach().to(device=hidden.device, dtype=torch.float32).view(-1)
    v = torch.nn.functional.normalize(v, dim=0)
    return v.to(dtype=hidden.dtype)


def steer_last_token_hidden(hidden: torch.Tensor, direction: torch.Tensor, signed_alpha: float) -> torch.Tensor:
    v = _normalize_direction(direction, hidden)
    token = hidden[:, -1:, :]
    scalar = torch.matmul(token.to(torch.float32), v.to(torch.float32).view(1, -1).t())
    projection = scalar.to(hidden.dtype) * v.view(1, 1, -1)
    steered = hidden.clone()
    steered[:, -1:, :] = token + float(signed_alpha) * projection
    return steered


@contextmanager
def layer_steering_hook(
    model: torch.nn.Module,
    layer: int,
    direction: torch.Tensor,
    signed_alpha: float,
) -> Iterator[None]:
    layers, _ = resolve_decoder_layers(model)
    layer = int(layer)
    if layer <= 0 or layer > len(layers):
        raise ValueError(f"Layer must be in [1, {len(layers)}], got {layer}.")
    target = layers[layer - 1]

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = steer_last_token_hidden(output[0], direction, signed_alpha)
            return (hidden,) + output[1:]
        return steer_last_token_hidden(output, direction, signed_alpha)

    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def forward_next_logits(
    model: torch.nn.Module,
    inputs: dict,
    *,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    signed_alpha: float = 0.0,
) -> torch.Tensor:
    context = (
        layer_steering_hook(model, layer=layer, direction=direction, signed_alpha=signed_alpha)
        if layer is not None and direction is not None and signed_alpha != 0.0
        else nullcontext()
    )
    with context:
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    return get_output_logits(outputs)


def aod_next_token_logits(
    model: torch.nn.Module,
    inputs: dict,
    cfg: AODDecodeConfig | None,
    *,
    return_fallback: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return the AOD-decoded next-token logits.

    If `return_fallback` is True, also returns the unmasked `logits_pos`
    (factual-direction single-forward logits) so callers can recover when
    APC's -inf masking excludes every token in a subset of interest
    (e.g. both Yes and No tokens during binary Yes/No evaluation).
    """
    if cfg is None or cfg.mode == "base":
        logits = forward_next_logits(model, inputs)
        return (logits, logits) if return_fallback else logits
    if cfg.mode == "direct":
        logits = forward_next_logits(
            model,
            inputs,
            layer=cfg.layer,
            direction=cfg.direction,
            signed_alpha=-float(cfg.alpha),
        )
        return (logits, logits) if return_fallback else logits
    if cfg.mode != "cd":
        raise ValueError(f"Unsupported AOD decode mode: {cfg.mode}")

    logits_pos = forward_next_logits(
        model,
        inputs,
        layer=cfg.layer,
        direction=cfg.direction,
        signed_alpha=-float(cfg.alpha),
    )
    logits_neg = forward_next_logits(
        model,
        inputs,
        layer=cfg.layer,
        direction=cfg.direction,
        signed_alpha=float(cfg.alpha),
    )
    logits_cd = (1.0 + float(cfg.beta)) * logits_pos - float(cfg.beta) * logits_neg
    probs_pos = torch.softmax(logits_pos, dim=-1)
    threshold = float(cfg.apc_alpha) * probs_pos.max(dim=-1, keepdim=True).values
    plausible = probs_pos >= threshold
    if cfg.apc_mode == "vcd":
        neg_inf = torch.full_like(logits_cd, float("-inf"))
        masked = torch.where(plausible, logits_cd, neg_inf)
    elif cfg.apc_mode == "fallback":
        masked = torch.where(plausible, logits_cd, logits_pos)
    else:
        raise ValueError(f"Unsupported apc_mode: {cfg.apc_mode!r} (expected 'vcd' or 'fallback')")
    return (masked, logits_pos) if return_fallback else masked


def greedy_generate_ids(
    model: torch.nn.Module,
    inputs: dict,
    *,
    cfg: AODDecodeConfig | None,
    max_new_tokens: int,
    eos_token_ids: Iterable[int],
) -> torch.Tensor:
    eos = set(int(x) for x in eos_token_ids if x is not None)
    current = dict(inputs)
    for _ in range(int(max_new_tokens)):
        logits = aod_next_token_logits(model, current, cfg)
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        current["input_ids"] = torch.cat([current["input_ids"], next_id], dim=1)
        if "attention_mask" in current:
            ones = torch.ones_like(next_id, device=current["attention_mask"].device)
            current["attention_mask"] = torch.cat([current["attention_mask"], ones], dim=1)
        if int(next_id.item()) in eos:
            break
    return current["input_ids"][:, inputs["input_ids"].shape[1] :]
