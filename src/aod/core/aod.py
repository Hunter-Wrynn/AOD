from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg().mul(ctx.lambda_), None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


def mlp_probe(input_dim: int, hidden_dim: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class AODDisentangler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        probe_hidden_dim: int = 512,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        # grl_lambda is kept in the meta for record-keeping, but the GRL itself
        # only performs the sign flip; the scalar weight is applied to L_adv
        # outside the module so the loss matches Eq. 4 of the paper.
        self._grl_lambda_meta = float(grl_lambda)
        self.v = nn.Parameter(torch.randn(1, input_dim))
        self.consistency_probe = mlp_probe(input_dim=input_dim, hidden_dim=probe_hidden_dim)
        self.residual_adversary = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            mlp_probe(input_dim=input_dim, hidden_dim=probe_hidden_dim),
        )

    def v_unit(self) -> torch.Tensor:
        return F.normalize(self.v, dim=1)

    @torch.no_grad()
    def renormalize_direction_(self) -> None:
        self.v.copy_(self.v_unit())

    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.v_unit()
        scalar = x.matmul(v.t())
        h_hallu = scalar * v
        h_res = x - h_hallu
        return h_res, h_hallu

    def steer(self, x: torch.Tensor, alpha: float = 1.0, polarity: float = -1.0) -> torch.Tensor:
        _, h_hallu = self.decompose(x)
        return x + float(polarity) * float(alpha) * h_hallu

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_res, h_hallu = self.decompose(x)
        pred_consistency = self.consistency_probe(h_hallu)
        pred_residual_adv = self.residual_adversary(h_res)
        return {
            "pred_consistency": pred_consistency,
            "pred_residual_adv": pred_residual_adv,
            "h_res": h_res,
            "h_hallu": h_hallu,
        }


@dataclass(frozen=True)
class AODCheckpoint:
    layer: int
    input_dim: int
    probe_hidden_dim: int
    grl_lambda: float
    seed: int
    label_name: str = "consistency"


def save_checkpoint(path: str, meta: AODCheckpoint, model: AODDisentangler) -> None:
    torch.save({"meta": meta.__dict__, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Tuple[AODCheckpoint, AODDisentangler]:
    ckpt = torch.load(path, map_location=map_location)
    meta_d = ckpt["meta"]
    meta_d.setdefault("label_name", "consistency")
    meta = AODCheckpoint(**meta_d)
    model = AODDisentangler(
        input_dim=int(meta.input_dim),
        probe_hidden_dim=int(meta.probe_hidden_dim),
        grl_lambda=float(meta.grl_lambda),
    )
    state_dict = ckpt["state_dict"]
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        # Detect early-prototype checkpoints with renamed branches and point
        # the user at the explicit migration rather than silently leaving
        # consistency_probe / residual_adversary at random init.
        legacy_prefixes = ("truth_probe.", "adversary.")
        looks_legacy = any(k.startswith(legacy_prefixes) for k in unexpected)
        hint = (
            "\nThis looks like an early-prototype checkpoint (keys start with "
            "'truth_probe.' or 'adversary.'). Re-train with the current "
            "train_layers.py, or manually rename keys:\n"
            "  truth_probe.* -> consistency_probe.*\n"
            "  adversary.*   -> residual_adversary.*"
            if looks_legacy
            else ""
        )
        raise RuntimeError(
            f"Checkpoint {path} is incompatible with AODDisentangler "
            f"(missing_keys={missing!r}, unexpected_keys={unexpected!r}).{hint}"
        )
    return meta, model
