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


def mlp_probe(input_dim: int, hidden_dim: int = 1024) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class AODDisentangler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        probe_hidden_dim: int = 1024,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.v = nn.Parameter(torch.randn(1, input_dim))
        self.truth_probe = mlp_probe(input_dim=input_dim, hidden_dim=probe_hidden_dim)
        self.adversary = nn.Sequential(
            GradientReversalLayer(lambda_=grl_lambda),
            mlp_probe(input_dim=input_dim, hidden_dim=probe_hidden_dim),
        )

    def v_unit(self) -> torch.Tensor:
        return F.normalize(self.v, dim=1)

    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.v_unit()
        scalar = x.matmul(v.t())
        h_hallu = scalar * v
        h_truth = x - h_hallu
        return h_truth, h_hallu

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_truth, h_hallu = self.decompose(x)
        pred_truth = self.truth_probe(h_truth)
        pred_adv = self.adversary(h_hallu)
        return {"pred_truth": pred_truth, "pred_adv": pred_adv, "h_truth": h_truth, "h_hallu": h_hallu}


@dataclass(frozen=True)
class AODCheckpoint:
    layer: int
    input_dim: int
    probe_hidden_dim: int
    grl_lambda: float
    seed: int


def save_checkpoint(path: str, meta: AODCheckpoint, model: AODDisentangler) -> None:
    torch.save({"meta": meta.__dict__, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Tuple[AODCheckpoint, AODDisentangler]:
    ckpt = torch.load(path, map_location=map_location)
    meta_d = ckpt["meta"]
    meta = AODCheckpoint(**meta_d)
    model = AODDisentangler(
        input_dim=int(meta.input_dim),
        probe_hidden_dim=int(meta.probe_hidden_dim),
        grl_lambda=float(meta.grl_lambda),
    )
    model.load_state_dict(ckpt["state_dict"])
    return meta, model

