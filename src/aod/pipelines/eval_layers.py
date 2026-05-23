from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from aod.core.aod import load_checkpoint
from aod.core.dataset import load_layer_dataset, split_indices


@dataclass(frozen=True)
class EvalResult:
    layer: int
    n_test: int
    base_consistency: float
    direction_acc: float
    residual_adv_acc: float
    best_alpha: float
    best_direction_acc: float


def _acc_binary(pred01: torch.Tensor, y01: torch.Tensor) -> float:
    return float((pred01 == y01).float().mean().item() * 100.0)


def _acc_from_logits(logits: torch.Tensor, y01: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).to(y01.dtype)
    return _acc_binary(pred, y01)


def parse_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to aod_<bench>_layer_*.pt")
    ap.add_argument(
        "--layers_dir",
        required=True,
        help="Directory of extracted hidden states matching this ckpt, "
             "e.g. output/layers/qwen2_5vl_pope.",
    )
    ap.add_argument(
        "--test_ratio",
        type=float,
        default=-1.0,
        help="Test ratio for split_indices. Default <0 means use the training "
             "default (0.2); must match the value used by train_layers.py for "
             "this ckpt.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for split_indices. Default <0 means read meta.seed from the "
             "ckpt so the held-out split matches the training run.",
    )
    ap.add_argument("--alphas", default="0.0,0.5,1.0,2.0,3.0,5.0")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--quiet_load_warning", action="store_true", help="Suppress torch.load FutureWarning output.")
    args = ap.parse_args(argv)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.quiet_load_warning:
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\\.load.*weights_only.*")
    meta, model = load_checkpoint(args.ckpt, map_location=device)
    model = model.to(device)
    layer = int(meta.layer)
    ds_path = os.path.join(args.layers_dir, f"layer_{layer}_dataset.json")
    ds = load_layer_dataset(ds_path, layer=layer, device=device)

    seed = int(args.seed) if int(args.seed) >= 0 else int(meta.seed)
    test_ratio = float(args.test_ratio) if float(args.test_ratio) >= 0.0 else 0.2
    print(f"[split] seed={seed} test_ratio={test_ratio} (meta.seed={meta.seed})")
    train_idx, test_idx = split_indices(n=int(ds.x.shape[0]), test_ratio=test_ratio, seed=seed)

    x_te = ds.x[test_idx]
    y_te = ds.y_consistency[test_idx]
    base_consistency = float(y_te.float().mean().item() * 100.0)

    model.eval()
    with torch.no_grad():
        out = model(x_te)
        direction_acc = _acc_from_logits(out["pred_consistency"], y_te)
        residual_adv_acc = _acc_from_logits(out["pred_residual_adv"], y_te)

        alphas = parse_floats(args.alphas)
        if not alphas:
            raise ValueError("Empty --alphas")
        best_alpha = float(alphas[0])
        best_direction_acc = float("-inf")
        for a in alphas:
            x_int = model.steer(x_te, alpha=float(a), polarity=-1.0)
            logits = model(x_int)["pred_consistency"]
            acc = _acc_from_logits(logits, y_te)
            if acc > best_direction_acc:
                best_direction_acc = float(acc)
                best_alpha = float(a)
            print(f"[alpha={a:g}] direction_probe_acc={acc:.2f}%")

    print("\n=== AOD extracted-layer diagnostic ===")
    print(f"ckpt={args.ckpt}")
    print(
        f"layer={layer} n_test={int(test_idx.size)} "
        f"base_consistency={base_consistency:.2f}% direction_acc={direction_acc:.2f}% "
        f"residual_adv_acc={residual_adv_acc:.2f}%"
    )
    print(f"best_alpha={best_alpha:g} best_direction_acc={best_direction_acc:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
