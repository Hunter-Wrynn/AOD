from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aod.core.aod import AODCheckpoint, AODDisentangler, save_checkpoint
from aod.core.dataset import LayerDataset, load_layer_dataset, split_indices


@dataclass(frozen=True)
class TrainResult:
    layer: int
    n_train: int
    n_test: int
    base_consistency: float
    direction_acc: float
    residual_adv_acc: float


def _acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).to(y.dtype)
    return float((pred == y).float().mean().item() * 100.0)


def _mean01(y: torch.Tensor) -> float:
    return float(y.float().mean().item() * 100.0)


def train_one_layer(
    ds: LayerDataset,
    *,
    seed: int,
    test_ratio: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    probe_hidden_dim: int,
    grl_lambda: float,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    device = ds.x.device
    train_idx, test_idx = split_indices(n=int(ds.x.shape[0]), test_ratio=test_ratio, seed=seed)
    x_tr, y_tr = ds.x[train_idx], ds.y_consistency[train_idx]
    x_te, y_te = ds.x[test_idx], ds.y_consistency[test_idx]

    model = AODDisentangler(
        input_dim=int(ds.x.shape[1]),
        probe_hidden_dim=int(probe_hidden_dim),
        grl_lambda=float(grl_lambda),
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    bce = nn.BCEWithLogitsLoss()

    n_tr = int(x_tr.shape[0])
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        total = 0.0
        for start in range(0, n_tr, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            xb = x_tr[idx]
            yb = y_tr[idx]

            out = model(xb)
            loss_task = bce(out["pred_consistency"], yb)
            loss_adv = bce(out["pred_residual_adv"], yb)
            loss = loss_task + loss_adv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.renormalize_direction_()
            total += float(loss.item()) * int(xb.shape[0])

        if epoch == 0 or (epoch + 1) % 25 == 0 or (epoch + 1) == int(epochs):
            model.eval()
            with torch.no_grad():
                out_te = model(x_te)
                base_consistency = _mean01(y_te)
                direction_acc = _acc_from_logits(out_te["pred_consistency"], y_te)
                residual_adv_acc = _acc_from_logits(out_te["pred_residual_adv"], y_te)
                avg_loss = total / max(1, n_tr)
            print(
                f"[Layer {ds.layer}] ep={epoch+1:03d}/{epochs} "
                f"train_loss={avg_loss:.4f} base_consistency={base_consistency:.2f}% "
                f"direction_acc={direction_acc:.2f}% residual_adv_acc={residual_adv_acc:.2f}%"
            )

    model.eval()
    with torch.no_grad():
        out_te = model(x_te)
        base_consistency = _mean01(y_te)
        direction_acc = _acc_from_logits(out_te["pred_consistency"], y_te)
        residual_adv_acc = _acc_from_logits(out_te["pred_residual_adv"], y_te)

    return {
        "model": model,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "result": TrainResult(
            layer=int(ds.layer),
            n_train=int(train_idx.size),
            n_test=int(test_idx.size),
                base_consistency=float(base_consistency),
                direction_acc=float(direction_acc),
                residual_adv_acc=float(residual_adv_acc),
        ),
    }


def parse_layers(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers_dir", default="output/layers/qwen2_5vl_hallusionbench")
    ap.add_argument("--layers", default="1,4,8,12,16,20,24,28")
    ap.add_argument("--output_dir", default="output/aod_ckpt/hallusionbench")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--probe_hidden_dim", type=int, default=512)
    ap.add_argument("--grl_lambda", type=float, default=1.0, help="Paper's adversarial weight lambda; applied inside the GRL backward.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
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

    os.makedirs(args.output_dir, exist_ok=True)
    layers = parse_layers(args.layers)
    if not layers:
        raise ValueError("Empty --layers")

    all_results: List[TrainResult] = []
    for layer in layers:
        ds_path = os.path.join(args.layers_dir, f"layer_{int(layer)}_dataset.json")
        ds = load_layer_dataset(ds_path, layer=int(layer), device=device)
        run = train_one_layer(
            ds,
            seed=int(args.seed),
            test_ratio=float(args.test_ratio),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            probe_hidden_dim=int(args.probe_hidden_dim),
            grl_lambda=float(args.grl_lambda),
        )
        model: AODDisentangler = run["model"]
        res: TrainResult = run["result"]
        all_results.append(res)

        ckpt_meta = AODCheckpoint(
            layer=int(layer),
            input_dim=int(ds.x.shape[1]),
            probe_hidden_dim=int(args.probe_hidden_dim),
            grl_lambda=float(args.grl_lambda),
            seed=int(args.seed),
            label_name="consistency",
        )
        ckpt_path = os.path.join(args.output_dir, f"aod_hallusionbench_layer_{int(layer)}.pt")
        save_checkpoint(ckpt_path, ckpt_meta, model)
        print(f"[Saved] {ckpt_path}")

    print("\n=== AOD direction training summary ===")
    for r in all_results:
        print(
            f"layer={r.layer:<2d} n_train={r.n_train:<4d} n_test={r.n_test:<4d} "
            f"base_consistency={r.base_consistency:>6.2f}% "
            f"direction_acc={r.direction_acc:>6.2f}% residual_adv_acc={r.residual_adv_acc:>6.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
