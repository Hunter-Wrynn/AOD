from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _as01(s: Any) -> int:
    ss = str(s).strip()
    if ss not in {"0", "1"}:
        raise ValueError(f"Expected '0'/'1' string, got: {s!r}")
    return int(ss)


@dataclass(frozen=True)
class HallusionBenchLayerDataset:
    layer: int
    x: torch.Tensor  # [N, D]
    y_gt: torch.Tensor  # [N, 1] ground-truth yes/no
    y_pred_base: torch.Tensor  # [N, 1] base model yes/no prediction
    meta: List[Dict[str, Any]]  # raw records


def load_layer_dataset(path: str, layer: int, device: torch.device) -> HallusionBenchLayerDataset:
    layer = int(layer)
    key = f"layer_{layer}_hidden"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    x_list: List[List[float]] = []
    y_list: List[int] = []
    base_list: List[int] = []
    kept: List[Dict[str, Any]] = []
    for rec in data:
        if key not in rec:
            continue
        vec = rec.get(key)
        if not isinstance(vec, list) or not vec:
            continue
        gt = rec.get("gt_answer")
        pred = rec.get("model_prediction")
        if gt is None or pred is None:
            continue
        try:
            y = _as01(gt)
            b = _as01(pred)
        except Exception:
            continue
        x_list.append(vec)
        y_list.append(y)
        base_list.append(b)
        kept.append(rec)

    if not x_list:
        raise ValueError(f"No usable records found in {path} for {key}")

    x = torch.tensor(np.asarray(x_list, dtype=np.float32), device=device)
    y_gt = torch.tensor(np.asarray(y_list, dtype=np.float32)[:, None], device=device)
    y_pred_base = torch.tensor(np.asarray(base_list, dtype=np.float32)[:, None], device=device)
    return HallusionBenchLayerDataset(layer=layer, x=x, y_gt=y_gt, y_pred_base=y_pred_base, meta=kept)


def split_indices(n: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("--test_ratio must be in (0,1)")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * float(test_ratio))))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if train_idx.size == 0:
        train_idx = idx[n_test - 1 :]
        test_idx = idx[: n_test - 1]
    return train_idx, test_idx

