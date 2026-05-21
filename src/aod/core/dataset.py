from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


ANSWER_FIELDS = ("gt_answer", "gt", "label", "answer")
PREDICTION_FIELDS = ("model_prediction", "base_pred", "prediction", "pred")


def normalize_binary_answer(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true", "correct"}:
        return 1
    if text in {"0", "no", "n", "false", "incorrect"}:
        return 0
    return None


def first_present(record: Dict[str, Any], fields: Tuple[str, ...]) -> Any:
    for field in fields:
        if field in record:
            return record[field]
    return None


@dataclass(frozen=True)
class LayerDataset:
    layer: int
    x: torch.Tensor
    y_consistency: torch.Tensor
    y_gt: torch.Tensor
    y_pred_base: torch.Tensor
    meta: List[Dict[str, Any]]


def load_layer_dataset(path: str, layer: int, device: torch.device) -> LayerDataset:
    layer = int(layer)
    key = f"layer_{layer}_hidden"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    x_list: List[List[float]] = []
    y_consistency_list: List[int] = []
    y_gt_list: List[int] = []
    y_pred_base_list: List[int] = []
    kept: List[Dict[str, Any]] = []

    for rec in data:
        if not isinstance(rec, dict) or key not in rec:
            continue
        vec = rec.get(key)
        if not isinstance(vec, list) or not vec:
            continue

        gt = normalize_binary_answer(first_present(rec, ANSWER_FIELDS))
        pred = normalize_binary_answer(first_present(rec, PREDICTION_FIELDS))
        if gt is None or pred is None:
            continue

        x_list.append(vec)
        y_gt_list.append(gt)
        y_pred_base_list.append(pred)
        y_consistency_list.append(int(gt == pred))
        kept.append(rec)

    if not x_list:
        raise ValueError(f"No usable records found in {path} for {key}")

    x = torch.tensor(np.asarray(x_list, dtype=np.float32), device=device)
    y_consistency = torch.tensor(np.asarray(y_consistency_list, dtype=np.float32)[:, None], device=device)
    y_gt = torch.tensor(np.asarray(y_gt_list, dtype=np.float32)[:, None], device=device)
    y_pred_base = torch.tensor(np.asarray(y_pred_base_list, dtype=np.float32)[:, None], device=device)
    return LayerDataset(
        layer=layer,
        x=x,
        y_consistency=y_consistency,
        y_gt=y_gt,
        y_pred_base=y_pred_base,
        meta=kept,
    )


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
