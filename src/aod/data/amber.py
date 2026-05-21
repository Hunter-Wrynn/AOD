"""AMBER (Wang et al. 2023) discriminative-subset loader.

The AMBER release ships two files:

  data/query/query_all.json   — list of {"id", "image", "query", "type": "discriminative"|"generative"}
  data/annotations.json       — dict keyed by str(id) with {"truth", "type"} where
                                "truth" ∈ {"yes","no"} for discriminative samples and
                                "type" ∈ {"existence","attribute","relation"}.

We join the two by id and emit a flat list of dicts that downstream loaders
can normalize with `aod.layer_dataset.normalize_binary_answer`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


_DISCRIMINATIVE_TYPES = {"existence", "attribute", "relation"}


@dataclass(frozen=True)
class AMBERRecord:
    id: int
    image: str
    query: str
    answer: str  # "yes" | "no"
    typology: str  # "existence" | "attribute" | "relation"


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_amber_discriminative(
    query_path: str,
    annotations_path: str,
    *,
    typology: Optional[str] = None,
) -> List[AMBERRecord]:
    """Return AMBER discriminative samples, optionally filtered by typology."""
    queries = _load_json(query_path)
    if not isinstance(queries, list):
        raise ValueError(f"Expected a JSON list in {query_path}")
    annotations = _load_json(annotations_path)
    if not isinstance(annotations, dict):
        raise ValueError(f"Expected a JSON object in {annotations_path}")

    typology_filter = None
    if typology is not None:
        typology = typology.strip().lower()
        if typology not in _DISCRIMINATIVE_TYPES:
            raise ValueError(
                f"--amber_typology must be one of {sorted(_DISCRIMINATIVE_TYPES)} or omitted; got {typology!r}"
            )
        typology_filter = typology

    out: List[AMBERRecord] = []
    for rec in queries:
        sid = rec.get("id")
        if sid is None:
            continue
        ann = annotations.get(str(sid)) or annotations.get(int(sid)) if isinstance(annotations, dict) else None
        if not isinstance(ann, dict):
            continue
        truth = str(ann.get("truth", "")).strip().lower()
        typ = str(ann.get("type", "")).strip().lower()
        if truth not in {"yes", "no"} or typ not in _DISCRIMINATIVE_TYPES:
            continue
        if typology_filter is not None and typ != typology_filter:
            continue
        out.append(
            AMBERRecord(
                id=int(sid),
                image=str(rec.get("image", "")),
                query=str(rec.get("query", "")),
                answer=truth,
                typology=typ,
            )
        )
    return out


def amber_to_records(records: Iterable[AMBERRecord]) -> List[Dict[str, Any]]:
    """Flatten to the dict form consumed by the generic binary path."""
    return [
        {
            "id": rec.id,
            "image": rec.image,
            "question": rec.query,
            "answer": rec.answer,
            "gt_answer": "1" if rec.answer == "yes" else "0",
            "amber_type": rec.typology,
        }
        for rec in records
    ]
