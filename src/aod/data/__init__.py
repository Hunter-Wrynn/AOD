"""Dataset loaders and benchmark-specific scoring."""

from aod.data.amber import (
    AMBERRecord,
    amber_to_records,
    load_amber_discriminative,
)
from aod.data.chair import (
    DEFAULT_COCO_SYNONYMS,
    ChairScore,
    chair_score,
    detect_objects_in_caption,
    load_coco_image_objects,
    load_coco_synonyms_json,
)

__all__ = [
    "AMBERRecord",
    "ChairScore",
    "DEFAULT_COCO_SYNONYMS",
    "amber_to_records",
    "chair_score",
    "detect_objects_in_caption",
    "load_amber_discriminative",
    "load_coco_image_objects",
    "load_coco_synonyms_json",
]
