"""CHAIR (Caption Hallucination Assessment with Image Relevance) scoring.

We implement CHAIR_S (sentence-level) and CHAIR_I (instance-level) on COCO
following Rohrbach et al., 2018:

    CHAIR_I = |{hallucinated objects mentioned}| / |{all objects mentioned}|
    CHAIR_S = |{captions with >=1 hallucination}| / |{captions}|

An object is considered "mentioned" in a generated caption if any of its
COCO synonyms (singular or plural) appears as a whole word in the caption.
The set of ground-truth objects per image is the union of category names
attached to the COCO `instances` annotations for that image_id.

The implementation purposefully avoids importing external dependencies. It
expects a thin synonym mapping (provided here as `DEFAULT_COCO_SYNONYMS`)
covering the 80 COCO categories used in the original CHAIR release. Callers
may pass a richer map (e.g. the one ships with the official CHAIR repo).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")


def _tokenize(caption: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(caption)]


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    if n <= 0:
        return ()
    return (" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))


# Minimal CHAIR-style synonym map. Keys are canonical COCO category names; values
# are lists of synonyms (whitespace-separated phrases) detected in captions.
# Multi-word entries are matched against contiguous n-grams of the same length.
DEFAULT_COCO_SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children", "kid", "kids"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "car": ["car", "cars", "automobile", "automobiles", "vehicle", "vehicles"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
    "airplane": ["airplane", "airplanes", "aeroplane", "aeroplanes", "plane", "planes", "jet", "jets"],
    "bus": ["bus", "buses"],
    "train": ["train", "trains"],
    "truck": ["truck", "trucks"],
    "boat": ["boat", "boats"],
    "traffic light": ["traffic light", "traffic lights", "stoplight", "stoplights"],
    "fire hydrant": ["fire hydrant", "fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop sign", "stop signs"],
    "parking meter": ["parking meter", "parking meters"],
    "bench": ["bench", "benches"],
    "bird": ["bird", "birds"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "horse": ["horse", "horses", "pony", "ponies"],
    "sheep": ["sheep", "lamb", "lambs"],
    "cow": ["cow", "cows", "cattle"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "backpacks", "rucksack", "rucksacks"],
    "umbrella": ["umbrella", "umbrellas"],
    "handbag": ["handbag", "handbags", "purse", "purses"],
    "tie": ["tie", "ties", "necktie", "neckties"],
    "suitcase": ["suitcase", "suitcases", "luggage"],
    "frisbee": ["frisbee", "frisbees"],
    "skis": ["ski", "skis"],
    "snowboard": ["snowboard", "snowboards"],
    "sports ball": ["sports ball", "ball", "balls"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "baseball bats", "bat", "bats"],
    "baseball glove": ["baseball glove", "baseball gloves", "glove", "gloves"],
    "skateboard": ["skateboard", "skateboards"],
    "surfboard": ["surfboard", "surfboards"],
    "tennis racket": ["tennis racket", "tennis rackets", "racket", "rackets"],
    "bottle": ["bottle", "bottles"],
    "wine glass": ["wine glass", "wine glasses"],
    "cup": ["cup", "cups", "mug", "mugs"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives"],
    "spoon": ["spoon", "spoons"],
    "bowl": ["bowl", "bowls"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hot dogs", "hotdog", "hotdogs"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "cake": ["cake", "cakes"],
    "chair": ["chair", "chairs"],
    "couch": ["couch", "couches", "sofa", "sofas"],
    "potted plant": ["potted plant", "potted plants", "houseplant", "houseplants", "plant", "plants"],
    "bed": ["bed", "beds"],
    "dining table": ["dining table", "dining tables", "table", "tables"],
    "toilet": ["toilet", "toilets"],
    "tv": ["tv", "tvs", "television", "televisions", "monitor", "monitors", "screen", "screens"],
    "laptop": ["laptop", "laptops"],
    "mouse": ["mouse", "mice"],
    "remote": ["remote", "remotes", "remote control", "remote controls"],
    "keyboard": ["keyboard", "keyboards"],
    "cell phone": ["cell phone", "cell phones", "phone", "phones", "smartphone", "smartphones"],
    "microwave": ["microwave", "microwaves"],
    "oven": ["oven", "ovens"],
    "toaster": ["toaster", "toasters"],
    "sink": ["sink", "sinks"],
    "refrigerator": ["refrigerator", "refrigerators", "fridge", "fridges"],
    "book": ["book", "books"],
    "clock": ["clock", "clocks"],
    "vase": ["vase", "vases"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy bears", "stuffed bear", "stuffed bears"],
    "hair drier": ["hair drier", "hair driers", "hair dryer", "hair dryers"],
    "toothbrush": ["toothbrush", "toothbrushes"],
}


def detect_objects_in_caption(
    caption: str,
    synonyms: Mapping[str, Sequence[str]] = DEFAULT_COCO_SYNONYMS,
) -> Set[str]:
    """Return canonical category names whose synonyms occur in the caption."""
    tokens = _tokenize(caption)
    if not tokens:
        return set()
    by_length: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for canonical, words in synonyms.items():
        for w in words:
            phrase_tokens = w.lower().split()
            if not phrase_tokens:
                continue
            by_length[len(phrase_tokens)][" ".join(phrase_tokens)].add(canonical)
    found: Set[str] = set()
    for n, phrase_map in by_length.items():
        for ng in _ngrams(tokens, n):
            for canon in phrase_map.get(ng, ()):  # type: ignore[arg-type]
                found.add(canon)
    return found


@dataclass(frozen=True)
class ChairScore:
    chair_s: float
    chair_i: float
    num_captions: int
    num_hallucinated_captions: int
    num_mentioned_objects: int
    num_hallucinated_objects: int


def load_coco_image_objects(instances_path: str) -> Dict[int, Set[str]]:
    """Parse a COCO `instances_*.json` file → {image_id: {category_name, ...}}.

    Designed for the discriminative-style usage in CHAIR: we only need the set
    of object categories present per image, not bounding boxes.
    """
    with open(instances_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    if not isinstance(coco, dict):
        raise ValueError(f"Expected a COCO json object in {instances_path}")
    cat_by_id: Dict[int, str] = {
        int(c["id"]): str(c["name"]).lower() for c in coco.get("categories", [])
    }
    img_objs: Dict[int, Set[str]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        cid = ann.get("category_id")
        iid = ann.get("image_id")
        if cid is None or iid is None:
            continue
        name = cat_by_id.get(int(cid))
        if name is None:
            continue
        img_objs[int(iid)].add(name)
    return img_objs


def chair_score(
    samples: Sequence[Tuple[int, str]],
    image_objects: Mapping[int, Set[str]],
    synonyms: Mapping[str, Sequence[str]] = DEFAULT_COCO_SYNONYMS,
) -> ChairScore:
    """Compute CHAIR_S / CHAIR_I.

    `samples` is an iterable of `(image_id, generated_caption)` pairs. Image
    ids missing from `image_objects` contribute nothing to the score and emit
    a per-caption skip — this is consistent with the official CHAIR script.
    """
    n_caps = 0
    n_hallu_caps = 0
    n_mentions = 0
    n_hallu_mentions = 0
    for image_id, caption in samples:
        if image_id not in image_objects:
            continue
        gt = image_objects[image_id]
        mentioned = detect_objects_in_caption(caption, synonyms=synonyms)
        if not mentioned:
            n_caps += 1
            continue
        hallu = mentioned - gt
        n_caps += 1
        n_mentions += len(mentioned)
        n_hallu_mentions += len(hallu)
        if hallu:
            n_hallu_caps += 1
    chair_s = (n_hallu_caps / n_caps) if n_caps else 0.0
    chair_i = (n_hallu_mentions / n_mentions) if n_mentions else 0.0
    return ChairScore(
        chair_s=chair_s,
        chair_i=chair_i,
        num_captions=n_caps,
        num_hallucinated_captions=n_hallu_caps,
        num_mentioned_objects=n_mentions,
        num_hallucinated_objects=n_hallu_mentions,
    )


def load_coco_synonyms_json(path: str) -> Dict[str, List[str]]:
    """Optional helper: load a synonyms JSON of shape `{category: [syn, ...]}`."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return {str(k).lower(): [str(s).lower() for s in v] for k, v in data.items()}
