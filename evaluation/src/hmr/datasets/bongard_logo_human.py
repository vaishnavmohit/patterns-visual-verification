"""Bongard-LOGO human-context sample reader.

The JSON files (bd.json / ff.json / hd_comb.json / hd_novel.json) contain
human-rated Bongard-LOGO puzzles. Each entry has a positive set, a negative
set, and a single test image, all encoded as action programs.
"""

from __future__ import annotations

import json
from typing import Any


def read_sample(json_path: str, test_id: str, m: int, n: int) -> tuple[list, str, str, str, str, Any]:
    """Read one puzzle from the JSON file.

    Returns
    -------
    image_paths : list
        Positive action programs (first m), then negative action programs
        (next n), then the test action program.
    concept : str
        Human-readable concept label for the positive set.
    caption : str
        Original caption (used to derive the concept for bd_* entries).
    uid : str
        Stable unique id used for output folder names.
    test_cat : str
        Ground-truth label for the test image ('pos' or 'neg').
    commonsense : None
        Placeholder for compatibility with other datasets.
    """

    with open(json_path) as f:
        data = json.load(f)

    sample = next((item for item in data if item["test_id"] == test_id), None)
    if sample is None:
        raise ValueError(f"Sample with test_id {test_id} not found in {json_path}.")

    pos = sample["action_prog"]["pos"][:m]
    neg = sample["action_prog"]["neg"][:n]
    test_image = sample["testfiles"]["action_prog"]
    test_cat = sample["testfiles"]["category"]
    concept = sample["concept_ui"]
    caption = sample["caption"]
    uid = sample["uid"]

    image_paths = pos + neg + [test_image]
    return image_paths, concept, caption, uid, test_cat, None
