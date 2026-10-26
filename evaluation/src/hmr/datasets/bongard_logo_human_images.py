"""Bongard-LOGO human-context image path reader.

Reads absolute image file paths for the DVRL (Direct Visual Rule Learning)
experiment, where all 13 puzzle images (6 pos + 6 neg + 1 test) are sent
together to a vision-language model.

The JSON files share the same structure as those used by bongard_logo_human.py.
Each entry has an ``images`` key (alongside ``action_prog``) that holds
relative paths to the rendered PNG files.
"""

from __future__ import annotations

import json
import os
from typing import Any


def read_sample(
    json_path: str,
    test_id: str,
    m: int,
    n: int,
    image_dir: str = "",
) -> tuple[list[str], str, str, str, str, Any]:
    """Read one puzzle and return absolute image paths.

    Parameters
    ----------
    json_path:
        Path to the puzzle JSON (bd.json / ff.json / …).
    test_id:
        Unique puzzle identifier.
    m:
        Number of positive images to include.
    n:
        Number of negative images to include.
    image_dir:
        Root directory that the relative image paths in the JSON are resolved
        against.  Typically the ``DVRL_IMAGE_DIR`` env var or the
        ``image_dir`` key from the YAML config.

    Returns
    -------
    image_paths : list[str]
        Absolute paths: first *m* positives, then *n* negatives, then the
        test image.  Length is always m + n + 1 = 13 for the default 6/6/1
        split.
    concept : str
        Human-readable concept label for the positive set.
    caption : str
        Original caption (used to derive the concept for bd_* entries).
    uid : str
        Stable unique id used for output folder names.
    test_cat : str
        Ground-truth label for the test image (``'pos'`` or ``'neg'``).
    commonsense : None
        Placeholder for compatibility with other dataset readers.
    """

    with open(json_path) as f:
        data = json.load(f)

    sample = next((item for item in data if item["test_id"] == test_id), None)
    if sample is None:
        raise ValueError(f"Sample with test_id {test_id} not found in {json_path}.")

    images = sample.get("images")
    if images is None:
        raise KeyError(
            f"Sample {test_id} in {json_path} has no 'images' key. "
            "Use the bongard_logo_human dataset for action-program experiments."
        )

    pos_paths = [_resolve(p, image_dir) for p in images["pos"][:m]]
    neg_paths = [_resolve(p, image_dir) for p in images["neg"][:n]]
    test_path = _resolve(images["testfiles"], image_dir)

    test_cat = sample["testfiles"]["category"]
    concept = sample["concept_ui"]
    caption = sample["caption"]
    uid = sample["uid"]

    image_paths = pos_paths + neg_paths + [test_path]
    return image_paths, concept, caption, uid, test_cat, None


def _resolve(relative_path: str, image_dir: str) -> str:
    """Join *image_dir* and *relative_path*, returning an absolute path."""
    path = os.path.join(image_dir, relative_path)
    return os.path.normpath(path)
