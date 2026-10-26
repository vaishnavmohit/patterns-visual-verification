"""I/O helpers: image encoding and global seed setup."""

from __future__ import annotations

import base64
import gc
import os
import random
from io import BytesIO

import numpy as np
from PIL import Image


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PYTHONHASHSEED for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its raw base64 string (no resizing)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_to_base64_shrink(
    image_path: str,
    max_dimension: int = 1024,
    max_size_mb: float = 4,
) -> str:
    """Resize-and-encode an image so the resulting base64 fits within max_size_mb."""

    image = Image.open(image_path)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    supported = {"jpeg", "jpg", "png", "bmp", "webp"}
    if ext not in supported:
        ext = "jpeg"
    elif ext == "jpg":
        ext = "jpeg"

    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    while True:
        buf = BytesIO()
        image.save(buf, format=ext.upper())
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        if len(b64) <= max_size_mb * 1_048_576 * 3 / 4:
            break
        image = image.resize(
            (int(image.width * 0.9), int(image.height * 0.9)),
            Image.Resampling.LANCZOS,
        )
        del buf
        gc.collect()

    return b64
