"""Lookup of human-designed Bongard-LOGO shape metadata.

The ``bd_*`` puzzles in Bongard-LOGO have hand-authored concept labels stored
in a TSV (``human_designed_shapes.tsv``). Given a puzzle's caption (e.g.
``bd_equilateral_traingle``), this module returns the structured shape
metadata used to inject "concept hints" into the LLM prompt.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


def _default_tsv_path() -> Path:
    """Locate the shapes TSV.

    Checks (in order):
      1. ``HMR_SHAPES_TSV`` env var
      2. ``<repo_root>/data/human_designed_shapes.tsv``

    Repo root is computed two levels up from this file
    (``src/hmr/datasets/shape_metadata.py`` -> ``src/hmr/datasets`` ->
    ``src/hmr`` -> ``src`` -> repo root).
    """
    env = os.getenv("HMR_SHAPES_TSV")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "human_designed_shapes.tsv"


def normalize_bd_key(key: str) -> str:
    """Strip the leading ``bd_`` prefix."""
    return key[3:] if key.startswith("bd_") else key


def _load_tsv(tsv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            rows.append({k.strip(): (v.strip() if v else v) for k, v in raw.items()})
    return rows


def _find_shape_row(shape_fn: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    key = "shape function name"
    for r in rows:
        if r.get(key) == shape_fn:
            return r
    sf_lower = shape_fn.lower()
    for r in rows:
        if r.get(key, "").lower() == sf_lower:
            return r
    return None


def _parse_base_actions(value: str) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _parse_turn_angles(value: str) -> list[str]:
    return [x.strip() for x in (value or "").split("--") if x.strip()]


def _row_to_dict(row: dict[str, str]) -> dict[str, Any]:
    return {
        "shape_name": row.get("shape name"),
        "super_class": row.get("super class"),
        "set_of_base_actions": _parse_base_actions(row.get("set of base actions", "")),
        "turn_angles": _parse_turn_angles(row.get("turn angles", "")),
    }


def process_bd_json(key: str, tsv_path: str | Path | None = None) -> list[dict[str, Any] | None]:
    """Look up shape metadata for one ``bd_*`` caption.

    A caption may encode multiple shapes joined by ``-`` (e.g.
    ``bd_asymmetric_unbala_x-no_two_parts_sector3``); each part is looked up
    independently.

    Returns a list of metadata dicts (or None for shapes not in the TSV).
    """
    path = Path(tsv_path) if tsv_path else _default_tsv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Shapes TSV not found at {path}. "
            f"Set HMR_SHAPES_TSV to override, or place the file at the default location."
        )

    rows = _load_tsv(path)
    shape_fn = normalize_bd_key(key)

    results: list[dict[str, Any] | None] = []
    for part in (p.strip() for p in shape_fn.split("-") if p.strip()):
        row = _find_shape_row(part, rows)
        results.append(_row_to_dict(row) if row else None)
    return results
