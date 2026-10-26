"""Stage 1: model-generated drawing descriptions from single images.

Each unique figure image (about 2,500 across the 200 items, because paired
positive/negative queries share a base problem) is captioned once per
captioner model, in isolation: no task context, no concept, no information
about which set the image belongs to. The caption is requested in the same
"drawing story" style as the oracle Action Descriptions so that the downstream
reasoning prompt is identical between caption and oracle conditions.

Captions are stored as JSONL keyed by the image's relative path, together
with the oracle description of the same image for later fidelity scoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..llm import _clean_json, _extract_json_block, _parse_raw_tags, _regex_repairs
from .backends import BackendError, ChatResult, chat, model_provenance
from .design import Item, slug

logger = logging.getLogger(__name__)

CAPTION_PROMPT_VERSION = "v3"
MAX_BUDGET = 32768

CAPTION_SYSTEM_PROMPT = """
**Objective:** Describe how to draw the single figure shown in the image.

The figure was produced by a LOGO-style turtle. It draws a sequence of strokes; each stroke is either a straight line or a circular arc. Every stroke carries a texture, one of: normal, zigzag, circle, triangle, square. The texture is the pattern the pen leaves along the path, so a "triangle line" is a straight path drawn with small triangles along it. Lengths and radii are relative, on a 0 to 1 scale. After each stroke the turtle either continues straight or turns left or right. Disconnected shapes are drawn one after another.

**What matters most:** enumerate EVERY stroke. Trace the outline all the way round and emit one step each time the path changes direction or changes texture. A closed hexagon has six steps, not two. A closed triangle has three steps. Do not summarise several edges as one step, and do not stop early: keep going until you return to where you started or the path ends. Getting the NUMBER of straight strokes and the NUMBER of arcs right is far more important than getting lengths or angles precise, so estimate the numbers loosely and spend your effort on the count.

**Format:** write a numbered recipe in exactly this style.

To draw the figure, follow these steps:

Step 1: draw a normal line of 0.500 units. After that, turn left by 120.0 degrees.

Step 2: draw a zigzag line of 0.500 units. After that, turn left by 120.0 degrees.

Step 3: draw a normal arc with a radius of 0.500 and sweeping 180.0 degrees. Then, continue straight without turning.

The figure is now complete.

**Worked example.** A closed six-sided figure whose edges are all plain straight lines is written as SIX steps, one per edge, each turning by roughly 60 degrees:

To draw the figure, follow these steps:

Step 1: draw a normal line of 0.500 units. After that, turn left by 60.0 degrees.

Step 2: draw a normal line of 0.500 units. After that, turn left by 60.0 degrees.

Step 3: draw a normal line of 0.500 units. After that, turn left by 60.0 degrees.

Step 4: draw a normal line of 0.500 units. After that, turn left by 60.0 degrees.

Step 5: draw a normal line of 0.500 units. After that, turn left by 60.0 degrees.

Step 6: draw a normal line of 0.500 units. After that, turn left by 60.0 degrees.

The figure is now complete.

**Rules:**
- Describe only this one image. Do not guess any rule, category, or meaning.
- Use "line" for straight strokes and "arc" for curved strokes; always name the texture.
- Almost every figure is ONE continuous path, so normally you will emit a single
  run of steps. Only write "Then, start a new, separate shape:" if the drawing
  clearly contains a second shape that does not touch the first, and restart the
  numbering when you do. Do not use it to mark a corner, a texture change, or the
  point where a closed outline meets its own start.
- Begin with "To draw the figure, follow these steps:" and end with "The figure is now complete."

**Required output:** respond only with a single JSON object with one key:
{"description": "<the complete recipe text, with newlines>"}
""".strip()

CAPTION_USER_PROMPT = "Here is the image. Trace the whole outline, emit one step per stroke, and return the JSON object."

# Open-weight replication: a contrasting caption format. The same single-image
# task, written as prose instead of the step recipe, so that the self-caption
# result can be checked against the recipe format it was measured with.
CAPTION_FREE_PROMPT_VERSION = "free-v1"

CAPTION_FREE_SYSTEM_PROMPT = """
**Objective:** Describe the single figure shown in the image, precisely enough that someone who cannot see it could tell it apart from similar figures.

The figure is a black line drawing made of straight lines and curved arcs. Each stroke carries a texture: plain, zigzag, or a chain of small circles, triangles or squares.

**Describe, in plain prose:**
- how many separate parts the drawing has, and how they are arranged;
- for each part, whether it is closed or open, how many straight strokes and how many curved strokes it has, and their textures;
- notable geometry: corners and their angles (acute, right, obtuse), parallel or crossing lines, symmetry, overall proportions (thin, wide, regular).

**Rules:**
- Describe only this one image. Do not guess any rule, category or meaning.
- Count strokes carefully; counts matter more than exact sizes.
- Write at most about 150 words.

**Required output:** respond only with a single JSON object with one key:
{"description": "<the prose description>"}
""".strip()

CAPTION_FREE_USER_PROMPT = "Here is the image. Describe the figure as instructed and return the JSON object."

STYLES = {
    "steps": (CAPTION_PROMPT_VERSION, CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT, ""),
    "free": (CAPTION_FREE_PROMPT_VERSION, CAPTION_FREE_SYSTEM_PROMPT, CAPTION_FREE_USER_PROMPT, "__free"),
}


def style_store(captioner: str, style: str, draw: str = "") -> str:
    """Store name under captions/: the captioner, suffixed for non-default styles and draws."""
    return captioner + STYLES[style][3] + (f"__{draw}" if draw else "")


_FIRST_LINE = re.compile(r"^\s*To draw (the|this) figure(?: \d+)?, follow these steps:\s*", re.IGNORECASE)


def render_caption(text: str, figure_index: int) -> str:
    """Normalize a caption into the oracle numbering: 'To draw figure i, ...'."""
    body = text.strip()
    if _FIRST_LINE.match(body):
        body = _FIRST_LINE.sub("", body, count=1)
    body = re.sub(r"\s*The figure is now complete\.?\s*$", "", body, flags=re.IGNORECASE)
    return f"To draw figure {figure_index}, follow these steps:\n\n{body.strip()}\n\nThe figure is now complete.\n\n"


def render_free_caption(text: str, figure_index: int) -> str:
    return f"Figure {figure_index}: {text.strip()}\n\n"


def caption_dir(out_root: str | Path, captioner: str) -> Path:
    return Path(out_root) / "captions" / slug(captioner)


def caption_store_path(out_root: str | Path, captioner: str) -> Path:
    return caption_dir(out_root, captioner) / "captions.jsonl"


def load_captions(path: str | Path, prompt_version: str | None = None) -> dict[str, dict[str, Any]]:
    """Return {relative_image_path: row} for successfully captioned images.

    ``prompt_version`` restricts the result to captions written by that version
    of the prompt. Resumption passes the current version so that changing the
    prompt does not silently reuse captions produced by the old one; readers
    that want everything on disk (analysis, fidelity scoring) omit it.
    """
    out: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if prompt_version is not None and row.get("prompt_version") != prompt_version:
                continue
            out[row["image"]] = row
    return out


def _parse_caption(raw: str) -> str | None:
    text = _parse_raw_tags(raw or "")[0]
    cleaned = _clean_json(text)
    for cand in (cleaned, _extract_json_block(cleaned) or "", _regex_repairs(cleaned)):
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            for k in ("description", "Description", "recipe"):
                if isinstance(data.get(k), str) and data[k].strip():
                    return data[k].strip()
    # Fallback: a bare recipe without JSON wrapper is still a usable caption.
    if "Step 1" in text and "follow these steps" in text:
        return text.strip()
    return None


def unique_images(items: list[Item]) -> dict[str, dict[str, Any]]:
    """Map relative image path -> {oracle description, first item, figure index}."""
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        for idx, (rel, oracle) in enumerate(zip(it.images, it.descriptions), start=1):
            out.setdefault(rel, {"image": rel, "oracle": oracle, "example_item": it.test_id, "example_index": idx})
    return out


def run_captioning(
    items: list[Item],
    captioner: str,
    model_cfg: dict[str, Any],
    image_dir: str | Path,
    out_root: str | Path,
    *,
    max_attempts: int = 3,
    workers: int = 1,
    max_tokens: int = 4096,
    think: bool = False,
    dry_run: bool = False,
    style: str = "steps",
    temperature: float = 0.0,
    seed: int = 42,
    draw: str = "",
) -> dict[str, int]:
    """Caption every unique image not yet captioned. Resumable and thread-safe.

    ``draw`` names an additional, independently sampled caption set (amendment
    A3 of docs/factorial_openweight_plan.md): it is stored apart from the
    primary set and generated at ``temperature``/``seed`` instead of 0/42.
    """
    version, system_prompt, user_prompt, _ = STYLES[style]
    store = caption_store_path(out_root, style_store(captioner, style, draw))
    store.parent.mkdir(parents=True, exist_ok=True)
    done = load_captions(store, prompt_version=version)
    stale = len(load_captions(store)) - len(done)
    if stale:
        logger.warning("[captions:%s] ignoring %d caption(s) from an older prompt version; "
                       "they stay in the file but are re-generated under %s",
                       captioner, stale, version)
    todo = {k: v for k, v in unique_images(items).items() if k not in done}
    stats = {"unique_images": len(done) + len(todo), "already_done": len(done), "to_run": len(todo), "ok": 0, "failed": 0}
    logger.info("[captions:%s] %d unique images, %d already captioned, %d to run", captioner, stats["unique_images"], len(done), len(todo))
    if dry_run or not todo:
        return stats

    manifest = {
        "captioner": captioner,
        "prompt_version": version,
        "style": style,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "temperature": temperature if model_cfg.get("supports_temperature", True) else None,
        "seed": seed,
        "draw": draw or "primary",
        "think": think,
        "max_tokens": max_tokens,
        "provenance": model_provenance(captioner),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(store.parent / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    (store.parent / "prompt_system.txt").write_text(system_prompt)

    lock = threading.Lock()
    every = max(1, min(25, len(todo) // 8))

    def one(rel: str, meta: dict[str, Any]) -> dict[str, Any]:
        abs_path = str(Path(image_dir) / rel)
        row: dict[str, Any] = {"image": rel, "captioner": captioner, "oracle": meta["oracle"], "prompt_version": version}
        last_err = None
        budget = max_tokens
        for attempt in range(1, max_attempts + 1):
            try:
                res: ChatResult = chat(
                    captioner, system_prompt, user_prompt, images=[abs_path],
                    temperature=temperature, seed=seed + 1000 * (attempt - 1), max_tokens=budget,
                    think=think, json_schema=None,
                    num_ctx=model_cfg.get("num_ctx"), supports_temperature=model_cfg.get("supports_temperature", True),
                    min_interval_s=float(model_cfg.get("min_interval_s", 0.0)),
                )
            except BackendError as exc:
                last_err = str(exc)
                continue
            if not (res.text or "").strip():
                # The reasoning trace consumed the whole output budget: grow it.
                budget = min(budget * 2, MAX_BUDGET)
                last_err = f"empty output (thought {len(res.thought or '')} chars); budget -> {budget}"
                continue
            desc = _parse_caption(res.text)
            row.update({
                "attempts": attempt, "max_tokens": budget, "usage": res.usage, "duration_s": round(res.duration_s, 3),
                "response_model": res.response_model, "raw": res.text, "thought_chars": len(res.thought or ""),
            })
            if desc:
                row.update({"status": "ok", "description": desc})
                return row
            last_err = "unparseable caption"
        row.update({"status": "failed", "error": last_err, "attempts": max_attempts})
        return row

    def record(row: dict[str, Any]) -> None:
        with lock:
            with open(store, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["ok" if row["status"] == "ok" else "failed"] += 1
            n = stats["ok"] + stats["failed"]
            if n % every == 0 or n == stats["to_run"]:
                logger.info("[captions:%s] %d/%d (failed %d)", captioner, n, stats["to_run"], stats["failed"])

    if workers <= 1:
        for rel, meta in todo.items():
            record(one(rel, meta))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, rel, meta) for rel, meta in todo.items()]
            for fut in as_completed(futs):
                record(fut.result())
    return stats
