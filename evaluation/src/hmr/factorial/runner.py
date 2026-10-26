"""Stage 2: reasoning over one representation of each item, per cell.

A *cell* is (model, representation, cue, run). Every cell writes

    <out_root>/<model>/<representation>/<cue>/<run>/
        results.jsonl      one row per item (resumable: existing ok rows are skipped)
        raw/<test_id>.json full raw output, thought text and usage per attempt
        manifest.json      snapshot provenance, design parameters, git commit, prompt hashes
        prompt_example.txt the complete prompt for the first item in the cell

Prompt construction reuses ``hmr.prompts`` so the new runs stay comparable
with the archived evaluation: only the evidence block differs between
representations, and the cue is injected through the same ``*_concept``
prompt variants.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from ..llm import parse_or_repair_json
from ..prompts import system_eval_prompt, user_eval_prompt, user_eval_prompt_dvrl
from ..schemas import ImageDescriptionSynthLogo
from .backends import BackendError, ChatResult, chat, model_provenance
from .scoring import extract_conclusion
from .captions import (CAPTION_PROMPT_VERSION, STYLES, caption_store_path, load_captions, render_caption,
                       render_free_caption, style_store)
from .design import DRAW_REPRESENTATIONS
from .design import (
    CUES,
    M_POS,
    N_NEG,
    REPRESENTATIONS,
    Cell,
    Item,
    RunSpec,
    enumerate_cells,
    load_wrong_cue_file,
    wrong_cue_for,
)

logger = logging.getLogger(__name__)

MAX_BUDGET = 32768

RESULT_FIELDS = (
    "test_id", "uid", "family", "polarity", "model", "representation", "cue", "run",
    "temperature", "seed", "captioner", "cue_text", "prediction", "correct", "status",
    "attempts", "analysis", "rule", "test_image", "usage", "duration_s", "response_model",
    "provider", "prompt_sha256", "caption_missing", "transient_retries", "timestamp",
)


# ---------------------------------------------------------------------------
# Design file
# ---------------------------------------------------------------------------

def load_design(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        design = yaml.safe_load(os.path.expandvars(f.read()))
    design["runs"] = [RunSpec(r["name"], float(r["temperature"]), int(r["seed"])) for r in design["runs"]]
    return design


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def cue_text_for(item: Item, cue: str, wrong_map: dict[str, str]) -> str | None:
    if cue == "none":
        return None
    if cue == "correct":
        return item.concept
    if cue == "wrong":
        return wrong_cue_for(item, wrong_map)
    if cue in ("wrong_clean", "irrelevant"):
        # Loaded from design["extra_cue_files"] under prefixed keys (see run_model).
        return wrong_map[f"{cue}::{item.family}::{item.concept}"]
    raise ValueError(cue)


def build_prompts(
    item: Item,
    cell: Cell,
    *,
    image_dir: str | Path,
    captions: dict[str, dict[str, Any]] | None,
    wrong_map: dict[str, str],
) -> tuple[str, str, list[str] | None, dict[str, Any]]:
    """Return (system_prompt, user_prompt, image_paths_or_None, meta)."""
    cue_text = cue_text_for(item, cell.cue, wrong_map)
    system = system_eval_prompt(cell.context, concept=cue_text or "")
    meta: dict[str, Any] = {"cue_text": cue_text, "caption_missing": 0}

    if cell.representation == "pixels":
        return system, user_eval_prompt_dvrl(M_POS, N_NEG), item.image_abs(image_dir), meta

    if cell.representation == "program":
        specs: list = [list(p) for p in item.programs]
        return system, user_eval_prompt(specs, M_POS, N_NEG), None, meta

    if cell.representation == "oracle_description":
        return system, user_eval_prompt(list(item.descriptions), M_POS, N_NEG), None, meta

    if cell.representation == "oracle_description_rendered":
        return system, user_eval_prompt(list(item.descriptions_rendered), M_POS, N_NEG), None, meta

    if cell.representation == "caption_self_free" or DRAW_REPRESENTATIONS.get(cell.representation, ("",))[0] == "free":
        if captions is None:
            raise RuntimeError("captions are required for caption representations")
        stories = []
        for idx, rel in enumerate(item.images, start=1):
            row = captions.get(rel)
            if row is None:
                meta["caption_missing"] += 1
                stories.append(f"Figure {idx}: (No description is available for this figure.)\n\n")
            else:
                stories.append(render_free_caption(row["description"], idx))
        return system, user_eval_prompt(stories, M_POS, N_NEG), None, meta

    if cell.representation in ("caption_self", "caption_strong") or DRAW_REPRESENTATIONS.get(cell.representation, ("",))[0] == "steps":
        if captions is None:
            raise RuntimeError("captions are required for caption representations")
        stories: list[str] = []
        for idx, rel in enumerate(item.images, start=1):
            row = captions.get(rel)
            if row is None:
                meta["caption_missing"] += 1
                stories.append(f"To draw figure {idx}, follow these steps:\n\n(No description is available for this figure.)\n\nThe figure is now complete.\n\n")
            else:
                stories.append(render_caption(row["description"], idx))
        return system, user_eval_prompt(stories, M_POS, N_NEG), None, meta

    raise ValueError(cell.representation)


def prompt_hash(system: str, user: str, images: list[str] | None) -> str:
    h = hashlib.sha256()
    h.update(system.encode())
    h.update(b"\x00")
    h.update(user.encode())
    for p in images or []:
        h.update(b"\x00")
        h.update(Path(p).as_posix().encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

def cell_dir(out_root: str | Path, cell: Cell) -> Path:
    return Path(out_root).joinpath(*cell.path_parts)


def load_done(results_file: Path) -> set[str]:
    done: set[str] = set()
    if not results_file.exists():
        return done
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("status") == "ok":
                    done.add(row["test_id"])
    return done


def run_cell(
    cell: Cell,
    items: list[Item],
    design: dict[str, Any],
    model_cfg: dict[str, Any],
    *,
    out_root: Path,
    image_dir: str | Path,
    captions: dict[str, dict[str, Any]] | None,
    captioner: str | None,
    wrong_map: dict[str, str],
    workers: int,
    dry_run: bool,
    repo_root: Path,
) -> dict[str, int]:
    d = cell_dir(out_root, cell)
    results_file = d / "results.jsonl"
    done = load_done(results_file)
    todo = [it for it in items if it.test_id not in done]
    stats = {"items": len(items), "already_done": len(done), "to_run": len(todo), "ok": 0, "failed": 0}
    tag = "/".join(cell.path_parts)
    logger.info("[%s] %d items, %d done, %d to run", tag, len(items), len(done), len(todo))
    if dry_run or not todo:
        return stats

    d.mkdir(parents=True, exist_ok=True)
    (d / "raw").mkdir(exist_ok=True)
    max_attempts = int(design.get("max_attempts", 3))
    max_tokens = int(model_cfg.get("max_tokens", design.get("max_tokens", 12288)))
    think = bool(model_cfg.get("think", False))
    supports_temp = bool(model_cfg.get("supports_temperature", True))
    schema = ImageDescriptionSynthLogo.model_json_schema()

    first_sys, first_user, first_imgs, _ = build_prompts(todo[0], cell, image_dir=image_dir, captions=captions, wrong_map=wrong_map)
    if not (d / "manifest.json").exists():
        manifest = {
            "cell": {"model": cell.model, "representation": cell.representation, "cue": cell.cue, "run": cell.run.name,
                     "temperature": cell.run.temperature if supports_temp else None, "seed": cell.run.seed,
                     "context": cell.context, "captioner": captioner},
            "model_cfg": {k: v for k, v in model_cfg.items()},
            "design": {k: v for k, v in design.items() if k not in ("models", "runs")},
            "runs": [r.__dict__ for r in design["runs"]],
            "provenance": model_provenance(cell.model),
            "git_commit": git_commit(repo_root),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "system_prompt_sha256": hashlib.sha256(first_sys.encode()).hexdigest(),
            "n_items_planned": len(items),
        }
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        with open(d / "prompt_example.txt", "w") as f:
            f.write(f"### item {todo[0].test_id}\n### SYSTEM\n{first_sys}\n\n### USER\n{first_user}\n")
            if first_imgs:
                f.write("\n### IMAGES\n" + "\n".join(first_imgs) + "\n")

    lock = threading.Lock()
    # Report often enough that a small cell does not look frozen, but not so
    # often that a 200-item cell floods the log.
    every = max(1, min(20, len(todo) // 8))

    def one(item: Item) -> dict[str, Any]:
        system, user, images, meta = build_prompts(item, cell, image_dir=image_dir, captions=captions, wrong_map=wrong_map)
        row: dict[str, Any] = {
            "test_id": item.test_id, "uid": item.uid, "family": item.family, "polarity": item.polarity,
            "model": cell.model, "representation": cell.representation, "cue": cell.cue, "run": cell.run.name,
            "temperature": cell.run.temperature if supports_temp else None, "seed": cell.run.seed,
            "captioner": captioner, "cue_text": meta["cue_text"], "caption_missing": meta["caption_missing"],
            "prompt_sha256": prompt_hash(system, user, images), "provider": None,
        }
        raw_log: list[dict[str, Any]] = []
        last_err: str | None = None
        budget = max_tokens
        use_think = think
        fills = 0  # consecutive attempts whose whole output budget went to a trace
        for attempt in range(1, max_attempts + 1):
            seed = cell.run.seed + 1000 * (attempt - 1)
            try:
                res: ChatResult = chat(
                    cell.model, system, user, images=images,
                    temperature=cell.run.temperature, seed=seed, max_tokens=budget, think=use_think,
                    json_schema=schema, num_ctx=model_cfg.get("num_ctx"), supports_temperature=supports_temp,
                    min_interval_s=float(model_cfg.get("min_interval_s", 0.0)),
                )
            except BackendError as exc:
                last_err = str(exc)
                raw_log.append({"attempt": attempt, "seed": seed, "error": last_err})
                logger.warning("[%s] %s attempt %d backend error: %s", tag, item.test_id, attempt, last_err[:200])
                continue
            raw_log.append({"attempt": attempt, "seed": seed, "text": res.text, "thought": res.thought,
                            "usage": res.usage, "duration_s": res.duration_s, "response_model": res.response_model,
                            "extra": res.extra, "transient_retries": res.transient_retries})
            row["provider"] = res.provider
            if not (res.text or "").strip():
                used = (res.usage or {}).get("completion_tokens") or 0
                full = used >= 0.95 * budget
                fills = fills + 1 if full else 0
                # Same policy as hmr.llm: a trace that fills the budget twice is
                # not converging. Try once without thinking, then record the item
                # as a non-terminating trace instead of doubling to the ceiling.
                if fills >= 2:
                    if use_think:
                        use_think = False
                        last_err = f"budget filled twice by a reasoning trace ({used}/{budget}); think disabled"
                        continue
                    last_err = f"non_terminating_trace: {used}/{budget} output tokens with no answer on {fills} consecutive attempts"
                    break
                budget = min(budget * 2, MAX_BUDGET)
                last_err = f"empty output (thought {len(res.thought or '')} chars); budget -> {budget}"
                continue
            parse_fallback = False
            try:
                parsed = json.loads(parse_or_repair_json(res.text, cell.model, allow_ai_repair=False))
            except (ValueError, json.JSONDecodeError) as exc:
                # Malformed envelope, well-formed final decision: recover it
                # deterministically (amendment A2), else retry as before.
                fallback = extract_conclusion(res.text)
                if fallback is not None and (res.extra or {}).get("done_reason") != "length":
                    parsed, parse_fallback = {"Conclusion": fallback}, True
                else:
                    parsed = None
            if parsed is None:
                last_err = f"parse: unparseable answer"
                # An answer cut off at the output budget never closes its JSON;
                # regenerating at the same budget truncates again, so grow it.
                if (res.extra or {}).get("done_reason") == "length":
                    budget = min(budget * 2, MAX_BUDGET)
                    last_err += f" (truncated at the output budget; budget -> {budget})"
                continue
            pred = parsed.get("Conclusion")
            row.update({
                "prediction": pred, "correct": int(pred == item.polarity), "status": "ok", "attempts": attempt, "max_tokens": budget,
                "parse_fallback": parse_fallback,
                "analysis": parsed.get("Analysis"), "rule": parsed.get("Rule"), "test_image": parsed.get("Test Image"),
                "usage": res.usage, "duration_s": round(res.duration_s, 3), "response_model": res.response_model,
                "transient_retries": res.transient_retries, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            break
        # Reached either by exhausting max_attempts or by the early break on a
        # non-terminating trace; a for/else would miss the break, so test the
        # row instead. `attempt` is the number actually made.
        if "status" not in row:
            row.update({"prediction": None, "correct": None, "status": "failed", "attempts": attempt,
                        "error": last_err, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(d / "raw" / f"{item.test_id}.json", "w") as f:
            json.dump({"row": {k: row.get(k) for k in ("test_id", "status", "prediction")}, "attempts": raw_log,
                       "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest()}, f, ensure_ascii=False)
        return row

    def record(row: dict[str, Any]) -> None:
        with lock:
            with open(results_file, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["ok" if row["status"] == "ok" else "failed"] += 1
            n = stats["ok"] + stats["failed"]
            if n % every == 0 or n == stats["to_run"]:
                logger.info("[%s] %d/%d done (failed %d)", tag, n, stats["to_run"], stats["failed"])

    if workers <= 1:
        for it in todo:
            record(one(it))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(one, it) for it in todo]):
                record(fut.result())
    return stats


# ---------------------------------------------------------------------------
# Model driver
# ---------------------------------------------------------------------------

def plan_cells(model: str, design: dict[str, Any], representations, cues, runs) -> tuple[list[Cell], list[str]]:
    model_cfg = design["models"][model]
    return enumerate_cells(model, model_cfg, representations, cues, runs, design["strong_captioner"])


def captions_for(cell: Cell, design: dict[str, Any], out_root: Path) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    if cell.representation == "caption_self_free" or cell.representation in DRAW_REPRESENTATIONS:
        style, draw = DRAW_REPRESENTATIONS.get(cell.representation, ("free", ""))
        store = style_store(cell.model, style, draw)
        return load_captions(caption_store_path(out_root, store), prompt_version=STYLES[style][0]), store
    if cell.representation == "caption_self":
        captioner = cell.model
    elif cell.representation == "caption_strong":
        captioner = design["strong_captioner"]
    else:
        return None, None
    # Only captions written by the current prompt version feed the reasoning
    # stage, so a mixed store never silently blends prompt generations.
    return load_captions(caption_store_path(out_root, captioner),
                         prompt_version=CAPTION_PROMPT_VERSION), captioner


def check_capabilities(model: str, model_cfg: dict[str, Any]) -> list[str]:
    """Compare declared vision support against what the backend advertises.

    Returns human-readable problems. An Ollama model that is not multimodal
    cannot serve the pixels or caption_self cells, and finding that out after a
    multi-hour job has started is expensive.
    """
    problems: list[str] = []
    prov = model_provenance(model)
    caps = (prov.get("ollama") or {}).get("capabilities")
    if caps is None:
        return problems
    declared = bool(model_cfg.get("vision", True))
    advertised = "vision" in caps
    if declared and not advertised:
        problems.append(
            f"{model}: design declares vision:true but the backend advertises {caps}. "
            f"Set vision:false, or drop pixels and caption_self for this model."
        )
    elif advertised and not declared:
        problems.append(f"{model}: backend advertises vision but the design declares vision:false.")
    return problems


def run_model(
    model: str,
    design: dict[str, Any],
    items: list[Item],
    *,
    out_root: Path,
    image_dir: str | Path,
    repo_root: Path,
    representations: list[str] | None = None,
    cues: list[str] | None = None,
    run_names: list[str] | None = None,
    workers: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if model not in design["models"]:
        raise KeyError(f"{model!r} is not in the design panel: {sorted(design['models'])}")
    model_cfg = design["models"][model]
    reps = representations or list(design.get("representations", REPRESENTATIONS))
    cue_list = cues or list(design.get("cues", CUES))
    runs = [r for r in design["runs"] if not run_names or r.name in run_names]
    # A model may restrict its own run set (cost, or provider rate limits).
    only = model_cfg.get("runs")
    if only:
        runs = [r for r in runs if r.name in only]
    cells, skipped = plan_cells(model, design, reps, cue_list, runs)
    for s in skipped:
        logger.info("[%s] skipped %s", model, s)
    for problem in check_capabilities(model, model_cfg):
        logger.warning("[preflight] %s", problem)
    wrong_map = load_wrong_cue_file(design["wrong_cue_file"])
    for cue_name, path in (design.get("extra_cue_files") or {}).items():
        for key, text in load_wrong_cue_file(path).items():
            wrong_map[f"{cue_name}::{key}"] = text
    missing = [c for c in cue_list if c in ("wrong_clean", "irrelevant") and c not in (design.get("extra_cue_files") or {})]
    if missing:
        raise KeyError(f"cues {missing} need extra_cue_files in the design")
    n_workers = workers if workers is not None else int(model_cfg.get("workers", 1))
    summary: list[dict[str, Any]] = []
    for cell in cells:
        captions, captioner = captions_for(cell, design, out_root)
        if captions is not None:
            n_img = len({rel for it in items for rel in it.images})
            have = sum(1 for it in items for rel in it.images if rel in captions)
            if have == 0:
                logger.error("[%s] no captions from %s at %s; run --stage captions for that model first. Skipping cell.",
                             "/".join(cell.path_parts), captioner, caption_store_path(out_root, captioner))
                summary.append({"cell": "/".join(cell.path_parts), "skipped": "captions missing"})
                continue
            logger.info("[%s] captions available for %d/%d figure slots (%d unique images)", "/".join(cell.path_parts), have, sum(len(it.images) for it in items), n_img)
        stats = run_cell(cell, items, design, model_cfg, out_root=out_root, image_dir=image_dir, captions=captions,
                         captioner=captioner, wrong_map=wrong_map, workers=n_workers, dry_run=dry_run, repo_root=repo_root)
        summary.append({"cell": "/".join(cell.path_parts), **stats})
    return summary
