"""Bongard-LOGO classification pipeline.

For each puzzle in the input JSON, run the LLM, validate the response,
and append a row to ``<output_base_dir>/<model>/<run_name>/results.xlsx``.

The pipeline supports ablations (shuffling, mixing, noise injection) and
the optional "think" mode used by reasoning-trace models.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .batch import (
    BATCH_TERMINAL_GEMINI,
    BATCH_TERMINAL_OPENAI,
    append_batch_results,
    batch_manifest_file,
    download_batch_output,
    get_batch_status,
    load_manifest,
    normalize_batch_mode,
    safe_custom_id,
    save_manifest,
    submit_batch_job,
    thought_token_count,
    write_batch_requests,
)
from .datasets.action_descriptions import process_all_image_specs, process_all_image_specs_rendered
from .datasets.shape_metadata import process_bd_json
from .io_utils import set_seed
from .llm import _backend, llm_inference, llm_inference_multiimage
from .prompts import system_eval_prompt, user_eval_prompt, user_eval_prompt_dvrl

set_seed(42)

EXPECTED_KEYS = ("Analysis", "Rule", "Test Image", "Conclusion")
EXCEL_CELL_LIMIT = 32767
RESULT_COLUMNS = [
    "concept_ui",
    "caption",
    "uid",
    "test_cat_label",
    "test_id",
    "m",
    "n",
    "test_analysis",
    "rule_identified",
    "test_image_summary",
    "test_category_identified",
    "complete_output",
    "commonsense",
    "model",
    "tokens_used",
    "thought_tokens",
    "duration_sec",
    "thinking_text",
    "thinking_text_path",
    "metadata_path",
]

# ---------------------------------------------------------------------------
# Dataset loader (dynamic import keeps configs declarative)
# ---------------------------------------------------------------------------

def _load_dataset_module(dataset_name: str):
    """Import a dataset module by short name.

    First tries ``hmr.datasets.<name>`` (in-package), then falls back to
    ``dataset.<name>`` for back-compat with the old layout.
    """
    for prefix in ("hmr.datasets", "dataset"):
        try:
            return importlib.import_module(f"{prefix}.{dataset_name}")
        except ImportError:
            continue
    raise ValueError(f"Dataset module {dataset_name!r} not found.")


# ---------------------------------------------------------------------------
# Concept extraction (caption -> human-readable concept hint)
# ---------------------------------------------------------------------------

def get_concept_from_caption(caption: str) -> str:
    """Derive a concept string from a Bongard-LOGO caption."""
    if caption.lower().startswith("hd_"):
        return caption.replace("hd_", "").replace("_", " ").replace("-", ", ")
    if caption.lower().startswith("ff_"):
        return caption.replace("ff_", "").replace("_", " ").replace("-", ", ")
    if caption.lower().startswith("bd_"):
        return json.dumps(process_bd_json(caption), indent=2) if caption else ""
    return "unknown concept"


# ---------------------------------------------------------------------------
# Ablations: shuffling and noise
# ---------------------------------------------------------------------------

def _shuffle_spec(spec):
    """Shuffle the action steps within one image's spec.

    Handles list-of-lists (multi-shape) and flat lists (with optional 'and'
    separators, in which case each group is shuffled independently).
    """
    if not isinstance(spec, list) or not spec:
        return spec

    if isinstance(spec[0], list):
        return [
            random.sample(shape, len(shape)) if isinstance(shape, list) else shape
            for shape in spec
        ]

    # flat list -- handle 'and' separators
    if "and" not in spec:
        return random.sample(spec, len(spec))

    groups: list[list] = [[]]
    for action in spec:
        if action == "and":
            groups.append([])
        else:
            groups[-1].append(action)

    out: list = []
    for i, group in enumerate(groups):
        random.shuffle(group)
        out.extend(group)
        if i < len(groups) - 1:
            out.append("and")
    return out


def _add_noise(program: list[str], noise_level: float) -> list[str]:
    """Add Gaussian noise to numerical fields in each action string.

    ``-`` is treated as a separator (e.g. between LENGTH and TURNANGLE),
    not a negative sign, so only positive floats are matched.
    """
    if noise_level <= 0:
        return program

    def replace_num(match: re.Match) -> str:
        val = float(match.group()) + random.gauss(0, noise_level)
        return f"{val:.3f}"

    return [re.sub(r"\d+\.\d+", replace_num, action) for action in program]


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def _run_one(
    all_image_specs: list,
    *,
    model: str,
    temp: float,
    num_ctx: int | None,
    max_tokens: int,
    m: int,
    n: int,
    bongard_context: str,
    concept_ui: str | list = "",
    shuffle_test_image: bool = False,
    shuffle_pos_steps: bool = False,
    shuffle_neg_steps: bool = False,
    mix_pos_neg: bool = False,
    think: bool | str = False,
) -> tuple[str, int | None, int | None, str | None, dict[str, Any] | None] | None:
    """Run one LLM call.

    Returns (content, tokens, duration_ns, thinking_text, usage_metadata), or
    None on error.
    """

    system_prompt = system_eval_prompt(bongard_context, concept=concept_ui)

    # ------------------------------------------------------------------
    # DVRL branch: all_image_specs contains absolute image file paths.
    # Pass them all to the VLM in a single multi-image call.
    # ------------------------------------------------------------------
    if "dvrl" in bongard_context:
        user_prompt = user_eval_prompt_dvrl(m, n)
        # Errors propagate so the caller can record the reason per item.
        result = llm_inference_multiimage(
            model_name=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=all_image_specs,
            temp=temp,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            think=think,
        )
        time.sleep(10)
        if isinstance(result, tuple):
            return (
                result[0],
                result[1] if len(result) > 1 else None,
                result[2] if len(result) > 2 else None,
                result[3] if len(result) > 3 else None,
                result[4] if len(result) > 4 else None,
            )
        return (result, None, None, None, None)

    # ------------------------------------------------------------------
    # Text-based branch (action programs / action descriptions / aps)
    # ------------------------------------------------------------------
    if "action_description_rendered" in bongard_context:
        all_image_specs = process_all_image_specs_rendered(all_image_specs)
    elif "action_description" in bongard_context:
        all_image_specs = process_all_image_specs(all_image_specs)

    if shuffle_pos_steps:
        for i in range(m):
            all_image_specs[i] = _shuffle_spec(all_image_specs[i])
    if shuffle_neg_steps:
        for i in range(m, m + n):
            all_image_specs[i] = _shuffle_spec(all_image_specs[i])
    if shuffle_test_image:
        all_image_specs[-1] = _shuffle_spec(all_image_specs[-1])

    if mix_pos_neg:
        context = all_image_specs[:m + n]
        random.shuffle(context)
        all_image_specs = context + all_image_specs[m + n:]

    user_prompt = user_eval_prompt(all_image_specs, m, n)

    # Errors propagate so the caller can record the reason per item.
    result = llm_inference(
        model_name=model,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temp=temp,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        think=think,
    )

    time.sleep(10)

    if isinstance(result, tuple):
        return (
            result[0],
            result[1] if len(result) > 1 else None,
            result[2] if len(result) > 2 else None,
            result[3] if len(result) > 3 else None,
            result[4] if len(result) > 4 else None,
        )
    return (result, None, None, None, None)


FAILURES_FILE = "failures.jsonl"
NON_TERMINATING_PREFIX = "non-terminating reasoning trace"


def _record_failure(out_dir: Path, *, test_id: str, uid: str, model: str,
                    bongard_context: str, reason: str) -> None:
    """Append one line per failed item, so a cell can report what it could not
    answer instead of leaving 'never attempted' and 'attempted and failed'
    indistinguishable in results.xlsx."""
    with open(out_dir / FAILURES_FILE, "a") as f:
        f.write(json.dumps({
            "test_id": test_id, "uid": uid, "model": model, "bongard_context": bongard_context,
            "reason": reason, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False) + "\n")


def _known_non_terminating(out_dir: Path) -> set[str]:
    """test_ids this cell already recorded as non-terminating. At temperature
    zero with the same budget ladder the outcome repeats, so a retry only
    re-spends the minutes. Transient errors are NOT in this set and do retry."""
    ids: set[str] = set()
    p = out_dir / FAILURES_FILE
    if p.exists():
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("reason", "")).startswith(NON_TERMINATING_PREFIX):
                    ids.add(r["test_id"])
    return ids


def _duration_sec(duration_ns: int | None) -> float | None:
    if duration_ns is None:
        return None
    return duration_ns / 1e9


# openpyxl accepts \t, \n and \r and rejects every other control character.
_ILLEGAL_XLSX_RE = re.compile(r"[\000-\010\013\014\016-\037]")


def _excel_safe_text(text: str | None) -> str | None:
    """Make model-authored text writable by openpyxl.

    Two hazards, both silent data loss: a control character raises
    IllegalCharacterError, and a cell over the limit is rejected outright. The
    answer itself is fine in either case -- only the spreadsheet cannot hold the
    prose -- so strip rather than drop the row. gemma4:e2b emitted a form feed
    inside a LaTeX fragment in its Rule field and cost a whole cell.
    """
    if not isinstance(text, str):
        return text
    text = _ILLEGAL_XLSX_RE.sub("", text)
    if len(text) <= EXCEL_CELL_LIMIT:
        return text
    suffix = "\n\n[truncated in results.xlsx; see per-sample artifact file]"
    return text[: EXCEL_CELL_LIMIT - len(suffix)] + suffix


def _write_sample_artifacts(
    sample_dir: Path,
    *,
    test_id: str,
    uid: str,
    model: str,
    result_dict: dict[str, Any],
    raw_output: str,
    thinking_text: str | None,
    tokens_used: int | None,
    duration_ns: int | None,
    usage_metadata: dict[str, Any] | None,
) -> tuple[str, str]:
    """Persist per-sample response, thoughts, and run metadata."""

    sample_dir.mkdir(parents=True, exist_ok=True)

    result_path = sample_dir / f"{test_id}_result.json"
    with open(result_path, "w") as jf:
        json.dump(result_dict, jf, indent=4)

    thought_path = sample_dir / f"{test_id}_thinking.txt"
    with open(thought_path, "w") as tf:
        tf.write(thinking_text or "")

    metadata = {
        "test_id": test_id,
        "uid": uid,
        "model": model,
        "tokens_used": tokens_used,
        "duration_sec": _duration_sec(duration_ns),
        "duration_ns": duration_ns,
        "thinking_text_path": thought_path.name,
        "result_path": result_path.name,
        "raw_output": raw_output,
        "usage": usage_metadata or {},
    }

    metadata_path = sample_dir / f"{test_id}_metadata.json"
    with open(metadata_path, "w") as mf:
        json.dump(metadata, mf, indent=4)

    return str(thought_path), str(metadata_path)


def _prepare_text_batch_prompts(
    all_image_specs: list,
    *,
    m: int,
    n: int,
    bongard_context: str,
    concept_ui: str | list,
    shuffle_test_image: bool,
    shuffle_pos_steps: bool,
    shuffle_neg_steps: bool,
    mix_pos_neg: bool,
) -> tuple[str, str]:

    system_prompt = system_eval_prompt(bongard_context, concept=concept_ui)

    if "action_description_rendered" in bongard_context:
        all_image_specs = process_all_image_specs_rendered(all_image_specs)
    elif "action_description" in bongard_context:
        all_image_specs = process_all_image_specs(all_image_specs)

    if shuffle_pos_steps:
        for i in range(m):
            all_image_specs[i] = _shuffle_spec(all_image_specs[i])
    if shuffle_neg_steps:
        for i in range(m, m + n):
            all_image_specs[i] = _shuffle_spec(all_image_specs[i])
    if shuffle_test_image:
        all_image_specs[-1] = _shuffle_spec(all_image_specs[-1])

    if mix_pos_neg:
        context = all_image_specs[:m + n]
        random.shuffle(context)
        all_image_specs = context + all_image_specs[m + n:]

    return system_prompt, user_eval_prompt(all_image_specs, m, n)


def _collect_batch_items(
    *,
    dataset_name: str,
    json_path: str,
    image_dir: str,
    df: pd.DataFrame,
    num_samples: int,
    m: int,
    n: int,
    bongard_context: str,
    shuffle_test_image: bool,
    shuffle_pos_steps: bool,
    shuffle_neg_steps: bool,
    mix_pos_neg: bool,
    noise_level: float,
) -> list[dict[str, Any]]:
    dataset = _load_dataset_module(dataset_name)
    with open(json_path) as f:
        data = json.load(f)

    processed_ids = set(df["test_id"].values)
    items: list[dict[str, Any]] = []

    for sample_info in data:
        if len(df) + len(items) >= num_samples:
            break

        test_id = sample_info["test_id"]
        if test_id in processed_ids:
            continue

        read_kwargs = {"image_dir": image_dir} if image_dir else {}
        img_arr, concept_ui, caption, uid, test_cat, commonsense = dataset.read_sample(
            json_path, test_id, m, n, **read_kwargs
        )

        if noise_level > 0 and img_arr and isinstance(img_arr[0], list):
            img_arr = [_add_noise(p, noise_level) for p in img_arr]

        if "dvrl" in bongard_context:
            system_prompt = system_eval_prompt(bongard_context, concept=concept_ui)
            user_prompt = user_eval_prompt_dvrl(m, n)
            image_paths = img_arr
        else:
            system_prompt, user_prompt = _prepare_text_batch_prompts(
                img_arr,
                m=m,
                n=n,
                bongard_context=bongard_context,
                concept_ui=concept_ui,
                shuffle_test_image=shuffle_test_image,
                shuffle_pos_steps=shuffle_pos_steps,
                shuffle_neg_steps=shuffle_neg_steps,
                mix_pos_neg=mix_pos_neg,
            )
            image_paths = None

        items.append({
            "custom_id": safe_custom_id(len(items), test_id),
            "test_id": test_id,
            "uid": uid,
            "caption": caption,
            "concept_ui": concept_ui,
            "test_cat_label": test_cat,
            "commonsense": commonsense,
            "m": m,
            "n": n,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_paths": image_paths,
        })

    return items


def _process_batch_mode(
    *,
    mode: str,
    provider: str,
    model: str,
    dataset_name: str,
    json_path: str,
    output_base_dir: str,
    out_dir: Path,
    results_file: Path,
    df: pd.DataFrame,
    num_samples: int,
    m: int,
    n: int,
    temp: float,
    max_tokens: int,
    bongard_context: str,
    image_dir: str,
    shuffle_test_image: bool,
    shuffle_pos_steps: bool,
    shuffle_neg_steps: bool,
    mix_pos_neg: bool,
    noise_level: float,
    batch_id: str | None,
    batch_manifest_path: str | None,
    batch_completion_window: str,
) -> None:
    if provider not in {"openai", "gemini"}:
        raise ValueError(f"Batch mode currently supports GPT/OpenAI and Gemini models only, got {model!r}.")

    manifest_path = batch_manifest_file(out_dir, batch_manifest_path)

    if mode == "submit":
        items = _collect_batch_items(
            dataset_name=dataset_name,
            json_path=json_path,
            image_dir=image_dir,
            df=df,
            num_samples=num_samples,
            m=m,
            n=n,
            bongard_context=bongard_context,
            shuffle_test_image=shuffle_test_image,
            shuffle_pos_steps=shuffle_pos_steps,
            shuffle_neg_steps=shuffle_neg_steps,
            mix_pos_neg=mix_pos_neg,
            noise_level=noise_level,
        )
        if not items:
            print(f"No new samples to submit; {results_file} already has {len(df)} rows.")
            return

        request_path = out_dir / "batch_requests.jsonl"
        write_batch_requests(
            request_path,
            provider=provider,
            items=items,
            model=model,
            temp=temp,
            max_tokens=max_tokens,
        )
        display_name = f"hmr-{Path(output_base_dir).name}-{model}-{int(time.time())}"[:120]
        submission = submit_batch_job(
            provider=provider,
            model=model,
            request_path=request_path,
            completion_window=batch_completion_window,
            display_name=display_name,
        )
        manifest = {
            "provider": provider,
            "model": model,
            "batch_id": submission["batch_id"],
            "input_file_id": submission.get("input_file_id"),
            "status": submission.get("status"),
            "request_file": str(request_path),
            "results_file": str(results_file),
            "created_at": time.time(),
            "num_requests": len(items),
            "items": items,
            "submission": submission.get("raw"),
        }
        save_manifest(manifest_path, manifest)
        print(f"Submitted {len(items)} requests to {provider} batch.")
        print(f"Batch id: {manifest['batch_id']}")
        print(f"Manifest: {manifest_path}")
        return

    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError:
        # In a multi-config sweep a missing manifest just means this config was
        # never submitted (or was submitted under different settings). Skip it
        # with a warning instead of crashing the whole run.
        print(f"No batch manifest at {manifest_path}; skipping (nothing submitted for this run).")
        return
    manifest_provider = manifest.get("provider", provider)
    active_batch_id = batch_id or manifest.get("batch_id")
    if not active_batch_id:
        raise ValueError("No batch_id provided and manifest has no batch_id.")

    status = get_batch_status(manifest_provider, active_batch_id)
    manifest["status"] = status.get("status")
    manifest["last_status"] = status
    if status.get("output_file_id"):
        manifest["output_file_id"] = status["output_file_id"]
    save_manifest(manifest_path, manifest)
    print(f"Batch {active_batch_id} status: {status.get('status')}")

    if mode == "status":
        counts = status.get("request_counts")
        if counts:
            print(f"Request counts: {counts}")
        return

    if mode != "download":
        raise ValueError("batch_mode must be one of: submit, status, download.")

    terminal = BATCH_TERMINAL_OPENAI if manifest_provider == "openai" else BATCH_TERMINAL_GEMINI
    success = "completed" if manifest_provider == "openai" else "JOB_STATE_SUCCEEDED"
    if status.get("status") not in terminal:
        print("Batch is not terminal yet; run again with batch_mode: download after it completes.")
        return
    if status.get("status") != success:
        # A cancelled/expired/failed batch can still have an output file with the
        # requests that finished before it stopped. Salvage those (ingestion is
        # resumable, so the remaining samples can be resubmitted later); only
        # give up if there is no output at all.
        if not status.get("output_file_id"):
            raise RuntimeError(f"Batch finished unsuccessfully with no output: {status}")
        counts = status.get("request_counts")
        print(f"Batch status is {status.get('status')!r}; salvaging partial output. "
              f"Request counts: {counts}")

    output_path = download_batch_output(manifest_provider, status, out_dir)
    manifest["output_file"] = str(output_path)
    save_manifest(manifest_path, manifest)
    added = append_batch_results(
        provider=manifest_provider,
        model=manifest.get("model", model),
        manifest=manifest,
        out_dir=out_dir,
        results_file=results_file,
        output_path=output_path,
        df=df,
        expected_keys=EXPECTED_KEYS,
        excel_safe_text=_excel_safe_text,
        write_sample_artifacts=_write_sample_artifacts,
    )
    print(f"Downloaded batch output and added {added} rows to {results_file}.")


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def _build_output_dir(base: str, model: str, bongard_context: str, temp: float,
                     noise_level: float, shuffle_test_image: bool,
                     shuffle_pos_steps: bool, shuffle_neg_steps: bool,
                     mix_pos_neg: bool) -> Path:
    name = f"{bongard_context}_temp_{temp}"
    if noise_level > 0:
        name += f"_noise_{noise_level}"
    if shuffle_test_image:
        name += "_shuffledTest"
    if shuffle_pos_steps:
        name += "_shuffledPos"
    if shuffle_neg_steps:
        name += "_shuffledNeg"
    if mix_pos_neg:
        name += "_mixedPosNegV2"
    return Path(base) / model / name


def process_samples(
    dataset_name: str,
    json_path: str,
    output_base_dir: str,
    num_samples: int,
    m: int,
    n: int,
    model: str,
    temp: float,
    max_tokens: int,
    bongard_context: str,
    num_ctx: int | None = None,
    image_dir: str = "",
    shuffle_test_image: bool = False,
    shuffle_pos_steps: bool = False,
    shuffle_neg_steps: bool = False,
    mix_pos_neg: bool = False,
    noise_level: float = 0.0,
    think: bool | str = False,
    batch_mode: bool | str | None = False,
    batch_id: str | None = None,
    batch_manifest_path: str | None = None,
    batch_completion_window: str = "24h",
    skip_non_terminating: bool = True,
) -> None:
    """Iterate over puzzles, run the LLM, and accumulate results to xlsx."""

    out_dir = _build_output_dir(
        output_base_dir, model, bongard_context, temp, noise_level,
        shuffle_test_image, shuffle_pos_steps, shuffle_neg_steps, mix_pos_neg,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"

    if results_file.exists():
        df = pd.read_excel(results_file)
        for column in RESULT_COLUMNS:
            if column not in df.columns:
                df[column] = None
    else:
        df = pd.DataFrame(columns=RESULT_COLUMNS)

    mode = normalize_batch_mode(batch_mode)
    if mode:
        _process_batch_mode(
            mode=mode,
            provider=_backend(model),
            model=model,
            dataset_name=dataset_name,
            json_path=json_path,
            output_base_dir=output_base_dir,
            out_dir=out_dir,
            results_file=results_file,
            df=df,
            num_samples=num_samples,
            m=m,
            n=n,
            temp=temp,
            max_tokens=max_tokens,
            bongard_context=bongard_context,
            image_dir=image_dir,
            shuffle_test_image=shuffle_test_image,
            shuffle_pos_steps=shuffle_pos_steps,
            shuffle_neg_steps=shuffle_neg_steps,
            mix_pos_neg=mix_pos_neg,
            noise_level=noise_level,
            batch_id=batch_id,
            batch_manifest_path=batch_manifest_path,
            batch_completion_window=batch_completion_window,
        )
        return

    if len(df) >= num_samples:
        print(f"Already have {len(df)} samples in {results_file}; target is {num_samples}. Nothing to do.")
        return

    print(f"Found {len(df)} existing results. Target {num_samples}. Need {num_samples - len(df)} more.")

    dataset = _load_dataset_module(dataset_name)

    with open(json_path) as f:
        data = json.load(f)

    processed_ids = set(df["test_id"].values)
    skip_ids = _known_non_terminating(out_dir) if skip_non_terminating else set()
    if skip_ids:
        print(f"Skipping {len(skip_ids)} item(s) this cell already recorded as non-terminating "
              f"(see {out_dir / FAILURES_FILE}).")

    with tqdm(total=num_samples, initial=len(df), desc="Processing samples") as pbar:
        for sample_info in data:
            if len(df) >= num_samples:
                print(f"\nReached target of {num_samples} samples. Stopping.")
                break

            test_id = sample_info["test_id"]
            if test_id in processed_ids or test_id in skip_ids:
                continue

            try:
                read_kwargs = {"image_dir": image_dir} if image_dir else {}
                img_arr, concept_ui, caption, uid, test_cat, commonsense = dataset.read_sample(
                    json_path, test_id, m, n, **read_kwargs
                )
            except Exception as e:
                print(f"Error reading sample {test_id}: {e}. Skipping.")
                continue

            if noise_level > 0 and img_arr and isinstance(img_arr[0], list):
                img_arr = [_add_noise(p, noise_level) for p in img_arr]

            try:
                pbar.set_postfix_str(f"UID: {uid} | running…")
                _iter_t0 = time.perf_counter()
                outcome = _run_one(
                    img_arr,
                    model=model, temp=temp, num_ctx=num_ctx, max_tokens=max_tokens,
                    m=m, n=n, bongard_context=bongard_context,
                    concept_ui=concept_ui,
                    shuffle_test_image=shuffle_test_image,
                    shuffle_pos_steps=shuffle_pos_steps,
                    shuffle_neg_steps=shuffle_neg_steps,
                    mix_pos_neg=mix_pos_neg,
                    think=think,
                )
                iter_sec = time.perf_counter() - _iter_t0
                if outcome is None:
                    pbar.set_postfix_str(f"UID: {uid} | {iter_sec:.0f}s | FAILED")
                    print(f"Error: no response for test_id {test_id} (UID {uid}). Skipping.")
                    _record_failure(out_dir, test_id=test_id, uid=uid, model=model,
                                    bongard_context=bongard_context, reason="no response")
                    continue
                result, total_tokens, duration, thinking_text, usage_metadata = outcome
                duration_sec = _duration_sec(duration)

                try:
                    result_dict = json.loads(result)
                except json.JSONDecodeError as je:
                    snippet = result[:200] + "..." if len(result) > 200 else result
                    print(f"Bad JSON for test_id {test_id} (UID {uid}): {je}. Snippet: {snippet!r}. Skipping.")
                    _record_failure(out_dir, test_id=test_id, uid=uid, model=model,
                                    bongard_context=bongard_context, reason=f"bad JSON: {je}")
                    continue

                missing = [k for k in EXPECTED_KEYS if k not in result_dict]
                if missing:
                    print(f"\nMissing keys {missing} in JSON for test_id {test_id} (UID {uid}). "
                          f"Found: {list(result_dict.keys())}. Skipping.")
                    _record_failure(out_dir, test_id=test_id, uid=uid, model=model,
                                    bongard_context=bongard_context, reason=f"missing keys: {missing}")
                    continue

                sample_dir = out_dir / "folders" / uid
                thinking_text_path, metadata_path = _write_sample_artifacts(
                    sample_dir,
                    test_id=test_id,
                    uid=uid,
                    model=model,
                    result_dict=result_dict,
                    raw_output=result,
                    thinking_text=thinking_text,
                    tokens_used=total_tokens,
                    duration_ns=duration,
                    usage_metadata=usage_metadata,
                )

                row = {
                    "concept_ui": _excel_safe_text(concept_ui),
                    "caption": _excel_safe_text(caption),
                    "uid": uid,
                    "test_cat_label": test_cat,
                    "test_id": test_id,
                    "m": m,
                    "n": n,
                    "test_analysis": _excel_safe_text(result_dict.get("Analysis")),
                    "rule_identified": _excel_safe_text(result_dict.get("Rule")),
                    "test_image_summary": _excel_safe_text(result_dict.get("Test Image")),
                    "test_category_identified": _excel_safe_text(result_dict.get("Conclusion")),
                    "complete_output": _excel_safe_text(result),
                    "commonsense": _excel_safe_text(commonsense),
                    "model": model,
                    "tokens_used": total_tokens,
                    "thought_tokens": thought_token_count(usage_metadata),
                    "duration_sec": duration_sec,
                    "thinking_text": _excel_safe_text(thinking_text),
                    "thinking_text_path": thinking_text_path,
                    "metadata_path": metadata_path,
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                processed_ids.add(test_id)
                try:
                    df.to_excel(results_file, index=False)
                except Exception:
                    # One unwritable row must not poison the rest of the cell.
                    # df and processed_ids are already updated, so every later
                    # item would fail on this row's write and the cell would end
                    # with no rows at all -- which is what happened to
                    # gemma4:e2b on hd_comb and hd_novel.
                    df = df.iloc[:-1].reset_index(drop=True)
                    processed_ids.discard(test_id)
                    raise
                pbar.set_postfix_str(
                    f"UID: {uid} | {iter_sec:.0f}s | -> {result_dict.get('Conclusion')}"
                )
                pbar.update(1)

            except Exception as e:
                # Inference errors (including NonTerminatingTrace) arrive here now
                # that _run_one no longer swallows them.
                pbar.set_postfix_str(f"UID: {uid} | FAILED")
                print(f"Error during LLM inference for test_id {test_id} (UID {uid}): {e}. Skipping.")
                _record_failure(out_dir, test_id=test_id, uid=uid, model=model,
                                bongard_context=bongard_context, reason=str(e))
                continue


def test_visual_concepts(
    dataset: str,
    json_path: str,
    output_base_dir: str,
    num_samples: int,
    m: int,
    n: int,
    model: str,
    temperature: float,
    max_tokens: int,
    bongard_context: str,
    num_ctx: int | None,
    image_dir: str = "",
    shuffle_test_image: bool = False,
    shuffle_pos_steps: bool = False,
    shuffle_neg_steps: bool = False,
    mix_pos_neg: bool = False,
    noise_level: float = 0.0,
    think: bool | str = False,
    batch_mode: bool | str | None = False,
    batch_id: str | None = None,
    batch_manifest_path: str | None = None,
    batch_completion_window: str = "24h",
    skip_non_terminating: bool = True,
) -> str:
    """Public entry point invoked by main.py via the YAML config."""
    process_samples(
        dataset_name=dataset,
        json_path=json_path,
        output_base_dir=output_base_dir,
        num_samples=num_samples,
        m=m,
        n=n,
        model=model,
        temp=temperature,
        max_tokens=max_tokens,
        bongard_context=bongard_context,
        num_ctx=num_ctx,
        image_dir=image_dir,
        shuffle_test_image=shuffle_test_image,
        shuffle_pos_steps=shuffle_pos_steps,
        shuffle_neg_steps=shuffle_neg_steps,
        mix_pos_neg=mix_pos_neg,
        noise_level=noise_level,
        think=think,
        batch_mode=batch_mode,
        batch_id=batch_id,
        batch_manifest_path=batch_manifest_path,
        batch_completion_window=batch_completion_window,
        skip_non_terminating=skip_non_terminating,
    )
    return "Test completed successfully."
