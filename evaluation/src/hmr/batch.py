"""Provider batch helpers for OpenAI and Gemini runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from google.genai import types as genai_types

from .io_utils import encode_image_to_base64_shrink, encode_image_to_base64
from .llm import _gemini, _openai, _openai_usage_metadata, parse_or_repair_json


def _openai_supports_temperature(model: str) -> bool:
    """Return False for o-series reasoning models that reject temperature != 1."""
    lower = model.lower()
    return not lower.startswith(("o1", "o3", "o4"))


BATCH_TERMINAL_OPENAI = {"completed", "failed", "expired", "cancelled"}
BATCH_TERMINAL_GEMINI = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def normalize_batch_mode(batch_mode: bool | str | None) -> str:
    if batch_mode is True:
        return "submit"
    if not batch_mode:
        return ""
    mode = str(batch_mode).strip().lower()
    if mode in {"false", "off", "none", "no", "0"}:
        return ""
    if mode in {"true", "on", "yes", "1"}:
        return "submit"
    aliases = {"check": "status", "retrieve": "download", "finalize": "download"}
    return aliases.get(mode, mode)


def batch_manifest_file(out_dir: Path, batch_manifest_path: str | None) -> Path:
    return Path(batch_manifest_path) if batch_manifest_path else out_dir / "batch_manifest.json"


def safe_custom_id(index: int, test_id: str) -> str:
    safe_test_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(test_id)).strip("_")
    return f"{index:06d}-{safe_test_id[:80]}"


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No batch manifest found at {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def write_batch_requests(
    request_path: Path,
    *,
    provider: str,
    items: list[dict[str, Any]],
    model: str,
    temp: float,
    max_tokens: int,
) -> None:
    request_path.parent.mkdir(parents=True, exist_ok=True)
    with open(request_path, "w") as f:
        for item in items:
            if provider == "openai":
                request = _openai_batch_request(item, model, temp, max_tokens)
            elif provider == "gemini":
                request = _gemini_batch_request(item, temp, max_tokens)
            else:
                raise ValueError(
                    "Batch mode is only supported for GPT/OpenAI and Gemini "
                    f"models, got {provider!r}."
                )
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


def submit_batch_job(
    *,
    provider: str,
    model: str,
    request_path: Path,
    completion_window: str,
    display_name: str,
) -> dict[str, Any]:
    if provider == "openai":
        client = _openai()
        with open(request_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={"description": display_name},
        )
        return {
            "batch_id": batch.id,
            "input_file_id": batch_input_file.id,
            "status": getattr(batch, "status", None),
            "raw": _json_safe(batch),
        }

    if provider == "gemini":
        client = _gemini()
        uploaded_file = client.files.upload(
            file=str(request_path),
            config=genai_types.UploadFileConfig(
                display_name=display_name,
                mime_type="jsonl",
            ),
        )
        batch = client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={"display_name": display_name},
        )
        state = _obj_get(_obj_get(batch, "state"), "name") or _obj_get(batch, "state")
        return {
            "batch_id": batch.name,
            "input_file_id": uploaded_file.name,
            "status": state,
            "raw": _json_safe(batch),
        }

    raise ValueError(f"Batch mode is only supported for GPT/OpenAI and Gemini models, got {provider!r}.")


def get_batch_status(provider: str, batch_id: str) -> dict[str, Any]:
    if provider == "openai":
        batch = _openai().batches.retrieve(batch_id)
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "request_counts": _json_safe(getattr(batch, "request_counts", None)),
            "raw": _json_safe(batch),
        }

    if provider == "gemini":
        batch = _gemini().batches.get(name=batch_id)
        state = _obj_get(_obj_get(batch, "state"), "name") or _obj_get(batch, "state")
        dest = _obj_get(batch, "dest")
        return {
            "batch_id": _obj_get(batch, "name") or batch_id,
            "status": state,
            "output_file_id": _obj_get(dest, "file_name", "fileName"),
            "error": _json_safe(_obj_get(batch, "error")),
            "raw": _json_safe(batch),
        }

    raise ValueError(f"Batch mode is only supported for GPT/OpenAI and Gemini models, got {provider!r}.")


def download_batch_output(provider: str, status: dict[str, Any], out_dir: Path) -> Path:
    output_file_id = status.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(f"Batch has no output file yet. Status: {status.get('status')}")

    output_path = out_dir / "batch_output.jsonl"
    if provider == "openai":
        response = _openai().files.content(output_file_id)
        if hasattr(response, "text"):
            content = response.text
        elif hasattr(response, "read"):
            payload = response.read()
            content = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        else:
            content = str(response)
        output_path.write_text(content)

        error_file_id = status.get("error_file_id")
        if error_file_id:
            error_response = _openai().files.content(error_file_id)
            error_content = error_response.text if hasattr(error_response, "text") else str(error_response)
            (out_dir / "batch_errors.jsonl").write_text(error_content)
        return output_path

    if provider == "gemini":
        payload = _gemini().files.download(file=output_file_id)
        output_path.write_text(payload.decode("utf-8") if isinstance(payload, bytes) else str(payload))
        return output_path

    raise ValueError(f"Batch mode is only supported for GPT/OpenAI and Gemini models, got {provider!r}.")


def _custom_id_to_test_id_suffix(custom_id: str) -> str:
    """Strip the ``NNNNNN-`` index prefix that ``safe_custom_id`` prepends."""
    return re.sub(r"^\d{6}-", "", custom_id)


def reconstruct_items_from_output(
    *,
    output_path: Path,
    dataset_module: Any,
    json_path: str,
    image_dir: str,
    m: int,
    n: int,
) -> list[dict[str, Any]]:
    """Rebuild manifest ``items`` from a batch output file with no manifest.

    Maps each result's ``custom_id`` back to a dataset ``test_id`` (the same id
    encoded by :func:`safe_custom_id` at submit time), then pulls per-sample
    metadata via ``read_sample`` -- exactly the source the non-batch Ollama path
    uses. Lets you ingest a provider output file downloaded directly from the
    web console when ``batch_manifest.json`` is unavailable (e.g. submitted on
    a different machine).
    """
    # 1. Collect the custom_ids present in the output (OpenAI and Gemini shapes).
    custom_ids: list[str] = []
    with open(output_path) as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            line = json.loads(raw_line)
            cid = line.get("custom_id") or line.get("key") or (line.get("metadata") or {}).get("key")
            if cid:
                custom_ids.append(cid)

    # 2. Map the sanitized id suffix back to the original dataset test_id.
    with open(json_path) as f:
        data = json.load(f)
    suffix_to_test_id: dict[str, str] = {}
    for sample in data:
        test_id = sample["test_id"]
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(test_id)).strip("_")[:80]
        suffix_to_test_id.setdefault(suffix, test_id)

    # 3. Pull metadata for each result via read_sample (no images needed).
    read_kwargs = {"image_dir": image_dir} if image_dir else {}
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for custom_id in custom_ids:
        if custom_id in seen:
            continue
        seen.add(custom_id)

        suffix = _custom_id_to_test_id_suffix(custom_id)
        test_id = suffix_to_test_id.get(suffix)
        if test_id is None:
            print(f"Could not map custom_id {custom_id!r} to a dataset test_id; skipping.")
            continue

        try:
            _, concept_ui, caption, uid, test_cat, commonsense = dataset_module.read_sample(
                json_path, test_id, m, n, **read_kwargs
            )
        except Exception as e:
            print(f"Failed to read sample {test_id} while reconstructing item: {e}; skipping.")
            continue

        items.append({
            "custom_id": custom_id,
            "test_id": test_id,
            "uid": uid,
            "caption": caption,
            "concept_ui": concept_ui,
            "test_cat_label": test_cat,
            "commonsense": commonsense,
            "m": m,
            "n": n,
        })

    return items


def append_batch_results(
    *,
    provider: str,
    model: str,
    manifest: dict[str, Any],
    out_dir: Path,
    results_file: Path,
    output_path: Path,
    df: pd.DataFrame,
    expected_keys: tuple[str, ...],
    excel_safe_text: Callable[[str | None], str | None],
    write_sample_artifacts: Callable[..., tuple[str, str]],
) -> int:
    item_by_id = {item["custom_id"]: item for item in manifest.get("items", [])}
    processed_ids = set(df["test_id"].values)
    added = 0

    with open(output_path) as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            line = json.loads(raw_line)

            try:
                custom_id, raw_output, usage_metadata = _batch_line_payload(provider, line)
            except Exception as e:
                print(f"Skipping failed batch line: {e}")
                continue

            item = item_by_id.get(custom_id)
            if item is None:
                print(f"Skipping unknown batch result custom_id={custom_id!r}")
                continue
            if item["test_id"] in processed_ids:
                continue

            try:
                repaired = parse_or_repair_json(raw_output, model, allow_ai_repair=False)
                result_dict = json.loads(repaired)
            except Exception as e:
                snippet = raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
                print(f"Bad JSON for batch result {custom_id}: {e}. Snippet: {snippet!r}. Skipping.")
                continue

            missing = [key for key in expected_keys if key not in result_dict]
            if missing:
                print(f"Missing keys {missing} in batch result {custom_id}. Skipping.")
                continue

            sample_dir = out_dir / "folders" / item["uid"]
            thinking_text_path, metadata_path = write_sample_artifacts(
                sample_dir,
                test_id=item["test_id"],
                uid=item["uid"],
                model=model,
                result_dict=result_dict,
                raw_output=repaired,
                thinking_text=None,
                tokens_used=_usage_total_from_dict(usage_metadata),
                duration_ns=None,
                usage_metadata=usage_metadata,
            )

            row = {
                "concept_ui": item["concept_ui"],
                "caption": item["caption"],
                "uid": item["uid"],
                "test_cat_label": item["test_cat_label"],
                "test_id": item["test_id"],
                "m": item["m"],
                "n": item["n"],
                "test_analysis": result_dict.get("Analysis"),
                "rule_identified": result_dict.get("Rule"),
                "test_image_summary": result_dict.get("Test Image"),
                "test_category_identified": result_dict.get("Conclusion"),
                "complete_output": excel_safe_text(repaired),
                "commonsense": item["commonsense"],
                "model": model,
                "tokens_used": _usage_total_from_dict(usage_metadata),
                "thought_tokens": thought_token_count(usage_metadata),
                "duration_sec": None,
                "thinking_text": None,
                "thinking_text_path": thinking_text_path,
                "metadata_path": metadata_path,
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            processed_ids.add(item["test_id"])
            added += 1

    df.to_excel(results_file, index=False)
    return added


def _openai_batch_request(item: dict[str, Any], model: str, temp: float, max_tokens: int) -> dict[str, Any]:
    content = [{"type": "text", "text": item["user_prompt"]}]
    if item.get("image_paths"):
        for path in item["image_paths"]:
            b64 = encode_image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"},
            })
    else:
        content = item["user_prompt"]

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
    }
    if _openai_supports_temperature(model):
        body["temperature"] = temp
    body[_openai_token_limit_param(model)] = max_tokens

    return {
        "custom_id": item["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _gemini_batch_request(item: dict[str, Any], temp: float, max_tokens: int) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "temperature": temp,
        "response_mime_type": "application/json",
    }
    if max_tokens:
        generation_config["max_output_tokens"] = max_tokens

    parts = [{"text": item["user_prompt"]}]
    if item.get("image_paths"):
        for path in item["image_paths"]:
            b64 = encode_image_to_base64(path)
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": b64
                }
            })

    return {
        "key": item["custom_id"],
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "system_instruction": {"parts": [{"text": item["system_prompt"]}]},
            "generation_config": generation_config,
        },
    }


def _batch_line_payload(provider: str, line: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if provider == "openai":
        custom_id = line.get("custom_id")
        if line.get("error"):
            raise RuntimeError(line["error"])
        response = line.get("response") or {}
        body = response.get("body") or {}
        status_code = response.get("status_code")
        if status_code and status_code >= 400:
            raise RuntimeError(body.get("error") or body)
        choices = body.get("choices") or []
        content = (((choices[0] if choices else {}).get("message") or {}).get("content") or "")
        usage = _openai_usage_metadata(body.get("usage") or {})
        return custom_id, content, usage

    if provider == "gemini":
        custom_id = line.get("key") or (line.get("metadata") or {}).get("key")
        if line.get("error"):
            raise RuntimeError(line["error"])
        response = line.get("response") or {}
        candidates = response.get("candidates") or []
        parts = (((candidates[0] if candidates else {}).get("content") or {}).get("parts") or [])
        content = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        return custom_id, content, _gemini_usage_from_response(response)

    raise ValueError(f"Batch mode is only supported for GPT/OpenAI and Gemini models, got {provider!r}.")


def _gemini_usage_from_response(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    prompt = usage.get("promptTokenCount") or usage.get("prompt_token_count")
    candidate = usage.get("candidatesTokenCount") or usage.get("candidates_token_count")
    total = usage.get("totalTokenCount") or usage.get("total_token_count")
    thoughts = usage.get("thoughtsTokenCount") or usage.get("thoughts_token_count")
    if thoughts is None and total is not None and prompt is not None and candidate is not None:
        thoughts = total - (prompt + candidate)
    return {
        "prompt_tokens": prompt,
        "candidate_tokens": candidate,
        "thoughts_tokens": thoughts,
        "total_tokens": total,
    }


def thought_token_count(usage: dict[str, Any] | None) -> int | None:
    """Reasoning/thinking token count across providers.

    Gemini reports it as ``thoughts_tokens``; OpenAI reasoning models report it
    as ``reasoning_tokens`` (under ``completion_tokens_details``). Returns the
    first one present, or None.
    """
    if not usage:
        return None
    for key in ("thoughts_tokens", "reasoning_tokens"):
        value = usage.get(key)
        if value is not None:
            return value
    return None


def _usage_total_from_dict(usage: dict[str, Any] | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens") or usage.get("totalTokenCount") or usage.get("total_token_count")
    if total is not None:
        return total
    parts = [
        usage.get("prompt_tokens") or usage.get("promptTokenCount") or usage.get("prompt_token_count"),
        usage.get("completion_tokens") or usage.get("candidatesTokenCount") or usage.get("candidates_token_count"),
    ]
    if any(part is not None for part in parts):
        return sum(part or 0 for part in parts)
    return None


def _openai_token_limit_param(model: str) -> str:
    lower = model.lower()
    if lower.startswith("gpt-5") or lower.startswith(("o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _obj_get(obj: Any, *names: str) -> Any:
    """Read a field from an SDK object or dict, accepting snake/camel names."""
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_json_dict"):
        return value.to_json_dict()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
