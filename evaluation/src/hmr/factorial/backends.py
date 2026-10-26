"""Thin provider call layer for the factorial experiment.

Differences from :mod:`hmr.llm`, which drives the archived runs:

* no hidden ``time.sleep`` per call (pacing is explicit and per model);
* every call returns provider provenance (response model id / fingerprint /
  Ollama digest) so snapshots are recorded per row;
* transient errors (rate limits, 5xx, connection resets) are retried with
  exponential backoff; content problems are NOT handled here -- the caller
  decides whether to regenerate.

The three backends are selected exactly as in ``hmr.llm._backend``.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import ollama
import openai
import PIL.Image
from google import genai
from google.genai import types as gtypes

from ..io_utils import encode_image_to_base64_shrink
from ..llm import _backend, _gemini, _gemini_text_and_thought, _ollama_for, _openai

logger = logging.getLogger(__name__)

TRANSIENT_ATTEMPTS = 6
BACKOFF_BASE_S = 4.0
BACKOFF_MAX_S = 120.0
OLLAMA_TIMEOUT_S = 1800


class BackendError(RuntimeError):
    """Raised after transient retries are exhausted or on a permanent error."""


@dataclass
class ChatResult:
    text: str
    thought: str
    usage: dict[str, Any]
    duration_s: float
    provider: str
    response_model: str | None
    extra: dict[str, Any] = field(default_factory=dict)
    transient_retries: int = 0


def provider_for(model: str) -> str:
    return _backend(model)


# ---------------------------------------------------------------------------
# Pacing (per model, thread-safe)
# ---------------------------------------------------------------------------

_pace_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _pace(model: str, min_interval_s: float) -> None:
    if min_interval_s <= 0:
        return
    with _pace_lock:
        now = time.monotonic()
        wait = _last_call.get(model, 0.0) + min_interval_s - now
        if wait > 0:
            time.sleep(wait)
        _last_call[model] = time.monotonic()


# ---------------------------------------------------------------------------
# Transient-error classification
# ---------------------------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500 or exc.status_code in (408, 409, 429)
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and (code == 429 or code >= 500):
        return True
    if isinstance(exc, ollama.ResponseError):
        return exc.status_code >= 500 if isinstance(exc.status_code, int) else False
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if any(k in name for k in ("timeout", "connect", "remoteprotocol", "readerror")):
        return True
    if any(k in text for k in ("rate limit", "resource exhausted", "overloaded", "temporarily", "503", "502", "connection reset")):
        return True
    return False


def _with_retries(fn, model: str):
    retries = 0
    for attempt in range(TRANSIENT_ATTEMPTS):
        try:
            result = fn()
            result.transient_retries = retries
            return result
        except Exception as exc:  # noqa: BLE001 - classified below
            if not _is_transient(exc) or attempt == TRANSIENT_ATTEMPTS - 1:
                raise BackendError(f"{model}: {type(exc).__name__}: {exc}") from exc
            delay = min(BACKOFF_BASE_S * (2 ** attempt), BACKOFF_MAX_S)
            retries += 1
            logger.warning("[%s] transient error (%s); retry %d in %.0fs", model, type(exc).__name__, retries, delay)
            time.sleep(delay)
    raise BackendError(f"{model}: unreachable")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def model_provenance(model: str) -> dict[str, Any]:
    """Static description of the model snapshot, captured once per cell."""
    provider = provider_for(model)
    info: dict[str, Any] = {"model": model, "provider": provider}
    if provider == "ollama":
        try:
            client = _ollama_for(model)
            show = client.show(model)
            details = getattr(show, "details", None)
            info["ollama"] = {
                "modified_at": str(getattr(show, "modified_at", "")),
                "family": getattr(details, "family", None) if details else None,
                "parameter_size": getattr(details, "parameter_size", None) if details else None,
                "quantization_level": getattr(details, "quantization_level", None) if details else None,
                "capabilities": list(getattr(show, "capabilities", []) or []),
            }
            for m in getattr(client.list(), "models", []) or []:
                if getattr(m, "model", None) == model:
                    info["ollama"]["digest"] = getattr(m, "digest", None)
                    info["ollama"]["size_bytes"] = getattr(m, "size", None)
            info["ollama"]["host"] = os.getenv("OLLAMA_HOST", "localhost:11434")
        except Exception as exc:  # noqa: BLE001
            info["ollama"] = {"error": f"{type(exc).__name__}: {exc}"}
    return info


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

def chat(
    model: str,
    system: str,
    user: str,
    *,
    images: list[str] | None = None,
    temperature: float | None,
    seed: int | None,
    max_tokens: int | None,
    think: bool,
    json_schema: dict | None = None,
    num_ctx: int | None = None,
    supports_temperature: bool = True,
    min_interval_s: float = 0.0,
) -> ChatResult:
    """One chat completion. ``images`` are absolute PNG paths (multi-image)."""
    provider = provider_for(model)
    _pace(model, min_interval_s)
    if provider == "openai":
        return _with_retries(lambda: _chat_openai(model, system, user, images, temperature, seed, max_tokens, supports_temperature), model)
    if provider == "gemini":
        return _with_retries(lambda: _chat_gemini(model, system, user, images, temperature, seed, max_tokens, think), model)
    if provider == "ollama":
        return _with_retries(lambda: _chat_ollama(model, system, user, images, temperature, seed, max_tokens, think, json_schema, num_ctx), model)
    raise BackendError(f"{model}: provider {provider!r} is not supported by the factorial runner")


def _b64_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _chat_openai(model, system, user, images, temperature, seed, max_tokens, supports_temperature) -> ChatResult:
    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    for p in images or []:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64_png(p)}", "detail": "auto"}})
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }
    if supports_temperature and temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    if max_tokens:
        lower = model.lower()
        key = "max_completion_tokens" if lower.startswith(("gpt-5", "o1", "o3", "o4")) else "max_tokens"
        kwargs[key] = max_tokens
    t0 = time.perf_counter()
    resp = _openai().chat.completions.create(**kwargs)
    dur = time.perf_counter() - t0
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    return ChatResult(
        text=resp.choices[0].message.content or "",
        thought="",
        usage={
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
        },
        duration_s=dur,
        provider="openai",
        response_model=getattr(resp, "model", None),
        extra={"system_fingerprint": getattr(resp, "system_fingerprint", None), "finish_reason": resp.choices[0].finish_reason},
    )


def _chat_gemini(model, system, user, images, temperature, seed, max_tokens, think) -> ChatResult:
    pil = [PIL.Image.open(p) for p in images or []]
    cfg = gtypes.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        seed=seed,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        thinking_config=gtypes.ThinkingConfig(include_thoughts=True) if think else None,
    )
    t0 = time.perf_counter()
    resp = _gemini().models.generate_content(model=model, config=cfg, contents=[user, *pil])
    dur = time.perf_counter() - t0
    for im in pil:
        im.close()
    text, thought = _gemini_text_and_thought(resp) if think else (resp.text or "", "")
    um = resp.usage_metadata
    usage = {
        "prompt_tokens": getattr(um, "prompt_token_count", None),
        "completion_tokens": getattr(um, "candidates_token_count", None),
        "thoughts_tokens": getattr(um, "thoughts_token_count", None),
        "total_tokens": getattr(um, "total_token_count", None),
    }
    finish = None
    if resp.candidates:
        fr = getattr(resp.candidates[0], "finish_reason", None)
        finish = getattr(fr, "name", None) or (str(fr) if fr is not None else None)
    return ChatResult(text=text or "", thought=thought or "", usage=usage, duration_s=dur, provider="gemini",
                      response_model=getattr(resp, "model_version", None), extra={"finish_reason": finish})


_think_unsupported: set[str] = set()


def _chat_ollama(model, system, user, images, temperature, seed, max_tokens, think, json_schema, num_ctx) -> ChatResult:
    client = _ollama_for(model)
    if getattr(client, "_client", None) is not None:
        try:
            client._client.timeout = OLLAMA_TIMEOUT_S  # httpx client
        except Exception:  # noqa: BLE001
            pass
    user_msg: dict[str, Any] = {"role": "user", "content": user}
    if images:
        user_msg["images"] = [encode_image_to_base64_shrink(p) for p in images]
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    if num_ctx:
        options["num_ctx"] = num_ctx
    if max_tokens:
        options["num_predict"] = max_tokens
    # Send think explicitly, False included. Leaving it unset (None) lets Ollama apply
    # the model default, which is thinking ON for gemma4 and qwen3.5/3.6: the
    # open-weight caption campaigns ran that way while recording think: false.
    use_think: bool | None = None if model in _think_unsupported else bool(think)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, user_msg],
        "options": options,
        "think": use_think,
        "keep_alive": "30m",
    }
    # Mirror the archived runs: schema-constrained decoding for text prompts only.
    if json_schema is not None and not images:
        kwargs["format"] = json_schema
    t0 = time.perf_counter()
    try:
        resp = client.chat(**kwargs)
    except ollama.ResponseError as exc:
        if use_think is not None and "think" in str(exc).lower():
            logger.warning("[%s] thinking not supported; retrying without think", model)
            _think_unsupported.add(model)
            kwargs["think"] = None
            resp = client.chat(**kwargs)
        else:
            raise
    dur = time.perf_counter() - t0
    msg = resp.message
    content = getattr(msg, "content", "") or ""
    thought = getattr(msg, "thinking", "") or ""
    if "<think>" in content and not thought:
        from ..llm import _parse_raw_tags
        content, thought = _parse_raw_tags(content)
    usage = {
        "prompt_tokens": getattr(resp, "prompt_eval_count", None),
        "completion_tokens": getattr(resp, "eval_count", None),
        "total_duration_ns": getattr(resp, "total_duration", None),
        "load_duration_ns": getattr(resp, "load_duration", None),
        "prompt_eval_duration_ns": getattr(resp, "prompt_eval_duration", None),
        "eval_duration_ns": getattr(resp, "eval_duration", None),
    }
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    usage["total_tokens"] = (pt or 0) + (ct or 0) if (pt is not None or ct is not None) else None
    return ChatResult(text=content, thought=thought, usage=usage, duration_s=dur, provider="ollama",
                      response_model=getattr(resp, "model", model),
                      extra={"done_reason": getattr(resp, "done_reason", None), "think_effective": bool(kwargs["think"])})
