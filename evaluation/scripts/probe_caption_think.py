#!/usr/bin/env python3
"""Does an explicit think=False make open-weight self-captioning feasible?

hmr.factorial.backends._chat_ollama sends think=None whenever think is False
(``use_think = think if (think and ...) else None``), so Ollama applied each
model's default, which is thinking ON for gemma4 and qwen3.5/qwen3.6. The
open-weight caption campaigns therefore ran with thinking enabled although
their manifests record think: false.

This probe sends the captioner's exact request (system and user prompt v3,
the image shrunk as in the campaign, temperature 0, seed 42, the same output
budget) under two settings: think unset (what the campaigns did) and
think=False. Per call it records wall time, output tokens, reasoning-trace
length, caption length, done_reason and whether the caption parses. One
untimed warm-up call per model loads the weights first.

    python scripts/probe_caption_think.py --models gemma4:31b qwen3.5:27b \
        --n-unset 6 --n-false 20 --out output/probe_caption_think
"""
import argparse
import json
import time
from pathlib import Path

from hmr.factorial.backends import _ollama_for, encode_image_to_base64_shrink
from hmr.factorial.captions import (CAPTION_PROMPT_VERSION, CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT,
                                    _parse_caption, unique_images)
from hmr.factorial.design import load_items

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--n-unset", type=int, default=6)
ap.add_argument("--n-false", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--num-ctx", type=int, default=16384)
ap.add_argument("--data-dir", default="data")
ap.add_argument("--image-dir", default=str(Path.home() / "ShapeBongard_V2"))
ap.add_argument("--out", default="output/probe_caption_think")
args = ap.parse_args()

images = list(unique_images(load_items(args.data_dir)).values())   # inventory order, same for every model
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)


def call(client, model, rel, think):
    kwargs = {"model": model,
              "messages": [{"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                           {"role": "user", "content": CAPTION_USER_PROMPT,
                            "images": [encode_image_to_base64_shrink(str(Path(args.image_dir) / rel))]}],
              "options": {"temperature": 0.0, "seed": 42, "num_ctx": args.num_ctx, "num_predict": args.max_tokens},
              "keep_alive": "30m"}
    if think is not None:
        kwargs["think"] = think
    t0 = time.perf_counter()
    resp = client.chat(**kwargs)
    dur = time.perf_counter() - t0
    content = getattr(resp.message, "content", "") or ""
    thought = getattr(resp.message, "thinking", "") or ""
    desc = _parse_caption(content)
    return {"model": model, "image": rel, "think": "unset" if think is None else think, "seconds": round(dur, 2),
            "output_tokens": getattr(resp, "eval_count", None), "thought_chars": len(thought),
            "content_chars": len(content), "caption_chars": len(desc or ""), "parsed": bool(desc),
            "steps": (desc or "").count("Step "), "done_reason": getattr(resp, "done_reason", None),
            "prompt_version": CAPTION_PROMPT_VERSION}


for model in args.models:
    client = _ollama_for(model)
    call(client, model, images[0]["image"], False)   # warm-up, untimed
    with open(out / f"{model.replace(':', '_')}.jsonl", "a") as f:
        for think, n in ((False, args.n_false), (None, args.n_unset)):
            for meta in images[:n]:
                try:
                    row = call(client, model, meta["image"], think)
                except Exception as exc:  # noqa: BLE001
                    row = {"model": model, "image": meta["image"], "think": "unset" if think is None else think, "error": str(exc)}
                f.write(json.dumps(row) + "\n"); f.flush()
                print(json.dumps(row), flush=True)
