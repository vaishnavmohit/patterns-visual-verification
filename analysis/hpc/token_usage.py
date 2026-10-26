#!/usr/bin/env python3
"""Recompute the token and caption counts quoted in the manuscript's cost and portability paragraphs.

Run ON THE HPC from the code repository root (reads output/factorial/):

    python3 token_usage.py

Prints, per factorial model, input and output token totals over reasoning and
captioning calls (output = answer + thinking tokens), the mean answer and
thinking tokens per Gemini reasoning call, Gemini thinking tokens by
representation, GPT mean tokens per decision by representation, the open-weight
caption counts, and median caption lengths for Gemma4:31b against GPT 5.5 on
the same images. Costs in EUR come from the provider invoices, not from here.
"""
import collections
import glob
import json
import statistics

F = "output/factorial"


def rows(pattern):
    for p in glob.glob(f"{F}/{pattern}", recursive=True):
        for line in open(p):
            if line.strip():
                yield json.loads(line)


def output_tokens(u):
    # Gemini reports thinking separately; OpenAI's completion count already includes reasoning.
    return u.get("completion_tokens", 0) + u.get("thoughts_tokens", 0)


for m in ("gemini-3.5-flash", "gpt-5.5"):
    res = [r for r in rows(f"{m}/**/*.jsonl") if r.get("usage")]
    cap = [r for r in rows(f"captions/{m}/captions.jsonl") if r.get("usage")]
    tin = sum(r["usage"]["prompt_tokens"] for r in res + cap)
    tout = sum(output_tokens(r["usage"]) for r in res + cap)
    ok = sum(1 for r in rows(f"{m}/**/*.jsonl") if r.get("status") == "ok")
    print(f"{m}: {ok} answered decisions, {len(cap)} caption calls with usage; "
          f"{tin / 1e6:.1f}M input, {tout / 1e6:.1f}M output tokens")
    if m.startswith("gemini"):
        n = sum(1 for _ in rows(f"{m}/**/*.jsonl"))  # every reasoning call, including any without usage
        print(f"  per reasoning call ({n}): {sum(r['usage']['completion_tokens'] for r in res) / n:.0f} answer, "
              f"{sum(r['usage']['thoughts_tokens'] for r in res) / n:.0f} thinking tokens")
        th = collections.Counter()
        for r in res:
            th[r["representation"]] += r["usage"]["thoughts_tokens"]
        print("  thinking tokens by representation (M):", {k: round(v / 1e6, 1) for k, v in th.items()})
    else:
        by = collections.defaultdict(list)
        for r in res:
            by[r["representation"]].append(r["usage"])
        print("  mean input/output per decision:",
              {k: (round(statistics.mean(u["prompt_tokens"] for u in v)),
                   round(statistics.mean(u["completion_tokens"] for u in v))) for k, v in by.items()})

caps = {m: list(rows(f"captions/{m}/captions.jsonl")) for m in ("qwen3-vl_30b", "gemma4_31b", "gpt-5.5")}
for m, rs in caps.items():
    print(f"{m}: {sum(r.get('status') == 'ok' for r in rs)} of 2504 images captioned")
gpt = {r["image"]: r["usage"]["completion_tokens"] for r in caps["gpt-5.5"] if r.get("status") == "ok" and r.get("usage")}
gem = {r["image"]: r["usage"]["completion_tokens"] for r in caps["gemma4_31b"] if r.get("status") == "ok" and r.get("usage")}
shared = [i for i in gem if i in gpt]
print(f"median caption output tokens on {len(shared)} shared images: "
      f"gemma4:31b {statistics.median(gem[i] for i in shared):.0f}, gpt-5.5 {statistics.median(gpt[i] for i in shared):.0f}")
