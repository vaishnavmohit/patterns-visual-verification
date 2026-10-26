#!/usr/bin/env python3
"""Export the open-weight replication (output/factorial_ow) to one trials CSV.

Run ON THE HPC from the code repository root, after
scripts/rescore_factorial.py output/factorial_ow:

    .venv/bin/python export_factorial_ow.py > factorial_ow_trials.csv

and copy the CSV into analysis/patterns_outputs/. One row per attempted item
and cell, with the decision under the original strict scoring
(results.jsonl) and under amendment A2 (results_rescored.jsonl; the
last well-formed "Conclusion" of a malformed answer). A decision is empty when
the item has none under that scoring. Plan: docs/factorial_openweight_plan.md.
"""
import csv
import glob
import json
import os
import sys


def rows(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[r["test_id"]] = r
    return out


def decision(r, strict=False):
    # Strict scoring excludes conclusions recovered from a malformed envelope,
    # whether recovered offline (rescore) or online (runner, after A2).
    if r and r.get("status") == "ok" and r.get("prediction") in ("pos", "neg"):
        if strict and r.get("parse_fallback"):
            return ""
        return r["prediction"]
    return ""


w = csv.writer(sys.stdout)
w.writerow(["model", "representation", "cue", "run", "test_id", "polarity", "prediction_strict",
            "prediction_rescored", "parse_fallback"])
for model_dir, model in (("gemma4_31b", "gemma4:31b"), ("qwen3.6_27b", "qwen3.6:27b")):
    for p in sorted(glob.glob(f"output/factorial_ow/{model_dir}/*/*/*/results.jsonl")):
        rep, cue, run = p.split("/")[-4:-1]
        strict = rows(p)
        resc = rows(os.path.join(os.path.dirname(p), "results_rescored.jsonl")) or strict
        for t, r in strict.items():
            rr = resc.get(t, r)
            w.writerow([model, rep, cue, run, t, r["polarity"], decision(r, strict=True), decision(rr),
                        int(bool(rr.get("parse_fallback")))])
