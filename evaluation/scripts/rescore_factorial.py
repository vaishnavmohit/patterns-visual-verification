#!/usr/bin/env python3
"""Rescore failed factorial items from their saved raw attempts (amendment A2).

For every cell under an output tree (<root>/<model>/<rep>/<cue>/<run>/), each
item recorded as failed is re-examined from raw/<test_id>.json. Attempts are
taken in the order they were made; the first that stopped normally (not at the
output budget) and contains a final "Conclusion": "pos|neg" supplies the
decision, which is what the runner now does online. No model is called.
Writes results_rescored.jsonl beside results.jsonl, which is left unchanged;
rescored rows carry parse_fallback=true and rescored_attempt. Idempotent.

    python scripts/rescore_factorial.py output/factorial_ow
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hmr.factorial.scoring import extract_conclusion  # noqa: E402

root = Path(sys.argv[1])
total = recovered = 0
for res in sorted(root.glob("*/*/*/*/results.jsonl")):
    rows = {}
    for line in open(res):
        if line.strip():
            r = json.loads(line)
            rows[r["test_id"]] = r
    n_fail = n_rec = 0
    for tid, r in rows.items():
        if r.get("status") == "ok":
            continue
        n_fail += 1
        raw = res.parent / "raw" / f"{tid}.json"
        if not raw.exists():
            continue
        for i, a in enumerate(json.load(open(raw))["attempts"], start=1):
            if (a.get("extra") or {}).get("done_reason") == "length":
                continue
            pred = extract_conclusion(a.get("text"))
            if pred:
                r.update({"prediction": pred, "correct": int(pred == r["polarity"]), "status": "ok",
                          "parse_fallback": True, "rescored_attempt": i, "usage": a.get("usage")})
                r.pop("error", None)
                n_rec += 1
                break
    with open(res.parent / "results_rescored.jsonl", "w") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total += n_fail; recovered += n_rec
    if n_fail:
        print(f"{res.parent.relative_to(root)}: failed {n_fail}, recovered {n_rec}, still failed {n_fail - n_rec}")
print(f"TOTAL failed {total}, recovered {recovered}")
