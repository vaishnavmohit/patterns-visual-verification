#!/usr/bin/env python3
"""Export the factorial results from the HPC run tree into the two CSVs the paper reads.

Run ON THE HPC, from the code repository root (~/code/human-machine-reasoning),
where output/factorial/<model>/<representation>/<cue>/<run>/results.jsonl lives:

    .venv/bin/python export_factorial.py trials > factorial_trials.csv
    .venv/bin/python export_factorial.py cells  > factorial_cells_all.csv

then copy both into analysis/patterns_outputs/ of the paper repository.

trials: one row per answered decision (status ok, prediction pos/neg), the last
        answered row per run x item, for both factorial models.
cells:  one row per model x representation x cue x run with at least 195
        answered items: n, accuracy, d' and c with the (x+0.5)/(n+1) correction.
"""
import csv, glob, json, math, sys


def answered(path):
    best = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("prediction") in ("pos", "neg"):
                best[r["test_id"]] = r
    return best


def erfinv(y):
    w = -math.log((1 - y) * (1 + y))
    if w < 5:
        w -= 2.5
        c = [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
             -0.00125372503, -0.00417768164, 0.246640727, 1.50140941]
    else:
        w = math.sqrt(w) - 3
        c = [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
             -0.0076224613, 0.00943887047, 1.00167406, 2.83297682]
    p = c[0]
    for k in c[1:]:
        p = p * w + k
    return p * y


def z(p):
    return math.sqrt(2) * erfinv(2 * p - 1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "trials"
    w = csv.writer(sys.stdout)
    w.writerow(["model", "representation", "cue", "run", "test_id", "polarity", "prediction", "correct"]
               if mode == "trials" else ["model", "representation", "cue", "run", "n", "accuracy", "dprime", "criterion"])
    for model in ("gemini-3.5-flash", "gpt-5.5"):
        for p in sorted(glob.glob(f"output/factorial/{model}/*/*/*/results.jsonl")):
            rep, cue, run = p.split("/")[-4:-1]
            best = answered(p)
            if mode == "trials":
                for t, r in best.items():
                    w.writerow([model, rep, cue, run, t, r["polarity"], r["prediction"], int(r["correct"])])
                continue
            rows = list(best.values())
            if len(rows) < 195:
                continue
            npos = sum(r["polarity"] == "pos" for r in rows); nneg = len(rows) - npos
            h = sum(r["polarity"] == "pos" and r["prediction"] == "pos" for r in rows)
            fa = sum(r["polarity"] == "neg" and r["prediction"] == "pos" for r in rows)
            zh, zf = z((h + .5) / (npos + 1)), z((fa + .5) / (nneg + 1))
            w.writerow([model, rep, cue, run, len(rows), round(sum(r["correct"] for r in rows) / len(rows), 4),
                        round(zh - zf, 4), round(-0.5 * (zh + zf), 4)])


if __name__ == "__main__":
    main()
