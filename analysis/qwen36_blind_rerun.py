#!/usr/bin/env python3
"""Qwen3.6:35b symbolic rows whose archived output said no input was provided.

81 of the 86 rows flagged by the text-based failure detector are Qwen3.6:35b
Action Description / Action Program outputs from the 7-14 May archive run in
which the model replied that the descriptions were missing and defaulted to
"neg". Missing-input reports do not establish whether inputs were actually
absent or incorrectly described as absent by the model. The items were
rerun on the HPC (jobs 998518/998519, ~/rerun_qwen36_blind, current pipeline,
temperature 0, thinking off); the rerun outputs are copied to
analysis/patterns_outputs/qwen36_rerun/<context>/<family>.xlsx.

Reports, for each symbolic cell, how many rerun outputs are themselves blind,
and the no-cue description-minus-pixel comparison of Table 2 under three
treatments of the blind rows: as archived (primary scoring), excluded, and
replaced by the rerun. Writes qwen36_blind_rerun.csv.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
RERUN = OUT / "qwen36_rerun"
CTX = {"action_description": "AD", "action_description_concept_v2": "ADC", "aps": "AP", "aps_concept": "APC"}
FAILURE_PATTERNS = ["cannot be determined", "missing input data", "cannot be analyzed", "impossible to analyze",
                    "insufficient information", "user has not provided", "unable to determine", "not enough information"]


def blind(text):
    t = str(text).lower()
    return any(p in t for p in FAILURE_PATTERNS)


d = pd.read_csv(ROOT / "machine_result/clean_machine_data.csv", low_memory=False)
q = d[d.model.eq("qwen3.6:35b") & d.prediction_valid].drop_duplicates(["experiment_type", "test_id"])

rr = []
for ctx, exp in CTX.items():
    for f in sorted((RERUN / ctx).glob("*.xlsx")):
        x = pd.read_excel(f)
        x["experiment_type"] = exp
        rr.append(x)
rr = pd.concat(rr, ignore_index=True)
assert not rr.duplicated(["experiment_type", "test_id"]).any()
targets = pd.read_csv(RERUN / "blind_items.csv")
target_keys = set(zip(targets.experiment_type, targets.test_id))
rerun_keys = set(zip(rr.experiment_type, rr.test_id))
assert len(target_keys) == len(targets) == 81
assert rerun_keys <= target_keys, "Rerun output outside the targeted item-condition ledger"
rr["prediction"] = rr.test_category_identified.astype(str).str.strip().str.lower()
rr["valid"] = rr.prediction.isin(["pos", "neg"])
rr["blind"] = rr.complete_output.map(blind)
lab = q.drop_duplicates("test_id").set_index("test_id").label
rr["correct_int"] = (rr.valid & rr.prediction.eq(rr.test_id.map(lab))).astype(int)

print("rerun outputs by cell (n, valid, still blind, correct):")
print(rr.groupby("experiment_type").agg(n=("test_id", "size"), valid=("valid", "sum"),
                                        still_blind=("blind", "sum"), correct=("correct_int", "sum")).to_string())

pix = q[q.experiment_type.eq("DVRL")].set_index("test_id").correct_int
ad = q[q.experiment_type.eq("AD")].set_index("test_id")
fresh = rr[rr.experiment_type.eq("AD") & rr.valid & ~rr.blind].set_index("test_id").correct_int
assert fresh.index.isin(ad.index[ad.generation_failure_flag]).all()
variants = {
    "archived (primary scoring)": ad.correct_int,
    "blind rows excluded": ad.correct_int[~ad.generation_failure_flag],
    "blind rows replaced by rerun": pd.concat([ad.correct_int[~ad.generation_failure_flag], fresh]),
}
rows = []
for name, a in variants.items():
    ids = pix.index.intersection(a.index)
    a, p = a[ids], pix[ids]
    b, c = int(((a == 1) & (p == 0)).sum()), int(((a == 0) & (p == 1)).sum())
    rows.append(dict(treatment=name, n_items=len(ids), pixel_acc=100 * p.mean(), description_acc=100 * a.mean(),
                     gain_pp=100 * (a.mean() - p.mean()), description_only=b, pixel_only=c,
                     exact_mcnemar_p=binomtest(b, b + c).pvalue if b + c else float("nan")))
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
res.to_csv(OUT / "qwen36_blind_rerun.csv", index=False)
