#!/usr/bin/env python3
"""Accuracy given a decision, decision rate, and operational success for the archived cells.

Every archived cell targets the same 200 queries. The coverage rule and the
paired comparisons of Table 2 are computed over returned decisions, so a query
that never produced one (a non-terminating trace, an unrecoverable parse
failure, a run that stopped) is outside both denominators. This reports the
three quantities separately, per model x interface x cue:

  conditional accuracy  correct / returned decisions
  decision rate         returned decisions / 200 attempted queries
  operational success   correct / 200 attempted queries

and, for the 13 aliases of Table 2, the no-cue pixel-to-description gain
under each: the matched conditional gain of Table 2, and the operational gain
over all 200 queries with a missing decision counted as not correct (exact
McNemar on the 200 pairs). A strict variant also counts the 86 failure-flagged
decisions as not correct; 81 of them are Qwen3.6:35b outputs that report the
descriptions as missing (see qwen36_blind_rerun.py).

Writes operational_cells.csv and operational_paired.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
N_ITEMS = 200

d = pd.read_csv(ROOT / "machine_result/clean_machine_data.csv", low_memory=False)
d = d[d.prediction_valid & d.clean_primary_keep.astype(bool)] if "clean_primary_keep" in d else d[d.prediction_valid]
d = d.drop_duplicates(["model", "experiment_type", "test_id"])
INTERFACE = {"DVRL": ("pixels", "none"), "DVRLC": ("pixels", "correct"), "AD": ("descriptions", "none"),
             "ADC": ("descriptions", "correct"), "AP": ("programs", "none"), "APC": ("programs", "correct")}

cells = (d.groupby(["model", "experiment_type"])
         .agg(decisions=("test_id", "size"), correct=("correct_int", "sum"), flagged=("generation_failure_flag", "sum"))
         .reset_index())
cells[["interface", "cue"]] = cells.experiment_type.map(INTERFACE).apply(pd.Series)
cells["conditional_acc"] = 100 * cells.correct / cells.decisions
cells["decision_rate"] = 100 * cells.decisions / N_ITEMS
cells["operational_success"] = 100 * cells.correct / N_ITEMS
cells = cells[["model", "interface", "cue", "decisions", "flagged", "conditional_acc", "decision_rate", "operational_success"]]
cells.sort_values(["model", "interface", "cue"]).to_csv(OUT / "operational_cells.csv", index=False)

paired = pd.read_csv(OUT / "machine_paired_representation_tests.csv")


def mcnemar(a, p):
    b, c = int(((a == 1) & (p == 0)).sum()), int(((a == 0) & (p == 1)).sum())
    return b, c, (binomtest(b, b + c).pvalue if b + c else np.nan)


rows = []
for m in paired.model:
    q = d[d.model.eq(m)]
    pix = q[q.experiment_type.eq("DVRL")].set_index("test_id")
    ad = q[q.experiment_type.eq("AD")].set_index("test_id")
    allids = sorted(set(d.test_id))
    assert len(allids) == N_ITEMS
    r = dict(model=m, pix_decisions=len(pix), desc_decisions=len(ad))
    shared = pix.index.intersection(ad.index)
    r["matched_items"] = len(shared)
    r["matched_gain_pp"] = 100 * (ad.correct_int[shared].mean() - pix.correct_int[shared].mean())
    for name, flagged_as_error in (("operational", False), ("operational_strict", True)):
        def score(x):
            s = x.correct_int.where(~(flagged_as_error & x.generation_failure_flag.astype(bool)), 0)
            return s.reindex(allids).fillna(0).astype(int)
        a, p = score(ad), score(pix)
        b, c, pv = mcnemar(a, p)
        r[f"{name}_pix_pct"] = 100 * p.mean()
        r[f"{name}_desc_pct"] = 100 * a.mean()
        r[f"{name}_gain_pp"] = 100 * (a.mean() - p.mean())
        r[f"{name}_mcnemar_p"] = pv
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(OUT / "operational_paired.csv", index=False)

pd.set_option("display.width", 200)
print(res[["model", "pix_decisions", "desc_decisions", "matched_items", "matched_gain_pp", "operational_pix_pct",
           "operational_desc_pct", "operational_gain_pp", "operational_mcnemar_p", "operational_strict_gain_pp"]]
      .round(2).to_string(index=False))
print("\noperational gain positive in", int((res.operational_gain_pp > 0).sum()), "of", len(res),
      "| strict:", int((res.operational_strict_gain_pp > 0).sum()))
print("\ncells with decision rate below 100%:")
print(cells[cells.decision_rate < 100].round(1).to_string(index=False))
