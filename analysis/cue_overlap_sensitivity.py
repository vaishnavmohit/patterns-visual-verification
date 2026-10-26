#!/usr/bin/env python3
"""Do the factorial cue results survive removing mismatched labels that are partly correct?

The wrong cue is a cyclic derangement of concept labels within a family
(code repository, data/factorial/wrong_cues.json). It guarantees a different
label string, not a false concept. Bongard-LOGO labels are compositions: bd
labels join two shapes with " AND ", hd_comb and hd_novel labels join two
attributes with ", ". When the wrong label shares a component with the item's
own label, half of the "wrong" cue is true of the positive set. This flags
those items (a lexical lower bound: semantic overlaps such as "exist triangle"
against "has three straight lines" are not caught) and recomputes, per model
and representation, the correct-minus-wrong d' and both cues' criterion shifts
on all items and on the items whose wrong label shares no component.

d' and c use the (x+0.5)/(n+1) correction, per run, averaged over runs, as in
the manuscript. Intervals resample base problems within family (positive and
negative query of a base together), 2,000 draws.

Writes cue_overlap_items.csv and cue_overlap_sensitivity.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
CODE = ROOT.parents[1] / "code" / "human-machine-reasoning"
SEP = {"bd": " AND ", "ff": None, "hd_comb": ", ", "hd_novel": ", "}
NBOOT, SEED = 2000, 20261008

wrong = json.load(open(CODE / "data/factorial/wrong_cues.json"))["families"]
rows = []
for fam, sep in SEP.items():
    for it in json.load(open(CODE / f"data/Bongard_Logo_human/{fam}.json")):
        own, wr = it["concept_ui"], wrong[fam][it["concept_ui"]]
        shared = sorted(set(x.strip() for x in own.split(sep)) & set(x.strip() for x in wr.split(sep))) if sep else []
        rows.append(dict(test_id=it["test_id"], uid=it["uid"], family=fam, polarity=it["testfiles"]["category"],
                         correct_label=own, wrong_label=wr, shared_component="; ".join(shared), overlap=bool(shared)))
items = pd.DataFrame(rows)
items.to_csv(OUT / "cue_overlap_items.csv", index=False)
print("items whose wrong label shares a component with the correct one:")
print(items.groupby("family").overlap.agg(["sum", "size"]).to_string(), "\n")


def sdt(g):
    t = g.polarity.eq("pos").to_numpy(); r = g.prediction.eq("pos").to_numpy()
    h = ((t & r).sum() + .5) / (t.sum() + 1); f = (((~t) & r).sum() + .5) / ((~t).sum() + 1)
    return norm.ppf(h) - norm.ppf(f), -(norm.ppf(h) + norm.ppf(f)) / 2


def contrasts(tr):
    """Run-averaged d' and c per cue, then the three contrasts."""
    per = {}
    for cue, g in tr.groupby("cue"):
        v = np.array([sdt(r) for _, r in g.groupby("run")])
        per[cue] = v.mean(axis=0)
    return dict(dprime_correct_minus_wrong=per["correct"][0] - per["wrong"][0],
                c_correct_minus_none=per["correct"][1] - per["none"][1],
                c_wrong_minus_none=per["wrong"][1] - per["none"][1])


tr = pd.read_csv(OUT / "factorial_trials.csv").merge(items[["test_id", "uid", "family", "overlap"]], on="test_id")
rng = np.random.default_rng(SEED)
bases = items.drop_duplicates("uid")[["uid", "family"]]
draws = []
for _ in range(NBOOT):
    pick = pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1 << 31)) for _, g in bases.groupby("family")])
    draws.append(pick.uid.value_counts())

out = []
for (m, rep), g in tr.groupby(["model", "representation"]):
    for subset, s in (("all items", g), ("no shared component", g[~g.overlap])):
        est = contrasts(s)
        boot = {k: [] for k in est}
        by_uid = {u: x for u, x in s.groupby("uid")}
        for w in draws:
            parts = []
            for u, k in w.items():
                if u in by_uid:
                    parts += [by_uid[u]] * k
            b = contrasts(pd.concat(parts))
            for k in est:
                boot[k].append(b[k])
        for k, v in est.items():
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            out.append(dict(model=m, representation=rep, subset=subset, n_items=s.test_id.nunique(),
                            contrast=k, estimate=v, ci_low=lo, ci_high=hi))
res = pd.DataFrame(out)
res.to_csv(OUT / "cue_overlap_sensitivity.csv", index=False)

pd.set_option("display.width", 200)
wide = res.pivot_table(index=["model", "representation", "contrast"], columns="subset", values="estimate").round(3)
print(wide.to_string())
for k in ("dprime_correct_minus_wrong", "c_correct_minus_none", "c_wrong_minus_none"):
    for sub in ("all items", "no shared component"):
        x = res[(res.contrast == k) & (res.subset == sub)]
        print(f"{k:28s} {sub:20s} positive {int((x.estimate > 0).sum())}/{len(x)}, interval excludes zero {int((x.ci_low > 0).sum())}/{len(x)}")
