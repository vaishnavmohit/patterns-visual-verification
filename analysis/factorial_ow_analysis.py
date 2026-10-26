#!/usr/bin/env python3
"""Primary and secondary analyses of the open-weight replication, as planned.

Plan: code repository docs/factorial_openweight_plan.md (commit ec3efd2, with
amendments A1 and A2). Input: patterns_outputs/factorial_ow_trials.csv
(analysis/hpc/export_factorial_ow.py). Models are analysed separately.

d' and c use the (x+0.5)/(n+1) correction, are computed per run on the
decisions available in that run, and are averaged over the two runs (greedy,
s1). Intervals: 10,000 family-stratified base-problem bootstrap resamples, the
positive and negative query of a base kept together, percentile 95%.

P1 recovered share (d'_caption_self - d'_pixels) / (d'_oracle - d'_pixels), no
   cue; large recovery excluded if the upper limit < 0.5, recovery shown if the
   lower limit > 0, not estimable if the gap interval includes zero.
P2 d'_caption_self_free - d'_caption_self, no cue.
P3 within pixels, caption_self, oracle_description: (a) d'_correct -
   d'_wrong_clean; (b) c_cue - c_none for correct, wrong_clean, irrelevant.
P4 oracle_description_rendered - oracle_description, no cue: d' (bootstrap) and
   accuracy (exact McNemar per run on items answered under both).
Secondary: d'_wrong - d'_wrong_clean and c_wrong - c_wrong_clean; the no-cue
ladder; decision rate per cell.

A3 (amendment): a second caption set of each protocol (temperature 1, seed 2);
   prose - step in the new set, replicated for Gemma if its lower limit > 0.
A4 (amendment): two further caption sets (temperature 1, seeds 3 and 4). Each
   caption set is one complete captioning of all 2,504 figures under one
   protocol. Primary: prose - step averaged over the four sets, all sets under
   the same resamples; robust to caption sampling if the lower limit > 0. Also
   the per-set contrasts, their between-set SD, a sets-as-random interval
   (mean +/- t(0.975, 3) x SD / 2), temperature-1 mean minus the temperature-0
   set, and per-set recovered shares. Written once all sets of a model are complete.

Scoring: primary = rescored decisions (amendment A2); --strict uses the
original scoring. Writes factorial_ow_results[_strict].csv and
factorial_ow_cells[_strict].csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from scipy.stats import t as tdist

OUT = Path(__file__).resolve().parent / "patterns_outputs"
NBOOT, SEED = 10_000, 20261012
RUNS = ("greedy", "s1")

ap = argparse.ArgumentParser()
ap.add_argument("--strict", action="store_true")
args = ap.parse_args()
col = "prediction_strict" if args.strict else "prediction_rescored"
tag = "_strict" if args.strict else ""

tr = pd.read_csv(OUT / "factorial_ow_trials.csv")
items = pd.read_csv(OUT / "cue_overlap_items.csv")[["test_id", "uid", "family"]].set_index("test_id")
ids = list(items.index)
idx = {t: i for i, t in enumerate(ids)}
pos = np.array([t.endswith("_pos") for t in ids])

# Bootstrap weights: base problems resampled within family; a base's queries share its weight.
rng = np.random.default_rng(SEED)
bases = items.reset_index().groupby(["family", "uid"]).test_id.apply(list).reset_index()
W = np.zeros((len(ids), NBOOT), dtype=np.float32)
for fam, g in bases.groupby("family"):
    draws = rng.integers(0, len(g), size=(len(g), NBOOT))
    counts = np.zeros((len(g), NBOOT), dtype=np.float32)
    for b in range(len(g)):
        counts[b] = (draws == b).sum(axis=0)
    for b, tids in enumerate(g.test_id):
        for t in tids:
            W[idx[t]] = counts[b]
ONE = np.ones((len(ids), 1), dtype=np.float32)


def vectors(model, rep, cue, run):
    g = tr[(tr.model == model) & (tr.representation == rep) & (tr.cue == cue) & (tr.run == run)]
    v = np.zeros(len(ids)); r = np.zeros(len(ids))
    for t, p in zip(g.test_id, g[col]):
        if isinstance(p, str):
            v[idx[t]] = 1; r[idx[t]] = p == "pos"
    return v, r, len(g)


def sdt(model, rep, cue, weights):
    """Run-averaged d' and c for every column of weights."""
    ds, cs = [], []
    for run in RUNS:
        v, r, n = vectors(model, rep, cue, run)
        if n == 0:
            return None, None
        # numpy 2.0 with Apple Accelerate raises spurious floating-point warnings
        # in matmul; results match a float64 einsum exactly (checked 2026-10-12).
        with np.errstate(all="ignore"):
            hit = weights.T @ (v * r * pos); npos = weights.T @ (v * pos)
            fa = weights.T @ (v * r * ~pos); nneg = weights.T @ (v * ~pos)
        zh, zf = norm.ppf((hit + .5) / (npos + 1)), norm.ppf((fa + .5) / (nneg + 1))
        ds.append(zh - zf); cs.append(-(zh + zf) / 2)
    return np.mean(ds, axis=0), np.mean(cs, axis=0)


def ci(x):
    return np.percentile(x, [2.5, 97.5])


res = []


def add(model, q, contrast, point, boot, **extra):
    lo, hi = ci(boot) if boot is not None else (np.nan, np.nan)
    res.append(dict(model=model, question=q, contrast=contrast, estimate=float(point), ci_low=float(lo), ci_high=float(hi), **extra))


cells = []
for model in sorted(tr.model.unique()):
    m = tr[tr.model == model]
    for (rep, cue), g in m.groupby(["representation", "cue"]):
        d, c = sdt(model, rep, cue, ONE)
        acc = [g[g.run == r_].pipe(lambda x: (x[col] == x.polarity).sum() / x[col].notna().sum()) for r_ in RUNS if (g.run == r_).any()]
        cells.append(dict(model=model, representation=rep, cue=cue, runs=g.run.nunique(), attempted=len(g),
                          decision_rate=g[col].notna().mean(), accuracy=float(np.mean(acc)),
                          dprime=None if d is None else float(d[0]), criterion=None if c is None else float(c[0])))

    def D(rep, cue, w=W):
        return sdt(model, rep, cue, w)

    # P1
    if all(D(r_, "none", ONE)[0] is not None for r_ in ("pixels", "caption_self", "oracle_description")):
        (dp, _), (dsf, _), (do, _) = D("pixels", "none", ONE), D("caption_self", "none", ONE), D("oracle_description", "none", ONE)
        (bp, _), (bs, _), (bo, _) = D("pixels", "none"), D("caption_self", "none"), D("oracle_description", "none")
        add(model, "P1", "d' own captions - pixels", dsf[0] - dp[0], bs - bp)
        add(model, "P1", "d' benchmark descriptions - pixels", do[0] - dp[0], bo - bp)
        gap_lo, _ = ci(bo - bp)
        share = (dsf[0] - dp[0]) / (do[0] - dp[0])
        bshare = (bs - bp) / (bo - bp)
        lo, hi = ci(bshare)
        verdict = ("not estimable" if gap_lo <= 0 else
                   "recovery shown" if lo > 0 else
                   "large recovery excluded" if hi < 0.5 else "inconclusive")
        add(model, "P1", "recovered share", share, bshare, verdict=verdict)
    # Exploratory (not in the plan): recovered share for the prose captions.
    if all(D(r_, "none", ONE)[0] is not None for r_ in ("pixels", "caption_self_free", "oracle_description")):
        dp, dfr, do = (D(r_, "none", ONE)[0][0] for r_ in ("pixels", "caption_self_free", "oracle_description"))
        bp, bf, bo = (D(r_, "none")[0] for r_ in ("pixels", "caption_self_free", "oracle_description"))
        add(model, "E", "recovered share, prose captions (exploratory)", (dfr - dp) / (do - dp), (bf - bp) / (bo - bp))
    # P2
    if D("caption_self_free", "none", ONE)[0] is not None and D("caption_self", "none", ONE)[0] is not None:
        add(model, "P2", "d' prose captions - step captions", D("caption_self_free", "none", ONE)[0][0] - D("caption_self", "none", ONE)[0][0],
            D("caption_self_free", "none")[0] - D("caption_self", "none")[0])
    # A3 (amendment, recorded before generation): independent temperature-1 caption draw.
    # Computed only when both new-draw cells have every item in both runs.
    def complete(rep):
        return all(vectors(model, rep, "none", run)[2] == len(ids) for run in RUNS)
    if all(complete(r_) for r_ in ("caption_self_d2", "caption_self_free_d2")):
        p2, s2_ = D("caption_self_free_d2", "none", ONE)[0][0], D("caption_self_d2", "none", ONE)[0][0]
        bp2, bs2 = D("caption_self_free_d2", "none")[0], D("caption_self_d2", "none")[0]
        p1, s1_ = D("caption_self_free", "none", ONE)[0][0], D("caption_self", "none", ONE)[0][0]
        bp1, bs1 = D("caption_self_free", "none")[0], D("caption_self", "none")[0]
        lo, _ = ci(bp2 - bs2)
        add(model, "A3", "d' prose - step captions, draw 2", p2 - s2_, bp2 - bs2,
            verdict=("replicated" if lo > 0 else "not replicated"))
        add(model, "A3", "d' prose - step captions, pooled over draws", ((p1 - s1_) + (p2 - s2_)) / 2, ((bp1 - bs1) + (bp2 - bs2)) / 2)
        add(model, "A3", "draw 2 minus draw 1 contrast", (p2 - s2_) - (p1 - s1_), (bp2 - bs2) - (bp1 - bs1))
        dp, do = D("pixels", "none", ONE)[0][0], D("oracle_description", "none", ONE)[0][0]
        bpx, bo = D("pixels", "none")[0], D("oracle_description", "none")[0]
        add(model, "A3", "recovered share, step captions, draw 2", (s2_ - dp) / (do - dp), (bs2 - bpx) / (bo - bpx))
        add(model, "A3", "recovered share, prose captions, draw 2", (p2 - dp) / (do - dp), (bp2 - bpx) / (bo - bpx))
    # A4 (amendment, recorded before generation): four caption sets per protocol.
    SETS = (("", "primary set"), ("_d2", "set 2"), ("_d3", "set 3"), ("_d4", "set 4"))
    if all(complete(f"caption_self{s_}") and complete(f"caption_self_free{s_}") for s_, _ in SETS[1:]):
        pt = np.array([D(f"caption_self_free{s_}", "none", ONE)[0][0] - D(f"caption_self{s_}", "none", ONE)[0][0] for s_, _ in SETS])
        bt = np.array([D(f"caption_self_free{s_}", "none")[0] - D(f"caption_self{s_}", "none")[0] for s_, _ in SETS])
        for (_, name), p_, b_ in zip(SETS, pt, bt):
            add(model, "A4", f"d' prose - step captions, {name}", p_, b_)
        lo, _ = ci(bt.mean(axis=0))
        add(model, "A4", "d' prose - step captions, mean of four sets", pt.mean(), bt.mean(axis=0),
            verdict=("robust to caption sampling" if lo > 0 else "not shown robust"))
        sd = pt.std(ddof=1); half = tdist.ppf(0.975, len(SETS) - 1) * sd / np.sqrt(len(SETS))
        res.append(dict(model=model, question="A4", contrast="d' prose - step captions, mean of four sets, sets as random",
                        estimate=float(pt.mean()), ci_low=float(pt.mean() - half), ci_high=float(pt.mean() + half)))
        res.append(dict(model=model, question="A4", contrast="between-set SD of d' prose - step", estimate=float(sd), ci_low=np.nan, ci_high=np.nan))
        add(model, "A4", "temperature-1 sets mean minus temperature-0 set", pt[1:].mean() - pt[0], bt[1:].mean(axis=0) - bt[0])
        for s_, name in SETS[2:]:
            for proto, rep in (("step captions", f"caption_self{s_}"), ("prose captions", f"caption_self_free{s_}")):
                add(model, "A4", f"recovered share, {proto}, {name}", (D(rep, "none", ONE)[0][0] - dp) / (do - dp), (D(rep, "none")[0] - bpx) / (bo - bpx))
    # P3 and secondary cue contrasts
    for rep in ("pixels", "caption_self", "oracle_description"):
        pt = {cue: D(rep, cue, ONE) for cue in ("none", "correct", "wrong", "wrong_clean", "irrelevant")}
        if any(v[0] is None for v in pt.values()):
            continue
        bt = {cue: D(rep, cue) for cue in pt}
        add(model, "P3a", f"{rep}: d' correct - wrong_clean", pt["correct"][0][0] - pt["wrong_clean"][0][0], bt["correct"][0] - bt["wrong_clean"][0])
        for cue in ("correct", "wrong_clean", "irrelevant"):
            add(model, "P3b", f"{rep}: c {cue} - none", pt[cue][1][0] - pt["none"][1][0], bt[cue][1] - bt["none"][1])
        add(model, "S", f"{rep}: d' wrong - wrong_clean", pt["wrong"][0][0] - pt["wrong_clean"][0][0], bt["wrong"][0] - bt["wrong_clean"][0])
        add(model, "S", f"{rep}: c wrong - wrong_clean", pt["wrong"][1][0] - pt["wrong_clean"][1][0], bt["wrong"][1] - bt["wrong_clean"][1])
    # P4
    if D("oracle_description_rendered", "none", ONE)[0] is not None:
        add(model, "P4", "d' rendered - archived descriptions", D("oracle_description_rendered", "none", ONE)[0][0] - D("oracle_description", "none", ONE)[0][0],
            D("oracle_description_rendered", "none")[0] - D("oracle_description", "none")[0])
        for run in RUNS:
            a = m[(m.representation == "oracle_description") & (m.cue == "none") & (m.run == run)].set_index("test_id")[col]
            b = m[(m.representation == "oracle_description_rendered") & (m.cue == "none") & (m.run == run)].set_index("test_id")[col]
            both = a.dropna().index.intersection(b.dropna().index)
            lab = pd.Series(["pos" if t.endswith("_pos") else "neg" for t in both], index=both)
            ca, cb = (a[both] == lab), (b[both] == lab)
            x, y = int((cb & ~ca).sum()), int((ca & ~cb).sum())
            add(model, "P4", f"accuracy rendered - archived ({run})", 100 * (cb.mean() - ca.mean()), None,
                n_items=len(both), mcnemar_p=binomtest(x, x + y).pvalue if x + y else np.nan)

R = pd.DataFrame(res); C = pd.DataFrame(cells)
R.to_csv(OUT / f"factorial_ow_results{tag}.csv", index=False)
C.to_csv(OUT / f"factorial_ow_cells{tag}.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(f"scoring: {col}")
print(R.round(3).to_string(index=False))
print(C[C.cue == "none"].round(3).to_string(index=False))
print(C[C.decision_rate < 1].round(3)[["model", "representation", "cue", "attempted", "decision_rate"]].to_string(index=False))
