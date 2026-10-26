#!/usr/bin/env python3
"""Headline recovered share on the d' scale, with base-problem bootstrap CIs.

share = (d'_own captions - d'_pixels) / (d'_benchmark descriptions - d'_pixels),
no cue, each d' computed per run and averaged over runs (four for Gemini, two
for GPT), so the point estimates equal the run means quoted in the text.
Resamples base problems within families, positive and negative queries of a
base together. Also reports the accuracy-scale share once, for the supplement.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "patterns_outputs"
NBOOT, SEED = 4000, 20260926
tr = pd.read_csv(OUT / "factorial_trials.csv"); tr = tr[tr.cue.eq("none")]
fam = pd.read_csv(OUT / "item_identifiability.csv").set_index("test_id").family.to_dict()
rng = np.random.default_rng(SEED)


def dprime(g):
    t = g.polarity.eq("pos").to_numpy(); r = g.prediction.eq("pos").to_numpy()
    return norm.ppf(((t & r).sum() + .5) / (t.sum() + 1)) - norm.ppf((((~t) & r).sum() + .5) / ((~t).sum() + 1))


def run_mean(sub, rep, metric):
    s = sub[sub.representation.eq(rep)]
    return np.mean([dprime(g) if metric == "d" else g.correct.mean() for _, g in s.groupby("run")])


def share(sub, metric):
    p, c, o = (run_mean(sub, r, metric) for r in ("pixels", "caption_self", "oracle_description"))
    return (c - p) / (o - p), p, c, o


for model, g in tr.groupby("model"):
    ids = sorted(set(g.test_id)); base = {t: t.rsplit("_", 1)[0] for t in ids}
    blocks = {}
    for t in ids: blocks.setdefault(fam[t], {}).setdefault(base[t], []).append(t)
    blocks = {f: list(b.values()) for f, b in blocks.items()}
    by_id = {t: grp for t, grp in g.groupby("test_id")}
    for metric in ("d", "acc"):
        point, p, c, o = share(g, metric)
        boot = []
        for _ in range(NBOOT):
            pick = [t for f, bl in blocks.items() for j in rng.integers(0, len(bl), len(bl)) for t in bl[j]]
            boot.append(share(pd.concat([by_id[t] for t in pick]), metric)[0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        unit = "d'" if metric == "d" else "acc"
        print(f"{model:17s} {unit:4s} pixels {p:.3f}  own captions {c:.3f}  benchmark {o:.3f}  "
              f"share {100*point:+.1f}%  95% CI [{100*lo:+.0f}%, {100*hi:+.0f}%]")
