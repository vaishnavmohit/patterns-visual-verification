#!/usr/bin/env python3
"""Structural fidelity of the factorial self-captions against the benchmark's own descriptions.

The planned secondary analysis (factorial plan, commit 46dfbb7): per-image
agreement between a model's caption and the oracle Action Description in
stroke count, line/arc counts and texture words, and whether per-item fidelity
predicts per-item accuracy within model.

Captions and oracle descriptions share one step grammar ("draw a <texture>
<line|arc> ... Then, continue straight" or "After that, turn ..."), so both
parse into a sequence of (texture, kind) strokes. An oracle boundary is
invisible when, within one shape, a straight line is followed by a straight line
of the same texture whose own turn is zero (the renderer turns before each
stroke; item_identifiability_v2.py); the visible edge count merges those
strokes. Shape boundaries come from the original ShapeBongard_V2 program files,
passed as the first argument. Per image:

  strokes_match   caption stroke count equals the program's
  visible_match   caption stroke count equals the visible edge count
  kinds_match     caption line and arc counts both equal the program's
  token_f1        multiset F1 of (texture, kind) tokens

Per item, metrics average over its 13 figures (6 positive, 6 negative, query).
Accuracy is the model's own-caption, no-cue accuracy pooled over runs.

Inputs: patterns_outputs/factorial_captions/<model>.jsonl (final caches,
prompt v3), factorial_trials.csv, item_identifiability_v2.csv, and the item files
in the code repository. Writes caption_fidelity_images.csv,
caption_fidelity_items.csv and caption_fidelity_summary.csv.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
CODE = ROOT.parents[1] / "code" / "human-machine-reasoning"
MODELS = ("gemini-3.5-flash", "gpt-5.5")
STEP = re.compile(r"draw an? (normal|zigzag|circle|square|triangle) (line|arc)\b[^\n]*?(continue straight|turn (?:left|right))",
                  re.I)


def parse(text):
    return [(t.lower(), k.lower(), "straight" in turn.lower()) for t, k, turn in STEP.findall(text or "")]


def hidden_boundaries(shapes):
    """Renderer convention: a stroke's turn precedes it; shapes are drawn separately."""
    n = 0
    for shape in shapes:
        s = [(x.split("_")[0], x.split("_")[1], float(x.split("-")[-1])) for x in shape]
        n += sum(1 for a, b in zip(s, s[1:]) if a[0] == b[0] == "line" and a[1] == b[1] and abs(b[2] - 0.5) < 1e-9)
    return n


PROGRAMS = Path(sys.argv[1]) if len(sys.argv) > 1 else None   # ShapeBongard_V2 *_action_programs.json directory
NESTED = {}
for src in ("bd", "ff", "hd"):
    for prob, sides in json.load(open(PROGRAMS / f"{src}_action_programs.json")).items():
        for f in sides[0] + sides[1]:
            NESTED[(prob, tuple(x for sh in f for x in sh))] = f


def token_f1(a, b):
    ca, cb = Counter((t, k) for t, k, _ in a), Counter((t, k) for t, k, _ in b)
    tot = sum(ca.values()) + sum(cb.values())
    return 2 * sum((ca & cb).values()) / tot if tot else np.nan


fig = []
for fam in ("bd", "ff", "hd_comb", "hd_novel"):
    for it in json.load(open(CODE / f"data/Bongard_Logo_human/{fam}.json")):
        imgs = it["images"]["pos"][:6] + it["images"]["neg"][:6] + [it["images"]["testfiles"]]
        progs = it["action_prog"]["pos"][:6] + it["action_prog"]["neg"][:6] + [it["testfiles"]["action_prog"]]
        roles = ["pos"] * 6 + ["neg"] * 6 + ["query"]
        fig += [dict(test_id=it["test_id"], uid=it["uid"], family=fam, image=p, role=r,
                     hidden=hidden_boundaries(NESTED[(p.split("/")[2], tuple(g))]))
                for p, g, r in zip(imgs, progs, roles)]
fig = pd.DataFrame(fig)
HIDDEN = fig.drop_duplicates("image").set_index("image").hidden.to_dict()

rows = []
for m in MODELS:
    caps = {}
    for line in open(OUT / "factorial_captions" / f"{m}.jsonl"):
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("prompt_version") == "v3":
            caps[r["image"]] = r
    assert len(caps) == 2504, (m, len(caps))
    for img, r in caps.items():
        o, c = parse(r["oracle"]), parse(r["description"])
        assert o, img
        vis = len(o) - HIDDEN[img]
        rows.append(dict(model=m, image=img, oracle_strokes=len(o), oracle_visible=vis,
                         caption_strokes=len(c), caption_parsed=bool(c),
                         strokes_match=len(c) == len(o), visible_match=len(c) == vis,
                         kinds_match=(sum(k == "line" for _, k, _ in c), sum(k == "arc" for _, k, _ in c))
                         == (sum(k == "line" for _, k, _ in o), sum(k == "arc" for _, k, _ in o)),
                         token_f1=token_f1(c, o), multipart="separate shape" in r["description"].lower()))
img = pd.DataFrame(rows)
img["ambiguous"] = img.oracle_visible < img.oracle_strokes
img.to_csv(OUT / "caption_fidelity_images.csv", index=False)

summ = []
for (m, amb), g in img.groupby(["model", "ambiguous"]):
    summ.append(dict(model=m, figures="under-determined" if amb else "no invisible boundary", n_images=len(g),
                     parsed=g.caption_parsed.mean(), strokes_match=g.strokes_match.mean(),
                     visible_match=g.visible_match.mean(), kinds_match=g.kinds_match.mean(),
                     token_f1=g.token_f1.mean()))
for m, g in img.groupby("model"):
    summ.append(dict(model=m, figures="all", n_images=len(g), parsed=g.caption_parsed.mean(),
                     strokes_match=g.strokes_match.mean(), visible_match=g.visible_match.mean(),
                     kinds_match=g.kinds_match.mean(), token_f1=g.token_f1.mean()))
summ = pd.DataFrame(summ)

# Item level: fidelity over the 13 figures, joined to own-caption accuracy without a cue.
tr = pd.read_csv(OUT / "factorial_trials.csv")
tr = tr[tr.representation.eq("caption_self") & tr.cue.eq("none")]
ident = pd.read_csv(OUT / "item_identifiability_v2.csv")[["test_id", "query_underdetermined"]]
items = []
for m in MODELS:
    f = fig.merge(img[img.model.eq(m)], on="image")
    agg = f.groupby(["test_id", "uid", "family"]).agg(item_token_f1=("token_f1", "mean"),
                                                       item_strokes_match=("strokes_match", "mean"),
                                                       item_visible_match=("visible_match", "mean")).reset_index()
    q = f[f.role.eq("query")].set_index("test_id")
    agg["query_token_f1"] = agg.test_id.map(q.token_f1)
    acc = tr[tr.model.eq(m)].groupby("test_id").correct.agg(["mean", "size"]).rename(columns={"mean": "accuracy", "size": "n_trials"})
    agg = agg.merge(acc, left_on="test_id", right_index=True).merge(ident, on="test_id")
    agg["model"] = m
    items.append(agg)
items = pd.concat(items)
items.to_csv(OUT / "caption_fidelity_items.csv", index=False)

# Does item fidelity predict own-caption correctness? Trial-level logistic GEE, family fixed
# effects, clustered by base problem; fidelity standardised within model.
pred = []
for m in MODELS:
    it = items[items.model.eq(m)].copy()
    t = tr[tr.model.eq(m)].merge(it[["test_id", "uid", "family", "item_token_f1", "query_underdetermined"]], on="test_id")
    t["fid_z"] = (t.item_token_f1 - it.item_token_f1.mean()) / it.item_token_f1.std()
    fit = smf.gee("correct ~ fid_z + C(family)", groups="uid", data=t, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    fit2 = smf.gee("correct ~ fid_z + query_underdetermined + C(family)", groups="uid", data=t,
                   family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    rho, p = spearmanr(it.item_token_f1, it.accuracy)
    ci = fit.conf_int().loc["fid_z"]
    pred.append(dict(model=m, spearman_item_f1_vs_accuracy=rho, spearman_p=p, logit_per_sd=fit.params["fid_z"],
                     ci_low=ci[0], ci_high=ci[1], p=fit.pvalues["fid_z"],
                     logit_per_sd_adj_underdetermined=fit2.params["fid_z"],
                     underdetermined_logit_adj=fit2.params["query_underdetermined[T.True]"],
                     underdetermined_p_adj=fit2.pvalues["query_underdetermined[T.True]"]))
pred = pd.DataFrame(pred)
summ.to_csv(OUT / "caption_fidelity_summary.csv", index=False)
pred.to_csv(OUT / "caption_fidelity_prediction.csv", index=False)

pd.set_option("display.width", 220)
print(summ.round(3).to_string(index=False))
print()
print(pred.round(3).to_string(index=False))
print()
for m in MODELS:
    g = img[img.model.eq(m) & img.ambiguous]
    print(f"{m}: on {len(g)} under-determined figures, caption count equals visible edges {g.visible_match.mean():.1%} "
          f"vs program strokes {g.strokes_match.mean():.1%}; multi-part captions overall {img[img.model.eq(m)].multipart.mean():.1%}")
