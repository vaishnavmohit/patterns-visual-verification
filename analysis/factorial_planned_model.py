#!/usr/bin/env python3
"""The factorial's planned trial-level analysis, fitted retrospectively.

The plan (code repository, docs/factorial_preregistration.md at 46dfbb7)
specified a logistic mixed model on trial correctness with item random
intercepts, model fixed effects and representation x cue x polarity
interactions, with run as a replicate, and five contrasts:

  1. pixels vs caption_self                 (no cue)
  2. caption_self vs oracle_description     (no cue)
  3. caption_strong vs caption_self         (no cue; GPT 5.5 only)
  4. oracle_description vs program          (no cue)
  5. correct vs wrong vs none within each representation, on d' and on c

Contrasts 1-4 are fitted per model on the two representations involved:
correct ~ rep + polarity + rep:polarity with polarity coded +-1/2, so the rep
coefficient is the log-odds change averaged over query polarity. Two
estimators: the planned crossed-intercept GLMM (statsmodels
BinomialBayesMixedGLM, variational Bayes, item random intercepts) and a GEE
with exchangeable working correlation clustered by base problem, whose robust
intervals are the primary ones because variational Bayes can understate
posterior variance. Contrast 5 uses the response-level decomposition
P(respond pos) = logistic(b0 + b1 s + b2 cue + b3 s cue), s = +-1/2, GEE by base
problem: b3 is the cue's change in logit-scale sensitivity and -b2 its change
in criterion (positive = more conservative). These are logit-scale analogues of
d' and c, not probit estimates. This analysis was fitted after the results were
known and is reported as the completed planned model, not as a prespecified
confirmation. Writes factorial_planned_model.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

OUT = Path(__file__).resolve().parent / "patterns_outputs"
tr = pd.read_csv(OUT / "factorial_trials.csv")
tr["s"] = np.where(tr.polarity.eq("pos"), 0.5, -0.5)
tr["resp_pos"] = tr.prediction.eq("pos").astype(int)
tr["uid"] = tr.test_id.str.replace(r"_(pos|neg)$", "", regex=True)
rows = []


def gee(formula, data):
    return smf.gee(formula, groups="uid", data=data, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


CONTRASTS = [("1 pixels -> own captions", "pixels", "caption_self"),
             ("2 own captions -> benchmark descriptions", "caption_self", "oracle_description"),
             ("3 own captions -> other model's captions", "caption_self", "caption_strong"),
             ("4 benchmark descriptions -> programs", "oracle_description", "program")]
for model, g in tr[tr.cue.eq("none")].groupby("model"):
    for label, a, b in CONTRASTS:
        d = g[g.representation.isin([a, b])].copy()
        if d.representation.nunique() < 2:
            continue
        d["rep"] = d.representation.eq(b).astype(float)
        f = gee("correct ~ rep * s", d)
        ci = f.conf_int().loc["rep"]
        vb = BinomialBayesMixedGLM.from_formula("correct ~ rep * s", {"item": "0 + C(test_id)"}, d).fit_vb()
        i = list(vb.model.exog_names).index("rep")
        rows.append(dict(model=model, contrast=label, estimand="log-odds correct (b minus a)",
                         gee=f.params["rep"], gee_low=ci[0], gee_high=ci[1], gee_p=f.pvalues["rep"],
                         glmm_vb=vb.fe_mean[i], glmm_vb_sd=vb.fe_sd[i], n_trials=len(d)))

for (model, rep), g in tr.groupby(["model", "representation"]):
    for a, b in (("none", "correct"), ("none", "wrong"), ("wrong", "correct")):
        d = g[g.cue.isin([a, b])].copy()
        d["cue_b"] = d.cue.eq(b).astype(float)
        f = gee("resp_pos ~ s * cue_b", d)
        ci = f.conf_int()
        for term, name, sign in (("s:cue_b", "logit sensitivity", 1), ("cue_b", "logit criterion", -1)):
            lo, hi = sorted((sign * ci.loc[term, 0], sign * ci.loc[term, 1]))
            rows.append(dict(model=model, contrast=f"5 {rep}: {b} vs {a}", estimand=name,
                             gee=sign * f.params[term], gee_low=lo, gee_high=hi, gee_p=f.pvalues[term], n_trials=len(d)))

res = pd.DataFrame(rows)
res.to_csv(OUT / "factorial_planned_model.csv", index=False)
pd.set_option("display.width", 220)
print(res[res.contrast.str[0].isin(list("1234"))].round(3).to_string(index=False))
c5 = res[res.contrast.str.startswith("5")]
for est in ("logit sensitivity", "logit criterion"):
    for cmp in ("correct vs none", "wrong vs none", "correct vs wrong"):
        x = c5[(c5.estimand == est) & c5.contrast.str.endswith(cmp)]
        print(f"{est:18s} {cmp:17s} positive {int((x.gee > 0).sum())}/{len(x)}, CI excludes zero {int(((x.gee_low > 0) | (x.gee_high < 0)).sum())}/{len(x)}")
