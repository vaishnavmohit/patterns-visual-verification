#!/usr/bin/env python3
"""Hierarchical signal-detection model for the human study.

Participant-level d' and c rest on 20 trials each, with unequal polarity counts
and different items per participant, so they are noisy and the log-linear
correction does real work. This fits the same decomposition as one binomial
GLMM on all 4,140 trials, with crossed random intercepts for participant and
item, by variational Bayes (statsmodels BinomialBayesMixedGLM, as the
manuscript's other crossed models):

    P(respond "positive") = logistic(b0 + b1*s + b2*cue + b3*s*cue)

with s = +1/2 for positive queries and -1/2 for negative ones. On the logit
scale b1 is sensitivity without a cue, b3 the cue's change in sensitivity,
-b0 the criterion without a cue and -b2 the cue's change in criterion
(positive = more conservative, matching the sign convention of c). Logit-scale
sensitivity is about 1.7 times probit-scale d'. The primary fit is a GEE; see below.
"""
from pathlib import Path
import json
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import patterns_statistics as ps

OUT = Path(__file__).resolve().parent / "patterns_outputs"
h = ps.prepare_human().copy()
h["y"] = h.response_positive.astype(int)
h["s"] = np.where(h.truth_positive, 0.5, -0.5)
h["cue"] = h.cue.astype(int)
# Primary: GEE with item fixed effects and participant-clustered robust covariance.
# s is constant within an item, so the item effects absorb it (and item difficulty);
# the cue's change in criterion (-cue) and in sensitivity (s:cue) remain identified.
gee = smf.gee("y ~ C(question_id) + cue + s:cue", groups="participant_id", data=h,
              family=sm.families.Binomial()).fit()
def gee_term(k, sign=1):
    lo, hi = sorted(sign * gee.conf_int().loc[k])
    return {"estimate": sign * gee.params[k], "ci95": [lo, hi], "p": gee.pvalues[k]}
# Sensitivity check: crossed random intercepts plus a participant random slope on
# signal, by variational Bayes. VB understates posterior variance, and with a
# large participant-slope SD its interval for s:cue is about half the robust one,
# so it is reported but not relied on.
vb = BinomialBayesMixedGLM.from_formula(
    "y ~ s * cue", {"p_int": "0 + C(participant_id)", "p_slope": "0 + C(participant_id):s",
                    "item": "0 + C(question_id)"}, h).fit_vb()
names = vb.model.exog_names
def vb_term(k, sign=1):
    i = names.index(k); m, sd = sign * vb.fe_mean[i], vb.fe_sd[i]
    return {"estimate": m, "ci95": [m - 1.96 * sd, m + 1.96 * sd]}
res = {"n_trials": len(h), "n_participants": int(h.participant_id.nunique()), "n_items": int(h.question_id.nunique()),
       "gee_itemFE_participant_clustered": {"cue_change_in_criterion_logit": gee_term("cue", -1),
                                            "cue_change_in_sensitivity_logit": gee_term("s:cue")},
       "vb_crossed_with_participant_slope": {"sensitivity_no_cue_logit": vb_term("s"),
                                             "cue_change_in_criterion_logit": vb_term("cue", -1),
                                             "cue_change_in_sensitivity_logit": vb_term("s:cue"),
                                             "random_effect_sd": dict(zip(["participant_intercept", "participant_slope", "item"],
                                                                          np.exp(vb.vcp_mean).round(3).tolist()))}}
(OUT / "human_hierarchical_sdt.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))
