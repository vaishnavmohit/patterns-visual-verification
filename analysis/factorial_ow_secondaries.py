#!/usr/bin/env python3
"""Complete the repository-planned open-weight GEE robustness analysis.

Uses the fixed export (16,000 primary attempts and 1,600 attempts per
additional caption set of amendments A3 and A4), not new model calls. Fits each model
separately, under primary A2 and strict scoring, among returned decisions.
Response models check P1--P4 on the logit sensitivity/criterion scales;
correctness models describe accuracy contrasts. These do not estimate probit
d-prime, recovered-share ratios or a model-by-protocol interaction.
The plan names the existing factorial_planned_model.py form; this adaptation
adds the replication's conditions and a fixed run effect. Fits were performed
after the primary results were known and are secondary robustness analyses.
The A3/A4 rows (prose minus step in each additional caption set) were added
after the A3 results; a set enters once both of its cells are complete.
"""
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT = Path(__file__).resolve().parent / "patterns_outputs"


def contrasts():
    yield "P1", "step minus pixels", "pixels", "none", "caption_self", "none"
    yield "P1", "descriptions minus pixels", "pixels", "none", "oracle_description", "none"
    yield "P1", "descriptions minus step", "caption_self", "none", "oracle_description", "none"
    yield "P2", "prose minus step", "caption_self", "none", "caption_self_free", "none"
    yield "A3", "prose minus step, second draw", "caption_self_d2", "none", "caption_self_free_d2", "none"
    yield "A4", "prose minus step, set 3", "caption_self_d3", "none", "caption_self_free_d3", "none"
    yield "A4", "prose minus step, set 4", "caption_self_d4", "none", "caption_self_free_d4", "none"
    yield "P4", "renderer-faithful minus archived", "oracle_description", "none", "oracle_description_rendered", "none"
    yield "secondary", "programs minus descriptions", "oracle_description", "none", "program", "none"
    for rep in ("pixels", "caption_self", "oracle_description"):
        yield "P3a", f"{rep}: correct minus clean", rep, "wrong_clean", rep, "correct"
        for cue in ("correct", "wrong_clean", "irrelevant"):
            yield "P3b", f"{rep}: {cue} minus none", rep, "none", rep, cue
        yield "secondary", f"{rep}: original minus clean", rep, "wrong_clean", rep, "wrong"


def main():
    source = OUT / "factorial_ow_trials.csv"
    trials = pd.read_csv(source)
    extra = trials.representation.str.contains(r"_d\d$")
    assert (~extra).sum() == 16_000 and trials.groupby(["model", "representation", "cue"]).size().eq(400).all()
    assert not trials.duplicated(["model", "representation", "cue", "run", "test_id"]).any()
    assert trials.polarity.eq(np.where(trials.test_id.str.endswith("_pos"), "pos", "neg")).all()
    trials["uid"] = trials.test_id.str.replace(r"_(pos|neg)$", "", regex=True)
    trials["s"] = np.where(trials.polarity.eq("pos"), 0.5, -0.5)
    rows = []
    for scoring, column in (("primary_A2", "prediction_rescored"), ("strict", "prediction_strict")):
        assert trials[column].dropna().isin(["pos", "neg"]).all()
        for model, panel in trials.groupby("model"):
            for question, label, rep_a, cue_a, rep_b, cue_b in contrasts():
                a = panel[panel.representation.eq(rep_a) & panel.cue.eq(cue_a)].copy()
                b = panel[panel.representation.eq(rep_b) & panel.cue.eq(cue_b)].copy()
                if question in ("A3", "A4") and (a.empty or b.empty):
                    continue   # caption set not (yet) complete for this model
                assert len(a) == len(b) == 400
                a["intervention"], b["intervention"] = 0.0, 1.0
                attempted = pd.concat([a, b], ignore_index=True)
                d = attempted[attempted[column].notna()].copy()
                d["resp_pos"] = d[column].eq("pos").astype(int)
                d["correct"] = d[column].eq(d.polarity).astype(int)
                for outcome in ("resp_pos", "correct"):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit = smf.gee(f"{outcome} ~ s * intervention + C(run)",
                                      groups="uid", data=d, family=sm.families.Binomial(),
                                      cov_struct=sm.cov_struct.Exchangeable()).fit()
                    assert fit.converged and np.isfinite(fit.params).all(), (model, label, outcome)
                    assert np.isfinite(fit.cov_params().to_numpy()).all(), (model, label, outcome)
                    ci = fit.conf_int()
                    terms = (("s:intervention", "logit sensitivity", 1),
                             ("intervention", "logit criterion", -1)) if outcome == "resp_pos" else (
                                 ("intervention", "polarity-averaged log-odds correct", 1),)
                    for term, estimand, sign in terms:
                        low, high = sorted(sign * ci.loc[term].to_numpy())
                        rows.append(dict(scoring=scoring, model=model, question=question,
                                         contrast=label, estimand=estimand,
                                         estimate=sign * fit.params[term], ci_low=low, ci_high=high,
                                         p=fit.pvalues[term], attempted=len(attempted), n_trials=len(d),
                                         n_base_problems=d.uid.nunique(),
                                         decisions_a=int(a[column].notna().sum()),
                                         decisions_b=int(b[column].notna().sum()), converged=fit.converged,
                                         warnings="; ".join(str(w.message) for w in caught)))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "factorial_ow_gee.csv", index=False)
    manifest = dict(analysis_date="2026-10-14", plan_commit="ec3efd2",
                    status="repository-planned secondary, fitted after primary results were known",
                    formula="outcome ~ polarity(+/-0.5) * intervention + C(run)",
                    correlation="exchangeable, robust sandwich intervals clustered by base problem",
                    missingness="returned decisions within each condition; no imputation",
                    scoring=["primary_A2", "strict"], python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, statsmodels=statsmodels.__version__,
                    sha256={str(p.relative_to(OUT.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source, Path(__file__), OUT / "factorial_ow_gee.csv")})
    (OUT / "factorial_ow_gee_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result.query("scoring == 'primary_A2' and question in ['P1', 'P2', 'A3', 'A4', 'P4'] and estimand == 'logit sensitivity'").round(3).to_string(index=False))
    print(f"{len(result)} estimates; all fits converged; {result.warnings.ne('').sum()} rows with warnings")


if __name__ == "__main__":
    main()
