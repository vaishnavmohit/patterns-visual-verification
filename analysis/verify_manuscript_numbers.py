#!/usr/bin/env python3
"""Check that numbers quoted in the manuscript match the analysis outputs.

Checks the covered statistical quantities against analysis/patterns_outputs/,
formats them as printed, and checks that each string appears in patterns_main.tex or
patterns_supplementary.tex. Exits non-zero on any mismatch. Run after the
analysis scripts and before building the PDFs:

    machine_result/.venv/bin/python analysis/verify_manuscript_numbers.py

A FAIL means an expected, context-specific numeric string was not found:
the text or output may be stale, or an editorial change may require updating
the checked phrase. Reconcile failures with the source outputs before changing
a check; do not replace contextual checks with searches for bare numbers. It does not check numbers taken from HPC run logs (token
counts, costs, caption-stage counts, probe timings); those are listed in
analysis/README.md with their source.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "analysis" / "patterns_outputs"
raw = (ROOT / "paper/patterns_main.tex").read_text() + (ROOT / "paper/patterns_supplementary.tex").read_text()
TEXT = re.sub(r"\s+", " ", raw.replace("$", "").replace("~", " ").replace("\\,", ""))
results, fails = [], 0
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--aggregate-human", action="store_true",
                    help="Check published human rates from aggregate counts; do not load participant records.")
args = parser.parse_args()


def check(label, *needles):
    global fails
    missing = [n for n in needles if re.sub(r"\s+", " ", n) not in TEXT]
    results.append((label, not missing, missing))
    fails += bool(missing)


# Round half up, as the manuscript does: GPT 5.5's pixel accuracy is exactly
# 285/400 = 71.25% and is printed 71.3, where Python's float formatting gives 71.2.
from decimal import Decimal, ROUND_HALF_UP
def _r(x, nd): return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
def f1(x): return f"{_r(x, 1)}"
def s1(x): return f"{_r(x, 1):+}"
def f2(x): return f"{_r(x, 2)}"
def s0(x): return f"{_r(x, 0):+}"
def s2(x): return f"{_r(x, 2):+}"


NAMES = {"gemini-3.5-flash": "Gemini 3.5 Flash", "gpt-5.5": "GPT 5.5"}
def disp(m):
    if m in NAMES: return NAMES[m]
    v, _, t = m.partition(":"); return (v[:1].upper() + v[1:]).replace("-vl", "-VL") + (":" + t if t else "")

# ---- Table 1 (human accuracy); polarity accuracies are trial-level means
sys.path.insert(0, str(ROOT / "analysis"))
import patterns_statistics as ps
st = json.loads((O / "patterns_statistics.json").read_text())
hp = st["human_participant_accuracy"]; ha = st["human_accuracy"]["primary_item_fixed_participant_clustered"]
check("Table 1 overall", f"Overall accuracy & {f2(100*hp['no_cue_mean_accuracy'])}\\% & {f2(100*hp['cue_mean_accuracy'])}\\%",
      f"p={hp['permutation_p']:.3f}")
fam = pd.read_csv(O / "human_condition_family_polarity.csv")
if args.aggregate_human:
    # Recover integer correct counts from the full-precision family summaries.
    counts = fam.assign(correct_count=np.rint(fam.trials * fam.accuracy).astype(int))
    assert np.allclose(counts.correct_count, counts.trials * counts.accuracy)
    counts = counts.groupby(["truth_positive", "cue"])[["correct_count", "trials"]].sum()
    pol = counts.correct_count / counts.trials
    print("Human checks use published aggregate summaries; participant-level model fits are not rerun.")
else:
    hum = ps.prepare_human()
    pol = hum.groupby(["truth_positive", "cue"]).correct.mean()
check("Table 1 positive row", f"Positive-query accuracy & {f2(100*pol[(True,0)])}\\% & {f2(100*pol[(True,1)])}\\%")
check("Table 1 negative row", f"Negative-query accuracy & {f2(100*pol[(False,0)])}\\% & {f2(100*pol[(False,1)])}\\%")
for key, lab in (("cue_positive_queries", "positive"), ("net_cue_effect_negative_queries", "negative"),
                 ("additional_cue_effect_negative_queries", "interaction")):
    o = ha[key]
    check(f"Table 1 OR {lab}", f"OR {f2(o['odds_ratio'])} [{f2(o['or_ci95'][0])}, {f2(o['or_ci95'][1])}]")

# ---- Results 2.1-2.2 human numbers quoted in prose
check("Results 2.1 polarity sentence", f"({f2(100*pol[(True,0)])}\\% versus {f2(100*pol[(True,1)])}\\% of pooled trials)", f"from {f2(100*pol[(False,0)])}\\% to {f2(100*pol[(False,1)])}\\%")
hn = fam[(fam.category == "hd_novel") & ~fam.truth_positive].set_index("cue").accuracy
check("Results 2.1 hd_novel negatives", f"from {f1(100*hn[0])}\\% without the cue to {f1(100*hn[1])}\\% with the cue")
cells_fam = fam.pivot_table(index=["category", "truth_positive"], columns="cue", values="accuracy")
assert (cells_fam[1] - cells_fam[0]).idxmax() == ("hd_novel", False), "largest family cue difference moved"
hs = st["human_signal_detection"]
dp, cr = hs["dprime"], hs["criterion"]
check("Results 2.2 d'", f"d'={dp['no_cue_mean']:.3f}", f"to {dp['cue_mean']:.3f}",
      f"Delta d'= {dp['difference']:.3f}".replace("Delta d'= ", ""), f"[{dp['ci95'][0]:.3f},{dp['ci95'][1]:.3f}]", f"p = {dp['permutation_p']:.3f}".replace(" ", ""))
check("Results 2.2 criterion", f"c={cr['no_cue_mean']:.3f} to {cr['cue_mean']:.3f}", f"{cr['difference']:+.3f}", f"[{cr['ci95'][0]:.3f},{cr['ci95'][1]:.3f}]")
check("Results 2.2 rates", f"{hs['hit_rate']['cue_mean']:.3f} with the cue versus {hs['hit_rate']['no_cue_mean']:.3f} without", f"fell from {hs['false_alarm_rate']['no_cue_mean']:.3f} to {hs['false_alarm_rate']['cue_mean']:.3f}", f"a reduction of {f1(-100*hs['false_alarm_rate']['difference'])} percentage points")

# ---- Table 3 (both factorial models, no cue): run means, SDs and false-alarm rates
trials = pd.read_csv(O / "factorial_trials.csv")
cells_all = pd.read_csv(O / "factorial_cells_all.csv")
for model in ("gemini-3.5-flash", "gpt-5.5"):
    model_cells = cells_all[cells_all.model.eq(model) & cells_all.cue.eq("none")]
    model_trials = trials[trials.model.eq(model) & trials.cue.eq("none")]
    for rep, lab in (("pixels", "Raw pixels"), ("caption_self", "Self-written descriptions"),
                     ("caption_strong", "Gemini descriptions"),
                     ("oracle_description", "Action Descriptions"), ("program", "Action Programs")):
        g = model_cells[model_cells.representation.eq(rep)]
        if g.empty:
            continue  # Gemini is the reference captioner: no duplicate transfer cell.
        fa = model_trials[model_trials.representation.eq(rep) & model_trials.polarity.eq("neg")].groupby("run").prediction.apply(lambda s: s.eq("pos").mean()).mean()
        check(f"Table 3 {model} {lab}", f"{lab} & {f1(100*g.accuracy.mean())} ({f1(100*g.accuracy.std(ddof=1))}) & "
              f"{f2(g.dprime.mean())} ({f2(g.dprime.std(ddof=1))}) & {s2(g.criterion.mean())} & {f2(fa)}")

# ---- coverage
c = pd.read_csv(O / "submission_coverage.csv"); ret = c[c.retained]
check("retained trials", f"{int(ret.valid.sum()):,} trials")
check("retained cells", f"{len(ret)} of the {len(c)} evaluation cells")
check("parseable decisions", f"{int(c.valid.sum()):,} parseable decisions")

# ---- Table 2 rows and pooled test
r = pd.read_csv(O / "machine_paired_representation_tests.csv")
for _, x in r.iterrows():
    check(f"Table 2 row {x.model}",
          f"{disp(x.model)} & {int(x.n_common_items)} & {f1(100*x.left_accuracy)}\\% & {f1(100*x.right_accuracy)}\\% & {s1(x.right_minus_left_pp)}")
d = r.right_minus_left_pp
check("paired median/mean", f"median of {s1(d.median())}", f"mean of {s1(d.mean())}")
p = binomtest(int((d > 0).sum()), len(d)).pvalue
check("sign test", f"p={p:.1e}".replace("e-0", "\\times10^{-").replace("p=", "p=") + "}")

# ---- paired d'
s = pd.read_csv(O / "machine_paired_sdt.csv").set_index("model")
for m in ("gemini-3.5-flash", "gpt-5.5", "qwen3-vl:30b"):
    check(f"paired d' {m}", f"from {f2(s.loc[m, 'pixel_dprime'])} to {f2(s.loc[m, 'description_dprime'])}")

# ---- pooled no-cue means
cells = pd.read_csv(O / "machine_signal_detection_cells.csv"); no = cells[cells.cue.eq("no_concept")]
mm = no.groupby("representation").dprime.mean()
check("pooled cell means", f"d'={f2(mm['AD'])} for Action Descriptions, {f2(mm['AP'])} for Action Programs, and {f2(mm['DVRL'])} for Direct Visual pixels")

# ---- cue effects by representation
cu = pd.read_csv(O / "machine_paired_cue_tests.csv")
for rep, lab in (("AD", "descriptions"), ("AP", "programs"), ("DVRL", "pixels")):
    g = cu[cu.representation.eq(rep)]
    check(f"cue mean {lab}", s1(g.right_minus_left_pp.mean()))

# ---- alignment
a = json.loads((O / "submission_checks.json").read_text())["alignment"]
check("alignment overall/pos/neg", f"r_s={a['overall']['r']:.3f}", f"r_s={a['positive']['r']:.3f}", f"r_s={a['negative']['r']:.3f}")
pm = a["positive_minus_negative"]
check("alignment difference", f"{pm['difference']:.3f}", f"[{pm['ci95'][0]:.3f},{pm['ci95'][1]:.3f}]")
sens = pd.read_csv(O / "alignment_sensitivity.csv").set_index("comparison")
check("alignment pixels vs descriptions", f"r_s={sens.loc['DVRL, no cue','overall']:.3f} versus {sens.loc['AD, no cue','overall']:.3f}")
fp = sens.loc["AD, fixed model panel (pooled cues)"]
check("alignment fixed panel", f"({fp.positive:.3f} versus {fp.negative:.3f})")

# ---- human hierarchical SDT (GEE)
h = json.loads((O / "human_hierarchical_sdt.json").read_text())["gee_itemFE_participant_clustered"]
cc, ss = h["cue_change_in_criterion_logit"], h["cue_change_in_sensitivity_logit"]
check("human GEE criterion (supplement)", f"logit criterion by {s2(cc['estimate'])} (95\\% CI [{cc['ci95'][0]:.2f},{cc['ci95'][1]:.2f}]")
check("human GEE sensitivity (supplement)", f"logit sensitivity by {s2(ss['estimate'])} (95\\% CI [{ss['ci95'][0]:.2f},{ss['ci95'][1]:.2f}]")

# ---- factorial ladder (run means, no cue)
fc = pd.read_csv(O / "factorial_cells_all.csv")
lad = fc[fc.cue.eq("none")].groupby(["model", "representation"])[["accuracy", "dprime"]].mean()
G, P = "gemini-3.5-flash", "gpt-5.5"
check("Gemini ladder d'", f"{f1(100*lad.loc[(G,'pixels'),'accuracy'])}\\% accuracy (d'={f2(lad.loc[(G,'pixels'),'dprime'])}) on pixels", f"{f1(100*lad.loc[(G,'caption_self'),'accuracy'])}\\% (d'={f2(lad.loc[(G,'caption_self'),'dprime'])}) on its captions", f"{f1(100*lad.loc[(G,'oracle_description'),'accuracy'])}\\% (d'={f2(lad.loc[(G,'oracle_description'),'dprime'])}) on Action Descriptions")
check("GPT ladder", f"GPT achieved {f1(100*lad.loc[(P,'pixels'),'accuracy'])}\\% (d'={f2(lad.loc[(P,'pixels'),'dprime'])}), {f1(100*lad.loc[(P,'caption_self'),'accuracy'])}\\% (d'={f2(lad.loc[(P,'caption_self'),'dprime'])}), {f1(100*lad.loc[(P,'oracle_description'),'accuracy'])}\\% (d'={f2(lad.loc[(P,'oracle_description'),'dprime'])}), and {f1(100*lad.loc[(P,'program'),'accuracy'])}\\% (d'={f2(lad.loc[(P,'program'),'dprime'])})")
for m, lab in ((G, "Gemini"), (P, "GPT")):
    sh = (lad.loc[(m, "caption_self"), "dprime"] - lad.loc[(m, "pixels"), "dprime"]) / (lad.loc[(m, "oracle_description"), "dprime"] - lad.loc[(m, "pixels"), "dprime"])
    check(f"recovered share {lab}", f"{100*sh:+.0f}\\%".replace("+-", "-"))

# ---- cue generalisation across arms
arm = fc.groupby(["model", "representation", "cue"])[["dprime", "criterion"]].mean()
cw, crit = [], []
for (m, rep), g in arm.groupby(level=[0, 1]):
    g = g.droplevel([0, 1])
    if {"none", "correct", "wrong"} <= set(g.index):
        cw.append(g.dprime["correct"] - g.dprime["wrong"])
        crit += [g.criterion[q] - g.criterion["none"] for q in ("correct", "wrong")]
check("cue generalisation", f"in all {['zero','one','two','three','four','five','six','seven','eight','nine'][len(cw)]} proprietary model-by-representation comparisons (margins {s2(min(cw))} to {s2(max(cw))})")
assert all(x > 0 for x in cw) and all(x > 0 for x in crit)

# ---- renderer convention: invisible boundaries (item_identifiability_v2.py)
it = pd.read_csv(O / "item_identifiability_v2.csv")
check("invisible boundaries (queries)", f"{int(it.query_underdetermined.sum())} of the 200 query figures")
check("invisible boundaries (images)", "104 of the 2,504 unique benchmark figures (4.2\\%)")

# ---- Table S12: archived vs renderer-faithful descriptions (complete aliases only)
NAME = {"gemini-3.5-flash": "Gemini 3.5 Flash", "gpt-5.5": "GPT 5.5"}
def name(m): return NAME.get(m, m.replace("gemma4", "Gemma4").replace("qwen3-vl", "Qwen3-VL").replace("qwen", "Qwen"))
def pfmt(p): return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
adr = pd.read_csv(O / "ad_rendered_comparison.csv")
for x in adr.itertuples():
    check(f"Table S12 {x.model}", f"{name(x.model)} & {x.n_both} & {f1(x.acc_original)}\\% & {f1(x.acc_rendered)}\\% & {s1(x.rendered_minus_original_pp)} & {x.mcnemar_p:.2f} & "
          f"{s1(x.gain_original_vs_pixels_pp)} \\to {s1(x.gain_rendered_vs_pixels_pp)}")

# ---- Table 2 answered column, Table S13 and text: answered queries per condition
op = pd.read_csv(O / "operational_paired.csv")
opc = pd.read_csv(O / "operational_cells.csv").query("cue == 'none'").set_index(["model", "interface"])
for x in op.itertuples():
    pa, da = opc.loc[(x.model, "pixels")], opc.loc[(x.model, "descriptions")]
    assert int(pa.decisions) == x.pix_decisions and int(da.decisions) == x.desc_decisions
    check(f"Table S13 {x.model}", f"{name(x.model)} & {x.pix_decisions} & {x.desc_decisions} & {x.matched_items} & {f1(pa.conditional_acc)}\\% & {f1(da.conditional_acc)}\\%")
    check(f"Table 2 answered {x.model}", f"& {x.pix_decisions} / {x.desc_decisions} \\\\")
full = op[(op.pix_decisions >= 197) & (op.desc_decisions >= 197)]
assert len(full) == 10
v30, v32, v8 = (op.set_index("model").loc[m] for m in ("qwen3-vl:30b", "qwen3-vl:32b", "qwen3-vl:8b"))
iv = lambda r: {k: (int(val) if k in ("pix_decisions", "desc_decisions", "matched_items") else val) for k, val in r.items()}
v30, v32, v8 = (pd.Series(iv(r), dtype=object) for r in (v30, v32, v8))
check("answered text (main)", "The other ten answered at least 197 of the 200 queries under both pixels and descriptions",
      f"Qwen3-VL:30b answered all {v30.pix_decisions} pixel queries but {v30.desc_decisions} description queries, and Qwen3-VL:32b answered {v32.pix_decisions} and {v32.desc_decisions}",
      f"({v8.desc_decisions} versus {v8.pix_decisions})")
check("answered text (supplement)", f"Qwen3-VL:30b answered {v30.desc_decisions} of 200 description queries", f"Qwen3-VL:32b answered {v32.pix_decisions} pixel and {v32.desc_decisions} description queries, and Qwen3-VL:8b {v8.pix_decisions} and {v8.desc_decisions}")
v30_desc, v30_pix = (opc.loc[("qwen3-vl:30b", interface)] for interface in ("descriptions", "pixels"))
v30_correct = [int(round(row.decisions * row.conditional_acc / 100)) for row in (v30_desc, v30_pix)]
check("Qwen operational example", f"Qwen3-VL:30b returned {v30_correct[0]} correct description answers across all 200 targets ({f1(v30_desc.operational_success)}\\%), compared with {v30_correct[1]} correct pixel answers ({f1(v30_pix.operational_success)}\\%)")

# ---- cue overlap (cue_overlap_sensitivity.py)
ci = pd.read_csv(O / "cue_overlap_items.csv"); co = pd.read_csv(O / "cue_overlap_sensitivity.csv")
clean = co[co.subset.eq("no shared component")]
dw = clean[clean.contrast.eq("dprime_correct_minus_wrong")]
W = ['zero','one','two','three','four','five','six','seven','eight','nine']
check("cue overlap counts", f"to {int(ci.overlap.sum())} of the 200 queries", f"the other {int((~ci.overlap).sum())} queries")
check("cue overlap d'", f"margins of {s2(dw.estimate.min())} to {s2(dw.estimate.max())}", f"{W[int((dw.ci_low>0).sum())]} of nine confidence intervals excluding zero")
cc = clean[clean.contrast.eq("c_correct_minus_none")]; cwn = clean[clean.contrast.eq("c_wrong_minus_none")]
check("cue overlap criterion", f"in {W[int((cc.estimate>0).sum())]} of nine conditions and mismatched labels in {W[int((cwn.estimate>0).sum())]} of nine")
assert int((cc.ci_low > 0).sum()) == 1 and int((cwn.ci_low > 0).sum()) == 1

# ---- caption fidelity (caption_fidelity.py)
fs = pd.read_csv(O / "caption_fidelity_summary.csv").query("figures == 'all'").set_index("model")
fp = pd.read_csv(O / "caption_fidelity_prediction.csv").set_index("model")
check("fidelity F1", f"Texture-by-kind F_1 was {fs.token_f1[G]:.2f} for Gemini and {fs.token_f1[P]:.2f} for GPT")
check("fidelity counts", f"exact stroke counts matched on only {100*fs.strokes_match[G]:.0f}\\% and {100*fs.strokes_match[P]:.0f}\\% of figures")
check("fidelity prediction", f"log-odds slopes {s2(fp.logit_per_sd[G])}, 95\\% CI [{fp.ci_low[G]:.2f},{fp.ci_high[G]:.2f}], and {s2(fp.logit_per_sd[P])}, [{fp.ci_low[P]:.2f},{fp.ci_high[P]:.2f}]")

# ---- planned trial-level model (factorial_planned_model.py)
pm = pd.read_csv(O / "factorial_planned_model.csv")
def est(m, c): return pm[(pm.model == m) & pm.contrast.str.startswith(c)].gee.iloc[0]
assert f2(est(G, "1")) == f2(est(P, "1"))
check("planned model", f"pixel-to-caption shifts of {s2(est(G, '1'))} log-odds in both models", f"caption-to-description shifts of {s2(est(G, '2'))} for Gemini and {s2(est(P, '2'))} for GPT")

# ---- open-weight replication (factorial_ow_analysis.py): Tables S19, S20 and text
owc = pd.read_csv(O / "factorial_ow_cells.csv").query("cue == 'none'").set_index(["model", "representation"])
owr = pd.read_csv(O / "factorial_ow_results.csv"); ows = pd.read_csv(O / "factorial_ow_results_strict.csv")
REPN = [("pixels", "Raw pixels"), ("caption_self", "Own step captions"), ("caption_self_free", "Own prose captions"),
        ("oracle_description", "Action Descriptions"), ("oracle_description_rendered", "Renderer-faithful descriptions"),
        ("program", "Action Programs")]
def pct(x): return "100\\%" if x >= 0.9995 else f"{f1(100*x)}\\%"
for mdl in ("gemma4:31b", "qwen3.6:27b"):
    for rep_, lab in REPN:
        x = owc.loc[(mdl, rep_)]
        check(f"Table S19 {mdl} {rep_}", f"{lab} & {f1(100*x.accuracy)}\\% & {f2(x.dprime)} & {s2(x.criterion)} & {pct(x.decision_rate)}")
def r_(df, mdl, con):
    return df[(df.model == mdl) & (df.contrast == con)].iloc[0]
def iv(x, sign=True):
    a = s2(x.estimate) if sign else f2(x.estimate)
    return f"{a} [{f2(x.ci_low)}, {f2(x.ci_high)}]".replace("[-", "[-").replace(", -", ", -")
for mdl in ("gemma4:31b", "qwen3.6:27b"):
    for con in ("d' own captions - pixels", "d' benchmark descriptions - pixels", "recovered share", "d' prose captions - step captions",
                "recovered share, prose captions (exploratory)", "d' rendered - archived descriptions"):
        check(f"Table S20 {mdl} {con}", iv(r_(owr, mdl, con)))
    check(f"Table S20 {mdl} strict share", iv(r_(ows, mdl, "recovered share")))
    assert r_(owr, mdl, "recovered share").verdict == "large recovery excluded" == r_(ows, mdl, "recovered share").verdict
g, q = r_(owr, "gemma4:31b", "recovered share"), r_(owr, "qwen3.6:27b", "recovered share")
check("replication text shares", f"{s0(100*g.estimate)}\\% (95\\% CI [{100*g.ci_low:+.0f}\\%,{100*g.ci_high:+.0f}\\%])", f"{s0(100*q.estimate)}\\% (95\\% CI [{100*q.ci_low:+.0f}\\%,{100*q.ci_high:+.0f}\\%])")
p3 = owr[owr.question.eq("P3a")]; assert (p3.estimate > 0).all() and len(p3) == 6 and int((p3.ci_low > 0).sum()) == 3
for x in p3.itertuples():
    check(f"Table S20 P3a {x.model} {x.contrast}", iv(x))
ir = owr[owr.contrast.str.endswith("c irrelevant - none")]; assert len(ir) == 6 and int(((ir.ci_low > 0) | (ir.ci_high < 0)).sum()) == 1
ww = owr[owr.contrast.str.endswith("c wrong - wrong_clean")]; assert len(ww) == 6 and (ww.estimate > 0).all()
check("replication partly correct labels", f"in all six conditions (\\Delta c from {s2(ww.estimate.min())} to {s2(ww.estimate.max())})")
cl = owr[owr.contrast.str.endswith("c wrong_clean - none")]
check("replication clean criterion", f"ranged from {s2(cl.estimate.min())} to {s2(cl.estimate.max())}")
assert ((cl.ci_low < 0) & (cl.ci_high > 0)).all()

# ---- replication recovery accounting and coverage (factorial_ow_trials.csv)
owt = pd.read_csv(O / "factorial_ow_trials.csv")
owt = owt[~owt.representation.str.contains(r"_d\d$")]   # the 40 primary cells; additional caption sets (A3, A4) are reported separately
qq = owt[owt.model.eq("qwen3.6:27b")]
empty_strict = int(qq.prediction_strict.isna().sum()); recovered = int((qq.prediction_strict.isna() & qq.prediction_rescored.notna()).sum())
left = int(qq.prediction_rescored.isna().sum()); assert empty_strict == recovered + left
assert int(owt[owt.model.eq("gemma4:31b")].prediction_strict.isna().sum()) == 0
check("replication recovery", f"recovered {recovered:,} of the {empty_strict:,} Qwen3.6:27b trials that failed strict JSON parsing", f"leaving {left} unanswered Qwen trials")
cov = pd.read_csv(O / "factorial_ow_cells.csv").query("model == 'qwen3.6:27b' and representation == 'caption_self' and cue != 'none'").decision_rate
check("replication cue coverage", f"{f1(100*cov.min())}--{f1(100*cov.max())}\\% of cued step-recipe trials")

# ---- exploratory common-returned-query sensitivity and operational replication reporting
matched = pd.read_csv(O / "factorial_ow_matched_sensitivity.csv")
qm = matched.query("model == 'qwen3.6:27b' and contrast == 'common-returned recipe recovery'").iloc[0]
check("Qwen matched recovery", f"{qm.estimate:.3f} (95\\% CI [{qm.ci_low:.3f},{qm.ci_high:.3f}])")
check("Qwen matched recovery denominators", f"{int(qm.greedy_queries)} greedy and {int(qm.sampled_queries)} sampled queries for Qwen3.6:27b")
qpair = matched.query("model == 'qwen3.6:27b' and contrast == 'common-returned prose minus step dprime'").iloc[0]
check("Qwen matched protocol sensitivity", f"{qpair.estimate:+.3f} ([{qpair.ci_low:.3f},{qpair.ci_high:.3f}])")
for mdl in ("gemma4:31b", "qwen3.6:27b"):
    mm = matched.query("model == @mdl and contrast == 'common-returned recipe recovery'").iloc[0]
    if mdl == "gemma4:31b":
        primary = r_(owr, mdl, "recovered share")
        assert np.allclose(mm[["estimate", "ci_low", "ci_high"]].astype(float),
                           primary[["estimate", "ci_low", "ci_high"]].astype(float), atol=1e-6)
operational = pd.read_csv(O / "factorial_ow_operational.csv")
qp = operational.query("scoring == 'primary_A2' and model == 'qwen3.6:27b' and cue == 'none'").set_index("representation")
assert (~operational.representation.str.contains(r"_d\d$")).sum() == 80 and operational.attempted.eq(400).all()   # 40 primary conditions, two scorings, plus extra caption sets
assert np.allclose(operational.operational_accuracy, operational.correct / operational.attempted)

# ---- open-weight caption fidelity (caption_structure_ow.py): Table S22 and text
cs = pd.read_csv(O / "caption_structure_ow_summary.csv").set_index(["model", "format"])
cf = pd.read_csv(O / "caption_structure_ow_figures.csv")
def said_closed(m, closed):
    p = cf[cf.model.eq(m) & cf.format.eq("prose") & cf.true_parts.eq(1) & cf.claimed_closed.notna()]
    return p[p.true_closed.astype(bool) == closed].claimed_closed.astype(bool).mean()
def closure_baseline(m):
    p = cf[cf.model.eq(m) & cf.format.eq("prose") & cf.true_parts.eq(1) & cf.claimed_closed.notna()]
    assert p.true_closed.notna().all() and set(p.true_closed.unique()) == {False, True}
    return p.true_closed.astype(bool).mean()
def closure_balanced_accuracy(m):
    return (said_closed(m, True) + 1 - said_closed(m, False)) / 2
GM, QM = "gemma4:31b", "qwen3.6:27b"
rows21 = [("Step recipes: exact stroke count", lambda m: f"{f1(100*cs.loc[(m,'step')].strokes_match)}\\%"),
          ("Step recipes: texture-by-kind F1", lambda m: f2(cs.loc[(m,'step')].token_f1)),
          ("Correct shape count: step recipes", lambda m: f"{f1(100*cs.loc[(m,'step')].parts_correct_overall)}\\%"),
          ("Correct shape count: prose", lambda m: f"{f1(100*cs.loc[(m,'prose')].parts_correct_overall)}\\%"),
          ("Prose closure stated (single-shape images)", lambda m: f"{f1(100*cs.loc[(m,'prose')].closure_stated)}\\%"),
          ("Prose closure correct, when stated", lambda m: f"{f1(100*cs.loc[(m,'prose')].closure_correct_of_stated)}\\%"),
          ("Always-closed baseline, when stated", lambda m: f"{f1(100*closure_baseline(m))}\\%"),
          ("Prose closure balanced accuracy, when stated", lambda m: f"{f1(100*closure_balanced_accuracy(m))}\\%"),
          ("Says closed: closed / open shapes", lambda m: f"{f1(100*said_closed(m, True))}\\% / {f1(100*said_closed(m, False))}\\%"),
          ("Slope of correctness gain on shape-count gain", lambda m: f"{s2(cs.loc[(m,'step')].slope_d_parts)} [{f2(cs.loc[(m,'step')].ci_low)}, {f2(cs.loc[(m,'step')].ci_high)}]")]
for lab, fn in rows21:
    check(f"Table S22 {lab}", f"{lab} & {fn(GM)} & {fn(QM)}")
check("fidelity text (main)", f"exact stroke counts on {f1(100*cs.loc[(GM,'step')].strokes_match)}\\% of Gemma's images and {f1(100*cs.loc[(QM,'step')].strokes_match)}\\% of Qwen's", f"{f1(100*cs.loc[(GM,'prose')].parts_correct_overall)}\\% versus {f1(100*cs.loc[(GM,'step')].parts_correct_overall)}\\% for Gemma", f"Gemma slope {s2(cs.loc[(GM,'step')].slope_d_parts)}, 95\\% CI [{f2(cs.loc[(GM,'step')].ci_low)},{f2(cs.loc[(GM,'step')].ci_high)}]")
check("fidelity text (supplement)", f"{f1(100*cs.loc[(GM,'prose')].parts_stated)}\\% of Gemma's and {f1(100*cs.loc[(QM,'prose')].parts_stated)}\\% of Qwen's captions")
check("Closure baseline text", f"an always-closed answer would be correct on {f1(100*closure_baseline(GM))}\\% of Gemma's figures and {f1(100*closure_baseline(QM))}\\% of Qwen's", f"balanced accuracies were {f1(100*closure_balanced_accuracy(GM))}\\% and {f1(100*closure_balanced_accuracy(QM))}\\%")

# ---- amendment A3: independent caption draw (factorial_ow_analysis.py, caption_structure_ow.py); Table S23 and text
A3K = {"c2": "d' prose - step captions, draw 2", "pool": "d' prose - step captions, pooled over draws", "dd": "draw 2 minus draw 1 contrast",
       "rst": "recovered share, step captions, draw 2", "rs": "recovered share, prose captions, draw 2"}
a3 = {m: owr.query("model == @m and question == 'A3'").set_index("contrast") for m in (GM, QM)}
a3s = {m: ows.query("model == @m and question == 'A3'").set_index("contrast") for m in (GM, QM)}
assert np.allclose(a3[GM][["estimate", "ci_low", "ci_high"]].astype(float), a3s[GM].loc[a3[GM].index, ["estimate", "ci_low", "ci_high"]].astype(float))
assert a3[GM].loc[A3K["c2"], "verdict"] == "not replicated" and a3[QM].loc[A3K["c2"], "verdict"] == "replicated"
d2c = pd.read_csv(O / "caption_structure_ow_d2_summary.csv").set_index(["model", "format"])
owcs = pd.read_csv(O / "factorial_ow_cells_strict.csv").query("cue == 'none'").set_index(["model", "representation"])
def tm(x): return f"-{f2(-x)}" if _r(x, 2) < 0 else f2(x)
def p0(x): return f"{_r(100*x, 0)}"
def a3row(m, k): x = a3[m].loc[A3K[k]]; return f"{s2(x.estimate)} [{tm(x.ci_low)}, {tm(x.ci_high)}]"
def a3ci(m, k): x = a3[m].loc[A3K[k]]; return f"[{tm(x.ci_low)},{tm(x.ci_high)}]"
def a3e(m, k): return a3[m].loc[A3K[k]]
cs2 = {m: (owc.loc[(m, "caption_self_d2")], owc.loc[(m, "caption_self_free_d2")]) for m in (GM, QM)}
def accd(m, i): c = cs2[m][i]; return f"{f1(100*c.accuracy)}\\% ({f2(c.dprime)})"
rows22 = [("Decisions returned: step / prose", lambda m: f"{f1(100*cs2[m][0].decision_rate)}\\% / {f1(100*cs2[m][1].decision_rate)}\\%"),
          ("Step recipes: accuracy (d')", lambda m: accd(m, 0)), ("Prose: accuracy (d')", lambda m: accd(m, 1)),
          ("d' prose - step", lambda m: a3row(m, "c2")), ("d' prose - step, mean of both draws", lambda m: a3row(m, "pool")),
          ("Prose - step, second minus primary draw", lambda m: a3row(m, "dd")),
          ("Recovered share, step recipes", lambda m: a3row(m, "rst")), ("Recovered share, prose", lambda m: a3row(m, "rs")),
          ("Step recipes: exact stroke count", lambda m: f"{f1(100*d2c.loc[(m,'step')].strokes_match)}\\%"),
          ("Step recipes: texture-by-kind F1", lambda m: f2(d2c.loc[(m,'step')].token_f1)),
          ("Correct shape count: step recipes", lambda m: f"{f1(100*d2c.loc[(m,'step')].parts_correct_overall)}\\%"),
          ("Correct shape count: prose", lambda m: f"{f1(100*d2c.loc[(m,'prose')].parts_correct_overall)}\\%"),
          ("Prose closure stated (single-shape images)", lambda m: f"{f1(100*d2c.loc[(m,'prose')].closure_stated)}\\%"),
          ("Prose closure correct, when stated", lambda m: f"{f1(100*d2c.loc[(m,'prose')].closure_correct_of_stated)}\\%"),
          ("Slope of correctness gain on shape-count gain", lambda m: f"{s2(d2c.loc[(m,'step')].slope_d_parts)} [{tm(d2c.loc[(m,'step')].ci_low)}, {tm(d2c.loc[(m,'step')].ci_high)}]")]
for lab, fn in rows22:
    check(f"Table S23 {lab}", f"{lab} & {fn(GM)} & {fn(QM)} \\\\")
gs, gp = cs2[GM]; qs_, qp = cs2[QM]
q_p2 = owr.query("model == @QM and question == 'P2'").estimate.iloc[0]
q_e = owr.query("model == @QM and question == 'E' and contrast.str.startswith('recovered share, prose')", engine="python").estimate.iloc[0]
slope2 = lambda m: f"{s2(d2c.loc[(m,'step')].slope_d_parts)} ([{tm(d2c.loc[(m,'step')].ci_low)},{tm(d2c.loc[(m,'step')].ci_high)}])"
# Main text now reports all four caption sets (A4); set-2 details are checked in the supplement.
A4R = owr.set_index(["model", "contrast"]); A4S = ows.set_index(["model", "contrast"])
A4M = "d' prose - step captions, mean of four sets"
SETN = ["primary set", "set 2", "set 3", "set 4"]
a4c = {m: [A4R.loc[(m, f"d' prose - step captions, {n}")] for n in SETN] for m in (GM, QM)}
a4ci = lambda x: f"[{tm(x.ci_low)},{tm(x.ci_high)}]"
a4 = lambda m, k: A4R.loc[(m, k)]
assert sum(x.estimate > 0 for m in (GM, QM) for x in a4c[m]) == 8 and sum(x.ci_low > 0 for m in (GM, QM) for x in a4c[m]) == 2
assert all(a4(m, A4M).ci_low > 0 for m in (GM, QM)) and all(a4(m, A4M).verdict == "robust to caption sampling" for m in (GM, QM))
assert all(a4(m, A4M + ", sets as random").ci_low < 0 for m in (GM, QM))
tG, tQ = a4(GM, "temperature-1 sets mean minus temperature-0 set"), a4(QM, "temperature-1 sets mean minus temperature-0 set")
assert a4c[GM][0].estimate == max(x.estimate for x in a4c[GM]) and a4c[QM][0].estimate == min(x.estimate for x in a4c[QM])
sdG, sdQ = a4(GM, "between-set SD of d' prose - step").estimate, a4(QM, "between-set SD of d' prose - step").estimate
lst = lambda m: f"{s2(a4c[m][1].estimate)}, {s2(a4c[m][2].estimate)} and {s2(a4c[m][3].estimate)}"
stp = {m: [a4(m, k) for k in ("recovered share, step captions, draw 2", "recovered share, step captions, set 3", "recovered share, step captions, set 4")] for m in (GM, QM)}
pro = {m: [a4(m, k) for k in ("recovered share, prose captions (exploratory)", "recovered share, prose captions, draw 2", "recovered share, prose captions, set 3", "recovered share, prose captions, set 4")] for m in (GM, QM)}
dd = {d: pd.read_csv(O / f"caption_structure_ow{d}_summary.csv").set_index(["model", "format"]) for d in ("", "_d2", "_d3", "_d4")}
pr = lambda m, fm: [100 * dd[d].loc[(m, fm)].parts_correct_overall for d in dd]
rg = lambda xs: f"{f1(min(xs))}--{f1(max(xs))}"
sq2, sq3 = dd["_d2"].loc[(QM, "step")], dd["_d3"].loc[(QM, "step")]
assert [(m, d) for m in (GM, QM) for d in dd if dd[d].loc[(m, "step")].ci_low > 0 or dd[d].loc[(m, "step")].ci_high < 0] == [(QM, "_d2")]
q3s = stp[QM][1]
assert sum(x.ci_high >= 0.5 for m in (GM, QM) for x in stp[m]) == 1 and q3s.ci_high >= 0.5
check("A4 text (main)",
      f"In Gemma, the prose-minus-recipe contrast was {lst(GM)} in these sets, against {s2(a4c[GM][0].estimate)} in the primary set",
      f"In Qwen, it was {lst(QM)}, against {s2(a4c[QM][0].estimate)}.",
      f"prose exceeded recipes by \\Delta d'={s2(a4(GM, A4M).estimate)} (95\\% CI {a4ci(a4(GM, A4M))}) in Gemma and {s2(a4(QM, A4M).estimate)} ({a4ci(a4(QM, A4M))}) in Qwen",
      f"gave intervals of {a4ci(a4(GM, A4M + ', sets as random'))} and {a4ci(a4(QM, A4M + ', sets as random'))}, with between-set standard deviations of {f2(sdG)} and {f2(sdQ)}",
      f"{f2(-tG.estimate)} above the mean of its sampled sets (95\\% CI [{f2(-tG.ci_high)},{f2(-tG.ci_low)}]), but Qwen's smallest",
      f"Qwen's four-set mean was {s2(A4S.loc[(QM, A4M)].estimate)} ({a4ci(A4S.loc[(QM, A4M)])})",
      f"recovered shares ranged from {s0(100*min(x.estimate for x in stp[GM]))}\\% to {s0(100*max(x.estimate for x in stp[GM]))}\\% in Gemma and from {s0(100*min(x.estimate for x in stp[QM]))}\\% to {s0(100*max(x.estimate for x in stp[QM]))}\\% in Qwen; only Qwen's third set had an upper limit above one half (95\\% CI [{_r(100*q3s.ci_low,0)}\\%,{_r(100*q3s.ci_high,0)}\\%])",
      f"shape counts were correct for {rg(pr(GM,'prose'))}\\% of prose and {rg(pr(GM,'step'))}\\% of recipe captions, while its prose contrast ranged from {s2(min(x.estimate for x in a4c[GM]))} to {s2(max(x.estimate for x in a4c[GM]))}",
      f"counted shapes correctly on {rg(pr(QM,'prose'))}\\% of images",
      f"excluded zero ({s2(sq2.slope_d_parts)}, [{tm(sq2.ci_low)},{tm(sq2.ci_high)}]); its third set gave {s2(sq3.slope_d_parts)} ([{tm(sq3.ci_low)},{tm(sq3.ci_high)}])",
      f"prose recovered between {_r(100*min(x.estimate for x in pro[GM]),0)}\\% and {_r(100*max(x.estimate for x in pro[GM]),0)}\\% of the sensitivity gap across Gemma's sets and between {_r(100*min(x.estimate for x in pro[QM]),0)}\\% and {_r(100*max(x.estimate for x in pro[QM]),0)}\\% across Qwen's",
      f"Between-set standard deviations of {f2(sdG)} and {f2(sdQ)} in the open-weight prose contrasts",
      "Across four tested caption sets, prose captions discriminated better than recipes on average in both open-weight models")
assert (a3e(GM, "c2").ci_low < 0 < a3e(QM, "c2").ci_low) and all(a3e(m, "c2").estimate > 0 for m in (GM, QM))
assert owr.query("model == @QM and question == 'P2'").ci_low.iloc[0] < 0 < owr.query("model == @GM and question == 'P2'").ci_low.iloc[0]
qd, qst = a3e(QM, "dd"), a3s[QM].loc[A3K["c2"]]
owt_d2 = pd.read_csv(O / "factorial_ow_trials.csv").query("representation.str.endswith('_d2')", engine="python")
fbq = owt_d2.query("model == @QM").groupby("representation").parse_fallback.sum()
check("A3 text (supplement)", f"The prose-minus-step contrast was {s2(a3e(GM,'c2').estimate)} ({a3ci(GM,'c2')}) and did not meet the criterion",
      f"It fell by {f2(-a3e(GM,'dd').estimate)} (95\\% CI [{f2(-a3e(GM,'dd').ci_high)},{f2(-a3e(GM,'dd').ci_low)}]) from the primary draw",
      f"lower prose sensitivity (d'={f2(gp.dprime)} versus {f2(owc.loc[(GM,'caption_self_free')].dprime)}) and higher recipe sensitivity ({f2(gs.dprime)} versus {f2(owc.loc[(GM,'caption_self')].dprime)})",
      f"prose exceeded recipes by {s2(a3e(GM,'pool').estimate)} ({a3ci(GM,'pool')})",
      f"correct for {f1(100*d2c.loc[(GM,'prose')].parts_correct_overall)}\\% of prose and {f1(100*d2c.loc[(GM,'step')].parts_correct_overall)}\\% of recipe captions, and prose stated closure for {f1(100*d2c.loc[(GM,'prose')].closure_stated)}\\%",
      f"less often in prose ({f1(100*d2c.loc[(QM,'prose')].parts_correct_overall)}\\%) and more often in recipes ({f1(100*d2c.loc[(QM,'step')].parts_correct_overall)}\\%)",
      f"the contrast was {s2(a3e(GM,'c2').estimate)} ({a3ci(GM,'c2')}) for Gemma4:31b and {s2(a3e(QM,'c2').estimate)} ({a3ci(QM,'c2')}) for Qwen3.6:27b (Table S23)",
      f"Qwen3.6:27b returned decisions on {f1(100*qs_.decision_rate)}\\% of second-draw recipe trials and {f1(100*qp.decision_rate)}\\% of prose trials, {int(fbq['caption_self_d2'])} and {int(fbq['caption_self_free_d2'])} of 400 through final-conclusion recovery",
      f"Its prose-minus-step contrast was {s2(a3e(QM,'c2').estimate)} ({a3ci(QM,'c2')}), {f2(qd.estimate)} ([{tm(qd.ci_low)},{tm(qd.ci_high)}]) above the primary draw, reflecting both higher prose sensitivity (d'={f2(qp.dprime)} versus {f2(owc.loc[(QM,'caption_self_free')].dprime)}) and lower recipe sensitivity ({f2(qs_.dprime)} versus {f2(owc.loc[(QM,'caption_self')].dprime)})",
      f"Strict scoring, which retained {f1(100*owcs.loc[(QM,'caption_self_d2')].decision_rate)}\\% and {f1(100*owcs.loc[(QM,'caption_self_free_d2')].decision_rate)}\\% of these trials, gave {s2(qst.estimate)} ([{tm(qst.ci_low)},{tm(qst.ci_high)}])",
      f"Averaged over the two draws, the contrast was {s2(a3e(QM,'pool').estimate)} ({a3ci(QM,'pool')})",
      f"(slope {s2(d2c.loc[(QM,'step')].slope_d_parts)}, [{tm(d2c.loc[(QM,'step')].ci_low)},{tm(d2c.loc[(QM,'step')].ci_high)}])")
assert all(d2c.loc[(m, 'step')].ci_low < 0 for m in (GM,)) and d2c.loc[(QM, 'step')].ci_low > 0
assert len(owt_d2) == 1600 and owt_d2.query("model == @GM").prediction_rescored.notna().all() and owt_d2.query("model == @GM").parse_fallback.eq(0).all()
check("A3 coverage (supplement)", "Gemma4:31b returned decisions on all 800 second-draw trials, none through final-conclusion recovery")
t4 = owt_all = pd.read_csv(O / "factorial_ow_trials.csv")
q34 = owt_all[owt_all.model.eq(QM) & owt_all.representation.str.contains(r"_d[34]$")]
g34 = owt_all[owt_all.model.eq(GM) & owt_all.representation.str.contains(r"_d[34]$")]
assert len(q34) == len(g34) == 1600 and g34.prediction_rescored.notna().all() and g34.parse_fallback.eq(0).all()
qrate = q34.groupby(["representation", "run"]).prediction_rescored.apply(lambda s: s.notna().mean())
cellA4 = lambda x: (f"{s2(x.estimate)} [{tm(x.ci_low)}, {tm(x.ci_high)}]")
owc_all = pd.read_csv(O / "factorial_ow_cells.csv").query("cue == 'none'").set_index(["model", "representation"])
for lab, suf, n in (("Primary (greedy)", "", "primary set"), ("Set 2", "_d2", "set 2"), ("Set 3", "_d3", "set 3"), ("Set 4", "_d4", "set 4")):
    vals = []
    for m in (GM, QM):
        vals += [f2(owc_all.loc[(m, "caption_self" + suf)].dprime), f2(owc_all.loc[(m, "caption_self_free" + suf)].dprime), cellA4(A4R.loc[(m, f"d' prose - step captions, {n}")])]
    check(f"Table S24 {lab}", f"{lab} & " + " & ".join(vals))
for lab, k_ in (("Mean of four sets", A4M), ("Mean, sets as random", A4M + ", sets as random"), ("Temperature-1 mean - primary", "temperature-1 sets mean minus temperature-0 set")):
    check(f"Table S24 {lab}", f"{lab} & & & {cellA4(A4R.loc[(GM, k_)])} & & & {cellA4(A4R.loc[(QM, k_)])}")
check("Table S24 SD", f"Between-set SD & & & {f2(sdG)} & & & {f2(sdQ)}")
mq3 = pd.read_csv(O / "factorial_ow_matched_sensitivity.csv").set_index(["model", "contrast"])
check("A4 text (supplement)",
      f"The four-set mean was {s2(A4R.loc[(GM, A4M)].estimate)} ({a4ci(A4R.loc[(GM, A4M)])}) for Gemma4:31b and {s2(A4R.loc[(QM, A4M)].estimate)} ({a4ci(A4R.loc[(QM, A4M)])}) for Qwen3.6:27b",
      "Gemma4:31b returned every decision in sets 3 and 4, none through final-conclusion recovery",
      f"Qwen3.6:27b returned {f1(100*qrate.min())}--{f1(100*qrate.max())}\\% of decisions per run, {_r(100*q34.parse_fallback.sum()/q34.prediction_rescored.notna().sum(),0)}\\% of them",
      f"with a sets-as-random interval of {a4ci(A4S.loc[(QM, A4M + ', sets as random')])}",
      f"set-3 and set-4 contrasts on queries answered under both protocols were {s2(mq3.loc[(QM, 'common-returned prose minus step dprime, set 3')].estimate)} ({a4ci(mq3.loc[(QM, 'common-returned prose minus step dprime, set 3')])}) and {s2(mq3.loc[(QM, 'common-returned prose minus step dprime, set 4')].estimate)} ({a4ci(mq3.loc[(QM, 'common-returned prose minus step dprime, set 4')])})",
      f"was {s2(tG.estimate)} ({a4ci(tG)}) for Gemma4:31b and {s2(tQ.estimate)} ({a4ci(tQ)}) for Qwen3.6:27b",
      f"Qwen3.6:27b's item-level slopes were {s2(sq3.slope_d_parts)} ([{tm(sq3.ci_low)},{tm(sq3.ci_high)}]) and {s2(dd['_d4'].loc[(QM,'step')].slope_d_parts)} ([{tm(dd['_d4'].loc[(QM,'step')].ci_low)},{tm(dd['_d4'].loc[(QM,'step')].ci_high)}])",
      f"Qwen3.6:27b's third set gave {s2(sq3.slope_d_parts)} ([{tm(sq3.ci_low)},{tm(sq3.ci_high)}])")

# ---- set 3-4 recovered shares and slopes (supplement source for main-text ranges); renderer rerun baselines
for m_, lab in ((GM, "Gemma4:31b"), (QM, "Qwen3.6:27b")):
    for proto in ("step", "prose"):
        x3, x4 = (A4R.loc[(m_, f"recovered share, {proto} captions, set {k}")] for k in (3, 4))
        check(f"A4 shares {m_} {proto}", f"{s2(x3.estimate)} ([{tm(x3.ci_low)},{tm(x3.ci_high)}]) and {s2(x4.estimate)} ([{tm(x4.ci_low)},{tm(x4.ci_high)}]) for {lab}'s {'recipes' if proto == 'step' else 'prose'}"
              if (m_, proto) == (GM, "step") else f"{s2(x3.estimate)} ([{tm(x3.ci_low)},{tm(x3.ci_high)}]) and {s2(x4.estimate)} ([{tm(x4.ci_low)},{tm(x4.ci_high)}]) for {'its prose' if proto == 'prose' else lab + chr(39) + 's recipes'}")
g3, g4 = dd["_d3"].loc[(GM, "step")], dd["_d4"].loc[(GM, "step")]
check("A4 Gemma slopes", f"Gemma4:31b's item-level slopes in sets 3 and 4 were {s2(g3.slope_d_parts)} ([{tm(g3.ci_low)},{tm(g3.ci_high)}]) and {s2(g4.slope_d_parts)} ([{tm(g4.ci_low)},{tm(g4.ci_high)}])")
hist = opc.xs("descriptions", level="interface")
check("renderer rerun baselines", "(" + ", ".join(f"{f1(hist.loc[m_].conditional_acc)}\\%" for m_ in ("gemma4:26b", "gemma4:31b", "qwen3.5:35b")) + f", and {f1(hist.loc['qwen3.6:27b'].conditional_acc)}\\% on answered queries)")
check("four-set estimand timing", "The four-set average was specified before sets 3 and 4 were generated", "The four-set estimand was specified after the second-set results were known and before sets 3 and 4 were generated")

# ---- Table S18: proprietary cue contrasts (cue_overlap_sensitivity.py); Table S20 criterion rows
cs18 = pd.read_csv(O / "cue_overlap_sensitivity.csv").set_index(["model", "representation", "subset", "contrast"])
def tcell(e, lo, hi):
    sg = lambda x: f"-{f2(-x)}" if _r(x, 2) < 0 else f"+{f2(x)}"
    bb = lambda x: f"-{f2(-x)}" if _r(x, 2) < 0 else f2(x)
    return f"{sg(e)} [{bb(lo)}, {bb(hi)}]"
REP18 = {"pixels": "Raw pixels", "caption_self": "Own step captions", "caption_strong": "Gemini captions", "oracle_description": "Action Descriptions", "program": "Action Programs"}
for (m_, rep, sub), g in cs18.groupby(level=[0, 1, 2]):
    vals = [tcell(*cs18.loc[(m_, rep, sub, k), ["estimate", "ci_low", "ci_high"]]) for k in ("dprime_correct_minus_wrong", "c_correct_minus_none", "c_wrong_minus_none")]
    check(f"Table S18 {m_} {rep} {sub}", f"{REP18[rep]} & " + " & ".join(vals))
allp = cs18.xs("all items", level="subset")
d18 = allp.xs("dprime_correct_minus_wrong", level="contrast").estimate
assert (d18 > 0).all() and (allp.xs("c_correct_minus_none", level="contrast").estimate > 0).all() and (allp.xs("c_wrong_minus_none", level="contrast").estimate > 0).all()
nine = d18
assert len(nine) == 9
check("proprietary cue margins", f"(margins {s2(nine.min())} to {s2(nine.max())})")
for rep, rl in (("pixels", "pixels"), ("caption_self", "step captions"), ("oracle_description", "descriptions")):
    for lab, key in (("c correct - no cue", "c correct - none"), ("c clean mismatched - no cue", "c wrong_clean - none"), ("c irrelevant text - no cue", "c irrelevant - none"), ("c original - clean mismatched", "c wrong - wrong_clean")):
        g_, q_ = (owr.query("model == @mm and contrast == @cc").iloc[0] for mm, cc in ((GM, f"{rep}: {key}"), (QM, f"{rep}: {key}")))
        check(f"Table S20 {lab} {rep}", f"{lab}: {rl} & {tcell(g_.estimate, g_.ci_low, g_.ci_high)} & {tcell(q_.estimate, q_.ci_low, q_.ci_high)}")
ocl = owr[owr.contrast.str.endswith("c wrong - wrong_clean")]
assert (ocl.estimate > 0).all() and (ocl.ci_low > 0).sum() == 3

# ---- Discussion and Results additions (audit revision): alignment, recipe range, worked examples
al = json.loads((O / "submission_checks.json").read_text())["alignment"]; apm = al["positive_minus_negative"]
check("alignment (main Results)", f"(Spearman r_s={al['overall']['r']:.2f}, 95\\% CI [{al['overall']['ci95'][0]:.2f},{al['overall']['ci95'][1]:.2f}]; exploratory)",
      f"(r_s={al['positive']['r']:.2f} versus {al['negative']['r']:.2f}; difference {apm['difference']:.2f}, [{apm['ci95'][0]:.2f},{apm['ci95'][1]:.2f}]; Figure S1)")
check("alignment (Discussion)", f"found largely the same positive queries difficult (r_s={al['positive']['r']:.2f}) but agreed less on negative ones ({al['negative']['r']:.2f})")
qrs = [A4R.loc[(QM, k)].estimate for k in ("recovered share", "recovered share, step captions, draw 2", "recovered share, step captions, set 3", "recovered share, step captions, set 4")]
check("Qwen recipe range (Discussion)", f"Qwen3.6:27b recovered {_r(100*min(qrs),0)}--{_r(100*max(qrs),0)}\\% across its four recipe sets")
cl = pd.read_csv(O / "factorial_cells_all.csv").query("model == 'gpt-5.5' and cue == 'none'").groupby("representation").dprime.mean()
check("GPT on Gemini captions (Discussion)", f"reached d'={f2(cl['caption_strong'])}, below its own pixels ({f2(cl['pixels'])}) and recipes ({f2(cl['caption_self'])})")
check("Discussion stroke-count fidelity", "Recipes matched the true stroke count on 39\\% and 52\\% of figures for Gemini and GPT")

# ---- Table S17 rows (non-termination; moved from main manuscript)
nt = pd.read_csv(O / "nontermination_by_cell.csv", comment="#")
LAB = {"Pix": "Raw pixels", "AD": "Descriptions", "AP": "Programs"}
for (m, i), g in nt.groupby(["model", "interface"]):
    g = g.set_index("family"); cellv = [f"{100*g.loc[f,'rate']:.0f}" if f in g.index else "--" for f in ("bd","ff","hd_comb","hd_novel")]
    tot = f"{int(g.non_terminating.sum())}/{int(g.attempted.sum())} ({100*g.non_terminating.sum()/g.attempted.sum():.0f}\\%)"
    check(f"Table S17 {m} {i}", f"{LAB[i]} & " + " & ".join(cellv) + f" & {tot}")
foot = open(O / "nontermination_by_cell.csv").read().strip().splitlines()[-1]
kv = dict(x.split("=") for x in foot.lstrip("# ").split())
check("non-termination totals", f"all {int(kv['records']):,} recorded", f"{int(kv['distinct_cells']):,} unique item-by-cue", f"{kv['reattempted']} out of {kv['reattempted']} failed to terminate again")

# ---- Qwen3.6:35b no-input rows and their rerun
qr = pd.read_csv(O / "qwen36_blind_rerun.csv").set_index("treatment")
ex, rp, ar = qr.loc["blind rows excluded"], qr.loc["blind rows replaced by rerun"], qr.loc["archived (primary scoring)"]
check("Qwen3.6 rerun (main)", f"The smallest gain, {s1(ar.gain_pp)} points for Qwen3.6:35b, includes 35 no-cue description trials", f"a description advantage of {s1(rp.gain_pp)} percentage points over pixels on {int(rp.n_items)} answered queries (p={rp.exact_mcnemar_p:.3f}, exploratory")
check("Qwen3.6 rerun (exclusion)", f"Omitting the affected trials instead gives {s1(ex.gain_pp)} percentage points on {int(ex.n_items)} shared queries")
assert int(ar.n_items) - int(ex.n_items) == 35 and int(rp.n_items) == int(ar.n_items) - 5
check("Qwen3.6 rerun (supplement)", f"was {s1(rp.gain_pp)} percentage points on {int(rp.n_items)} answered queries (p={rp.exact_mcnemar_p:.3f}, exploratory)",
      f"the initial evaluation: {s1(ar.gain_pp)} percentage points on {int(ar.n_items)} shared queries (p={ar.exact_mcnemar_p:.3f})")

# ---- Table S21: repository-planned open-weight GEE secondary
owg = pd.read_csv(O / "factorial_ow_gee.csv")
assert (~owg.question.isin(["A3", "A4"])).sum() == 252 and owg.converged.all() and owg.warnings.isna().all()
assert np.isfinite(owg[["estimate", "ci_low", "ci_high", "p"]].to_numpy()).all()
primary_gee = owg.query("scoring == 'primary_A2' and estimand == 'logit sensitivity'").set_index(["model", "contrast"])
gee_rows = {
    "step minus pixels": "Step captions $-$ pixels",
    "descriptions minus pixels": "Descriptions $-$ pixels",
    "descriptions minus step": "Descriptions $-$ step captions",
    "prose minus step": "Prose $-$ step captions",
    "pixels: correct minus clean": "Correct $-$ clean label: pixels",
    "caption_self: correct minus clean": "Correct $-$ clean label: step captions",
    "oracle_description: correct minus clean": "Correct $-$ clean label: descriptions",
    "renderer-faithful minus archived": "Renderer-faithful $-$ original",
}
def gee_cell(row):
    def endpoint(x):
        return f"${f2(x)}$" if x < 0 else f2(x)
    return f"${s2(row.estimate)}$ [{endpoint(row.ci_low)}, {endpoint(row.ci_high)}]"
for contrast, label in gee_rows.items():
    a, b = (primary_gee.loc[(model, contrast)] for model in ("gemma4:31b", "qwen3.6:27b"))
    check(f"Table S21 {contrast}", f"{label} & {gee_cell(a)} & {gee_cell(b)}".replace("$", ""))
strict_protocol = owg.query("scoring == 'strict' and model == 'qwen3.6:27b' and contrast == 'prose minus step' and estimand == 'logit sensitivity'").iloc[0]
check("GEE strict Qwen protocol", f"{s2(strict_protocol.estimate)}", f"[{f2(strict_protocol.ci_low)},{f2(strict_protocol.ci_high)}]")
msq = pd.read_csv(O / "factorial_ow_matched_sensitivity.csv").set_index(["model", "contrast"]).loc[(QM, "common-returned prose minus step dprime, second draw")]
opq = operational.query("scoring == 'primary_A2' and model == @QM and cue == 'none'").set_index("representation")
ga3 = owg.query("question == 'A3' and estimand == 'logit sensitivity'").set_index(["scoring", "model"])
gci = lambda r: f"{s2(r.estimate)} ([{('-' + f2(-r.ci_low)) if r.ci_low < 0 else f2(r.ci_low)},{f2(r.ci_high)}])"
check("A3 robustness (supplement)", f"({int(msq.greedy_queries)} greedy and {int(msq.sampled_queries)} sampled), Qwen3.6:27b's second-draw contrast was {s2(msq.estimate)} ([{f2(msq.ci_low)},{f2(msq.ci_high)}])",

      f"second draw were {gci(ga3.loc[('primary_A2', GM)])} for Gemma4:31b and {gci(ga3.loc[('primary_A2', QM)])} for Qwen3.6:27b; strict scoring gave {gci(ga3.loc[('strict', QM)])}")
check("GEE completion", f"{len(owg)} estimates from {len(owg) // 3 * 2} fits")
base_polarity = owt[["test_id", "polarity"]].drop_duplicates()
base_polarity = base_polarity.groupby(base_polarity.test_id.str.replace(r"_(pos|neg)$", "", regex=True)).polarity.nunique()
assert (base_polarity == 1).sum() == 184 and (base_polarity == 2).sum() == 8
check("GEE polarity across base problems", f"within {int((base_polarity == 1).sum())} of the {len(base_polarity)} base problems")


# Additional prose checks protect quantities and denominators obscured by the
# incoming language revision; table checks alone cannot catch these errors.
main_text = re.sub(r"\s+", " ", (ROOT / "paper/patterns_main.tex").read_text().replace("$", "").replace("~", " "))
def check_main(label, *needles):
    global TEXT
    combined = TEXT
    try:
        TEXT = main_text
        check(label, *needles)
    finally:
        TEXT = combined
gm_prose = owc.loc[(GM, "caption_self_free")]
check_main("Main Gemma prose sensitivity", f"{f1(100*gm_prose.accuracy)}\\% (d'={f2(gm_prose.dprime)}) on prose")
check_main("Main prose closure denominator", f"Gemma's prose stated closure on {f1(100*cs.loc[(GM,'prose')].closure_stated)}\\% and classified it correctly on {f1(100*cs.loc[(GM,'prose')].closure_correct_of_stated)}\\% of those where closure was stated")
inventory = cf[cf.model.eq(GM) & cf.format.eq("prose")]
check("Closure inventory denominator", f"{int(inventory.true_parts.eq(1).sum()):,} single-shape and {int(inventory.true_parts.eq(2).sum()):,} two-shape figures")

for label, ok, missing in results:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"   missing: {missing}"))
print(f"\n{len(results) - fails}/{len(results)} checks pass")
sys.exit(1 if fails else 0)
