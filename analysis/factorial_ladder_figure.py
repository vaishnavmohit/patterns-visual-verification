#!/usr/bin/env python3
"""Figure 4: representation ladder and cue controls for all four factorial models.

Inputs: patterns_outputs/factorial_cells_all.csv (Gemini 3.5 Flash and GPT 5.5,
one row per cell and run) and patterns_outputs/factorial_ow_trials.csv (the
open-weight replication, rescored decisions; d' and c computed here per run with
the (x+0.5)/(n+1) correction). Writes paper/figures/patterns_factorial_ladder.pdf
and a PNG copy.

A  No-cue d' at each representation: mean over runs with run-to-run SD. Prose
   self-captions exist only for the open-weight models, other-model captions
   only for GPT 5.5.
B  Change in d' from no cue, and C  change in criterion c from no cue, under
   each cue type, within pixels, own step captions and benchmark descriptions.
   One point per model-by-representation condition; clean mismatched labels and
   irrelevant text exist only for the open-weight models.
Colours follow the model; markers follow the model too, so identity never rests
on colour alone (palette validated with the dataviz skill's checker).
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
MODELS = [("gemini-3.5-flash", "Gemini 3.5 Flash", "#2a78d6", "o"),
          ("gpt-5.5", "GPT 5.5", "#eb6834", "s"),
          ("gemma4:31b", "Gemma4:31b", "#1baf7a", "^"),
          ("qwen3.6:27b", "Qwen3.6:27b", "#4a3aa7", "D")]
REPS = [("pixels", "Raw\npixels"), ("caption_self", "Own captions\n(steps)"),
        ("caption_self_free", "Own captions\n(prose)"), ("caption_strong", "Other model's\ncaptions"),
        ("oracle_description", "Benchmark\ndescriptions"), ("program", "Benchmark\nprograms")]
CUE_REPS = ("pixels", "caption_self", "oracle_description")
CUES = [("correct", "Correct\nlabel"), ("wrong", "Mis-\nmatched,\noriginal"),
        ("wrong_clean", "Mis-\nmatched,\nclean"), ("irrelevant", "Irrel-\nevant\ntext")]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"


def per_run():
    """One row per model x representation x cue x run with d' and c."""
    a = pd.read_csv(OUT / "factorial_cells_all.csv")[["model", "representation", "cue", "run", "dprime", "criterion"]]
    t = pd.read_csv(OUT / "factorial_ow_trials.csv").dropna(subset=["prediction_rescored"])
    rows = []
    for (m, rep, cue, run), g in t.groupby(["model", "representation", "cue", "run"]):
        pos, resp = g.polarity.eq("pos"), g.prediction_rescored.eq("pos")
        h = ((pos & resp).sum() + .5) / (pos.sum() + 1); f = ((~pos & resp).sum() + .5) / ((~pos).sum() + 1)
        rows.append(dict(model=m, representation=rep, cue=cue, run=run,
                         dprime=norm.ppf(h) - norm.ppf(f), criterion=-(norm.ppf(h) + norm.ppf(f)) / 2))
    return pd.concat([a, pd.DataFrame(rows)], ignore_index=True)


def main(wide=False):
    d = per_run()
    means = d.groupby(["model", "representation", "cue"])[["dprime", "criterion"]].mean()
    base = 11 if wide else 8
    plt.rcParams.update({"font.size": base, "pdf.fonttype": 42, "axes.edgecolor": MUTED, "xtick.color": INK,
                         "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK})
    if wide:   # widescreen slides: three panels in one row
        fig = plt.figure(figsize=(15.5, 5.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.9, 1, 1], wspace=0.42)
        ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    else:      # one text column: A across the top, B and C side by side below
        fig = plt.figure(figsize=(6.6, 5.7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.55, wspace=0.42)
        ax_a = fig.add_subplot(gs[0, :]); ax_b = fig.add_subplot(gs[1, 0]); ax_c = fig.add_subplot(gs[1, 1])
    fs = base - 0.5

    # A: ladder
    for i, (model, name, col, mk) in enumerate(MODELS):
        sub = d[(d.model == model) & (d.cue == "none")]
        xs, ms, sds = [], [], []
        for k, (rep, _) in enumerate(REPS):
            g = sub[sub.representation == rep].dprime
            if len(g):
                xs.append(k + (i - 1.5) * 0.08); ms.append(g.mean()); sds.append(g.std(ddof=1) if len(g) > 1 else 0.0)
        ax_a.errorbar(xs, ms, yerr=sds, color=col, marker=mk, ms=8 if wide else 5.5, lw=0, elinewidth=1.1, capsize=2.5,
                      markeredgecolor="white", markeredgewidth=0.8, label=name, zorder=3)
        idx = [round(x - (i - 1.5) * 0.08) for x in xs]
        for (x0, m0, k0), (x1, m1, k1) in zip(zip(xs, ms, idx), zip(xs[1:], ms[1:], idx[1:])):
            if k1 - k0 == 1:   # connect only adjacent rungs a model has
                ax_a.plot([x0, x1], [m0, m1], color=col, lw=1.6, zorder=2)
    ax_a.set_xticks(range(len(REPS))); WIDE = {"Benchmark\ndescriptions": "Supplied\ntext", "Benchmark\nprograms": "Supplied\nprograms",
                                             "Other model's\ncaptions": "Other\ncaptions"}
    ax_a.set_xticklabels([WIDE.get(l, l.replace(" ", "\n", 1)) if wide else l for _, l in REPS], fontsize=fs - (1 if wide else 0))
    # The legend identifies every model; leave the last rung clear of labels.
    ax_a.set_xlim(-0.4, len(REPS) - 1 + 0.4)
    ax_a.set_ylabel("Sensitivity $d'$ without a cue")
    ax_a.set_title("A  Representation ladder", loc="left", fontsize=base + 1, color=INK)
    ax_a.legend(frameon=False, fontsize=fs, loc="upper left", handletextpad=0.4, ncol=2, columnspacing=1.2)

    # B, C: cue effects, one point per model x representation condition
    rng = np.random.default_rng(4)
    for ax, var, title, ylab in ((ax_b, "dprime", "B  Cue effect on sensitivity", "Change in $d'$ from no cue"),
                                 (ax_c, "criterion", "C  Cue effect on criterion", "Change in $c$ from no cue\n(positive = more conservative)")):
        for k, (cue, _) in enumerate(CUES):
            for i, (model, name, col, mk) in enumerate(MODELS):
                for j, rep in enumerate(CUE_REPS):
                    try:
                        v = means.loc[(model, rep, cue), var] - means.loc[(model, rep, "none"), var]
                    except KeyError:
                        continue
                    x = k + (i - 1.5) * 0.15 + (j - 1) * 0.04
                    ax.scatter(x, v, color=col, marker=mk, s=60 if wide else 26, edgecolor="white", linewidth=0.7, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        ax.set_xticks(range(len(CUES))); ax.set_xticklabels([l for _, l in CUES], fontsize=fs - (1.5 if wide else 0)); ax.set_xlim(-0.5, len(CUES) - 0.5)
        ax.set_title(title, loc="left", fontsize=base + 1, color=INK); ax.set_ylabel(ylab)
    for ax in (ax_a, ax_b, ax_c):
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if wide:
        fig.savefig(OUT / "patterns_factorial_ladder_wide.png", dpi=150, bbox_inches="tight")
    else:
        (ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(ROOT / "paper" / "figures" / "patterns_factorial_ladder.pdf", bbox_inches="tight")
        fig.savefig(OUT / "patterns_factorial_ladder.png", dpi=180, bbox_inches="tight")
    print("figure written")


if __name__ == "__main__":
    import sys
    main(wide="--wide" in sys.argv)
