#!/usr/bin/env python3
"""Figure 5: prose-minus-step sensitivity in each of the four open-weight caption sets.

Input: patterns_outputs/factorial_ow_results.csv (factorial_ow_analysis.py, question
A4). One panel per model: the primary (greedy) set and sets 2-4 (temperature 1,
seeds 2-4) with 95% base-problem bootstrap intervals, then the four-set mean with
its specified interval (thick, sets held fixed) and the sets-as-random interval
(thin, mean +/- t(0.975, 3) x SD / 2). Values are those of Table S23.

    machine_result/.venv/bin/python analysis/factorial_caption_sets_figure.py
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
MODELS = [("gemma4:31b", "Gemma4:31b", "#1baf7a"), ("qwen3.6:27b", "Qwen3.6:27b", "#4a3aa7")]
SETS = [("primary set", "Primary set (greedy)"), ("set 2", "Set 2"), ("set 3", "Set 3"), ("set 4", "Set 4")]
MEAN = "d' prose - step captions, mean of four sets"
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e3"


def main():
    r = pd.read_csv(OUT / "factorial_ow_results.csv").query("question == 'A4'").set_index(["model", "contrast"])
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED, "xtick.color": INK,
                         "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True, sharex=True)
    ys = [5.6, 4.6, 3.6, 2.6]           # caption sets, top to bottom
    y_mean = 1.0                        # summary row, set apart
    shown = [f"d' prose - step captions, {k}" for k, _ in SETS] + [MEAN, MEAN + ", sets as random"]
    sub = r[r.index.get_level_values("contrast").isin(shown)]
    lo_all = min(sub.ci_low.min(), -0.2); hi_all = sub.ci_high.max()
    for ax, (model, name, col) in zip(axes, MODELS):
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
        for y, (key, _) in zip(ys, SETS):
            x = r.loc[(model, f"d' prose - step captions, {key}")]
            ax.plot([x.ci_low, x.ci_high], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
            greedy = key == "primary set"
            ax.plot(x.estimate, y, "o", ms=6.5, mfc="white" if greedy else col, mec=col, mew=1.6, zorder=3)
            ax.text(hi_all + 0.12, y, f"{x.estimate:+.2f}", va="center", ha="left", fontsize=7.5, color=INK)
        fixed, rand = r.loc[(model, MEAN)], r.loc[(model, MEAN + ", sets as random")]
        ax.plot([rand.ci_low, rand.ci_high], [y_mean, y_mean], color=col, lw=1, zorder=2)
        ax.plot([fixed.ci_low, fixed.ci_high], [y_mean, y_mean], color=col, lw=4.5, solid_capstyle="round", zorder=2, alpha=0.9)
        ax.plot(fixed.estimate, y_mean, "D", ms=7.5, mfc=col, mec="white", mew=1.4, zorder=4)
        ax.text(hi_all + 0.12, y_mean, f"{fixed.estimate:+.2f}", va="center", ha="left", fontsize=7.5, color=INK, fontweight="bold")
        ax.axhline(1.8, color=GRID, lw=0.8)
        ax.set_title(name, fontsize=8.5, color=INK, pad=4)
        ax.set_xlim(lo_all - 0.08, hi_all + 0.42)
        ax.set_ylim(0.3, 6.2)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(r"Prose $-$ step captions, $\Delta d'$")
    axes[0].set_yticks(ys + [y_mean])
    axes[0].set_yticklabels([lab for _, lab in SETS] + ["Mean of four sets"])
    fig.tight_layout(w_pad=1.2)
    fig.savefig(ROOT / "paper" / "figures" / "patterns_caption_sets.pdf", bbox_inches="tight")
    fig.savefig(OUT / "patterns_caption_sets.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
