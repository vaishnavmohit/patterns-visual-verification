#!/usr/bin/env python3
"""Patterns-oriented statistical analysis for the matched human-machine study.

The script separates finite-sample claims about the evaluated models from
population claims about participants. It adds item-adjusted human inference,
crossed random-intercept sensitivity checks, robust RT models,
signal-detection measures, and within-model/item representation comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest, norm, spearmanr
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "data" / "public" / "human_trials.csv"
MACHINE = ROOT / "machine_result" / "clean_machine_data.csv"
OUT = ROOT / "analysis" / "patterns_outputs"
PAPER_FIGURES = ROOT / "paper" / "figures"
SEED = 20260807


def native(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [native(item) for item in value]
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def model_term(result, term: str) -> dict:
    ci = result.conf_int().loc[term]
    beta = float(result.params[term])
    return {
        "beta": beta,
        "se": float(result.bse[term]),
        "p": float(result.pvalues[term]),
        "odds_ratio": float(math.exp(beta)),
        "or_ci95": [float(math.exp(ci.iloc[0])), float(math.exp(ci.iloc[1]))],
    }


def linear_contrast(result, weights: dict[str, float], scale: str = "odds_ratio") -> dict:
    vector = np.zeros(len(result.params))
    for term, weight in weights.items():
        vector[result.params.index.get_loc(term)] = weight
    estimate = float(vector @ result.params)
    se = float(np.sqrt(vector @ result.cov_params() @ vector))
    z_value = estimate / se
    p_value = float(2 * norm.sf(abs(z_value)))
    lower, upper = estimate - 1.96 * se, estimate + 1.96 * se
    output = {"beta": estimate, "se": se, "p": p_value}
    if scale == "odds_ratio":
        output.update(
            {
                "odds_ratio": float(math.exp(estimate)),
                "or_ci95": [float(math.exp(lower)), float(math.exp(upper))],
            }
        )
    elif scale == "percent_change":
        output.update(
            {
                "percent_change": float((math.exp(estimate) - 1) * 100),
                "ci95_percent_change": [
                    float((math.exp(lower) - 1) * 100),
                    float((math.exp(upper) - 1) * 100),
                ],
            }
        )
    return output


def permutation_bootstrap_difference(a, b, rng, n_iter=20_000):
    """Return b-a with a two-sided permutation p and percentile bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = float(b.mean() - a.mean())
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
    permutations = np.empty(n_iter)
    bootstraps = np.empty(n_iter)
    for index in range(n_iter):
        shuffled = rng.permutation(labels)
        permutations[index] = values[shuffled == 1].mean() - values[shuffled == 0].mean()
        bootstraps[index] = rng.choice(b, len(b), replace=True).mean() - rng.choice(
            a, len(a), replace=True
        ).mean()
    p_value = (np.sum(np.abs(permutations) >= abs(observed)) + 1) / (n_iter + 1)
    return {
        "difference": observed,
        "ci95": np.quantile(bootstraps, [0.025, 0.975]).tolist(),
        "permutation_p": float(p_value),
    }


def signal_detection(frame: pd.DataFrame, truth_col: str, response_col: str) -> dict:
    positive = frame[frame[truth_col]]
    negative = frame[~frame[truth_col]]
    hits = int(positive[response_col].sum())
    false_alarms = int(negative[response_col].sum())
    hit_rate = (hits + 0.5) / (len(positive) + 1)
    false_alarm_rate = (false_alarms + 0.5) / (len(negative) + 1)
    z_hit = norm.ppf(hit_rate)
    z_false_alarm = norm.ppf(false_alarm_rate)
    return {
        "n_positive": int(len(positive)),
        "n_negative": int(len(negative)),
        "hit_rate": float(hit_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "dprime": float(z_hit - z_false_alarm),
        "criterion": float(-0.5 * (z_hit + z_false_alarm)),
    }


def prepare_human() -> pd.DataFrame:
    frame = pd.read_csv(HUMAN, low_memory=False)
    frame["correct"] = pd.to_numeric(frame["is_correct"])
    frame["cue"] = frame["concept_condition"].eq("concept_shown").astype(int)
    frame["truth_positive"] = ~frame["question_id"].str.endswith("_neg")
    frame["negative"] = (~frame["truth_positive"]).astype(int)
    frame["response_positive"] = frame["answer"].eq("positive")
    frame["position_z"] = (
        frame["question_number"] - frame["question_number"].mean()
    ) / frame["question_number"].std()
    frame["rt"] = pd.to_numeric(frame["derived_rt_seconds"], errors="coerce")
    return frame


def human_accuracy_models(frame: pd.DataFrame) -> dict:
    primary = smf.gee(
        "correct ~ cue + cue:negative + C(question_id) + position_z",
        groups="participant_id",
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit(maxiter=100)

    participant_cluster = smf.gee(
        "correct ~ cue * negative + C(category) + position_z",
        groups="participant_id",
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    item_cluster = smf.gee(
        "correct ~ cue * negative + C(category) + position_z",
        groups="question_id",
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()

    crossed = BinomialBayesMixedGLM.from_formula(
        "correct ~ cue * negative + C(category) + position_z",
        {
            "participant": "0 + C(participant_id)",
            "item": "0 + C(question_id)",
        },
        frame,
        vcp_p=0.5,
        fe_p=2,
    )
    crossed_result = crossed.fit_vb(verbose=False)
    crossed_terms = {
        name: {
            "posterior_mean": float(mean),
            "posterior_sd": float(sd),
            "or": float(math.exp(mean)),
            "or_interval95_normal": [
                float(math.exp(mean - 1.96 * sd)),
                float(math.exp(mean + 1.96 * sd)),
            ],
        }
        for name, mean, sd in zip(
            crossed.exog_names, crossed_result.fe_mean, crossed_result.fe_sd
        )
    }

    return {
        "primary_item_fixed_participant_clustered": {
            "description": "GEE logistic model with item fixed effects and participant-clustered robust covariance",
            "cue_positive_queries": model_term(primary, "cue"),
            "additional_cue_effect_negative_queries": model_term(primary, "cue:negative"),
            "net_cue_effect_negative_queries": linear_contrast(
                primary, {"cue": 1, "cue:negative": 1}
            ),
            "n_trials": int(len(frame)),
            "n_participants": int(frame["participant_id"].nunique()),
            "n_items": int(frame["question_id"].nunique()),
        },
        "sensitivity_participant_clustered": {
            term: model_term(participant_cluster, term)
            for term in ["cue", "negative", "cue:negative"]
        },
        "sensitivity_item_clustered": {
            term: model_term(item_cluster, term)
            for term in ["cue", "negative", "cue:negative"]
        },
        "sensitivity_crossed_random_intercepts_vb": crossed_terms,
    }


def human_participant_accuracy(frame: pd.DataFrame, rng) -> dict:
    """Reproduce the unadjusted participant-level primary contrast."""
    participants = (
        frame.groupby(["participant_id", "cue"], as_index=False)["correct"].mean()
    )
    hidden = participants.loc[participants["cue"] == 0, "correct"]
    shown = participants.loc[participants["cue"] == 1, "correct"]
    comparison = permutation_bootstrap_difference(hidden, shown, rng)
    return {
        "no_cue_mean_accuracy": float(hidden.mean()),
        "cue_mean_accuracy": float(shown.mean()),
        "cue_minus_no_cue_percentage_points": float(comparison["difference"] * 100),
        "ci95_percentage_points": [float(value * 100) for value in comparison["ci95"]],
        "permutation_p": comparison["permutation_p"],
        "n_no_cue": int(len(hidden)),
        "n_cue": int(len(shown)),
    }

def human_rt_models(frame: pd.DataFrame) -> dict:
    results = {}
    for cap in [120, 180, 300]:
        subset = frame[
            (frame["question_number"] > 1)
            & frame["rt"].between(1, cap, inclusive="both")
        ].copy()
        subset["log_rt"] = np.log(subset["rt"])
        fitted = smf.mixedlm(
            "log_rt ~ cue * negative + C(category) + position_z",
            subset,
            # A constant group makes both variance components genuinely crossed.
            # Grouping by participant would instead nest items within people.
            groups=np.ones(len(subset)),
            vc_formula={
                "participant": "0 + C(participant_id)",
                "item": "0 + C(question_id)",
            },
            re_formula="0",
            use_sparse=True,
        ).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        terms = {}
        for term in ["cue", "negative", "cue:negative"]:
            ci = fitted.conf_int().loc[term]
            beta = float(fitted.params[term])
            terms[term] = {
                "beta_log_seconds": beta,
                "percent_change": float((math.exp(beta) - 1) * 100),
                "ci95_percent_change": [
                    float((math.exp(ci.iloc[0]) - 1) * 100),
                    float((math.exp(ci.iloc[1]) - 1) * 100),
                ],
                "p": float(fitted.pvalues[term]),
            }
        terms["net_cue_effect_negative"] = linear_contrast(
            fitted, {"cue": 1, "cue:negative": 1}, scale="percent_change"
        )
        results[f"cap_{cap}_seconds"] = {
            "n_trials": int(len(subset)),
            "converged": bool(fitted.converged),
            "random_effects": "crossed participant and item intercepts",
            "variance_components": dict(zip(fitted.model.exog_vc.names, fitted.vcomp.tolist())),
            "residual_variance": float(fitted.scale),
            "terms": terms,
        }
    return results


def human_sdt(frame: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (participant, cue), subset in frame.groupby(["participant_id", "cue"]):
        values = signal_detection(subset, "truth_positive", "response_positive")
        rows.append({"participant_id": participant, "cue": int(cue), **values})
    participant_sdt = pd.DataFrame(rows)
    summary = {}
    for metric in ["dprime", "criterion", "hit_rate", "false_alarm_rate"]:
        hidden = participant_sdt.loc[participant_sdt["cue"] == 0, metric]
        shown = participant_sdt.loc[participant_sdt["cue"] == 1, metric]
        comparison = permutation_bootstrap_difference(hidden, shown, rng)
        summary[metric] = {
            "no_cue_mean": float(hidden.mean()),
            "cue_mean": float(shown.mean()),
            **comparison,
        }
    return participant_sdt, summary


def prepare_machine() -> pd.DataFrame:
    frame = pd.read_csv(MACHINE, low_memory=False)
    frame = frame[frame["prediction_valid"].astype(str).str.lower().eq("true")].copy()
    frame = frame[
        frame["clean_primary_keep"].astype(str).str.lower().eq("true")
    ].copy()
    frame["correct"] = pd.to_numeric(frame["correct_int"])
    frame["truth_positive"] = frame["polarity_label"].eq("Positive")
    frame["response_positive"] = frame["prediction"].eq("pos")
    return frame


def human_machine_item_alignment(
    human: pd.DataFrame, machine: pd.DataFrame
) -> dict:
    """Correlate human and clean-primary machine AD accuracy by matched item."""
    human_items = (
        human.groupby(["question_id", "truth_positive"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "human_accuracy"})
    )
    machine_items = (
        machine[machine["representation"].eq("AD")]
        .groupby("test_id", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "machine_accuracy"})
    )
    matched = human_items.merge(
        machine_items, left_on="question_id", right_on="test_id", how="inner"
    )

    def correlation(subset: pd.DataFrame) -> dict:
        estimate, p_value = spearmanr(
            subset["human_accuracy"], subset["machine_accuracy"]
        )
        return {
            "n_items": int(len(subset)),
            "spearman_r": float(estimate),
            "p": float(p_value),
        }

    return {
        "overall": correlation(matched),
        "positive_queries": correlation(matched[matched["truth_positive"]]),
        "negative_queries": correlation(matched[~matched["truth_positive"]]),
    }


def mcnemar_paired(left: pd.Series, right: pd.Series) -> dict:
    left = left.astype(int)
    right = right.astype(int)
    left_only = int(((left == 1) & (right == 0)).sum())
    right_only = int(((left == 0) & (right == 1)).sum())
    discordant = left_only + right_only
    p_value = (
        float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "left_accuracy": float(left.mean()),
        "right_accuracy": float(right.mean()),
        "right_minus_left_pp": float((right.mean() - left.mean()) * 100),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "exact_mcnemar_p": p_value,
    }


def machine_representation_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    no_cue = frame[frame["concept_flag"].eq("no_concept")]
    for model, subset in no_cue.groupby("model"):
        pixel = (
            subset[subset["representation"].eq("DVRL")]
            .drop_duplicates("test_id")
            .set_index("test_id")
        )
        description = (
            subset[subset["representation"].eq("AD")]
            .drop_duplicates("test_id")
            .set_index("test_id")
        )
        common = pixel.index.intersection(description.index)
        if len(common) < 40:
            continue
        comparison = mcnemar_paired(
            pixel.loc[common, "correct"], description.loc[common, "correct"]
        )
        rows.append({"model": model, "n_common_items": len(common), **comparison})
    return pd.DataFrame(rows)


def machine_cue_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (representation, model), subset in frame.groupby(["representation", "model"]):
        no_cue = (
            subset[subset["concept_flag"].eq("no_concept")]
            .drop_duplicates("test_id")
            .set_index("test_id")
        )
        cue = (
            subset[subset["concept_flag"].eq("concept")]
            .drop_duplicates("test_id")
            .set_index("test_id")
        )
        common = no_cue.index.intersection(cue.index)
        if len(common) < 40:
            continue
        comparison = mcnemar_paired(
            no_cue.loc[common, "correct"], cue.loc[common, "correct"]
        )
        rows.append(
            {
                "representation": representation,
                "model": model,
                "n_common_items": len(common),
                **comparison,
            }
        )
    return pd.DataFrame(rows)


def machine_sdt(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, representation, cue), subset in frame.groupby(
        ["model", "representation", "concept_flag"]
    ):
        if len(subset) < 40:
            continue
        values = signal_detection(subset, "truth_positive", "response_positive")
        rows.append(
            {
                "model": model,
                "representation": representation,
                "cue": cue,
                "n_trials": int(len(subset)),
                **values,
            }
        )
    return pd.DataFrame(rows)


def make_sdt_figure(participant_sdt: pd.DataFrame, machine_cells: pd.DataFrame) -> None:
    colors = {"AD": "#4C78A8", "AP": "#59A14F", "DVRL": "#E15759"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    human_means = participant_sdt.groupby("cue")[["dprime", "criterion"]].mean()
    human_sem = participant_sdt.groupby("cue")[["dprime", "criterion"]].sem()
    labels = ["No cue", "Cue"]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(
        x - width / 2,
        human_means["dprime"],
        width,
        label="$d'$",
        color="#4C78A8",
    )
    axes[0].bar(
        x + width / 2,
        human_means["criterion"],
        width,
        label="Criterion $c$",
        color="#F28E2B",
    )
    axes[0].errorbar(
        x - width / 2,
        human_means["dprime"],
        yerr=1.96 * human_sem["dprime"],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].errorbar(
        x + width / 2,
        human_means["criterion"],
        yerr=1.96 * human_sem["criterion"],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].axhline(0, color="0.3", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Human decisions")
    axes[0].set_ylabel("Signal-detection estimate")
    axes[0].legend(frameon=False)

    no_cue = machine_cells[machine_cells["cue"].eq("no_concept")]
    for index, representation in enumerate(["AD", "AP", "DVRL"]):
        values = no_cue[no_cue["representation"].eq(representation)]
        jitter = np.linspace(-0.08, 0.08, max(len(values), 1))
        axes[1].scatter(
            index + jitter,
            values["dprime"],
            color=colors[representation],
            alpha=0.65,
            s=28,
        )
        axes[1].plot(
            index,
            values["dprime"].mean(),
            marker="D",
            color="black",
            markersize=6,
        )
    axes[1].set_xticks(range(3), ["Description", "Program", "Raw pixels"])
    axes[1].set_title("Machine sensitivity (no cue)")
    axes[1].set_ylabel("Sensitivity $d'$; points are model cells")
    axes[1].axhline(0, color="0.3", linewidth=0.8)
    axes[1].grid(axis="y", color="0.9", linewidth=0.7)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure_signal_detection.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "figure_signal_detection.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure_signal_detection.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_latex_summary(results: dict) -> None:
    accuracy = results["human_accuracy"]["primary_item_fixed_participant_clustered"]
    rt = results["human_rt"]["cap_180_seconds"]["terms"]
    sdt = results["human_signal_detection"]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Analysis & Estimate & 95\% CI & $p$ \\",
        r"\midrule",
    ]
    for label, value in [
        ("Accuracy: cue on positive queries", accuracy["cue_positive_queries"]),
        (
            "Accuracy: additional cue effect on negative queries",
            accuracy["additional_cue_effect_negative_queries"],
        ),
    ]:
        lo, hi = value["or_ci95"]
        lines.append(
            f"{label} & OR {value['odds_ratio']:.2f} & [{lo:.2f}, {hi:.2f}] & {value['p']:.3f} \\\\"
        )
    lines.extend(
        [
            f"RT: cue on positive queries & {rt['cue']['percent_change']:+.1f}\\% & [{rt['cue']['ci95_percent_change'][0]:+.1f}, {rt['cue']['ci95_percent_change'][1]:+.1f}]\\% & {rt['cue']['p']:.3f} \\\\",
            f"Signal detection: criterion shift & {sdt['criterion']['difference']:+.2f} & [{sdt['criterion']['ci95'][0]:+.2f}, {sdt['criterion']['ci95'][1]:+.2f}] & {sdt['criterion']['permutation_p']:.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (OUT / "table_patterns_key_statistics.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(machine_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)  # Also fix the variational-Bayes optimizer initialization.
    if machine_only:
        machine = prepare_machine()
        representation_pairs = machine_representation_pairs(machine)
        cue_pairs = machine_cue_pairs(machine)
        machine_cells = machine_sdt(machine)
        representation_pairs.to_csv(OUT / "machine_paired_representation_tests.csv", index=False)
        cue_pairs.to_csv(OUT / "machine_paired_cue_tests.csv", index=False)
        machine_cells.to_csv(OUT / "machine_signal_detection_cells.csv", index=False)
        results = {
            "analysis_version": "patterns_machine_only",
            "random_seed": SEED,
            "machine_representation_pairs": representation_pairs.to_dict("records"),
            "machine_cue_pairs": cue_pairs.to_dict("records"),
            "machine_signal_detection_cells": machine_cells.to_dict("records"),
        }
        (OUT / "machine_statistics.json").write_text(json.dumps(native(results), indent=2) + "\n")
        print(f"Machine-only analysis: {len(machine)} decisions; {len(representation_pairs)} representation pairs, {len(cue_pairs)} cue pairs and {len(machine_cells)} SDT cells.")
        return
    accuracy_rng = np.random.default_rng(SEED)
    signal_detection_rng = np.random.default_rng(SEED)
    human = prepare_human()
    machine = prepare_machine()

    participant_sdt, human_sdt_summary = human_sdt(human, signal_detection_rng)
    representation_pairs = machine_representation_pairs(machine)
    cue_pairs = machine_cue_pairs(machine)
    machine_cells = machine_sdt(machine)

    results = {
        "analysis_version": "patterns_v2_20260907",
        "random_seed": SEED,
        "human_participant_accuracy": human_participant_accuracy(
            human, accuracy_rng
        ),
        "human_accuracy": human_accuracy_models(human),
        "human_rt": human_rt_models(human),
        "human_signal_detection": human_sdt_summary,
        "machine_representation_pairs": representation_pairs.to_dict("records"),
        "machine_cue_pairs": cue_pairs.to_dict("records"),
        "machine_signal_detection_no_cue_cell_means": machine_cells[
            machine_cells["cue"].eq("no_concept")
        ]
        .groupby("representation")[
            ["dprime", "criterion", "hit_rate", "false_alarm_rate"]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
        "human_machine_item_alignment": human_machine_item_alignment(
            human, machine
        ),
    }

    participant_sdt.to_csv(
        OUT / "human_participant_signal_detection.csv", index=False
    )
    representation_pairs.to_csv(
        OUT / "machine_paired_representation_tests.csv", index=False
    )
    cue_pairs.to_csv(OUT / "machine_paired_cue_tests.csv", index=False)
    machine_cells.to_csv(OUT / "machine_signal_detection_cells.csv", index=False)
    (OUT / "patterns_statistics.json").write_text(
        json.dumps(native(results), indent=2), encoding="utf-8"
    )
    make_sdt_figure(participant_sdt, machine_cells)
    write_latex_summary(results)
    print(json.dumps(native(results), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-only", action="store_true",
                        help="Regenerate archived machine results without loading human data.")
    main(machine_only=parser.parse_args().machine_only)
