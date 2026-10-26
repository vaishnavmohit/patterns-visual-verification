#!/usr/bin/env python3
"""Operational summaries and exploratory common-returned-query sensitivities.

Uses the fixed completed export; makes no model calls and does not replace
the planned representation-specific returned-decision estimand. The paired
bootstrap reproduces the 12 October independent review audit. Recovered share
uses the same three returned sets within each run; the protocol contrast uses
the two caption sets' common returned queries. Resample base problems within
family, retaining both query polarities, with the primary seed and count.

Additional caption sets (amendments A3 and A4: caption_self_dN and
caption_self_free_dN, no cue) are summarised the same way and reported in
additional rows; the 40 primary conditions are unchanged by their presence.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "patterns_outputs"
NBOOT, SEED = 10_000, 20261012
RUNS = ("greedy", "s1")


def main():
    trials = pd.read_csv(OUT / "factorial_ow_trials.csv")
    items = pd.read_csv(OUT / "cue_overlap_items.csv")[["test_id", "uid", "family"]]
    ids = items.test_id.tolist()
    assert len(ids) == len(set(ids)) == 200
    assert not trials.duplicated(["model", "representation", "cue", "run", "test_id"]).any()
    assert trials.polarity.eq(np.where(trials.test_id.str.endswith("_pos"), "pos", "neg")).all()
    for _, cell in trials.groupby(["model", "representation", "cue", "run"]):
        assert len(cell) == 200 and set(cell.test_id) == set(ids)
    for column in ("prediction_strict", "prediction_rescored"):
        assert trials[column].dropna().isin(["pos", "neg"]).all()

    operational = []
    for scoring, column in (("primary_A2", "prediction_rescored"), ("strict", "prediction_strict")):
        for (model, representation, cue), cell in trials.groupby(["model", "representation", "cue"]):
            assert set(cell.run) == set(RUNS)
            run_metrics = []
            for _, run in cell.groupby("run"):
                valid = run[column].notna()
                correct = run[column].eq(run.polarity).sum()
                run_metrics.append((correct / valid.sum(), valid.mean(), correct / len(run)))
            conditional, decision_rate, operational_accuracy = np.mean(run_metrics, axis=0)
            operational.append(dict(scoring=scoring, model=model, representation=representation, cue=cue,
                                    attempted=len(cell), decisions=int(cell[column].notna().sum()),
                                    correct=int(cell[column].eq(cell.polarity).sum()),
                                    conditional_accuracy=conditional, decision_rate=decision_rate,
                                    operational_accuracy=operational_accuracy))
    op = pd.DataFrame(operational)
    primary = pd.read_csv(OUT / "factorial_ow_cells.csv")
    joined = op[op.scoring.eq("primary_A2")].merge(primary, on=["model", "representation", "cue"])
    extra = primary.representation.str.contains(r"_d\d$").sum()
    assert len(joined) == len(primary) and len(primary) - extra == 40   # 20 primary conditions per model, plus extra caption sets
    assert np.allclose(joined.conditional_accuracy, joined.accuracy)
    assert np.allclose(joined.decision_rate_x, joined.decision_rate_y)
    op.to_csv(OUT / "factorial_ow_operational.csv", index=False)

    positions = {item: i for i, item in enumerate(ids)}
    bases = items.groupby(["family", "uid"]).test_id.apply(list).reset_index()
    rng = np.random.default_rng(SEED)
    weights = np.zeros((200, NBOOT), dtype=np.float64)
    for _, family in bases.groupby("family"):
        draws = rng.integers(0, len(family), size=(len(family), NBOOT))
        for base, queries in enumerate(family.test_id):
            for query in queries:
                weights[positions[query]] = (draws == base).sum(axis=0)
    weights = np.column_stack([np.ones(200), weights])
    positive = np.array([item.endswith("_pos") for item in ids])

    def sensitivity(model, representation, common):
        values, counts = [], []
        for run in RUNS:
            cell = trials[trials.model.eq(model) & trials.cue.eq("none") & trials.run.eq(run)]
            predictions = cell.pivot(index="test_id", columns="representation", values="prediction_rescored").reindex(ids)
            valid = predictions[list(common)].notna().all(axis=1).to_numpy()
            counts.append(int(valid.sum()))
            response = predictions[representation].eq("pos").to_numpy()
            data = np.column_stack([valid & positive, valid & response & positive,
                                    valid & ~positive, valid & response & ~positive]).astype(float)
            sums = np.einsum("ij,ik->kj", weights, data, optimize=False)
            values.append(norm.ppf((sums[1] + .5) / (sums[0] + 1)) -
                          norm.ppf((sums[3] + .5) / (sums[2] + 1)))
        return np.mean(values, axis=0), counts

    checks = []
    for model in sorted(trials.model.unique()):
        estimates = []
        present = set(trials[trials.model.eq(model)].representation)
        for draw, tag in (("", ""), ("_d2", ", second draw"), ("_d3", ", set 3"), ("_d4", ", set 4")):
            if draw and not {"caption_self" + draw, "caption_self_free" + draw} <= present:
                continue
            common = ("pixels", "caption_self" + draw, "oracle_description")
            pixel, counts = sensitivity(model, "pixels", common)
            step, _ = sensitivity(model, "caption_self" + draw, common)
            description, _ = sensitivity(model, "oracle_description", common)
            pair = ("caption_self" + draw, "caption_self_free" + draw)
            paired_step, pair_counts = sensitivity(model, "caption_self" + draw, pair)
            prose, _ = sensitivity(model, "caption_self_free" + draw, pair)
            estimates += [(f"common-returned recipe recovery{tag}", (step - pixel) / (description - pixel), counts),
                          (f"common-returned prose minus step dprime{tag}", prose - paired_step, pair_counts)]
        for contrast, value, count in estimates:
            low, high = np.percentile(value[1:], [2.5, 97.5])
            checks.append(dict(model=model, contrast=contrast, estimate=value[0], ci_low=low, ci_high=high,
                               greedy_queries=count[0], sampled_queries=count[1], exploratory=True))
    pd.DataFrame(checks).to_csv(OUT / "factorial_ow_matched_sensitivity.csv", index=False)
    sources = [OUT / "factorial_ow_trials.csv", OUT / "cue_overlap_items.csv", Path(__file__)]
    manifest = dict(seed=SEED, resamples=NBOOT, scope="Fixed outputs; exploratory common-returned-query sensitivities",
                    sha256={str(path.relative_to(OUT.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sources})
    (OUT / "factorial_ow_reporting_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(checks).to_string(index=False))
    print(f"Operational summaries verified for all {len(primary)} aggregate conditions (40 primary, {extra} extra caption sets) under primary scoring; strict summaries also written.")


if __name__ == "__main__":
    main()
