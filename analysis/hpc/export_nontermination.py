#!/usr/bin/env python3
"""Export the non-termination record behind Table 5 from the archive ledgers.

Run ON THE HPC from the code repository root (reads ~/output_human_context):

    .venv/bin/python export_nontermination.py > nontermination_by_cell.csv

and copy the CSV into analysis/patterns_outputs/. One row per Qwen3-VL alias x
interface x family: item-by-cue cells that never terminated (a failures.jsonl
record whose reason starts "non-terminating reasoning trace"), cells answered
(parsed pos/neg rows in results.xlsx), and the attempted total. A trailing
comment line gives the ledger totals: records, distinct cells, and the
confirmatory re-attempts (cells with two or more records) and how many of those
were later answered.
"""
import collections, csv, json, os, sys
import pandas as pd

A = os.path.expanduser("~/output_human_context")
FAM = ("bd", "ff", "hd_comb", "hd_novel")
PROGS = {"action_description_temp_0.0": "AD", "action_description_concept_v2_temp_0.0": "AD",
         "aps_temp_0.0": "AP", "aps_concept_temp_0.0": "AP",
         "dvrl_temp_0": "Pix", "dvrl_temp_0.0": "Pix", "dvrl_concept_temp_0": "Pix", "dvrl_concept_temp_0.0": "Pix"}
MODELS = ("qwen3-vl:8b", "qwen3-vl:30b", "qwen3-vl:32b")

ok, nt = collections.Counter(), collections.Counter()
records = reattempted = recovered = 0
for fam in FAM:
    for model in MODELS:
        for prog, tier in PROGS.items():
            d = f"{A}/bongard_logo_human_{fam}/{model}/{prog}"
            answered = set()
            if os.path.exists(f"{d}/results.xlsx"):
                x = pd.read_excel(f"{d}/results.xlsx")
                a = x.test_category_identified.astype(str).str.strip().str.lower()
                answered = set(x.test_id[a.isin(["pos", "neg"])].astype(str))
                ok[(model, tier, fam)] += len(answered)
            if os.path.exists(f"{d}/failures.jsonl"):
                per_id = collections.Counter()
                for line in open(f"{d}/failures.jsonl"):
                    if line.strip():
                        r = json.loads(line)
                        if str(r.get("reason", "")).startswith("non-terminating"):
                            per_id[r["test_id"]] += 1
                records += sum(per_id.values())
                nt[(model, tier, fam)] += len(per_id)
                reattempted += sum(1 for t, k in per_id.items() if k >= 2 or t in answered)
                recovered += sum(1 for t in per_id if t in answered)
w = csv.writer(sys.stdout)
w.writerow(["model", "interface", "family", "non_terminating", "answered", "attempted", "rate"])
for model in MODELS:
    for tier in ("Pix", "AD", "AP"):
        for fam in FAM:
            n, o = nt[(model, tier, fam)], ok[(model, tier, fam)]
            if n + o:
                w.writerow([model, tier, fam, n, o, n + o, round(n / (n + o), 4)])
print(f"# records={records} distinct_cells={sum(nt.values())} reattempted={reattempted} recovered={recovered}")
