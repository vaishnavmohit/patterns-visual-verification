#!/usr/bin/env python3
"""Do Action Descriptions that follow the rendered geometry change the description advantage?

The archived Action Descriptions (generate_drawing_story) place each turn
after the stroke it precedes and run a figure's separate shapes together. The
code repository's action_description_rendered context (commit b560bf5) states
each turn before its stroke and marks new shapes. Four open-weight aliases
with complete archived description cells reran both versions under the same
current pipeline and settings (no cue, temperature 0, thinking off; HPC jobs
1005297-1005300, ~/rerun_ad_rendered). Outputs are copied to
patterns_outputs/ad_rendered/<model>/<context>/<family>.xlsx.

Per model: accuracy with original and corrected descriptions on the items both
answered, exact McNemar test, and each version's gain over the model's archived
no-cue pixel cell on the shared items (the Table 2 contrast). Writes
ad_rendered_comparison.csv.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
SRC = OUT / "ad_rendered"

arch = pd.read_csv(ROOT / "machine_result/clean_machine_data.csv", low_memory=False)
arch = arch[arch.prediction_valid & arch.clean_primary_keep.astype(bool)].drop_duplicates(["model", "experiment_type", "test_id"])
label = arch.drop_duplicates("test_id").set_index("test_id").label


def load(model_dir, ctx):
    frames = [pd.read_excel(f) for f in sorted((model_dir / ctx).glob("*.xlsx"))]
    if not frames:
        return pd.Series(dtype=int)
    d = pd.concat(frames, ignore_index=True).drop_duplicates("test_id")
    d["pred"] = d.test_category_identified.astype(str).str.strip().str.lower()
    d = d[d.pred.isin(["pos", "neg"])]
    return pd.Series((d.pred == d.test_id.map(label)).astype(int).values, index=d.test_id.values)


def mcnemar(a, b):
    x, y = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return x, y, (binomtest(x, x + y).pvalue if x + y else float("nan"))


rows = []
for mdir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    model = mdir.name.replace("_", ":", 1) if ":" not in mdir.name else mdir.name
    orig, rend = load(mdir, "action_description"), load(mdir, "action_description_rendered")
    ids = orig.index.intersection(rend.index)
    b, c, p = mcnemar(rend[ids], orig[ids])
    pix = arch[arch.model.eq(model) & arch.experiment_type.eq("DVRL")].set_index("test_id").correct_int
    shared = ids.intersection(pix.index)
    rows.append(dict(model=model, n_original=len(orig), n_rendered=len(rend), n_both=len(ids),
                     acc_original=100 * orig[ids].mean(), acc_rendered=100 * rend[ids].mean(),
                     rendered_minus_original_pp=100 * (rend[ids].mean() - orig[ids].mean()),
                     rendered_only=b, original_only=c, mcnemar_p=p, n_pixel_shared=len(shared),
                     gain_original_vs_pixels_pp=100 * (orig[shared].mean() - pix[shared].mean()),
                     gain_rendered_vs_pixels_pp=100 * (rend[shared].mean() - pix[shared].mean())))
res = pd.DataFrame(rows)
res.to_csv(OUT / "ad_rendered_comparison.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
