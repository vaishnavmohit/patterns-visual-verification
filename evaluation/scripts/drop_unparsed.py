#!/usr/bin/env python
"""Drop rows whose answer never parsed, so a re-run can re-attempt them.

A row lands in results.xlsx with an empty ``test_category_identified`` when the
model filled its budget before emitting the Conclusion field: the analysis and
the rule are there, the decision is not. Scored as wrong, those rows understate
the model badly (gemma4:e2b reads as 20% accurate on Action Descriptions and is
79% over the rows that parsed).

They also block their own repair. Resumption skips any test_id already present
in results.xlsx, and process_samples returns early once ``len(df) >=
num_samples``, so re-running the cell prints "Nothing to do" and exits. Removing
the rows is what lets the normal top-up path pick them up again.

    python scripts/drop_unparsed.py --model gemma4:e2b            # what would go
    python scripts/drop_unparsed.py --model gemma4:e2b --apply    # do it

Every file it rewrites is backed up next to itself as results.xlsx.bak-<stamp>.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import pandas as pd

FAMILIES = ("bd", "ff", "hd_comb", "hd_novel")
ANSWER = "test_category_identified"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default=os.environ.get("OUTPUT_DIR", str(Path.home() / "output_human_context")))
    ap.add_argument("--model", action="append", default=[], help="repeatable; default every model")
    ap.add_argument("--program", action="append", default=[], help="repeatable; default every program")
    ap.add_argument("--apply", action="store_true", help="rewrite the files (default: report only)")
    args = ap.parse_args()

    archive = Path(args.archive)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    total_dropped = total_files = 0

    for fam in FAMILIES:
        fam_dir = archive / f"bongard_logo_human_{fam}"
        if not fam_dir.is_dir():
            continue
        for model_dir in sorted(fam_dir.iterdir()):
            if not model_dir.is_dir() or (args.model and model_dir.name not in args.model):
                continue
            for prog_dir in sorted(model_dir.iterdir()):
                if not prog_dir.is_dir() or (args.program and prog_dir.name not in args.program):
                    continue
                f = prog_dir / "results.xlsx"
                if not f.exists():
                    continue
                df = pd.read_excel(f)
                if ANSWER not in df.columns:
                    continue
                answer = df[ANSWER].astype(str).str.strip().str.lower()
                bad = ~answer.isin(["pos", "neg"])
                if not bad.any():
                    continue
                total_files += 1
                total_dropped += int(bad.sum())
                print(f"  {int(bad.sum()):3d} of {len(df):3d} rows  {model_dir.name}/{prog_dir.name}/{fam}")
                if args.apply:
                    shutil.copy2(f, f.with_suffix(f".xlsx.bak-{stamp}"))
                    df[~bad].to_excel(f, index=False)

    verb = "dropped" if args.apply else "would drop"
    print(f"\n{verb} {total_dropped} row(s) across {total_files} file(s)")
    if total_dropped and not args.apply:
        print("re-run with --apply, then submit the cell again to re-attempt them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
