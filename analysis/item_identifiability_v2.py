#!/usr/bin/env python3
"""Invisible stroke boundaries with the renderer's turn convention and shape boundaries.

Corrects item_identifiability.py, which read each stroke's turn parameter as
the turn taken AFTER the stroke. The Bongard-LOGO renderer
(NVlabs/Bongard-LOGO, bongard/bongard_painter.py, BongardShapePainter.draw)
turns the turtle by an action's angle BEFORE drawing it, and draws each shape
of a figure separately from its own start point and heading. A boundary is
therefore invisible when, within one shape, a straight stroke is followed by a
straight stroke of the same texture whose own turn is zero (parameter 0.5).
The first stroke of a shape sets its heading and creates no boundary.

Shape boundaries are absent from the item files used elsewhere (each figure's
program is flattened). They are recovered from the original
ShapeBongard_V2 <family>_action_programs.json files, which nest strokes by
shape; every figure is matched by its flattened program.

--normal-only restricts boundaries to the untextured ("normal") line, whose
joins leave no trace; stamped textures restart their pattern at a stroke and
may reveal the join.

Writes item_identifiability_v2.csv (and _normal.csv with --normal-only).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("bd", "ff", "hd_comb", "hd_novel")
SOURCE = {"bd": "bd", "ff": "ff", "hd_comb": "hd", "hd_novel": "hd"}


def parse(stroke):
    kind, texture, rest = stroke.split("_", 2)
    return kind, texture, float(rest.split("-")[-1])


def invisible(shapes, normal_only=False):
    n = 0
    for shape in shapes:
        s = [parse(x) for x in shape]
        for a, b in zip(s, s[1:]):
            if a[0] == b[0] == "line" and a[1] == b[1] and abs(b[2] - 0.5) < 1e-9 and (not normal_only or a[1] == "normal"):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(ROOT.parents[1] / "code/human-machine-reasoning/data/Bongard_Logo_human"))
    ap.add_argument("--programs", required=True, help="directory with {bd,ff,hd}_action_programs.json from ShapeBongard_V2")
    ap.add_argument("--normal-only", action="store_true")
    args = ap.parse_args()

    nested = {}
    for src in ("bd", "ff", "hd"):
        for prob, sides in json.load(open(Path(args.programs) / f"{src}_action_programs.json")).items():
            for fig in sides[0] + sides[1]:
                nested[(prob, tuple(x for shape in fig for x in shape))] = fig

    rows, strokes, hidden, slots, under, images, missing = [], 0, 0, 0, 0, {}, 0
    for fam in FAMILIES:
        for it in json.load(open(Path(args.items) / f"{fam}.json")):
            progs = it["action_prog"]["pos"] + it["action_prog"]["neg"] + [it["testfiles"]["action_prog"]]
            paths = it["images"]["pos"] + it["images"]["neg"] + [it["images"]["testfiles"]]
            prob = paths[0].split("/")[2]
            flags, ks = [], []
            for path, prog in zip(paths, progs):
                fig = nested.get((prob, tuple(prog)))
                if fig is None:
                    missing += 1
                    fig = [prog]
                k = invisible(fig, args.normal_only)
                strokes += len(prog); hidden += k; slots += 1; under += k > 0
                images[path] = k > 0; flags.append(k > 0); ks.append(k)
            rows.append({"test_id": it["test_id"], "family": fam, "query_underdetermined": flags[-1],
                         "support_underdetermined_fraction": sum(flags[:-1]) / 12,
                         "item_underdetermined_fraction": sum(flags) / 13,
                         "query_invisible_boundaries": ks[-1]})
    assert missing == 0, f"{missing} figures not matched to a nested program"
    out = ROOT / "analysis" / "patterns_outputs" / ("item_identifiability_v2_normal.csv" if args.normal_only else "item_identifiability_v2.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    d = pd.DataFrame(rows)
    print(f"strokes {hidden}/{strokes} = {100*hidden/strokes:.1f}% invisible boundaries; "
          f"slots {under}/{slots} = {100*under/slots:.1f}%; "
          f"unique images {sum(images.values())}/{len(images)} = {100*sum(images.values())/len(images):.1f}%")
    print(f"queries: {int(d.query_underdetermined.sum())}/{len(d)} under-determined; by family: "
          f"{d.groupby('family').query_underdetermined.sum().to_dict()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
