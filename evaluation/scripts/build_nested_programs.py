#!/usr/bin/env python3
"""Recover per-shape stroke lists for the 200 human-study items.

data/Bongard_Logo_human/<family>.json stores each figure's action program as
one flat list, losing where one shape ends and the next begins. The original
ShapeBongard_V2 <source>_action_programs.json files keep the nesting
(figure -> shapes -> strokes). This matches every figure of the 200 items to
its nested program by problem name and flattened strokes and writes
data/Bongard_Logo_human/nested_programs.json, a list of nested figures used by
hmr.datasets.action_descriptions.process_all_image_specs_rendered.

    python scripts/build_nested_programs.py --shapebongard ~/ShapeBongard_V2
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {"bd": "bd", "ff": "ff", "hd_comb": "hd", "hd_novel": "hd"}

ap = argparse.ArgumentParser()
ap.add_argument("--shapebongard", required=True)
args = ap.parse_args()

nested = {}
for src in ("bd", "ff", "hd"):
    for prob, sides in json.load(open(Path(args.shapebongard) / src / f"{src}_action_programs.json")).items():
        for fig in sides[0] + sides[1]:
            nested[(prob, tuple(x for shape in fig for x in shape))] = fig

out, seen = [], {}
for fam in SOURCE:
    for it in json.load(open(ROOT / "data" / "Bongard_Logo_human" / f"{fam}.json")):
        paths = it["images"]["pos"] + it["images"]["neg"] + [it["images"]["testfiles"]]
        progs = it["action_prog"]["pos"] + it["action_prog"]["neg"] + [it["testfiles"]["action_prog"]]
        for path, prog in zip(paths, progs):
            fig = nested[(path.split("/")[2], tuple(prog))]
            key = tuple(prog)
            if key in seen:
                assert seen[key] == fig, f"one flattened program, two nestings: {path}"
                continue
            seen[key] = fig
            out.append(fig)
dest = ROOT / "data" / "Bongard_Logo_human" / "nested_programs.json"
json.dump(out, open(dest, "w"))
print(f"{len(out)} distinct figures; {sum(len(f) > 1 for f in out)} with more than one shape -> {dest}")
