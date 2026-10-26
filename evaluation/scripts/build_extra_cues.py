#!/usr/bin/env python3
"""Build the two additional cue tables for the open-weight replication.

wrong_clean  A mismatched label that shares no component with the item's own
             label. The original derangement (data/factorial/wrong_cues.json)
             only guarantees a different string: for 64 of the 200 items the
             "wrong" label shares a component with the correct one (all of
             hd_novel, whose labels all contain "has eight straight lines").
             Here hd labels are mapped to attribute classes and a candidate
             must share no class; bd labels must share no content word; ff has
             two exclusive labels. hd_novel draws from hd_comb labels, since
             every hd_novel label shares a class with every other. Whether the
             chosen label is false of the figures is not evaluated.
irrelevant   A phrase with no geometric content, the nearest in length to the
             item's own label, so the cue slot is filled without information.

Both are written in the format of wrong_cues.json, keyed by family and label.

    python scripts/build_extra_cues.py
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
FAMILIES = ("bd", "ff", "hd_comb", "hd_novel")

CLASSES = {
    "curve": ("has curve",),
    "straight": ("straight line", "straight lines", "exist triangle", "exist quadrangle", "exist regular", "line crossing"),
    "symmetry": ("symmetric", "transposed"),
    "angle": ("acute angle", "obtuse angle"),
    "closure": ("closed shape", "convex"),
    "thin": ("thin shape",),
    "parts": ("two parts", "balanced two"),
}
STOP = {"and", "of", "two", "three", "four", "five", "six", "the", "a", "with"}

IRRELEVANT = [
    "rainy Tuesday", "a cup of warm tea", "the library opens at nine", "a quiet walk by the river",
    "fresh bread from the corner bakery", "an old song playing on the radio downstairs",
    "the train to Tartu leaves in the early morning", "a postcard from a friend who moved abroad last year",
    "the committee meeting was moved to next Thursday afternoon", "someone left an umbrella by the door",
    "the garden needs watering after a long dry week", "notes from yesterday's history lecture",
    "a crossword puzzle half finished on the kitchen table", "the museum is closed on public holidays",
    "coffee", "winter holidays", "the bus was late again", "a letter about the new parking rules",
    "the neighbours are painting their fence a pale blue colour this weekend",
    "a recipe for soup that uses whatever vegetables are left in the fridge",
]


def hd_classes(label):
    low = label.lower()
    return {c for c, keys in CLASSES.items() if any(k in low for k in keys)}


def words(label):
    return {w for w in re.findall(r"[a-z]+", label.lower()) if w not in STOP}


def main():
    labels = {}
    for fam in FAMILIES:
        labels[fam] = sorted({it["concept_ui"] for it in json.load(open(ROOT / "data/Bongard_Logo_human" / f"{fam}.json"))})
    rng = random.Random(SEED)
    clean = {fam: {} for fam in FAMILIES}
    for fam in FAMILIES:
        pool = labels["hd_comb"] if fam == "hd_novel" else labels[fam]
        used = {}
        for lab in labels[fam]:
            if fam == "ff":
                cands = [x for x in pool if x != lab]
            elif fam == "bd":
                cands = [x for x in pool if x != lab and not (words(x) & words(lab))]
            else:
                cands = [x for x in pool if x != lab and not (hd_classes(x) & hd_classes(lab))]
            if not cands:
                raise SystemExit(f"no clean mismatched label for {fam}: {lab}")
            rng.shuffle(cands)
            cands.sort(key=lambda x: used.get(x, 0))   # spread choices across labels
            clean[fam][lab] = cands[0]
            used[cands[0]] = used.get(cands[0], 0) + 1
    irrelevant = {fam: {} for fam in FAMILIES}
    for fam in FAMILIES:
        for lab in labels[fam]:
            irrelevant[fam][lab] = min(IRRELEVANT, key=lambda p: (abs(len(p) - len(lab)), p))
    out = ROOT / "data" / "factorial"
    json.dump({"seed": SEED, "method": "mismatched label sharing no attribute class (hd), content word (bd) or label (ff); hd_novel drawn from hd_comb",
               "families": clean}, open(out / "wrong_cues_clean.json", "w"), indent=2, ensure_ascii=False)
    json.dump({"seed": SEED, "method": "non-geometric phrase nearest in length to the item's own label",
               "families": irrelevant}, open(out / "irrelevant_cues.json", "w"), indent=2, ensure_ascii=False)
    for fam in FAMILIES:
        print(fam, len(clean[fam]), "labels; e.g.", list(clean[fam].items())[0], "|", list(irrelevant[fam].items())[0])


if __name__ == "__main__":
    main()
