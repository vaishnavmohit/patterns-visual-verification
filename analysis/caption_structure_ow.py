#!/usr/bin/env python3
"""Do open-weight step and prose captions preserve global structure, and does that predict the prose gain?

Experiment 1 of 14 October 2026. Captions: the primary open-weight caption sets
(patterns_outputs/factorial_captions/{gemma4_31b,qwen3.6_27b}[__free].jsonl).

Ground truth comes from the original per-shape programs (ShapeBongard_V2,
passed as the first argument), drawn with the renderer's geometry
(NVlabs bongard_painter.py): each stroke's turn precedes it, a line moves
forward by its length (zigzag and stamped lines have the same net
displacement), an arc is turtle.circle(radius, extent) with the centre on the
turtle's left, and every shape starts its own path.

  parts      number of separate shapes in the figure
  closed     single-shape figures only: the path ends within 5% of the
             figure's extent of where it started

Caption claims:
  step   parts = 1 + number of "start a new, separate shape" markers.
         Closure is not compared: simulated from the recipe's own loosely
         estimated lengths and angles, only 13-24% of single-shape recipes
         close against 79% in truth, so it measures numeric precision rather
         than whether the captioner saw a closed shape (kept in the figure file
         as recipe_geometry_closed).
  prose  parts from the first statement of a count of parts/shapes/figures;
         closure from the words "closed" / "open" in a single-part caption
         (unclear when both or neither occur)

Also: the structural audit of caption_fidelity.py (stroke count and
texture-by-kind F1 against the program) for the open-weight step captions.

Association: per item, the change from step to prose in the fraction of its 13
figures with a correct part count (and correct closure), against the change in
the model's no-cue correctness (prose minus step, both runs, rescored), OLS
with family fixed effects and base-problem clustered errors. Descriptive and
post hoc: it can show an association, not a mechanism.

Writes caption_structure_ow_figures.csv, caption_structure_ow_items.csv and
caption_structure_ow_summary.csv; for the A3 draw (captions {stem}[__free]__d2.jsonl)
the same files with a _d2 tag, and likewise _d3 and _d4 (amendment A4). The item association needs both no-cue cells of a
draw complete and skips a model otherwise.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "patterns_outputs"
CODE = ROOT.parents[1] / "code" / "human-machine-reasoning"
PROGRAMS = Path(sys.argv[1])
MODELS = {"gemma4:31b": "gemma4_31b", "qwen3.6:27b": "qwen3.6_27b"}
TOL = 0.05
NUM = {"a single": 1, "single": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


# ---------- ground truth from programs (renderer geometry) ----------
def denorm_turn(p):
    return p * 360 - 180 if p >= 0.5 else -(180 - p * 360)   # positive = left


def simulate(shapes_turn_first):
    """shapes: list of list of (kind, length_or_radius, arc_extent, turn_deg). Returns per-shape (start, end, extent)."""
    out = []
    for shape in shapes_turn_first:
        x = y = 0.0; h = 0.0; pts = [(0.0, 0.0)]
        for kind, a, ext, turn, turn_first in shape:
            if turn_first:
                h += turn
            if kind == "line":
                x += a * math.cos(math.radians(h)); y += a * math.sin(math.radians(h))
            else:   # turtle.circle(radius, extent): centre on the left
                cx, cy = x - a * math.sin(math.radians(h)), y + a * math.cos(math.radians(h))
                for k in range(1, 13):
                    th = math.radians(h + ext * k / 12)
                    pts.append((cx + a * math.sin(th), cy - a * math.cos(th)))
                h += ext
                x, y = pts[-1]
            if not turn_first:
                h += turn
            pts.append((x, y))
        xs, ys = zip(*pts)
        extent = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        out.append(math.hypot(x, y) / extent)
    return out


def program_shape(stroke):
    kind, _tex, rest = stroke.split("_", 2)
    move, turn = rest.split("-")[0], float(rest.split("-")[-1])
    if kind == "line":
        return ("line", float(move), 0.0, denorm_turn(turn), True)
    radius, arc = move.split("_")
    return ("arc", float(radius), float(arc) * 720 - 360, denorm_turn(turn), True)


nested = {}
for src in ("bd", "ff", "hd"):
    for prob, sides in json.load(open(PROGRAMS / f"{src}_action_programs.json")).items():
        for fig in sides[0] + sides[1]:
            nested[(prob, tuple(x for sh in fig for x in sh))] = fig

figs = []
for fam in ("bd", "ff", "hd_comb", "hd_novel"):
    for it in json.load(open(CODE / f"data/Bongard_Logo_human/{fam}.json")):
        imgs = it["images"]["pos"][:6] + it["images"]["neg"][:6] + [it["images"]["testfiles"]]
        progs = it["action_prog"]["pos"][:6] + it["action_prog"]["neg"][:6] + [it["testfiles"]["action_prog"]]
        for img, prog in zip(imgs, progs):
            shapes = nested[(img.split("/")[2], tuple(prog))]
            gaps = simulate([[program_shape(s) for s in sh] for sh in shapes])
            figs.append(dict(test_id=it["test_id"], uid=it["uid"], family=fam, image=img, true_parts=len(shapes),
                             true_closed=(gaps[0] < TOL) if len(shapes) == 1 else None))
figs = pd.DataFrame(figs)
truth = figs.drop_duplicates("image").set_index("image")

# ---------- caption claims ----------
STEP = re.compile(r"draw an? (normal|zigzag|circle|square|triangle) (line|arc)\b(.*?)(?:(continue straight)|turn (left|right) by ([\d.]+))", re.I)
LEN = re.compile(r"line of ([\d.]+)"); ARC = re.compile(r"radius of ([\d.]+) and sweeping (-?[\d.]+)")


def step_claims(text):
    blocks = re.split(r"start a new, separate shape", text or "", flags=re.I)
    shapes = []
    for b in blocks:
        sh = []
        for m in STEP.finditer(b):
            kind, rest = m.group(2).lower(), m.group(3)
            turn = 0.0 if m.group(4) else (float(m.group(6)) * (1 if m.group(5).lower() == "left" else -1))
            if kind == "line":
                ml = LEN.search(rest); sh.append(("line", float(ml.group(1)) if ml else 0.5, 0.0, turn, False))
            else:
                ma = ARC.search(rest)
                sh.append(("arc", float(ma.group(1)) if ma else 0.5, float(ma.group(2)) if ma else 180.0, turn, False))
        if sh:
            shapes.append(sh)
    parts = len(shapes) if shapes else None
    closed = (simulate(shapes)[0] < TOL) if parts == 1 else None
    return parts, closed


PARTS = re.compile(r"\b(a single|single|one|two|three|four|five|six|seven|eight)\b[\s,]+(?:[\w-]+[\s,]+){0,5}?(parts?|shapes?|figures?|components?|elements?|pieces?|objects?|paths?|loops?|outlines?|contours?|continuous lines?)\b", re.I)


def prose_claims(text):
    t = (text or "").strip()
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    m = PARTS.search(first) or PARTS.search(t[:400])
    parts = NUM[m.group(1).lower()] if m else None
    closed = None
    if parts == 1:
        head = " ".join(re.split(r"(?<=[.!?])\s", t)[:2]).lower()
        c, o = bool(re.search(r"\bclosed\b", head)), bool(re.search(r"\bopen\b", head))
        closed = True if c and not o else (False if o and not c else None)
    return parts, closed


def token_f1(cap, prog):
    ca = Counter((s[0], s[1]) for s in re.findall(r"draw an? (normal|zigzag|circle|square|triangle) (line|arc)", cap or "", re.I))
    cb = Counter((x.split("_")[1], x.split("_")[0]) for x in prog)
    ca = Counter({(t.lower(), k.lower()): v for (t, k), v in ca.items()})
    tot = sum(ca.values()) + sum(cb.values())
    return 2 * sum((ca & cb).values()) / tot if tot else np.nan


prog_of = {}
for fam in ("bd", "ff", "hd_comb", "hd_novel"):
    for it in json.load(open(CODE / f"data/Bongard_Logo_human/{fam}.json")):
        for img, prog in zip(it["images"]["pos"][:6] + it["images"]["neg"][:6] + [it["images"]["testfiles"]],
                             it["action_prog"]["pos"][:6] + it["action_prog"]["neg"][:6] + [it["testfiles"]["action_prog"]]):
            prog_of[img] = prog

def score_figures(draw):
    """Score every caption of one draw ('' = primary, '__d2' = amendment A3) against the renderer truth."""
    rows = []
    for model, stem in MODELS.items():
        for fmt, suffix in (("step", ""), ("prose", "__free")):
            path = OUT / "factorial_captions" / f"{stem}{suffix}{draw}.jsonl"
            if not path.exists():
                continue
            for line in open(path):
                r = json.loads(line)
                if r.get("status") != "ok":
                    continue
                parts, closed = (step_claims if fmt == "step" else prose_claims)(r["description"])
                recipe_closed = closed if fmt == "step" else None
                if fmt == "step":
                    closed = None
                tr = truth.loc[r["image"]]
                row = dict(model=model, format=fmt, image=r["image"], true_parts=tr.true_parts, true_closed=tr.true_closed,
                           claimed_parts=parts, claimed_closed=closed, recipe_geometry_closed=recipe_closed,
                           parts_correct=(parts == tr.true_parts) if parts is not None else None,
                           closed_correct=(closed == tr.true_closed) if (closed is not None and tr.true_closed is not None) else None)
                if fmt == "step":
                    prog = prog_of[r["image"]]
                    n = len(re.findall(r"draw an? (?:normal|zigzag|circle|square|triangle) (?:line|arc)", r["description"], re.I))
                    row.update(strokes_match=n == len(prog), token_f1=token_f1(r["description"], prog))
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(F):
    summ = []
    for (m, fmt), g in F.groupby(["model", "format"]):
        single = g[g.true_parts.eq(1)]
        summ.append(dict(model=m, format=fmt, images=len(g),
                         parts_stated=g.claimed_parts.notna().mean(),
                         parts_correct_of_stated=g.parts_correct.dropna().astype(float).mean(),
                         parts_correct_overall=g.parts_correct.fillna(False).astype(float).mean(),
                         single_shape_images=len(single),
                         closure_stated=single.claimed_closed.notna().mean(),
                         closure_correct_of_stated=single.closed_correct.dropna().astype(float).mean(),
                         strokes_match=g.strokes_match.mean() if fmt == "step" else np.nan,
                         token_f1=g.token_f1.mean() if fmt == "step" else np.nan))
    return pd.DataFrame(summ)


# ---------- item-level association with the prose gain ----------
TRIALS = pd.read_csv(OUT / "factorial_ow_trials.csv")


def associate(F, draw):
    """Item-level OLS of the prose-minus-step change in correctness on the change in part-count accuracy.
    A model enters only when both no-cue cells of the draw are complete (200 items, both runs)."""
    rep = {"caption_self" + draw.replace("__", "_"): "acc_step", "caption_self_free" + draw.replace("__", "_"): "acc_prose"}
    attempted = TRIALS[TRIALS.cue.eq("none") & TRIALS.representation.isin(list(rep))]
    tr = attempted.dropna(subset=["prediction_rescored"]).copy()
    tr["correct"] = (tr.prediction_rescored == tr.polarity).astype(float)
    items = []
    for model in MODELS:
        if attempted[attempted.model.eq(model)].groupby(["representation", "run"]).test_id.nunique().reindex(
                pd.MultiIndex.from_product([list(rep), ["greedy", "s1"]])).fillna(0).lt(200).any():
            continue
        tm = tr[tr.model.eq(model)]
        acc = tm.groupby(["test_id", "representation"]).correct.mean().unstack().rename(columns=rep)
        fm = F[F.model.eq(model)]
        fx = figs.merge(fm, on="image", suffixes=("", "_c"))
        agg = fx.groupby(["test_id", "uid", "family", "format"]).agg(parts=("parts_correct", lambda s: s.fillna(False).astype(float).mean()),
                                                                closed=("closed_correct", lambda s: s.dropna().astype(float).mean() if s.notna().any() else np.nan)).unstack("format")
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reset_index().merge(acc, left_on="test_id", right_index=True)
        agg["model"] = model
        agg["d_acc"] = agg.acc_prose - agg.acc_step
        agg["d_parts"] = agg.parts_prose - agg.parts_step
        items.append(agg)
    if not items:
        return pd.DataFrame(), pd.DataFrame(columns=["model"])
    I = pd.concat(items)
    assoc = []
    for model, g in I.groupby("model"):
        g = g.dropna(subset=["d_acc", "d_parts"])
        fit = smf.ols("d_acc ~ d_parts + C(family)", data=g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.uid)[0]})
        lo, hi = fit.conf_int().loc["d_parts"]
        assoc.append(dict(model=model, items=len(g), mean_d_parts=g.d_parts.mean(), mean_d_acc=g.d_acc.mean(),
                          slope_d_parts=fit.params["d_parts"], ci_low=lo, ci_high=hi, p=fit.pvalues["d_parts"]))
    return I, pd.DataFrame(assoc)


pd.set_option("display.width", 250)
# Primary caption sets, then the independent draw of amendment A3 (temperature 1, seed 2) when present.
for draw, tag in (("", ""), ("__d2", "_d2"), ("__d3", "_d3"), ("__d4", "_d4")):
    F = score_figures(draw)
    if F.empty:
        continue
    F.to_csv(OUT / f"caption_structure_ow{tag}_figures.csv", index=False)
    I, A = associate(F, draw)
    if not I.empty:
        I.to_csv(OUT / f"caption_structure_ow{tag}_items.csv", index=False)
    S = summarise(F).merge(A, on="model", how="left")
    S.to_csv(OUT / f"caption_structure_ow{tag}_summary.csv", index=False)
    print(f"== draw {draw or 'primary'}")
    print(S.round(3).to_string(index=False))
