#!/usr/bin/env python
"""Run the factorial experiment for one model.

Examples
--------
# stage 1 only: caption every unique image with the strong captioner
python scripts/run_factorial.py --model gemini-3.5-flash --stage captions

# pilot: 5 items per family, every cell, no writes to the full-run tree
python scripts/run_factorial.py --model qwen3-vl:30b --pilot 5

# count calls without running anything
python scripts/run_factorial.py --model gpt-5.5 --dry-run

# full run (captions for caption_self are produced automatically if missing)
python scripts/run_factorial.py --model qwen3-vl:30b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from hmr.factorial.captions import run_captioning  # noqa: E402
from hmr.factorial.design import CUES, DRAW_REPRESENTATIONS, REPRESENTATIONS, load_items, pilot_subset  # noqa: E402
from hmr.factorial.runner import load_design, plan_cells, run_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model alias as listed under models: in the design file")
    ap.add_argument("--design", default=str(REPO / "config/factorial/design.yaml"))
    ap.add_argument("--stage", choices=["all", "captions", "reason"], default="all")
    ap.add_argument("--representations", default=None, help=f"comma list from {','.join(REPRESENTATIONS)}")
    ap.add_argument("--cues", default=None, help=f"comma list from {','.join(CUES)}")
    ap.add_argument("--runs", default=None, help="comma list of run names from the design file (e.g. greedy,s1)")
    ap.add_argument("--pilot", type=int, default=0, help="items per family; writes under <out>/pilot instead of the full tree")
    ap.add_argument("--workers", type=int, default=None, help="override per-model concurrency")
    ap.add_argument("--out", default=None, help="override output root (default: design output_root)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    design = load_design(args.design)
    data_dir = Path(os.path.expandvars(design["data_dir"]))
    image_dir = Path(os.path.expandvars(design["image_dir"]))
    out_root = Path(args.out or os.path.expandvars(design["output_root"]))
    if args.pilot:
        out_root = out_root / "pilot"
    design["wrong_cue_file"] = str(Path(os.path.expandvars(design["wrong_cue_file"])))
    if not Path(design["wrong_cue_file"]).is_absolute():
        design["wrong_cue_file"] = str(REPO / design["wrong_cue_file"])
    for cue_name, path in list((design.get("extra_cue_files") or {}).items()):
        path = Path(os.path.expandvars(path))
        design["extra_cue_files"][cue_name] = str(path if path.is_absolute() else REPO / path)

    # Persist the run log next to the results, so an interactive session that
    # scrolls away or drops still leaves a record of the retry ladder.
    log_dir = out_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{args.model.replace('/', '_').replace(':', '_')}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    fh = logging.FileHandler(log_path); fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logging.info("log file: %s", log_path)

    items = load_items(data_dir)
    if args.pilot:
        items = pilot_subset(items, args.pilot)
    model = args.model
    if model not in design["models"]:
        ap.error(f"{model!r} not in design panel: {sorted(design['models'])}")
    model_cfg = design["models"][model]
    reps = args.representations.split(",") if args.representations else list(design.get("representations", REPRESENTATIONS))
    cues = args.cues.split(",") if args.cues else list(design.get("cues", CUES))
    run_names = args.runs.split(",") if args.runs else None
    workers = args.workers if args.workers is not None else int(model_cfg.get("workers", 1))

    logging.info("model=%s provider-vision=%s items=%d out=%s", model, model_cfg.get("vision", True), len(items), out_root)

    # ---- stage 1: captions -------------------------------------------------
    self_captions = bool(model_cfg.get("self_captions", True))
    need_self_captions = (args.stage == "captions" and not ({"caption_self_free"} | set(DRAW_REPRESENTATIONS)) & set(reps)) or (
        "caption_self" in reps and model_cfg.get("vision", True) and self_captions)
    if args.stage in ("all", "captions") and need_self_captions:
        if not model_cfg.get("vision", True):
            logging.info("[%s] text-only model: no captioning stage", model)
        elif not self_captions:
            logging.info("[%s] self_captions disabled; caption_strong will supply descriptions", model)
        else:
            stats = run_captioning(items, model, model_cfg, image_dir, out_root,
                                   max_attempts=int(design.get("max_attempts", 3)), workers=workers,
                                   max_tokens=int(design.get("caption_max_tokens", 4096)),
                                   think=bool(model_cfg.get("caption_think", design.get("caption_think", False))),
                                   dry_run=args.dry_run)
            logging.info("[captions:%s] %s", model, json.dumps(stats))
    # Open-weight replication: the prose caption style feeds caption_self_free.
    if args.stage in ("all", "captions") and "caption_self_free" in reps and model_cfg.get("vision", True) and self_captions:
        stats = run_captioning(items, model, model_cfg, image_dir, out_root,
                               max_attempts=int(design.get("max_attempts", 3)), workers=workers,
                               max_tokens=int(design.get("caption_max_tokens", 4096)),
                               think=bool(model_cfg.get("caption_think", design.get("caption_think", False))),
                               dry_run=args.dry_run, style="free")
        logging.info("[captions-free:%s] %s", model, json.dumps(stats))
    # Amendments A3/A4: independent sampled draws of either caption protocol.
    for rep, (style, draw) in DRAW_REPRESENTATIONS.items():
        if args.stage in ("all", "captions") and rep in reps and model_cfg.get("vision", True) and self_captions:
            spec = design["caption_draws"][draw]
            stats = run_captioning(items, model, model_cfg, image_dir, out_root,
                                   max_attempts=int(design.get("max_attempts", 3)), workers=workers,
                                   max_tokens=int(design.get("caption_max_tokens", 4096)),
                                   think=bool(model_cfg.get("caption_think", design.get("caption_think", False))),
                                   dry_run=args.dry_run, style=style, temperature=float(spec["temperature"]),
                                   seed=int(spec["seed"]), draw=draw)
            logging.info("[captions-%s-%s:%s] %s", style, draw, model, json.dumps(stats))
    if args.stage == "captions":
        return 0

    # ---- stage 2: reasoning ------------------------------------------------
    runs = [r for r in design["runs"] if not run_names or r.name in run_names]
    only = model_cfg.get("runs")
    if only:
        runs = [r for r in runs if r.name in only]
    cells, skipped = plan_cells(model, design, reps, cues, runs)
    logging.info("[%s] %d cells x %d items = %d reasoning calls (plus retries)", model, len(cells), len(items), len(cells) * len(items))
    for s in skipped:
        logging.info("[%s] skipped: %s", model, s)
    summary = run_model(model, design, items, out_root=out_root, image_dir=image_dir, repo_root=REPO,
                        representations=reps, cues=cues, run_names=run_names, workers=workers, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    failed = sum(s.get("failed", 0) for s in summary)
    return 1 if failed and not args.dry_run and failed > 0.2 * max(1, sum(s.get("to_run", 0) for s in summary)) else 0


if __name__ == "__main__":
    sys.exit(main())
