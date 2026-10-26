#!/usr/bin/env python
"""Fold pilot output into the full-run tree so the campaign does not redo it.

    python scripts/merge_pilot.py --from output/factorial/pilot --to output/factorial

Pilot runs write under <output_root>/pilot, which is a separate tree, so by
default a full run repeats work the pilot already did. This copies across only
what is genuinely reusable:

  captions   appended per captioner, skipping any (image, prompt_version)
             already present in the destination
  cells      result rows appended per cell, skipping test_ids already scored;
             raw artifacts and the manifest copied when absent

Rows are moved verbatim, so provenance (prompt hash, snapshot, seed, git
commit) travels with them. Nothing is overwritten and nothing is deleted, so
running this twice is a no-op. Use --dry-run to see the counts first.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _keys(path: Path, fields: tuple[str, ...]) -> set[tuple]:
    out: set[tuple] = set()
    if path.exists():
        for line in path.open():
            if line.strip():
                r = json.loads(line)
                out.add(tuple(r.get(f) for f in fields))
    return out


def merge_jsonl(src: Path, dst: Path, fields: tuple[str, ...], dry: bool) -> tuple[int, dict[str, int]]:
    """Append rows from src not already in dst. Returns (count, by prompt version)."""
    if not src.exists():
        return 0, {}
    have = _keys(dst, fields)
    new, by_ver = [], {}
    for line in src.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok" or tuple(r.get(f) for f in fields) in have:
            continue
        new.append(line)
        v = r.get("prompt_version")
        if v:
            by_ver[v] = by_ver.get(v, 0) + 1
    if new and not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("a") as f:
            f.writelines(new)
    return len(new), by_ver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if src.resolve() == dst.resolve():
        ap.error("source and destination are the same tree")

    total_c = total_r = 0
    for s in sorted(src.glob("captions/*/captions.jsonl")):
        d = dst / s.relative_to(src)
        n, by_ver = merge_jsonl(s, d, ("image", "prompt_version"), a.dry_run)
        total_c += n
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(by_ver.items())) if by_ver else ""
        print(f"captions  {s.parent.name:22s} +{n}" + (f"   ({detail})" if detail else ""))
        for extra in ("manifest.json", "prompt_system.txt"):
            if (s.parent / extra).exists() and not (d.parent / extra).exists() and not a.dry_run:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s.parent / extra, d.parent / extra)

    for s in sorted(src.glob("*/*/*/*/results.jsonl")):
        rel = s.relative_to(src)
        d = dst / rel
        n, _ = merge_jsonl(s, d, ("test_id",), a.dry_run)
        total_r += n
        print(f"cell      {'/'.join(rel.parts[:4]):48s} +{n}")
        if not a.dry_run and n:
            for extra in ("manifest.json", "prompt_example.txt"):
                if (s.parent / extra).exists() and not (d.parent / extra).exists():
                    shutil.copy2(s.parent / extra, d.parent / extra)
            raw_s, raw_d = s.parent / "raw", d.parent / "raw"
            if raw_s.exists():
                raw_d.mkdir(parents=True, exist_ok=True)
                for f in raw_s.glob("*.json"):
                    if not (raw_d / f.name).exists():
                        shutil.copy2(f, raw_d / f.name)

    print(f"\n{'would move' if a.dry_run else 'moved'}: {total_c} captions, {total_r} result rows")
    print("captions from superseded prompt versions are carried over for the prompt-development")
    print("record; the runner only feeds the current version to the reasoning stage.")
    if a.dry_run:
        print("re-run without --dry-run to apply")


if __name__ == "__main__":
    main()
