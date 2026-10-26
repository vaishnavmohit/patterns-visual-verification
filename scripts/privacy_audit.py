#!/usr/bin/env python3
"""Fail when the distributable tree contains direct human-participant identifiers."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
FORBIDDEN_COLUMNS = {
    "email",
    "invite_code",
    "enrollment_number",
    "metadata_json",
    "mouse_data_json",
    "created_at",
    "updated_at",
    "used_at",
    "invite_used_at",
    "first_response_at",
    "last_response_at",
}
SKIP_SUFFIXES = {".png", ".pdf", ".pickle", ".xlsx", ".svg"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("archive/human_analysis/human_result"), Path("data/public")])
    args = parser.parse_args()
    failures: list[str] = []

    for root in args.roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            emails = [value for value in EMAIL_RE.findall(text) if not value.endswith("@example.com")]
            if emails:
                failures.append(f"{path}: contains {len(emails)} email address(es)")
            uuids = UUID_RE.findall(text)
            if uuids:
                failures.append(f"{path}: contains {len(uuids)} raw UUID(s)")
            if path.suffix.lower() == ".csv":
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    header = next(csv.reader(stream), [])
                forbidden = sorted(set(header) & FORBIDDEN_COLUMNS)
                if forbidden:
                    failures.append(f"{path}: forbidden columns {forbidden}")

    if failures:
        print("Privacy audit FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Privacy audit passed: no direct emails, raw UUIDs, or forbidden columns found.")


if __name__ == "__main__":
    main()
