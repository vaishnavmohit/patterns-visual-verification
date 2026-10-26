#!/usr/bin/env python3
"""Archive the original human-data tree and regenerate anonymized working copies.

The archive is stored under .private/ with restrictive permissions and is ignored
by Git.  Public/working CSVs retain analysis variables but remove direct contact
fields, invitation secrets, raw metadata, mouse traces, and absolute timestamps.
Stable pseudonyms preserve joins across tables without publishing the source IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
PARTICIPANT_RE = re.compile(r"\bpid_\d+\b", re.IGNORECASE)

DROP_COLUMNS = {
    "email",
    "invite_code",
    "enrollment_number",
    "metadata_json",
    "mouse_data_json",
    "consent",
    "share_data",
}

ABSOLUTE_TIME_COLUMNS = {
    "created_at",
    "updated_at",
    "used_at",
    "invite_used_at",
    "first_response_at",
    "last_response_at",
}

TEXT_SUFFIXES = {".md", ".txt", ".json", ".tex"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chmod_private(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)
    root.chmod(0o700)


def collect_values(archive: Path) -> tuple[dict[str, str], dict[str, str]]:
    participants: set[str] = set()
    uuids: set[str] = set()
    for path in archive.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames:
                    continue
                for row in reader:
                    value = (row.get("participant_id") or "").strip()
                    if value:
                        participants.add(value)
                    for cell in row.values():
                        if cell:
                            uuids.update(UUID_RE.findall(cell))
        except (UnicodeDecodeError, csv.Error):
            continue

    participant_map = {
        value: f"P{index:04d}"
        for index, value in enumerate(sorted(participants), start=1)
    }
    uuid_map = {
        value: f"U{index:06d}"
        for index, value in enumerate(sorted(uuids), start=1)
    }
    return participant_map, uuid_map


def replace_identifiers(
    value: str, participant_map: dict[str, str], uuid_map: dict[str, str]
) -> str:
    if not value:
        return value
    if value in participant_map:
        return participant_map[value]
    if value in uuid_map:
        return uuid_map[value]
    value = EMAIL_RE.sub("[redacted-email]", value)
    value = PARTICIPANT_RE.sub(lambda m: participant_map.get(m.group(0), "P_REDACTED"), value)
    value = UUID_RE.sub(lambda m: uuid_map.get(m.group(0), "U_REDACTED"), value)
    return value


def sanitize_csv(
    source: Path,
    target: Path,
    participant_map: dict[str, str],
    uuid_map: dict[str, str],
) -> dict[str, int]:
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            shutil.copy2(source, target)
            return {"rows": 0, "columns_removed": 0}
        removed = set(reader.fieldnames) & (DROP_COLUMNS | ABSOLUTE_TIME_COLUMNS)
        fieldnames = [name for name in reader.fieldnames if name not in removed]
        rows = list(reader)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: replace_identifiers(row.get(name, ""), participant_map, uuid_map)
                    for name in fieldnames
                }
            )
    return {"rows": len(rows), "columns_removed": len(removed)}


def sanitize_text(
    source: Path,
    target: Path,
    participant_map: dict[str, str],
    uuid_map: dict[str, str],
) -> None:
    text = source.read_text(encoding="utf-8")
    text = replace_identifiers(text, participant_map, uuid_map)
    target.write_text(text, encoding="utf-8")


def write_private_manifest(archive: Path, participant_map: dict[str, str], uuid_map: dict[str, str]) -> None:
    manifest = {
        "archive": str(archive),
        "files": {
            str(path.relative_to(archive)): sha256(path)
            for path in sorted(archive.rglob("*"))
            if path.is_file()
        },
        "participant_mapping": participant_map,
        "uuid_mapping": uuid_map,
    }
    manifest_path = archive.parent / "archive_manifest_and_private_mapping.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_path.chmod(0o600)


def publish_curated_tables(target_root: Path, public_root: Path) -> None:
    public_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "human_trials.csv": target_root / "publication" / "trial_level_clean.csv",
        "human_participants.csv": target_root / "publication" / "participant_level_clean.csv",
        "human_items.csv": target_root / "publication" / "item_level_clean.csv",
        "participant_flow.csv": target_root / "publication" / "participant_flow.csv",
    }
    for name, source in sources.items():
        shutil.copy2(source, public_root / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("archive/human_analysis/human_result"))
    parser.add_argument(
        "--private-archive",
        type=Path,
        default=Path(".private/original_human_result_2026-08-07"),
    )
    parser.add_argument("--public-dir", type=Path, default=Path("data/public"))
    args = parser.parse_args()

    source = args.source.resolve()
    archive = args.private_archive.resolve()
    removal_record = Path(__file__).resolve().parents[1] / "analysis/patterns_outputs/identifier_removal_20261014.json"
    removed_archive = Path(__file__).resolve().parents[1] / ".private/original_human_result_2026-08-07"
    if archive == removed_archive and removal_record.exists() and not archive.exists():
        raise SystemExit(
            "The original export archive and identifier mapping were intentionally removed on "
            "14 October 2026. This legacy entry point cannot recreate those originals. "
            "Use the preserved pseudonymized analysis tables."
        )
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, archive)
        chmod_private(archive.parent)

    participant_map, uuid_map = collect_values(archive)
    write_private_manifest(archive, participant_map, uuid_map)

    stats: defaultdict[str, int] = defaultdict(int)
    for original in sorted(archive.rglob("*")):
        if not original.is_file():
            continue
        relative = original.relative_to(archive)
        target = source / relative
        if original.suffix.lower() == ".csv":
            result = sanitize_csv(original, target, participant_map, uuid_map)
            stats["csv_files"] += 1
            stats["csv_rows"] += result["rows"]
            stats["columns_removed"] += result["columns_removed"]
        elif original.suffix.lower() in TEXT_SUFFIXES:
            sanitize_text(original, target, participant_map, uuid_map)
            stats["text_files"] += 1

    publish_curated_tables(source, args.public_dir.resolve())
    print(json.dumps(dict(stats), indent=2))
    print(f"Private archive: {archive}")
    print(f"Curated public tables: {args.public_dir.resolve()}")


if __name__ == "__main__":
    main()
