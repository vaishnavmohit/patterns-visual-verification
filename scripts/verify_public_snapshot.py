#!/usr/bin/env python3
"""Verify this frozen snapshot's hashes and exclusion of participant records."""
import csv
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile
root = Path(__file__).resolve().parents[1]
private_path = re.compile(r'/(?:Users|home|gpfs|scratch|smbhome)/|/private/(?:var/folders|tmp)/|[A-Za-z]:\\Users\\', re.I)
text_suffixes = {'.py', '.md', '.txt', '.json', '.jsonl', '.csv', '.yaml', '.yml', '.toml', '.sbatch', '.sh', '.cff', '.example'}
manifest = json.loads((root / 'SNAPSHOT_MANIFEST.json').read_text())
for name, expected in manifest['sha256'].items():
    path = root / name
    assert path.is_file() and not path.is_symlink(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
for path in root.rglob('*'):
    if any(part in {'.git', '.venv', '__pycache__'} for part in path.relative_to(root).parts):
        continue
    assert path.name not in {'patterns_main.tex', 'patterns_supplementary.tex', 'patterns_main.pdf', 'patterns_supplementary.pdf', 'patterns_statistics.json', 'submission_checks.json', 'factorial_preregistration.md', 'factorial_openweight_plan.md'}, path
    assert not (path.name.startswith(('human_', 'alignment_')) and path.suffix != '.py'), path
    assert '.private' not in path.parts and 'human_result' not in path.parts, path
    if path.is_file() and path.suffix in text_suffixes:
        assert not private_path.search(path.read_text()), f'Private filesystem path in {path.relative_to(root)}'
    if path.is_file() and path.suffix == '.xlsx':
        with ZipFile(path) as workbook:
            for name in workbook.namelist():
                assert not private_path.search(workbook.read(name).decode('utf-8', errors='replace')), f'Private filesystem path in {path.relative_to(root)}:{name}'
    if path.suffix == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as f:
            header = next(csv.reader(f), [])
        assert not (set(header) & {'participant_id', 'email', 'invite_code', 'enrollment_number'}), path
subprocess.run([sys.executable, str(root / 'scripts/privacy_audit.py'), str(root / 'analysis/patterns_outputs'), str(root / 'data/public'), str(root / 'machine_result')], check=True)
print(f"Verified {len(manifest['sha256'])} file hashes; manuscripts, human data/results and private filesystem paths excluded.")
