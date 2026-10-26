#!/usr/bin/env python3
"""Prepare a code and machine-output snapshot without manuscripts, human results or Git history.

This only writes a local folder and ZIP. It never pushes or alters a remote.
"""
import argparse
import csv
import hashlib
from io import BytesIO
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'output/release/patterns_public_snapshot'
EXCLUDE = {
    'human_participant_signal_detection.csv', 'identifiability_recovered_share.csv',
    'identifiability_stratified_archive.csv', 'identifiability_stratified_factorial.csv',
    'patterns_factorial_ladder_wide.png', 'figure_signal_detection.pdf',
    'figure_signal_detection.png', 'table_patterns_key_statistics.tex',
    # Withdrawn first identifiability analysis (reversed turn convention; not cited).
    'item_identifiability.csv',
}
README = '''# From supplied structure to self-generated captions in visual rule verification

Code and frozen machine outputs accompanying *From supplied structure to self-generated captions in visual rule verification* (Mohit Vaishnav, Tanel Tammet, Halil Ibrahim Aysel and Meelis Kull).

We compare visual rule verification from images, model-generated captions and descriptions supplied from the benchmark's drawing programs. Controlled experiments in four models examine how much of the supplied-description advantage a captioning pipeline recovers. A broader 15-configuration survey and a complementary human study motivate reporting discrimination, response criterion and answer coverage alongside accuracy. Start with the [implemented evaluation protocol](evaluation/docs/README.md); a plain-language summary is in [`docs/blog.md`](docs/blog.md).

## Highlights from the machine experiments

| Finding | Evidence |
|---|---|
| Program-derived descriptions raise accuracy over images | all 13 paired configurations, +7.0 to +29.1 percentage points (median +17.5) on queries answered under both |
| Answer rates can differ by representation | Qwen3-VL:30b answered 200 image queries but 120 description queries |
| Models' own drawing recipes recover little of that advantage | recovered share near -5% for Gemini 3.5 Flash and GPT 5.5, -14% for Gemma4:31b; Qwen3.6:27b 10-29% across caption sets |
| Prose captions beat recipes on average, with large set-to-set variation | prose minus step d' averaged over four caption sets: +0.43 [0.16, 0.73] (Gemma4:31b), +0.24 [0.02, 0.46] (Qwen3.6:27b) |
| Correct rule cues give higher discrimination estimates than mismatched cues | all nine proprietary and all six open-weight point estimates; uncertainty varies by contrast |

![Representation ladder and cue controls](analysis/patterns_outputs/patterns_factorial_ladder.png)

*No-cue sensitivity across input representations (A) and cue effects on sensitivity (B) and criterion (C) in four models.*

![Similar accuracy, opposite criteria](analysis/patterns_outputs/patterns_model_profiles.png)

*Models with similar image accuracy can lean in opposite directions: one accepts most queries, another rejects most.*

![Prose minus recipes in four caption sets](analysis/patterns_outputs/patterns_caption_sets.png)

*Prose-minus-recipe sensitivity in each open-weight caption set and the four-set mean. Mean-contrast intervals hold captions fixed; descriptive intervals for future sets span zero in both models.*

## Contents

- `evaluation/`: experiment implementation, prompts, configurations and the [implemented protocol](evaluation/docs/README.md).
- `analysis/`: statistical analysis and figure-generation code, plus frozen machine predictions, all caption sets and machine-only analysis outputs.
- `machine_result/clean_machine_data.csv`: consolidated predictions of the 15-model experiment.
- `docs/blog.md`: plain-language summary of the study.
- `scripts/verify_public_snapshot.py`: integrity and exclusion checks.
- `SNAPSHOT_MANIFEST.json`: SHA-256 inventory.

Manuscript sources/PDFs, human datasets, human-derived result files and figures, and institutional correspondence are excluded.

Personal filesystem prefixes in machine CSV and workbook provenance columns are replaced by `external-results/`; model-cache provenance uses `external-model-cache/`. These are logical source labels, not bundled raw files. Filenames, relative directory suffixes, predictions and model-manifest hashes are retained. SLURM log destinations are relative to the job's working directory. `SNAPSHOT_MANIFEST.json` records the affected files and their original SHA-256 digests; the `sha256` inventory describes the distributed copies.

## Install and verify

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-analysis.txt
python scripts/verify_public_snapshot.py
```

## Reproduce machine analyses

These commands use the included frozen machine predictions and make no model API calls:

```sh
python analysis/patterns_statistics.py --machine-only
python analysis/factorial_ow_secondaries.py
python analysis/factorial_ow_analysis.py
python analysis/factorial_ow_analysis.py --strict
python analysis/factorial_ow_reporting_checks.py
python analysis/factorial_ladder_figure.py
python analysis/factorial_caption_sets_figure.py
python analysis/operational_denominators.py
python analysis/factorial_planned_model.py
```

Bootstrap analyses may take several minutes. The integrity check verifies the original snapshot before regeneration; regenerating outputs can change their hashes. See `analysis/README.md` for additional entry points and dependencies. The frozen provider-model outputs are the reference; new API calls may return different results.

To run new experiments, install `evaluation/` with `pip install -e evaluation`, obtain the benchmark inputs and configure provider credentials. See `evaluation/README.md`. Provider keys, benchmark images and original per-shape programs are excluded.

## Human participant data

Human participant data and human-derived outputs are not distributed in this repository. The human-analysis code is included; running it requires permitted participant records. Contact the lead author, Mohit Vaishnav (`mohit.vaishnav@taltech.ee`), about access. Access is subject to permitted sharing arrangements; unrestricted release is not implied.

## Licensing and citation

The MIT license applies to software. Research outputs and third-party benchmark content are not automatically covered by that software license. Obtain Bongard-LOGO inputs from their original distributors and follow their terms. Citation metadata for this software is in `CITATION.cff`.
'''
ANALYSIS_README = '''# Machine results and analysis code

The frozen files in `patterns_outputs/` contain machine predictions, captions and machine-only statistics. No human result tables, human datasets, mixed human/machine summaries or human-derived figures are included.

`patterns_statistics.py --machine-only` regenerates the 15-model comparisons and SDT cell summaries without loading human records. The default command, `human_hierarchical_sdt.py` and the human-dependent figure/alignment routines require participant data obtained through the lead contact and are not runnable from this public snapshot alone.

| Entry point | Purpose and inputs |
|---|---|
| `patterns_statistics.py --machine-only` | Paired representation/cue comparisons and SDT cells of the 15-model experiment from `machine_result/clean_machine_data.csv`. |
| `patterns_submission_assets.py --model-profiles-only` | Model response profiles from the machine CSV alone, without participant records; creates missing figure/output directories. |
| `factorial_planned_model.py` | Proprietary factorial response/correctness robustness from `factorial_trials.csv`. |
| `factorial_ow_analysis.py` | Open-weight primary/strict SDT bootstrap contrasts from `factorial_ow_trials.csv` and `cue_overlap_items.csv`. |
| `factorial_ow_secondaries.py` | Open-weight response/correctness GEE robustness under final-conclusion recovery and strict scoring, including each additional caption set. |
| `factorial_ow_reporting_checks.py` | Decision rates and exploratory common-answered-query sensitivities, for the primary and additional caption sets. |
| `factorial_ladder_figure.py` | Machine representation/cue figure from frozen cells and contrasts. Creates its figure output directory if absent. |
| `operational_denominators.py` | Answered queries per condition and accuracy on answered queries (Table S13). |
| `caption_structure_ow.py` | Open-weight caption fidelity (shape counts, closure, strokes) for every caption set; requires original per-shape programs. |
| `factorial_caption_sets_figure.py` | Prose-minus-step sensitivity across the four caption sets. |
| `qwen36_blind_rerun.py` | Exploratory substitution of successful machine rerun decisions. |
| `ad_rendered_comparison.py` | Renderer-faithful vs original descriptions from frozen machine outputs. |
| `cue_overlap_sensitivity.py` | Requires benchmark query JSONs and cue files in the evaluation source/data layout. |
| `caption_fidelity.py` | Proprietary structural audit; additionally requires original per-shape programs and benchmark query JSONs. |
| `item_identifiability_v2.py` | Renderer-convention boundary diagnostic; requires benchmark JSONs (`--items`) and per-shape programs (`--programs`). |
| `verify_manuscript_numbers.py` | Private-paper consistency checker; requires manuscript sources and human records or their full-precision aggregate summaries, which are excluded here. |

Some scripts retain the original evaluation-code/data layout; supply their documented arguments and external benchmark inputs. The figure-generation source is provided, but its human-dependent entry points require permitted human data and generate local human outputs only after those data are supplied.

Bootstrap intervals condition on fixed caption sets and observed reasoning runs; the four open-weight caption sets per protocol show how much results vary between sets. Prose recovered-share ratios and common-answered-query sensitivities are exploratory. GEE adaptations and the caption-fidelity audits were completed after primary results were known and are robustness checks.

The [implemented evaluation protocol](../evaluation/docs/README.md) describes the conditions, scoring and analysis qualifications. The proprietary execution chronology and deviations are preserved in [`factorial_protocol_provenance.md`](factorial_protocol_provenance.md); names of plans in that historical record refer to the original research repository, not files distributed here. Historical source hashes identify the original analysis environment and are not commits in this clean publication repository. Token/cost/caption-duration figures come from execution records and are not regenerated by these scripts. All caption sets, proprietary and open-weight, are included under `patterns_outputs/factorial_captions/`.
'''
EVALUATION_README = """# Bongard-LOGO evaluation implementation

See the [implemented protocol](docs/README.md) for representations, cue controls,
model settings, caption sets, scoring and analysis qualifications.

Install from the snapshot root with `pip install -e evaluation`. Execute new
experiments from this directory using `scripts/run_factorial.py` and the supplied
configurations. Configure provider credentials through `.env.example`. Obtain
benchmark query JSONs, images and original per-shape programs separately, and set
the input paths expected by the configurations. Benchmark inputs and credentials
are not distributed here.

The frozen predictions used by the analyses are included in the parent snapshot.
Reproducing analyses from those predictions requires no model calls; see the
[parent README](../README.md) and [analysis entry points](../analysis/README.md).
New provider calls may return different outputs. The general design configuration
also retains candidate models from incomplete campaigns; use the completed
condition grid in the protocol to identify the reported experiments.

Software is MIT licensed; third-party benchmark inputs retain their original terms.
"""

EVALUATION_PROTOCOL = """# Implemented evaluation protocol

This guide describes the completed machine experiments for *From supplied
structure to self-generated captions in visual rule verification*. It is a guide
to the implemented methods, not a preregistration or a plan for future work.

## Task and representations

Each problem contains six positive examples, six negative examples and one query
to classify as positive or negative. The shared inventory has 200 queries from
192 Bongard-LOGO base problems, with 50 queries from each of four families and
102 positive and 98 negative queries. Caption sets cover 2,504 distinct figures.

| Input | Source and interpretation |
|---|---|
| Pixels | Benchmark images supplied directly to the model. |
| Self-generated drawing recipes | Each figure captioned independently, without its rule, class membership or task context. |
| Self-generated prose | Geometry, parts and layout described in prose; tested in the open-weight experiments. |
| Supplied Action Descriptions | Descriptions derived from benchmark drawing programs; these provide structure an image-only pipeline must extract. |
| Renderer-faithful descriptions | A control preserving the renderer's turn order and separate shapes; tested without cues in the open-weight experiments. |
| Action Programs | Benchmark drawing programs supplied directly to the reasoning model. |

Supplied descriptions and programs are reference conditions, not outputs of a
learned image-to-text system or theoretical performance ceilings. Caption prompts
can change content, length and structure together, so comparisons do not isolate
syntax alone.

## Completed condition grid

| Campaign | Implemented conditions and reasoning runs |
|---|---|
| Gemini 3.5 Flash | Pixels, own recipes, supplied descriptions and programs; no cue, correct cue and original mismatched cue; one greedy and three sampled reasoning runs (48 condition-run combinations). |
| GPT 5.5 | The same four representations plus Gemini recipes; the same three cue levels; two provider-default sampled reasoning runs (30 combinations). |
| Gemma4:31b and Qwen3.6:27b | Pixels, own recipes and supplied descriptions crossed with five cue levels: none, correct, original mismatched, clean mismatched and irrelevant text. Programs use the three original cue levels; prose and renderer-faithful descriptions use no cue. Twenty conditions with one greedy and one sampled run (40 combinations per model, before extra caption sets). |

The proprietary reasoning output budget was 12,288 tokens. The open-weight
replication used Ollama with thinking explicitly disabled, an 8,192-token
reasoning budget with increases after truncation, and a 32,768-token context.
There were at most three attempts per query, including the initial call. Consult
cell manifests for provider settings and model provenance; a model alias does
not guarantee unchanged weights across evaluations.

Original mismatched labels differ textually from the correct label but may share
meaning. Clean mismatched labels avoid specified attribute-class, content-word
or label overlap; this does not certify that each cue is false for its query.
Irrelevant text controls for placing similarly sized non-geometric text in the
cue slot. The cue files are in `../data/factorial/`.

## Caption sets

Proprietary caption sets were generated once and held fixed across reasoning
runs. GPT also reasoned over Gemini's captions. Each open-weight model received
four complete sets per caption protocol: one greedy set and three sampled sets
at temperature 1 with seeds 2, 3 and 4. Each set was evaluated without cues in
the greedy and sampled reasoning runs. Repeating reasoning over fixed captions
does not measure variability from generating new captions.

## Scoring and comparisons

Primary scores use returned positive/negative decisions, with missing or
unparseable decisions reported through answer coverage. Open-weight primary
scoring additionally recovers the last well-formed `Conclusion` from a malformed
JSON envelope when generation stopped normally. Strict scoring excludes this
fallback, including online and retrospective recoveries. Retrospective recovery
uses saved attempts without new model calls.

Report accuracy, hit and false-alarm rates, signal-detection sensitivity (d'),
response criterion (c), answer coverage and correct answers over the full target
inventory. Use the log-linear rate correction (x + 0.5) / (n + 1). Conventional
d' and c assume equal-variance Gaussian evidence distributions, which a single
binary operating point cannot establish. Compare paired conditions on common
answered queries as well as reporting their individual coverage.

Uncertainty is resampled by base problem within the four families, retaining
both query polarities when present. Proprietary recipe-recovery intervals use
4,000 resamples; open-weight contrasts use 10,000. These intervals condition on
the recorded reasoning runs and captions. The four-set average weights sets
equally; an interval for these fixed sets does not establish an advantage across
future caption sets.

## Analysis status and protocol qualifications

Neither campaign was externally preregistered. The proprietary protocol was
finalized during data collection and retained compatible pilot observations.
Its original mixed-model plan was replaced by post hoc GEE and item-intercept
robustness analyses. Early open-weight campaigns did not complete because the
backend failed to transmit the disabled-thinking setting; their data do not
enter the reported factorial results. The completed open-weight replication used
the corrected backend and a separate design.

Open-weight representation/cue contrasts and the one-half recipe-recovery
threshold were specified in advance. Final-conclusion recovery was introduced
during collection and applied to saved attempts and online in the four remaining
cells. The second caption set supplied the specified Gemma replication test;
the four-set average was an adaptive extension after the first two sets were
evaluated. Caption-fidelity audits, trial-level robustness analyses, prose
recovered-share ratios, common-answered-query checks and later model-profile
contrasts must be interpreted with their post hoc or exploratory status.

The [proprietary execution record](../../analysis/factorial_protocol_provenance.md)
preserves detailed timing and deviations. Commit IDs and original plan names in
that record identify the original research environment, not revisions or
additional files in this publication repository.

## Files and reproduction

- [`../config/factorial/design.yaml`](../config/factorial/design.yaml): proprietary design configuration, including historical candidate models; use the completed grid above.
- [`../config/factorial/design_openweight.yaml`](../config/factorial/design_openweight.yaml): open-weight settings and additional caption draws.
- [`../scripts/run_factorial.py`](../scripts/run_factorial.py): experiment runner.
- [`../src/hmr/factorial/scoring.py`](../src/hmr/factorial/scoring.py): scoring implementation.
- [`../../analysis/README.md`](../../analysis/README.md): analysis entry points, frozen inputs and external requirements.
- `../../analysis/patterns_outputs/factorial_captions/`: frozen proprietary and open-weight caption sets.

Human participant records and human-derived outputs are excluded. Human-analysis
code requires records obtained through the lead contact under permitted sharing
arrangements; it is not runnable from this machine-only snapshot alone.
"""

VERIFY = r'''#!/usr/bin/env python3
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
'''


def sanitize_release_paths(dest):
    """Normalize only release-copy provenance, preserving scientific values."""
    home_prefix = re.compile(r'^/(?:gpfs/[^/]+/(?:home|smbhome)|Users|home)/[^/]+/')
    changes = []
    path = dest / 'machine_result/clean_machine_data.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    path_columns = ('source_results_path', 'metadata_path', 'thinking_text_path')
    assert set(path_columns) <= set(columns)
    changed_cells = 0
    for row in rows:
        for column in path_columns:
            before = row[column]
            row[column] = home_prefix.sub('external-results/', before)
            changed_cells += row[column] != before
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    changes.append({'file': str(path.relative_to(dest)), 'source_sha256': digest,
                    'change': 'personal home prefixes replaced by external-results/',
                    'columns': list(path_columns), 'rows': len(rows), 'changed_cells': changed_cells})
    # Frozen workbooks store the same provenance in inline-string cells. Edit
    # only those XML prefixes, retaining all other XML/ZIP entries byte-for-byte.
    # A workbook engine round trip could alter unrelated numeric/string values.
    xml_prefix = re.compile(rb'(?<=>)/(?:gpfs/[^/<]+/(?:home|smbhome)|Users|home)/[^/<]+/')
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    for path in sorted(dest.rglob('*.xlsx')):
        original = path.read_bytes()
        buffer = BytesIO()
        changed_cells = 0
        with ZipFile(BytesIO(original)) as source, ZipFile(buffer, 'w') as target:
            for info in source.infolist():
                payload = source.read(info.filename)
                if info.filename.startswith('xl/worksheets/') and info.filename.endswith('.xml'):
                    sheet = ET.fromstring(payload)
                    rows = sheet.findall('s:sheetData/s:row', ns)
                    headers = {re.sub(r'\d+$', '', cell.attrib['r']): ''.join(cell.itertext())
                               for cell in rows[0]} if rows else {}
                    allowed = {column for column, name in headers.items() if name in path_columns}
                    expected = 0
                    for row in rows[1:]:
                        for cell in row:
                            value = ''.join(t.text or '' for t in cell.findall('.//s:t', ns))
                            if home_prefix.match(value):
                                assert re.sub(r'\d+$', '', cell.attrib['r']) in allowed, (path.name, cell.attrib['r'])
                                expected += 1
                    payload, count = xml_prefix.subn(b'external-results/', payload)
                    assert count == expected, (path.name, info.filename, count, expected)
                    changed_cells += count
                target.writestr(info, payload)
        if changed_cells:
            path.write_bytes(buffer.getvalue())
            changes.append({'file': str(path.relative_to(dest)),
                            'source_sha256': hashlib.sha256(original).hexdigest(),
                            'change': 'personal home prefixes in workbook provenance cells replaced by external-results/; other XML retained',
                            'columns': ['metadata_path', 'thinking_text_path'],
                            'changed_cells': changed_cells})
    for name in ('run_rerun_prog.sbatch', 'run_rerun_desc.sbatch', 'model_manifest_sha256.txt'):
        path = dest / 'analysis/patterns_outputs/qwen36_rerun' / name
        before = path.read_text()
        if path.suffix == '.sbatch':
            after = re.sub(r'(?m)^(#SBATCH -[oe] )(/\S+)$',
                           lambda match: match[1] + Path(match[2]).name, before)
            change = 'SLURM log destinations made relative to job working directory'
        else:
            digest_text, source_path = before.strip().split(maxsplit=1)
            normalized = home_prefix.sub('external-model-cache/', source_path)
            after = f'{digest_text}  {normalized}\n'
            change = 'personal home prefix replaced by external-model-cache/; model hash retained'
        if after != before:
            changes.append({'file': str(path.relative_to(dest)),
                            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'change': change})
            path.write_text(after)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_evaluation = ROOT / 'evaluation' if (ROOT / 'evaluation').is_dir() else ROOT.parents[1] / 'code/human-machine-reasoning'
    parser.add_argument('--evaluation-root', type=Path, default=default_evaluation)
    parser.add_argument('--repository-url', default='')
    parser.add_argument('--dest', type=Path, default=DEST, help='output folder (default: output/release/patterns_public_snapshot)')
    args = parser.parse_args()
    dest = args.dest.resolve()
    evaluation = args.evaluation_root.resolve()
    assert (evaluation / 'src/hmr/factorial/runner.py').is_file(), 'Evaluation source missing'
    if dest.exists():
        assert (dest / 'SNAPSHOT_MANIFEST.json').is_file(), 'Refuse to replace an unrelated folder'
        assert not (dest / '.git').exists(), 'Refuse to overwrite a Git checkout; move the prepared checkout before rebuilding'
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    sources = set()
    for name in ('scripts/privacy_audit.py','scripts/archive_and_anonymize_human_data.py',
                 'scripts/prepare_patterns_public_snapshot.py',
                 'machine_result/clean_machine_data.csv',
                 'analysis/factorial_protocol_provenance.md'):
        sources.add(ROOT / name)
    for pattern in ('analysis/*.py','analysis/hpc/*.py'):
        sources.update(ROOT.glob(pattern))
    sources.discard(ROOT / 'analysis/identifiability_stratified.py')
    sources.discard(ROOT / 'analysis/item_identifiability.py')
    mixed_or_human = {
        'patterns_statistics.json','submission_checks.json','submission_manifest.json',
        'table_patterns_key_statistics.tex','figure_signal_detection.pdf','figure_signal_detection.png',
    }
    machine_figures = {'patterns_model_profiles.png','patterns_factorial_ladder.png','patterns_caption_sets.png'}
    for path in (ROOT / 'analysis/patterns_outputs').rglob('*'):
        if not path.is_file() or 'superseded' in path.parts or path.name in EXCLUDE or path.name in mixed_or_human:
            continue
        if path.name.startswith(('identifier_removal_', 'additional_review_', 'journal_voice_', 'manuscript_voice_',
                                 'public_snapshot_', 'human_', 'alignment_')):
            continue
        if path.suffix in {'.png','.pdf','.tex'} and path.name not in machine_figures:
            continue
        sources.add(path)
    for path in sorted(sources):
        assert path.is_file() and not path.is_symlink(), path
        target = dest / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    path_changes = sanitize_release_paths(dest)
    # Select only source/configuration files from the separately licensed experiment code.
    for pattern in ('src/hmr/**/*.py','scripts/*.py','config/bongard-logo-human/*.yaml',
                    'config/factorial/*.yaml','data/factorial/*.json'):
        for path in evaluation.glob(pattern):
            assert not path.is_symlink(), path
            target = dest / 'evaluation' / path.relative_to(evaluation)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for name in ('main.py','pyproject.toml','LICENSE','.env.example'):
        path = evaluation / name
        target = dest / 'evaluation' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(evaluation / 'LICENSE', dest / 'LICENSE')
    (dest / 'README.md').write_text(README)
    (dest / 'docs').mkdir(exist_ok=True)
    shutil.copyfile(ROOT / 'submission/blog.md', dest / 'docs/blog.md')
    (dest / 'analysis/README.md').write_text(ANALYSIS_README)
    (dest / 'evaluation/README.md').write_text(EVALUATION_README)
    (dest / 'evaluation/docs').mkdir(exist_ok=True)
    (dest / 'evaluation/docs/README.md').write_text(EVALUATION_PROTOCOL)
    (dest / 'requirements-analysis.txt').write_text('numpy==2.0.2\npandas==2.3.3\nscipy==1.13.1\nstatsmodels==0.14.6\nmatplotlib==3.9.4\nopenpyxl>=3.1\nPillow>=10\n')
    (dest / '.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\n.env\n.DS_Store\n.private/\nhuman_result/\ndata/public/\npaper/\noutput/\n*.zip\n.matplotlib_cache/\n.chart-data-*/\nmachine_result/*/\nmachine_result/*.xlsx\nanalysis/patterns_outputs/superseded/\nanalysis/patterns_outputs/*_manifest_draft*\nanalysis/patterns_outputs/human_*\nanalysis/patterns_outputs/alignment_*\nanalysis/patterns_outputs/patterns_statistics.json\nanalysis/patterns_outputs/submission_checks.json\nanalysis/patterns_outputs/submission_manifest.json\nanalysis/patterns_outputs/figure_signal_detection.*\nanalysis/patterns_outputs/patterns_signal_detection.*\nanalysis/patterns_outputs/patterns_alignment.*\nanalysis/patterns_outputs/patterns_task_design.*\npaper/*.aux\npaper/*.bbl\npaper/*.blg\npaper/*.fdb_latexmk\npaper/*.fls\npaper/*.log\npaper/*.out\npaper/*.spl\n')
    (dest / 'scripts/verify_public_snapshot.py').write_text(VERIFY)
    (dest / 'CITATION.cff').write_text('cff-version: 1.2.0\nmessage: "Please cite the accompanying manuscript when using this code and its research outputs."\ntype: software\ntitle: "From supplied structure to self-generated captions in visual rule verification"\nauthors:\n  - family-names: Vaishnav\n    given-names: Mohit\n  - family-names: Tammet\n    given-names: Tanel\n  - family-names: Aysel\n    given-names: Halil Ibrahim\n  - family-names: Kull\n    given-names: Meelis\n'+('repository-code: "'+args.repository_url+'"\n' if args.repository_url else '')+'license: MIT\n')
    # Compile sources without importing modules or making model calls.
    for path in dest.rglob('*.py'):
        compile(path.read_text(), str(path.relative_to(dest)), 'exec')
    manifest = {'status':'code and frozen machine-output snapshot; manuscripts and all human result/data files excluded',
                'repository_url':args.repository_url, 'git_history_included':False,
                'participant_level_records_included':False, 'human_aggregate_outputs_included':False,
                'manuscripts_included':False,
                'provenance_path_normalization': path_changes,
                'sha256':{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(dest.rglob('*')) if p.is_file()}}
    (dest / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run([sys.executable,str(dest / 'scripts/verify_public_snapshot.py')],check=True)
    archive = dest.parent / 'patterns_public_snapshot.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as bundle:
        for path in sorted(dest.rglob('*')):
            if path.is_file(): bundle.write(path,Path(dest.name) / path.relative_to(dest))
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    print(f'Prepared {len(manifest["sha256"])} files plus manifest: {archive}')


if __name__ == '__main__':
    main()
