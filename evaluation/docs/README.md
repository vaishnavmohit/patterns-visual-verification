# Implemented evaluation protocol

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
