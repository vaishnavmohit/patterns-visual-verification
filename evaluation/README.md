# Bongard-LOGO evaluation implementation

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
