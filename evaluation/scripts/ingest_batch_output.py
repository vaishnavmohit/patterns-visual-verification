"""Ingest a provider batch output JSONL into results.xlsx + folders/.

Use this when a batch was submitted on another machine (e.g. the HPC) and you
downloaded the provider's output file directly from the web console, so there
is no local ``batch_manifest.json``. Per-sample metadata (uid, concept,
caption, category) is reconstructed from the dataset JSON keyed by each
result's ``custom_id`` -- the same source the non-batch path uses via
``read_sample`` -- and the results are written exactly as ``download`` mode
would have written them.

There are two ways to point it at the results:

  * ``--output FILE``   -- ingest a JSONL you already downloaded (no manifest,
                           no API key needed).
  * ``--batch-id ID``   -- fetch the batch status and auto-download the output
                           from the provider first, then ingest it (needs the
                           provider API key in the environment / .env).

Usage:
    # ingest a file you downloaded from the web console
    python scripts/ingest_batch_output.py \
        --config config/bongard-logo-human/ff_dvrl.yaml \
        --output ~/Downloads/batch_..._output.jsonl \
        --model gpt-5.5

    # let the script download the output itself, given the batch id
    python scripts/ingest_batch_output.py \
        --config config/bongard-logo-human/ff_dvrl.yaml \
        --batch-id batch_abc123 \
        --model gpt-5.5

The ``--config`` should be the same YAML used at submit time (so dataset,
json_path, num samples, m/n, temperature, and context match). Override the
model with ``--model`` if the config still points at the default Ollama model.
The output directory is derived from the config the same way the pipeline does;
pass ``--out-dir`` to write somewhere else.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


def _load_config(path: str) -> dict:
    """Load YAML, expanding ${VAR} -- mirrors main.py."""
    with open(path) as f:
        raw = f.read()
    return yaml.safe_load(os.path.expandvars(raw))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config used at submit time.")
    parser.add_argument("--output", default=None,
                        help="Provider batch output .jsonl already on disk.")
    parser.add_argument("--batch-id", default=None,
                        help="Batch id to fetch + auto-download (instead of --output).")
    parser.add_argument("--model", default=None, help="Override the model name from the config.")
    parser.add_argument("--provider", default=None, choices=["openai", "gemini"],
                        help="Force the provider (default: inferred from the model).")
    parser.add_argument("--out-dir", default=None, help="Override the output run directory.")
    args = parser.parse_args()

    if not args.output and not args.batch_id:
        raise SystemExit("Provide either --output (a downloaded file) or --batch-id (to auto-download).")

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    load_dotenv()

    from hmr.batch import (
        BATCH_TERMINAL_GEMINI,
        BATCH_TERMINAL_OPENAI,
        append_batch_results,
        download_batch_output,
        get_batch_status,
        reconstruct_items_from_output,
    )
    from hmr.llm import _backend
    from hmr.pipeline import (
        EXPECTED_KEYS,
        RESULT_COLUMNS,
        _build_output_dir,
        _excel_safe_text,
        _load_dataset_module,
        _write_sample_artifacts,
    )

    params = _load_config(args.config)["model"]["parameters"]

    model = args.model or params["model"]
    provider = args.provider or _backend(model)
    if provider not in {"openai", "gemini"}:
        raise SystemExit(
            f"Ingestion supports OpenAI and Gemini batch outputs only; model "
            f"{model!r} resolves to provider {provider!r}."
        )

    dataset_name = params["dataset"]
    json_path = params["json_path"]
    image_dir = params.get("image_dir", "")
    m = params["m"]
    n = params["n"]
    bongard_context = params["bongard_context"]
    temp = params.get("temperature", 0.0)
    noise_level = params.get("noise_level", 0.0)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = _build_output_dir(
            params["output_base_dir"], model, bongard_context, temp, noise_level,
            params.get("shuffle_test_image", False),
            params.get("shuffle_pos_steps", False),
            params.get("shuffle_neg_steps", False),
            params.get("mix_pos_neg", False),
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.xlsx"

    if results_file.exists():
        df = pd.read_excel(results_file)
        for column in RESULT_COLUMNS:
            if column not in df.columns:
                df[column] = None
    else:
        df = pd.DataFrame(columns=RESULT_COLUMNS)

    if args.output:
        output_path = Path(args.output)
        if not output_path.exists():
            raise SystemExit(f"Output file not found: {output_path}")
    else:
        status = get_batch_status(provider, args.batch_id)
        print(f"Batch {args.batch_id} status: {status.get('status')}")
        terminal = BATCH_TERMINAL_OPENAI if provider == "openai" else BATCH_TERMINAL_GEMINI
        success = "completed" if provider == "openai" else "JOB_STATE_SUCCEEDED"
        if status.get("status") not in terminal:
            raise SystemExit("Batch has not finished yet; try again once it is terminal.")
        if status.get("status") != success:
            # cancelled/expired/failed batches still expose the requests that
            # finished; salvage them if an output file exists.
            if not status.get("output_file_id"):
                raise SystemExit(f"Batch finished unsuccessfully with no output: {status.get('status')}")
            print(f"Batch status is {status.get('status')!r}; salvaging partial output. "
                  f"Request counts: {status.get('request_counts')}")
        output_path = download_batch_output(provider, status, out_dir)
        print(f"Downloaded batch output to {output_path}.")

    dataset_module = _load_dataset_module(dataset_name)
    items = reconstruct_items_from_output(
        output_path=output_path,
        dataset_module=dataset_module,
        json_path=json_path,
        image_dir=image_dir,
        m=m,
        n=n,
    )
    if not items:
        raise SystemExit("No items could be reconstructed from the output file; nothing to ingest.")

    manifest = {"provider": provider, "model": model, "items": items}

    added = append_batch_results(
        provider=provider,
        model=model,
        manifest=manifest,
        out_dir=out_dir,
        results_file=results_file,
        output_path=output_path,
        df=df,
        expected_keys=EXPECTED_KEYS,
        excel_safe_text=_excel_safe_text,
        write_sample_artifacts=_write_sample_artifacts,
    )
    print(f"Reconstructed {len(items)} items; added {added} new rows to {results_file}.")
    print(f"Per-sample artifacts under {out_dir / 'folders'}.")


if __name__ == "__main__":
    main()
