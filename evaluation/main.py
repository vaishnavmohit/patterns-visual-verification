"""Entry point: load a YAML config and dispatch to the named module/function.

Usage:
    python main.py --config config/bongard_logo_human/bd.yaml

Env vars in YAML paths are expanded with ${VAR} syntax. This lets the same
config work on different machines: set DATA_DIR / OUTPUT_DIR locally or on HPC.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _load_config(path: str) -> dict:
    """Load YAML, expanding ${VAR} in every string value."""
    with open(path) as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)


def _dispatch(module_name: str, function_name: str, parameters: dict):
    """Import ``module_name`` and call ``function_name(**parameters)``."""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load {function_name} from {module_name}: {e}") from e
    return func(**parameters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model script with a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # Make src/ importable when running from a checkout (no pip install needed)
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    load_dotenv()

    # Surface the retry ladder (budget growth, think disabled, reseed) in the
    # run log. Root stays at WARNING so httpx/openai chatter is not printed.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("hmr").setLevel(logging.INFO)

    config = _load_config(args.config)
    model_cfg = config["model"]
    module_name = model_cfg["module"]
    function_name = model_cfg["function"]
    parameters = model_cfg["parameters"]

    print(f"Running {function_name} from {module_name} with parameters: {parameters}")
    result = _dispatch(module_name, function_name, parameters)
    print(f"Execution complete. Result: {result}")


if __name__ == "__main__":
    main()
