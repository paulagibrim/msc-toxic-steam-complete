#!/usr/bin/env python3
"""
Stage 5 — Final BERTopic Training
====================================
Trains the final BERTopic model using:
  - The best hyperparameters from Stage 3 (best_params.json).
  - The recommended sample size from Stage 4 (stability_report.json), or a
    manually supplied --sample-size value, or the full toxic dataset.

Usage:
    python run/05_train.py                              # both en and pt
    python run/05_train.py --lang en                     # just en
    python run/05_train.py --lang en --sample-size 200000
    python run/05_train.py --lang en --sample-size all

Expected inputs:
    embeddings_pca.npy       (from Stage 2)
    toxic_index.parquet      (from Stage 2)
    best_params.json         (from Stage 3)
    stability_report.json    (from Stage 4, optional)

Outputs (config_<lang>.yaml → paths.models_dir):
    final_model/   — BERTopic model in safetensors format
    mlruns/        — MLflow training run

IMPORTANT:
    The embedding model is NOT saved with BERTopic. Stage 6 (inference)
    reloads it from the config YAML's embedding.model_name, which must
    remain identical to the model used in Stage 2.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import Settings
from src.training import run_training
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 5: final BERTopic model training."
    )
    p.add_argument(
        "--lang",
        action="append",
        dest="languages",
        default=None,
        help="Language(s) to run (repeatable: --lang en --lang pt). Defaults to both.",
    )
    p.add_argument(
        "--sample-size",
        default="auto",
        help=(
            "Number of documents to train on. "
            "'auto' reads the Stage 4 recommendation; "
            "'all' or 0 trains on the full toxic dataset; "
            "any positive integer uses that exact count."
        ),
    )
    return p.parse_args()


def _parse_sample_size(value: str) -> Optional[int]:
    """Convert the --sample-size string to an int or None (= full dataset)."""
    if value in ("auto",):
        return "auto"
    if value in ("all", "0", "full"):
        return None
    try:
        n = int(value)
        return None if n <= 0 else n
    except ValueError:
        raise ValueError(f"Invalid --sample-size value: '{value}'")


def run_one(lang: str, sample_size_arg: str) -> None:
    settings = Settings.from_lang(lang)
    settings.create_directories()

    log_file = settings.results_dir / "logs" / "05_train.log"
    setup_logging(log_file)

    sample_size = _parse_sample_size(sample_size_arg)

    print(f"\n=== [{lang}] Stage 5 — Final Training ===")
    print(f"Best params path : {settings.best_params_path}")
    print(f"Sample size      : {sample_size_arg}")
    print(f"Model output     : {settings.final_model_path}")
    print()

    run_training(settings, sample_size=sample_size)
    print(f"\n[{lang}] Stage 5 complete. Model saved to: {settings.final_model_path}")


def main() -> None:
    args = parse_args()
    for lang in args.languages or ["en", "pt"]:
        run_one(lang, args.sample_size)


if __name__ == "__main__":
    main()
