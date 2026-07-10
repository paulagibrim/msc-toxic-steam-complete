#!/usr/bin/env python3
"""
Stage 6 — Batch Inference
===========================
Classifies every toxic document in the cleaned dataset using the trained model.

Processing is done file by file to keep memory usage proportional to the
largest single partition rather than the full dataset.

Each classified batch is written to results_dir/batches/ as a separate parquet.
Stage 7 merges these batches into the final output.

Usage:
    python run/06_infer.py                # both en and pt
    python run/06_infer.py --lang en      # just en
    python run/06_infer.py --lang en --no-resume

Expected inputs:
    cleaned_data_dir/*.parquet                (from Stage 1)
    models_dir/final_model/                    (from Stage 5)
    embeddings_dir/pca_model.joblib             (from Stage 2)

Outputs (config_<lang>.yaml → paths.results_dir):
    batches/*.parquet
        Columns: review_url, game_id, review_text_clean, topic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import run_inference
from src.settings import Settings
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 6: classify all toxic documents with the trained model."
    )
    p.add_argument(
        "--lang",
        action="append",
        dest="languages",
        default=None,
        help="Language(s) to run (repeatable: --lang en --lang pt). Defaults to both.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip batch files that already exist (default: True).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Reclassify all files from scratch.",
    )
    return p.parse_args()


def run_one(lang: str, resume: bool) -> None:
    settings = Settings.from_lang(lang)
    settings.create_directories()

    log_file = settings.results_dir / "logs" / "06_infer.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 6 — Batch Inference ===")
    print(f"Model            : {settings.final_model_path}")
    print(f"PCA model        : {settings.pca_model_path}")
    print(f"Input            : {settings.cleaned_data_dir}")
    print(f"Output (batches) : {settings.results_dir / 'batches'}")
    print(f"Resume           : {resume}")
    print()

    batch_dir = run_inference(settings, resume=resume)
    print(f"\n[{lang}] Stage 6 complete. Batches written to: {batch_dir}")


def main() -> None:
    args = parse_args()
    for lang in args.languages or ["en", "pt"]:
        run_one(lang, args.resume)


if __name__ == "__main__":
    main()
