#!/usr/bin/env python3
"""
Stage 7 — Export Final Results
================================
Merges all classified batch parquets from Stage 6, recomputes per-topic
document counts from the full dataset, and writes the final output artefacts.

Usage:
    python run/07_export.py              # both en and pt
    python run/07_export.py --lang en    # just en

Expected inputs:
    results_dir/batches/*.parquet   (from Stage 6)
    models_dir/final_model/          (from Stage 5)

Final outputs (config_<lang>.yaml → paths.results_dir):
    classified_toxic.parquet
        Columns: review_url, game_id, review_text_clean, topic

    topic_info_real_counts.csv
        Topic summary with counts from the FULL classified dataset.

    topic_info.csv
        Topic summary with counts from the training sample (as-is from BERTopic).

All files are also logged as MLflow artefacts.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export import run_export
from src.settings import Settings
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 7: merge batch results and export final artefacts."
    )
    p.add_argument(
        "--lang",
        action="append",
        dest="languages",
        default=None,
        help="Language(s) to run (repeatable: --lang en --lang pt). Defaults to both.",
    )
    return p.parse_args()


def run_one(lang: str) -> None:
    settings = Settings.from_lang(lang)
    settings.create_directories()

    log_file = settings.results_dir / "logs" / "07_export.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 7 — Export ===")
    print(f"Batches dir  : {settings.results_dir / 'batches'}")
    print(f"Output dir   : {settings.results_dir}")
    print()

    run_export(settings)


def main() -> None:
    args = parse_args()
    for lang in args.languages or ["en", "pt"]:
        run_one(lang)


if __name__ == "__main__":
    main()
