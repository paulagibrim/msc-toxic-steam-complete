#!/usr/bin/env python3
"""
Stage 4 — Sample-Size Stability Analysis
==========================================
Trains BERTopic with the best hyperparameters (from Stage 3) at increasing
sample sizes and measures whether the discovered topic structure converges.

This stage answers the question: "How many documents do we actually need to
train on?" — avoiding unnecessary cost if topic structure stabilises early.

Usage:
    python run/04_stability.py              # both en and pt
    python run/04_stability.py --lang en    # just en

Expected inputs:
    embeddings_pca.npy       (from Stage 2)
    toxic_index.parquet      (from Stage 2)
    best_params.json         (from Stage 3)

Outputs (config_<lang>.yaml → paths.stability_dir):
    stability_report.json
        → "recommended_sample_size" key used by Stage 5
    mlruns/  — one nested MLflow run per sample size
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import Settings
from src.stability_analysis import run_stability_analysis
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 4: sample-size stability analysis for BERTopic."
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

    log_file = settings.results_dir / "logs" / "04_stability.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 4 — Stability Analysis ===")
    print(f"Sample fractions to test : {[f'{f:.0%}' for f in settings.sample_fractions]}")
    print(f"Stability threshold  : {settings.stability_threshold}")
    print(f"Best params          : {settings.best_params_path}")
    print()

    report = run_stability_analysis(settings)
    rec    = report["recommended_sample_size"]
    total  = report["total_toxic_available"]

    if rec < total:
        print(f"\n[{lang}] Recommendation: train Stage 5 on {rec:,} documents.")
    else:
        print(f"\n[{lang}] Recommendation: train Stage 5 on the full dataset ({total:,} docs).")


def main() -> None:
    args = parse_args()
    for lang in args.languages or ["en", "pt"]:
        run_one(lang)


if __name__ == "__main__":
    main()
