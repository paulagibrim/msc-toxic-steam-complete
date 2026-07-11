#!/usr/bin/env python3
"""
Stage 3 — Hyperparameter Search (Optuna)
==========================================
Runs Optuna to find the best UMAP and HDBSCAN hyperparameters using a
sample of the toxic dataset. The study is persisted to an SQLite database
so it can be resumed after an interruption without losing completed trials.

Usage:
    python run/03_search.py                      # both en and pt
    python run/03_search.py --lang en             # just en
    python run/03_search.py --lang en --no-resume
    python run/03_search.py --lang en --n-trials 30

Expected inputs:
    embeddings_pca.npy    (from Stage 2)
    toxic_index.parquet   (from Stage 2)

Outputs (paths from config_<lang>.yaml):
    optuna/study.db        — Optuna SQLite study (all trials)
    optuna/best_params.json — best hyperparameters (JSON)
    mlruns/                — MLflow experiment tracking
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hyperparameter_search import run_search
from src.settings import Settings
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: Optuna hyperparameter search for BERTopic."
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
        help="Resume an existing Optuna study (default: True).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Delete any existing study and start fresh.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override the config YAML's optuna.n_trials.",
    )
    return p.parse_args()


def run_one(lang: str, resume: bool, n_trials: int | None) -> None:
    settings = Settings.from_lang(lang)
    settings.create_directories()

    if n_trials is not None:
        settings.optuna_n_trials = n_trials

    if not resume and settings.optuna_db.exists():
        import optuna
        try:
            optuna.delete_study(
                study_name=settings.optuna_study_name,
                storage=settings.optuna_storage,
            )
            print(f"[{lang}] Deleted existing study '{settings.optuna_study_name}'.")
        except Exception:
            pass

    log_file = settings.results_dir / "logs" / "03_search.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 3 — Hyperparameter Search ===")
    print(f"Optuna study     : {settings.optuna_study_name}")
    print(f"Trials           : {settings.optuna_n_trials}")
    print(f"Sample fraction  : {settings.optuna_sample_fraction:.0%}")
    print(f"Coherence metric : {settings.coherence_metric}")
    print(f"DB               : {settings.optuna_db}")
    print()

    best = run_search(settings, resume=resume)
    print(f"\n[{lang}] Best params: {best}")
    print(f"[{lang}] Saved to: {settings.best_params_path}")


def main() -> None:
    args = parse_args()
    for lang in args.languages or ["en", "pt"]:
        run_one(lang, args.resume, args.n_trials)


if __name__ == "__main__":
    main()
