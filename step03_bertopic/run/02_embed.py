#!/usr/bin/env python3
"""
Stage 2 — Embedding Generation
================================
Reads all cleaned parquet files, encodes toxic review text with a
SentenceTransformer model, optionally applies PCA, and saves all artefacts
to disk so every downstream stage can reuse them without re-encoding.

Usage:
    python run/02_embed.py              # both en and pt
    python run/02_embed.py --lang en    # just en

Expected inputs (config_<lang>.yaml → paths.cleaned_data_dir):
    *.parquet files with columns:
        review_text_clean  (str)
        is_toxic           (bool)
        review_url         (str)
        game_id            (str)

Outputs (config_<lang>.yaml → paths.embeddings_dir):
    embeddings_raw.npy   — float32 array of shape (n_toxic, embedding_dim)
    embeddings_pca.npy   — float32 array of shape (n_toxic, pca_components)
    pca_model.joblib     — fitted sklearn PCA model (used in Stage 6 inference)
    toxic_index.parquet  — DataFrame mapping row index → review_url, game_id

IMPORTANT:
    Re-running this stage overwrites all saved embeddings. Only re-run if you
    change the config YAML's embedding.model_name or embedding.pca_components,
    since changing the model makes all downstream artefacts invalid.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import run_embedding
from src.settings import Settings
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2: generate and persist text embeddings."
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

    log_file = settings.results_dir / "logs" / "02_embed.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 2 — Embedding Generation ===")
    print(f"Embedding model  : {settings.embedding_model_name}")
    print(f"PCA components   : {settings.pca_components}")
    print(f"Input            : {settings.cleaned_data_dir}")
    print(f"Output           : {settings.embeddings_dir}")
    print()

    run_embedding(settings)
    print(f"\n[{lang}] Stage 2 complete.")


def main() -> None:
    args = parse_args()
    for lang in args.languages or ["en", "pt"]:
        run_one(lang)


if __name__ == "__main__":
    main()
