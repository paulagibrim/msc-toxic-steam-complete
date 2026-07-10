"""
inference.py — Stage 6: classify all toxic documents with the trained model.

Ported unchanged from dissertacao-steam/bertopic_pipeline/src/inference.py -
generic over settings and Stage 1/2/5's saved artefacts, no project-specific
paths or column names.

Why not use .fit_transform() again?
  .fit_transform() would alter the model's learned topic structure.
  .transform() applies the FIXED model (UMAP projection + cosine similarity
  to topic centroids) to new documents without changing any parameters.

Pipeline per batch:
  1. Read one cleaned parquet file.
  2. Filter to toxic rows with valid text.
  3. Encode text with SentenceTransformer → 384-dim embeddings.
  4. Apply the saved PCA model → 50-dim embeddings (same space as training).
  5. Call topic_model.transform(texts, pca_embeddings) → topic ids.
  6. Append result rows (with review_url, game_id) to an output parquet.

Why process file by file?
  Reading and filtering file by file keeps peak RAM proportional to the
  largest single partition rather than the full dataset.

Resume support:
  Each processed file is written to settings.results_dir / "batches/".
  The final export (Stage 7) merges these batches.  On restart, already-
  written batch files are skipped.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .embeddings import load_pca_model
from .settings import Settings
from .training import load_trained_model
from .utils import detect_hardware, get_embedding_device, timer

logger = logging.getLogger(__name__)

# Columns read from each cleaned parquet file.
_READ_COLUMNS = ["review_text_clean", "is_toxic", "review_url", "game_id"]

# Columns written to each batch output file.
_OUTPUT_COLUMNS = ["review_url", "game_id", "review_text_clean", "topic"]


def _get_batch_path(results_dir: Path, source_file: Path) -> Path:
    """Return the output path for the classified batch of a given source file."""
    batch_dir = results_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir / source_file.name


def run_inference(settings: Settings, resume: bool = True) -> Path:
    """Classify all toxic documents file by file and write batch results.

    Args:
        settings: pipeline configuration.
        resume:   if True, skip batch files that already exist on disk.

    Returns:
        Path to the directory containing classified batch parquet files.
    """
    hw     = detect_hardware()
    device = get_embedding_device(hw)

    # Load the trained BERTopic model (embedding model is attached inside).
    topic_model     = load_trained_model(settings)
    pca_model       = load_pca_model(settings)
    embedding_model = SentenceTransformer(
        settings.embedding_model_name, device=device
    )

    input_files = sorted(settings.cleaned_data_dir.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(
            f"No cleaned parquet files found in {settings.cleaned_data_dir}. "
            "Run Stage 1 (01_clean.py) first."
        )

    logger.info(
        "Stage 6 — Inference | %d input files | device: %s",
        len(input_files),
        device,
    )

    batch_dir = settings.results_dir / "batches"
    total_classified = 0

    with timer("Stage 6 — Inference"):
        for f in input_files:
            batch_path = _get_batch_path(settings.results_dir, f)

            if resume and batch_path.exists():
                logger.info("SKIP (already classified): %s", f.name)
                continue

            # Read only the columns we need to minimise memory.
            available = [c for c in _READ_COLUMNS if c in pd.read_parquet(f, columns=None).columns]
            df_file   = pd.read_parquet(f, columns=available)

            # Filter to toxic rows with non-empty text.
            mask = (
                df_file["is_toxic"].fillna(False)
                & df_file["review_text_clean"].notna()
                & (df_file["review_text_clean"].str.strip() != "")
            )
            df_tox = df_file[mask].copy()

            if df_tox.empty:
                logger.info("No toxic rows in %s — skipping.", f.name)
                continue

            texts = df_tox["review_text_clean"].tolist()

            # Encode → PCA → classify.
            raw_emb = embedding_model.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=settings.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

            pca_emb = pca_model.transform(raw_emb).astype(np.float32)

            topics, _ = topic_model.transform(texts, pca_emb)
            df_tox["topic"] = topics

            # Write only the output columns (drop is_toxic, not needed downstream).
            out_cols = [c for c in _OUTPUT_COLUMNS if c in df_tox.columns]
            df_tox[out_cols].to_parquet(batch_path, index=False)

            total_classified += len(df_tox)
            logger.info(
                "Classified %s — %d toxic rows → %s",
                f.name,
                len(df_tox),
                batch_path.name,
            )

    logger.info(
        "Stage 6 complete — total classified: %d | batches dir: %s",
        total_classified,
        batch_dir,
    )
    return batch_dir
