"""
embeddings.py — Stage 2: generate and persist text embeddings with optional PCA.

Ported unchanged from dissertacao-steam/bertopic_pipeline/src/embeddings.py -
generic over settings.cleaned_data_dir, no project-specific paths or column
names beyond what Stage 1's output already fixes (review_text_clean, is_toxic,
review_url, game_id).

Design decisions:
  - Embeddings are generated once and saved to disk as .npy arrays.
    Re-running any downstream stage (Optuna, stability, training, inference)
    never needs to re-encode text, which on GPU typically takes 10–30 minutes.

  - PCA is applied to reduce embedding dimensionality from 384 to pca_components
    (default 50) before UMAP.  UMAP's time and memory scale roughly O(n^2),
    so this reduction is critical for datasets with hundreds of thousands of
    documents.  PCA itself is O(n * d^2) and fast for d=384.

  - The PCA model is fitted on ALL toxic documents and saved with joblib.
    The same fitted PCA is applied in Stage 6 (batch inference) so that the
    UMAP model always receives vectors in the same 50-dimensional space it
    was trained in.

  - A toxic_index parquet is saved alongside the embeddings to record which
    rows (review_url, game_id) correspond to each embedding position.  This
    index is required to reassemble the final classified dataset in Stage 7.

  - The device (CPU vs GPU) is auto-detected at runtime so the same code
    works on a workstation and on a GPU-equipped server without changes.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from .settings import Settings
from .utils import detect_hardware, get_embedding_device, log_hardware, set_global_seed, timer

logger = logging.getLogger(__name__)

# Columns loaded from each cleaned parquet file during embedding generation.
# Keeping the list minimal reduces peak RAM when reading many partition files.
_LOAD_COLUMNS = ["review_text_clean", "is_toxic", "review_url", "game_id"]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_all_toxic_texts(settings: Settings) -> pd.DataFrame:
    """Read all cleaned parquet files and return a DataFrame of toxic rows only.

    Only the columns needed for embedding and for the output index are loaded.
    """
    files = sorted(settings.cleaned_data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No cleaned parquet files found in {settings.cleaned_data_dir}. "
            "Run Stage 1 (01_clean.py) first."
        )

    frames = []
    for f in files:
        available = [c for c in _LOAD_COLUMNS if c in pd.read_parquet(f, columns=None).columns]
        df = pd.read_parquet(f, columns=available)
        mask = df["is_toxic"].fillna(False) & df["review_text_clean"].notna()
        mask &= df["review_text_clean"].str.strip() != ""
        frames.append(df[mask])

    df_all = pd.concat(frames, ignore_index=True)
    logger.info("Toxic documents loaded: %d", len(df_all))
    return df_all


def _encode_texts(
    texts: list,
    model_name: str,
    batch_size: int,
    device: str,
    normalize: bool,
) -> np.ndarray:
    """Encode a list of strings into a float32 embedding matrix."""
    logger.info(
        "Loading embedding model '%s' on device '%s'.", model_name, device
    )
    model = SentenceTransformer(model_name, device=device)

    logger.info("Encoding %d texts (batch_size=%d)…", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def _fit_and_apply_pca(
    embeddings: np.ndarray,
    n_components: int,
    pca_model_path: Path,
    seed: int,
) -> np.ndarray:
    """Fit PCA on embeddings, save the model, and return reduced embeddings."""
    logger.info(
        "Fitting PCA: %d dims → %d dims on %d documents.",
        embeddings.shape[1],
        n_components,
        embeddings.shape[0],
    )
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(embeddings).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA fitted — explained variance retained: %.1f%%", explained * 100
    )

    joblib.dump(pca, pca_model_path)
    logger.info("PCA model saved to %s.", pca_model_path)

    return reduced


# ── Public API ─────────────────────────────────────────────────────────────────

def load_pca_model(settings: Settings) -> PCA:
    """Load the fitted PCA model from disk.

    Called by inference.py so the same transformation applied during training
    is also applied during batch inference.
    """
    if not settings.pca_model_path.exists():
        raise FileNotFoundError(
            f"PCA model not found at {settings.pca_model_path}. "
            "Run Stage 2 (02_embed.py) first."
        )
    return joblib.load(settings.pca_model_path)


def load_pca_embeddings(settings: Settings) -> np.ndarray:
    """Load the PCA-reduced embedding matrix saved by Stage 2."""
    if not settings.pca_embeddings_path.exists():
        raise FileNotFoundError(
            f"PCA embeddings not found at {settings.pca_embeddings_path}. "
            "Run Stage 2 (02_embed.py) first."
        )
    return np.load(settings.pca_embeddings_path)


def load_toxic_index(settings: Settings) -> pd.DataFrame:
    """Load the toxic document index saved by Stage 2.

    The index maps each row position in the embedding matrix to the
    corresponding review_url and game_id.
    """
    if not settings.toxic_index_path.exists():
        raise FileNotFoundError(
            f"Toxic index not found at {settings.toxic_index_path}. "
            "Run Stage 2 (02_embed.py) first."
        )
    return pd.read_parquet(settings.toxic_index_path)


def run_embedding(settings: Settings) -> None:
    """Generate embeddings for all toxic documents and persist all artefacts.

    Outputs written to settings.embeddings_dir:
      - embeddings_raw.npy   : float32 array of shape (n_toxic, 384)
      - embeddings_pca.npy   : float32 array of shape (n_toxic, pca_components)
      - pca_model.joblib     : fitted sklearn PCA model
      - toxic_index.parquet  : DataFrame with review_url, game_id, text columns
    """
    set_global_seed(settings.seed)

    hw = detect_hardware()
    log_hardware(hw)
    device = get_embedding_device(hw)

    with timer("Stage 2 — Load toxic texts"):
        df_toxic = _load_all_toxic_texts(settings)

    texts = df_toxic["review_text_clean"].tolist()

    with timer("Stage 2 — Encode embeddings"):
        embeddings = _encode_texts(
            texts=texts,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            device=device,
            normalize=settings.normalize_embeddings,
        )

    logger.info(
        "Raw embeddings shape: %s | size on disk: ~%.1f MB",
        embeddings.shape,
        embeddings.nbytes / (1024 ** 2),
    )

    np.save(settings.raw_embeddings_path, embeddings)
    logger.info("Raw embeddings saved to %s.", settings.raw_embeddings_path)

    # ── PCA reduction ──────────────────────────────────────────────────────────
    if settings.pca_components is not None:
        with timer("Stage 2 — PCA"):
            embeddings_pca = _fit_and_apply_pca(
                embeddings=embeddings,
                n_components=settings.pca_components,
                pca_model_path=settings.pca_model_path,
                seed=settings.seed,
            )
        np.save(settings.pca_embeddings_path, embeddings_pca)
        logger.info("PCA embeddings saved to %s.", settings.pca_embeddings_path)
    else:
        # If PCA is disabled, the "PCA embeddings" file is just a copy of the raw.
        # Downstream stages always load pca_embeddings_path so the API stays uniform.
        np.save(settings.pca_embeddings_path, embeddings)
        logger.info("PCA disabled — raw embeddings copied to pca_embeddings_path.")

    # ── Toxic index ────────────────────────────────────────────────────────────
    # Keep only the metadata columns needed for the final output.
    index_cols = [c for c in ["review_url", "game_id", "review_text_clean"] if c in df_toxic.columns]
    df_toxic[index_cols].reset_index(drop=True).to_parquet(
        settings.toxic_index_path, index=True  # row position = embedding row position
    )
    logger.info(
        "Toxic index saved (%d rows) to %s.", len(df_toxic), settings.toxic_index_path
    )
