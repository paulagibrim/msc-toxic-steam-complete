"""
training.py — Stage 5: train the final BERTopic model.

Ported unchanged from dissertacao-steam/bertopic_pipeline/src/training.py -
generic over settings and Stage 2/3/4's saved artefacts, no project-specific
paths or column names.

Input:
  - PCA embeddings from Stage 2 (or a subset, determined by Stage 4).
  - Best hyperparameters from Stage 3 (best_params.json).
  - Recommended sample size from Stage 4 (stability_report.json), or the
    --sample-size CLI flag in run/05_train.py.

The model is saved in safetensors format, which is safer than pickle and
compatible with BERTopic's .load() method.  The c-TF-IDF weights are also
saved (save_ctfidf=True) so topic representations are preserved without
needing to re-fit the model.

The embedding model is NOT saved with BERTopic (by design of the library).
It must be re-instantiated from settings.embedding_model_name during
inference.  Stage 6 handles this automatically.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .embeddings import load_pca_embeddings, load_toxic_index
from .settings import Settings
from .utils import build_stop_words, scale_min_cluster_size, set_global_seed, timer

logger = logging.getLogger(__name__)


def _load_recommended_size(settings: Settings) -> Optional[int]:
    """Read the recommended sample size from the Stage 4 stability report."""
    report_path = settings.stability_dir / "stability_report.json"
    if not report_path.exists():
        logger.warning(
            "Stability report not found at %s. "
            "Will train on the full toxic dataset.",
            report_path,
        )
        return None

    with open(report_path) as f:
        report = json.load(f)

    size = report.get("recommended_sample_size")
    total = report.get("total_toxic_available")

    if size is not None and total is not None and size >= total:
        # Sentinel value: train on everything.
        return None

    return size


def run_training(settings: Settings, sample_size: Optional[int] = "auto") -> BERTopic:
    """Train the final BERTopic model and save it to settings.final_model_path.

    Args:
        settings:    pipeline configuration.
        sample_size: number of documents to train on.
                     - "auto": read from Stage 4 stability_report.json.
                     - None:   use the full toxic dataset.
                     - int:    use exactly this many documents.

    Returns:
        The trained BERTopic model.
    """
    set_global_seed(settings.seed)
    best_params = settings.load_best_params()

    # ── Resolve training size ──────────────────────────────────────────────────
    all_pca_embeddings = load_pca_embeddings(settings)
    df_index           = load_toxic_index(settings)
    texts_all          = df_index["review_text_clean"].tolist()
    total_available    = len(texts_all)

    if sample_size == "auto":
        sample_size = _load_recommended_size(settings)

    if sample_size is None or sample_size >= total_available:
        # Train on everything.
        emb_train   = all_pca_embeddings
        texts_train = texts_all
        actual_size = total_available
        logger.info("Training on full toxic dataset: %d documents.", actual_size)
    else:
        actual_size = min(sample_size, total_available)
        rng = np.random.default_rng(settings.seed)
        idx = rng.choice(total_available, size=actual_size, replace=False)
        idx.sort()
        emb_train   = all_pca_embeddings[idx]
        texts_train = [texts_all[i] for i in idx]
        logger.info("Training on %d / %d documents.", actual_size, total_available)

    # ── Build sub-models ───────────────────────────────────────────────────────
    # min_cluster_size is rescaled to actual_size - see
    # scale_min_cluster_size()'s docstring. Without this, training on the
    # full corpus by default would reuse the raw Optuna-tuned count (~1.2%
    # of its ~30% search sample) as an absolute value representing a much
    # smaller, more permissive fraction (~0.36%) of the full corpus.
    min_cluster_size = scale_min_cluster_size(best_params, actual_size)
    min_samples      = min(best_params.get("min_samples", 5), min_cluster_size)

    umap_model = UMAP(
        n_neighbors=best_params["n_neighbors"],
        n_components=best_params["n_components"],
        min_dist=best_params.get("min_dist", 0.1),
        metric="cosine",
        random_state=settings.seed,
        n_jobs=1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,  # required for .transform() in Stage 6
        core_dist_n_jobs=1,
    )
    # Stop words are applied at the c-TF-IDF vectorization step so that common
    # words in the target language (plus domain terms from the config YAML) do
    # not appear as topic keywords.  This is the correct BERTopic integration
    # point; the text cleaning step removes noise but does not filter stop words.
    stop_words = build_stop_words(settings.language, settings.extra_stop_words)
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    # .as_uri() (not str()) - str() on Windows gives "C:\...\mlruns", and
    # mlflow treats everything before the first ":" as a URI scheme, failing
    # on scheme "c". as_uri() gives "file:///C:/.../mlruns", correct on both
    # Windows and POSIX.
    mlflow.set_tracking_uri(settings.mlruns_dir.as_uri())
    mlflow.set_experiment("bertopic_final_training")

    with mlflow.start_run(run_name="final_training") as run:
        mlflow.log_params({
            **{f"best_{k}": v for k, v in best_params.items()},
            "actual_training_size": actual_size,
            "embedding_model":      settings.embedding_model_name,
            "pca_components":       settings.pca_components,
            "seed":                 settings.seed,
        })

        with timer("Stage 5 — BERTopic Training"):
            topics_, _ = topic_model.fit_transform(texts_train, emb_train)

        info = topic_model.get_topic_info()
        n_topics     = len(info[info["Topic"] != -1])
        outlier_rows = info[info["Topic"] == -1]
        n_outliers   = int(outlier_rows["Count"].sum()) if not outlier_rows.empty else 0
        outlier_rate = n_outliers / max(actual_size, 1)

        mlflow.log_metrics({
            "n_topics":            n_topics,
            "outlier_rate":        round(outlier_rate, 4),
            "training_size":       actual_size,
        })

        logger.info(
            "Training complete — %d topics | outlier rate: %.1f%%",
            n_topics,
            outlier_rate * 100,
        )
        print(info.to_string(index=False))

        # ── Save model ─────────────────────────────────────────────────────────
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        topic_model.save(
            str(settings.final_model_path),
            serialization="safetensors",
            save_ctfidf=True,
        )
        logger.info("Model saved to %s.", settings.final_model_path)
        mlflow.log_artifact(str(settings.final_model_path), artifact_path="model")

        # Record the MLflow run_id in a sidecar file so other stages can
        # attach their artefacts to the same experiment.
        run_id_path = settings.models_dir / "training_run_id.txt"
        run_id_path.write_text(run.info.run_id)

    return topic_model


def load_trained_model(settings: Settings) -> BERTopic:
    """Load the final trained BERTopic model from disk.

    The embedding model is NOT attached here because BERTopic does not save
    it.  Stage 6 attaches the embedding model separately before calling
    .transform().
    """
    from sentence_transformers import SentenceTransformer

    if not settings.final_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {settings.final_model_path}. "
            "Run Stage 5 (05_train.py) first."
        )

    embedding_model = SentenceTransformer(settings.embedding_model_name)
    model = BERTopic.load(str(settings.final_model_path), embedding_model=embedding_model)
    logger.info("Model loaded from %s.", settings.final_model_path)
    return model
