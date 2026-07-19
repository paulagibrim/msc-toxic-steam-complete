"""
stability_analysis.py — Stage 4: empirical sample-size stability analysis.

Ported unchanged from dissertacao-steam/bertopic_pipeline/src/stability_analysis.py -
generic over settings and Stage 2/3's saved artefacts, no project-specific
paths or column names.

Purpose:
  Training BERTopic on the full toxic corpus might be unnecessarily
  expensive if the topic structure converges at a smaller scale.  This stage
  trains the model at increasing sample sizes using the best hyperparameters
  found in Stage 3, and measures whether the discovered topics stabilise.

Stability criterion:
  Two models trained on sizes N and N+k are considered stable if the mean
  cosine similarity between matched topics (via c-TF-IDF representations)
  exceeds settings.stability_threshold (default 0.85).

  Matching is done greedily: for each topic in the smaller model, the most
  similar topic in the larger model is identified.  The mean of these maximum
  similarities is the stability score.

Output:
  - Per-sample-size metrics logged to MLflow.
  - stability_report.json written to settings.stability_dir.
  - Recommended training size printed to stdout and stored in the report.

The recommended size is passed to Stage 5 via the JSON report, but Stage 5
also accepts a --sample-size CLI flag to override it.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import gc
import mlflow
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP

from .embeddings import load_pca_embeddings, load_toxic_index
from .settings import Settings
from .utils import build_stop_words, set_global_seed, timer

logger = logging.getLogger(__name__)


# ── Topic similarity ───────────────────────────────────────────────────────────

def _compute_topic_overlap(model_a: BERTopic, model_b: BERTopic, top_n: int) -> float:
    """Return the mean maximum cosine similarity between topics of two models.

    For each non-outlier topic in model_a, we find the most similar topic in
    model_b (by cosine similarity of their c-TF-IDF row vectors) and record
    that maximum similarity.  Averaging across topics gives the stability score.

    A score near 1.0 means the two models discovered nearly identical topics.
    A score near 0.0 means the topic structures are completely different.
    """
    def _get_ctfidf(model: BERTopic) -> np.ndarray:
        mat = model.c_tf_idf_
        if issparse(mat):
            mat = mat.toarray()
        else:
            mat = np.array(mat)
        # Exclude the outlier row (topic -1 is the first row in BERTopic's matrix).
        mat = mat[1:, :]
        return normalize(mat, norm="l2")

    ctfidf_a = _get_ctfidf(model_a)
    ctfidf_b = _get_ctfidf(model_b)

    # Truncate to the top_n topics in each model for a fair comparison.
    n = min(top_n, ctfidf_a.shape[0], ctfidf_b.shape[0])
    if n == 0:
        return 0.0

    ctfidf_a = ctfidf_a[:n, :]
    ctfidf_b = ctfidf_b[:n, :]

    # Handle vocabulary mismatch: both c-TF-IDF matrices must share the same
    # column space.  BERTopic can produce different vocabularies at different
    # sample sizes.  We skip the similarity calculation in that case and return
    # NaN so the caller can flag it.
    if ctfidf_a.shape[1] != ctfidf_b.shape[1]:
        logger.warning(
            "Vocabulary size mismatch (%d vs %d); stability score will be NaN.",
            ctfidf_a.shape[1],
            ctfidf_b.shape[1],
        )
        return float("nan")

    sim_matrix = ctfidf_a @ ctfidf_b.T  # shape (n, n)
    max_sims   = sim_matrix.max(axis=1)  # best match per topic in model_a
    return float(np.mean(max_sims))


# ── Single-size run ────────────────────────────────────────────────────────────

def _run_one_size(
    sample_size: int,
    all_pca_embeddings: np.ndarray,
    texts: list,
    best_params: dict,
    settings: Settings,
) -> tuple[BERTopic, dict]:
    """Train a BERTopic model on sample_size documents and return it with metrics."""
    rng = np.random.default_rng(settings.seed)
    idx = rng.choice(len(texts), size=min(sample_size, len(texts)), replace=False)
    idx.sort()

    emb_sub   = all_pca_embeddings[idx]
    texts_sub = [texts[i] for i in idx]

    umap_ = UMAP(
        n_neighbors=best_params["n_neighbors"],
        n_components=best_params["n_components"],
        min_dist=best_params.get("min_dist", 0.1),
        metric="cosine",
        random_state=settings.seed,
        n_jobs=1,
    )
    # min_samples is constrained to not exceed min_cluster_size.
    min_cluster_size = best_params["min_cluster_size"]
    min_samples      = min(best_params.get("min_samples", 5), min_cluster_size)
    hdbscan_ = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    stop_words = build_stop_words(settings.language, settings.extra_stop_words)
    model = BERTopic(
        embedding_model=None,
        umap_model=umap_,
        hdbscan_model=hdbscan_,
        vectorizer_model=CountVectorizer(stop_words=stop_words),
        verbose=False,
    )

    with timer(f"Stability — sample_size={sample_size:,}"):
        topics_, _ = model.fit_transform(texts_sub, emb_sub)

    info  = model.get_topic_info()
    n_topics = len(info[info["Topic"] != -1])

    outlier_rows  = info[info["Topic"] == -1]
    n_outliers    = int(outlier_rows["Count"].sum()) if not outlier_rows.empty else 0
    outlier_rate  = n_outliers / max(len(texts_sub), 1)

    non_outlier = info[info["Topic"] != -1]
    min_topic_size = int(non_outlier["Count"].min()) if not non_outlier.empty else 0

    metrics = {
        "sample_size":    sample_size,
        "n_topics":       n_topics,
        "outlier_rate":   round(outlier_rate, 4),
        "min_topic_size": min_topic_size,
        "stability_score": None,  # filled in by the caller after comparison
    }
    return model, metrics


# ── Public entry point ─────────────────────────────────────────────────────────

def run_stability_analysis(settings: Settings) -> dict:
    """Run BERTopic at each sample size and report which size is sufficient.

    Returns:
        A dict with per-size metrics and the recommended training sample size.
        Also writes stability_report.json to settings.stability_dir.
    """
    set_global_seed(settings.seed)
    best_params = settings.load_best_params()

    logger.info(
        "Stage 4 — Stability Analysis | sample fractions: %s | best_params: %s",
        settings.sample_fractions,
        best_params,
    )

    all_pca_embeddings = load_pca_embeddings(settings)
    df_index           = load_toxic_index(settings)
    texts              = df_index["review_text_clean"].tolist()
    total_available    = len(texts)

    # .as_uri() (not str()) - str() on Windows gives "C:\...\mlruns", and
    # mlflow treats everything before the first ":" as a URI scheme, failing
    # on scheme "c". as_uri() gives "file:///C:/.../mlruns", correct on both
    # Windows and POSIX.
    mlflow.set_tracking_uri(settings.mlruns_dir.as_uri())
    mlflow.set_experiment("bertopic_stability_analysis")

    results       = []
    models_by_size: dict[int, BERTopic] = {}

    with mlflow.start_run(run_name="stability_analysis") as parent_run:
        mlflow.log_params({
            **{f"best_{k}": v for k, v in best_params.items()},
            "sample_fractions": str(settings.sample_fractions),
            "total_toxic":  total_available,
        })

        for fraction in settings.sample_fractions:
            # Same rationale as Stage 3: a fraction of the corpus, not a
            # fixed count, keeps the ladder proportionally identical across
            # languages regardless of each corpus's absolute size.
            size = max(1, round(total_available * fraction))
            actual_size = min(size, total_available)
            logger.info("--- Running sample_fraction=%.0f%% (%d docs) ---", 100 * fraction, actual_size)

            with mlflow.start_run(run_name=f"size_{actual_size}", nested=True):
                model, metrics = _run_one_size(
                    sample_size=actual_size,
                    all_pca_embeddings=all_pca_embeddings,
                    texts=texts,
                    best_params=best_params,
                    settings=settings,
                )
                metrics["sample_fraction"] = fraction
                models_by_size[actual_size] = model

                # Compute stability score relative to the previous size.
                if len(results) > 0:
                    prev_size  = results[-1]["sample_size"]
                    prev_model = models_by_size[prev_size]
                    score = _compute_topic_overlap(
                        prev_model, model, settings.top_n_topics_comparison
                    )
                    metrics["stability_score"] = round(score, 4) if np.isfinite(score) else None
                    logger.info(
                        "Stability score %d→%d: %.4f (threshold: %.2f)",
                        prev_size,
                        actual_size,
                        score,
                        settings.stability_threshold,
                    )

                mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
                results.append(metrics)

                # Release the previous model once we have computed its similarity;
                # we only need to keep the most recent model for the next comparison.
                if len(models_by_size) > 1:
                    prev = models_by_size.pop(list(models_by_size.keys())[0])
                    del prev
                    gc.collect()

        # ── Determine recommended training size ────────────────────────────────
        recommended_size: Optional[int] = None
        for i, r in enumerate(results):
            score = r.get("stability_score")
            if score is not None and np.isfinite(score) and score >= settings.stability_threshold:
                recommended_size = r["sample_size"]
                logger.info(
                    "Topics stabilised at sample_size=%d (score=%.4f ≥ %.2f).",
                    recommended_size,
                    score,
                    settings.stability_threshold,
                )
                break

        if recommended_size is None:
            logger.info(
                "Topics did not stabilise across all tested sizes. "
                "Recommending full dataset training (size=%d).",
                total_available,
            )
            recommended_size = total_available  # sentinel: train on everything

        report = {
            "recommended_sample_size": recommended_size,
            "total_toxic_available":   total_available,
            "stability_threshold":     settings.stability_threshold,
            "best_params":             best_params,
            "per_size_metrics":        results,
        }

        mlflow.log_metric("recommended_sample_size", recommended_size)
        mlflow.log_dict(report, "stability_report.json")

    # ── Persist report ─────────────────────────────────────────────────────────
    settings.stability_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.stability_dir / "stability_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Stability report saved to %s.", report_path)

    # Print a human-readable summary.
    print("\n=== STABILITY ANALYSIS SUMMARY ===")
    for r in results:
        stab = r["stability_score"]
        stab_str = "N/A" if stab is None else f"{stab:.4f}"
        print(
            f"  sample={r['sample_size']:>7,} | topics={r['n_topics']:>3} | "
            f"outliers={r['outlier_rate']*100:5.1f}% | stability={stab_str}"
        )
    if recommended_size < total_available:
        print(f"\n→ Recommended training size: {recommended_size:,} documents.")
    else:
        print(f"\n→ Topics did not stabilise. Train on full dataset ({total_available:,} docs).")

    return report
