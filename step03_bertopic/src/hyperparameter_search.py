"""
hyperparameter_search.py — Stage 3: Optuna-based hyperparameter optimisation.

Ported unchanged from dissertacao-steam/bertopic_pipeline/src/hyperparameter_search.py -
generic over settings and Stage 2's saved embeddings/index, no project-specific
paths or column names.

What is being optimised?
  BERTopic's quality depends heavily on two sub-models:
    - UMAP (n_neighbors, n_components, min_dist): controls how the high-
      dimensional embeddings are projected into a lower-dimensional space
      before clustering.
    - HDBSCAN (min_cluster_size, min_samples): controls the granularity and
      density sensitivity of the topic clustering.

  The objective function minimises:
      outlier_rate - coherence_weight * coherence_score + min_topics_penalty

  Where:
    - outlier_rate    : fraction of documents assigned to topic -1 (no cluster).
                        Lower is better.
    - coherence_score : c_npmi (or c_v) coherence over the top-N words per
                        topic, computed with gensim.  Higher is better.
    - min_topics_penalty : 0 if n_topics >= settings.min_topics, otherwise
                        1.0 + (min_topics - n_topics) - large enough to always
                        outweigh a normal trial's outlier_rate/coherence terms,
                        growing with how far short the trial falls.

  Below settings.min_topics, the number of topics IS penalised - added after
  an observed failure mode where minimising outlier_rate alone rewards
  collapsing the whole corpus into a handful of huge topics (a pt run
  converged on n_neighbors=45, n_components=5, min_samples=12 and produced
  just 3 topics for hundreds of thousands of toxic reviews: absorbing nearly
  everything into a few dense blobs minimises outlier_rate, and nothing
  countered that incentive). Above the floor, topic count is still not
  otherwise rewarded or penalised - it emerges from the data as before.

Key design decisions:
  - Uses PCA-reduced embeddings (from Stage 2) as UMAP input.  This is
    consistent with how the final model will be trained.
  - Each trial reuses the same pre-encoded, pre-reduced embeddings to avoid
    re-running the encoder or PCA.
  - n_jobs=1 is enforced in UMAP to guarantee reproducibility.
  - The Optuna study is persisted to an SQLite database, allowing the search
    to be resumed after an interruption without losing completed trials.
  - Each trial is logged as a nested MLflow run under the parent search run.
"""

import gc
import json
import logging
from typing import Optional

import mlflow
import numpy as np
import optuna
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .embeddings import load_pca_embeddings, load_toxic_index
from .settings import Settings
from .utils import build_stop_words, set_global_seed, timer

logger = logging.getLogger(__name__)

# Suppress Optuna's per-trial INFO logs; progress is tracked via MLflow.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Coherence computation ──────────────────────────────────────────────────────

def compute_coherence(
    topic_model: BERTopic,
    tokenized: list,
    dictionary: Dictionary,
    metric: str = "c_npmi",
    top_n: int = 10,
) -> float:
    """Compute gensim topic coherence over the topics found by a BERTopic model.

    Args:
        topic_model: a fitted BERTopic instance.
        tokenized:   pre-tokenized corpus (list of list of strings).
                     Must be built ONCE outside the trial loop and reused.
        dictionary:  pre-built gensim Dictionary for the same corpus.
                     Must be built ONCE outside the trial loop and reused.
        metric:      "c_npmi" (faster) or "c_v" (slower, higher quality).
        top_n:       number of top words per topic passed to the coherence model.

    Returns:
        Mean coherence score across all non-outlier topics.
        Returns 0.0 if fewer than 2 valid topics exist (degenerate outcome).

    Performance note:
        tokenized and dictionary are intentionally passed in rather than
        built here because CoherenceModel spawns multiprocessing workers to
        scan the corpus. Rebuilding them inside each trial multiplies that
        cost by n_trials, causing the ~minutes-per-trial slowdown observed
        in production. Building them once and passing them in reduces
        per-trial coherence time to seconds.
    """
    topic_info = topic_model.get_topic_info()
    topic_words = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(topic_id)[:top_n] if w]
        if words:
            topic_words.append(words)

    if len(topic_words) < 2:
        return 0.0

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence=metric,
        processes=1,  # single process avoids multiprocessing overhead per trial
    )
    try:
        score = cm.get_coherence()
    except Exception as exc:
        logger.warning("Coherence computation failed: %s. Returning 0.0.", exc)
        score = 0.0

    return float(score) if np.isfinite(score) else 0.0


# ── Optuna objective ───────────────────────────────────────────────────────────

class _Objective:
    """Callable passed to optuna.study.optimize().

    Encapsulates all trial-invariant data (embeddings, texts, config) to avoid
    global variables and to make the objective unit-testable.
    """

    def __init__(
        self,
        embeddings_pca: np.ndarray,
        texts_sample: list,
        settings: Settings,
        parent_run_id: str,
        total_toxic_available: int,
    ) -> None:
        self.emb           = embeddings_pca
        self.texts         = texts_sample
        self.cfg           = settings
        self.parent_run_id = parent_run_id

        # Pre-compute once; reused by every trial to avoid rebuilding per call.
        # CoherenceModel spawns worker processes to scan the corpus — doing this
        # N_TRIALS times multiplied the cost unnecessarily.
        logger.info("Pre-computing tokenized corpus and gensim dictionary…")
        self.tokenized  = [t.split() for t in texts_sample]
        self.dictionary = Dictionary(self.tokenized)
        logger.info(
            "Dictionary built: %d unique tokens from %d documents.",
            len(self.dictionary),
            len(self.tokenized),
        )

        # min_cluster_size is searched as a FRACTION of the language's total
        # toxic corpus (not the search sample), so the same relative search
        # space applies regardless of how large that corpus is. Resolved to
        # absolute counts once here rather than per-trial. Bounds are computed
        # against total_toxic_available (the full corpus), not len(texts_sample),
        # because that absolute count is what final training (Stage 5, which
        # trains on the full corpus by default) will actually use.
        frac_low, frac_high = self.cfg.hdbscan_search_space["min_cluster_size_fraction"]
        self.min_cluster_size_low  = max(2, round(total_toxic_available * frac_low))
        self.min_cluster_size_high = max(
            self.min_cluster_size_low + 1, round(total_toxic_available * frac_high)
        )
        logger.info(
            "min_cluster_size search bounds resolved to [%d, %d] "
            "(%.4f%%-%.4f%% of %d total toxic documents).",
            self.min_cluster_size_low,
            self.min_cluster_size_high,
            100 * frac_low,
            100 * frac_high,
            total_toxic_available,
        )

    def __call__(self, trial: optuna.Trial) -> float:
        sp = self.cfg

        # ── Sample hyperparameters ─────────────────────────────────────────────
        n_neighbors       = trial.suggest_int("n_neighbors",       *sp.umap_search_space["n_neighbors"])
        n_components      = trial.suggest_int("n_components",      *sp.umap_search_space["n_components"])
        min_dist          = trial.suggest_float("min_dist",        *sp.umap_search_space["min_dist"])
        min_cluster_size  = trial.suggest_int(
            "min_cluster_size", self.min_cluster_size_low, self.min_cluster_size_high
        )
        min_samples       = trial.suggest_int("min_samples",       *sp.hdbscan_search_space["min_samples"])
        # min_samples must not exceed min_cluster_size (HDBSCAN constraint)
        min_samples = min(min_samples, min_cluster_size)

        umap_ = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric="cosine",
            random_state=sp.seed,
            n_jobs=1,        # required for reproducibility
            low_memory=True, # avoids allocating the full distance matrix
        )
        hdbscan_ = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            core_dist_n_jobs=1,
        )
        stop_words_ = build_stop_words(self.cfg.language, self.cfg.extra_stop_words)
        model_ = BERTopic(
            embedding_model=None,  # embeddings are pre-computed
            umap_model=umap_,
            hdbscan_model=hdbscan_,
            vectorizer_model=CountVectorizer(stop_words=stop_words_),
            verbose=False,
        )

        try:
            topics_, _ = model_.fit_transform(self.texts, self.emb)

            topic_info = model_.get_topic_info()
            n_topics   = len(topic_info[topic_info["Topic"] != -1])

            # Outlier rate: fraction of documents in topic -1.
            outlier_rows = topic_info[topic_info["Topic"] == -1]
            n_outliers   = int(outlier_rows["Count"].sum()) if not outlier_rows.empty else 0
            outlier_rate = n_outliers / max(len(self.texts), 1)

            # Coherence over non-outlier topics.
            coherence = compute_coherence(
                model_,
                tokenized=self.tokenized,
                dictionary=self.dictionary,
                metric=sp.coherence_metric,
                top_n=sp.coherence_top_n,
            )

            objective_value = outlier_rate - sp.coherence_weight * coherence

            # Penalise trials below the minimum topic count - see module
            # docstring. The penalty (>= 1.0) always outweighs a normal
            # trial's outlier_rate/coherence terms, so any trial meeting
            # min_topics ranks strictly better than any trial that doesn't,
            # while still growing with the shortfall so Optuna has a
            # gradient to climb back toward min_topics from below.
            min_topics_penalty = 0.0
            if n_topics < sp.min_topics:
                min_topics_penalty = 1.0 + (sp.min_topics - n_topics)
                objective_value += min_topics_penalty

            logger.info(
                "Trial %d — n_topics=%d | outliers=%.1f%% | "
                "coherence=%.4f | min_topics_penalty=%.4f | objective=%.4f",
                trial.number,
                n_topics,
                outlier_rate * 100,
                coherence,
                min_topics_penalty,
                objective_value,
            )

            # Log trial metrics to the nested MLflow run.
            with mlflow.start_run(
                run_name=f"trial_{trial.number:03d}",
                nested=True,
                tags={"parent_run_id": self.parent_run_id},
            ):
                mlflow.log_params({
                    "n_neighbors":      n_neighbors,
                    "n_components":     n_components,
                    "min_dist":         round(min_dist, 4),
                    "min_cluster_size": min_cluster_size,
                    "min_samples":      min_samples,
                })
                mlflow.log_metrics({
                    "n_topics":            n_topics,
                    "outlier_rate":        round(outlier_rate, 4),
                    "coherence":           round(coherence, 4),
                    "min_topics_penalty":  round(min_topics_penalty, 4),
                    "objective":           round(objective_value, 4),
                })

            return objective_value

        finally:
            # Explicit cleanup prevents memory from accumulating across 60 trials.
            del model_, umap_, hdbscan_
            gc.collect()


# ── Public entry point ─────────────────────────────────────────────────────────

def run_search(settings: Settings, resume: bool = True) -> dict:
    """Run the Optuna hyperparameter search (or resume a previous study).

    Args:
        settings: pipeline configuration.
        resume:   if True and the SQLite study already exists, previous trials
                  are preserved and only the remaining n_trials are run.

    Returns:
        best_params dict (also written to settings.best_params_path).
    """
    set_global_seed(settings.seed)

    mlflow.set_tracking_uri(str(settings.mlruns_dir))
    mlflow.set_experiment("bertopic_hyperparameter_search")

    logger.info(
        "Stage 3 — Hyperparameter Search | %d trials | sample fraction: %.0f%%",
        settings.optuna_n_trials,
        100 * settings.optuna_sample_fraction,
    )

    # ── Load pre-computed PCA embeddings ───────────────────────────────────────
    all_pca_embeddings = load_pca_embeddings(settings)
    df_index           = load_toxic_index(settings)

    total_available = len(all_pca_embeddings)
    # A fraction of the corpus, not a fixed count: keeps search coverage
    # proportionally identical across languages/corpus sizes (a fixed N
    # represents a different share of each language's total), and avoids
    # an arbitrary absolute cutoff - see config_<lang>.yaml's optuna section.
    n_sample = max(1, min(total_available, round(total_available * settings.optuna_sample_fraction)))
    logger.info(
        "Sampling %d / %d (%.0f%%) toxic documents for Optuna trials.",
        n_sample, total_available, 100 * settings.optuna_sample_fraction,
    )

    # Fixed-seed sample for reproducibility across trials and reruns.
    rng = np.random.default_rng(settings.seed)
    sample_idx = rng.choice(total_available, size=n_sample, replace=False)
    sample_idx.sort()

    emb_sample   = all_pca_embeddings[sample_idx]
    texts_sample = df_index["review_text_clean"].iloc[sample_idx].tolist()

    # ── Create or resume Optuna study ─────────────────────────────────────────
    settings.optuna_db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=settings.optuna_study_name,
        storage=settings.optuna_storage,
        direction="minimize",
        load_if_exists=resume,   # preserves completed trials on restart
    )
    already_done = len(study.trials)
    remaining    = max(0, settings.optuna_n_trials - already_done)
    logger.info(
        "Optuna study '%s': %d trials already completed, %d remaining.",
        settings.optuna_study_name,
        already_done,
        remaining,
    )

    if remaining == 0:
        logger.info("All trials already completed. Skipping optimisation.")
    else:
        with mlflow.start_run(run_name="optuna_search") as parent_run:
            mlflow.log_params({
                "n_trials":              settings.optuna_n_trials,
                "optuna_sample_fraction": settings.optuna_sample_fraction,
                "optuna_sample_n":       n_sample,
                "coherence_metric":  settings.coherence_metric,
                "coherence_weight":  settings.coherence_weight,
                "embedding_model":   settings.embedding_model_name,
                "pca_components":    settings.pca_components,
            })

            objective = _Objective(
                embeddings_pca=emb_sample,
                texts_sample=texts_sample,
                settings=settings,
                parent_run_id=parent_run.info.run_id,
                total_toxic_available=total_available,
            )

            with timer("Stage 3 — Optuna"):
                study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=settings.optuna_n_jobs,
                    show_progress_bar=True,
                )

            mlflow.log_metrics({
                "best_outlier_rate_approx": study.best_value,
            })
            mlflow.log_params(
                {f"best_{k}": v for k, v in study.best_params.items()}
            )

    best_params = study.best_params
    logger.info("Best hyperparameters: %s", best_params)
    logger.info("Best objective value: %.4f", study.best_value)

    # ── Persist best params ────────────────────────────────────────────────────
    settings.best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best params saved to %s.", settings.best_params_path)

    return best_params
