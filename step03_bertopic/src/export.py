"""
export.py — Stage 7: merge batch results and write final output artefacts.

Ported unchanged from dissertacao-steam/bertopic_pipeline/src/export.py -
generic over settings and Stage 5/6's saved artefacts, no project-specific
paths or column names.

Outputs:
  - classified_toxic.parquet   : one row per toxic document with topic label,
                                 review_url, game_id, and review_text_clean.
  - topic_info.csv             : topic summary table with real counts from the
                                 full classified dataset (not just training sample).
  - topic_info_real_counts.csv : same as above, with Count_Real replacing the
                                 original training-sample Count.

All outputs are logged as MLflow artefacts so they are versioned alongside the
model and hyperparameters.

Why update the topic counts?
  The BERTopic model is trained on at most a few hundred thousand documents
  (or fewer, if topics stabilised earlier). After classifying the full toxic
  dataset, the per-topic document counts will differ from the training-sample
  counts. Replacing the counts ensures downstream analysis uses accurate
  frequencies.
"""

import logging
from pathlib import Path

import mlflow
import pandas as pd

from .settings import Settings
from .training import load_trained_model
from .utils import timer

logger = logging.getLogger(__name__)


def _merge_batches(batch_dir: Path) -> pd.DataFrame:
    """Concatenate all classified batch parquet files into one DataFrame."""
    batch_files = sorted(batch_dir.glob("*.parquet"))
    if not batch_files:
        raise FileNotFoundError(
            f"No batch files found in {batch_dir}. "
            "Run Stage 6 (06_infer.py) first."
        )

    frames = [pd.read_parquet(f) for f in batch_files]
    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Merged %d batch files → %d rows total.", len(batch_files), len(df)
    )
    return df


def _update_topic_counts(
    df_final: pd.DataFrame,
    topic_model,
) -> pd.DataFrame:
    """Replace training-sample topic counts with counts from the full dataset.

    The model's get_topic_info() still reports counts from the training sample.
    We recompute them from df_final (the fully classified dataset).
    """
    info = topic_model.get_topic_info()

    real_counts = (
        df_final["topic"]
        .value_counts()
        .rename_axis("Topic")
        .reset_index(name="Count_Real")
    )

    # Merge real counts and drop the stale training-sample Count column.
    info_updated = (
        info.drop(columns=["Count"])
        .merge(real_counts, on="Topic", how="left")
        .sort_values("Count_Real", ascending=False)
        .reset_index(drop=True)
    )

    # Reorder columns for readability.
    col_order = ["Topic", "Count_Real", "Name", "Representation", "Representative_Docs"]
    available = [c for c in col_order if c in info_updated.columns]
    return info_updated[available]


def run_export(settings: Settings) -> None:
    """Merge batches, recompute topic counts, and write final artefacts.

    All output files are saved to settings.results_dir and logged to MLflow.
    """
    batch_dir = settings.results_dir / "batches"

    with timer("Stage 7 — Merge batches"):
        df_final = _merge_batches(batch_dir)

    topic_model  = load_trained_model(settings)
    info_updated = _update_topic_counts(df_final, topic_model)
    info_original = topic_model.get_topic_info()

    # ── Write output files ─────────────────────────────────────────────────────
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    df_final.to_parquet(settings.final_results_path, index=False)
    logger.info("Final parquet saved: %s", settings.final_results_path)

    info_updated.to_csv(
        settings.results_dir / "topic_info_real_counts.csv",
        index=False,
        encoding="utf-8-sig",  # BOM for Excel compatibility
    )
    info_original.to_csv(
        settings.topic_info_path,
        index=False,
        encoding="utf-8-sig",
    )
    logger.info("Topic info CSVs saved to %s.", settings.results_dir)

    # ── MLflow ─────────────────────────────────────────────────────────────────
    # .as_uri() (not str()) - str() on Windows gives "C:\...\mlruns", and
    # mlflow treats everything before the first ":" as a URI scheme, failing
    # on scheme "c". as_uri() gives "file:///C:/.../mlruns", correct on both
    # Windows and POSIX.
    mlflow.set_tracking_uri(settings.mlruns_dir.as_uri())
    mlflow.set_experiment("bertopic_export")

    with mlflow.start_run(run_name="final_export"):
        mlflow.log_metrics({
            "total_classified":   len(df_final),
            "n_topics":           len(info_updated[info_updated["Topic"] != -1]),
            "outlier_count":      int(
                info_updated.loc[info_updated["Topic"] == -1, "Count_Real"].sum()
            ),
        })
        mlflow.log_artifact(str(settings.final_results_path))
        mlflow.log_artifact(str(settings.results_dir / "topic_info_real_counts.csv"))
        mlflow.log_artifact(str(settings.topic_info_path))

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n=== EXPORT COMPLETE ===")
    print(f"Total classified:  {len(df_final):,}")
    print(
        f"Topics found:      "
        f"{len(info_updated[info_updated['Topic'] != -1])}"
    )
    print(f"\nTop topics by document count:")
    display_cols = ["Topic", "Count_Real", "Name"]
    available = [c for c in display_cols if c in info_updated.columns]
    print(
        info_updated[info_updated["Topic"] != -1][available]
        .head(20)
        .to_string(index=False)
    )
    print(f"\nOutput files:")
    print(f"  {settings.final_results_path}")
    print(f"  {settings.results_dir / 'topic_info_real_counts.csv'}")
    print(f"  {settings.topic_info_path}")
