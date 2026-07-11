"""Identifies rows in step02's own scored output whose perspective_score
and/or detoxify_score fall outside the valid [0, 1] range - Detoxify's
-1.0 "failed to score" sentinel (see detoxify_scoring.py's batch-failure
handling) is the main case this catches, but both columns are checked
symmetrically/defensively, matching dissertacao-steam's toxicity_mask.py
convention of never trusting a score outside its valid range.

Report only, same reasoning as step01's agreement_mask.py: the mask is a
cheap `.between(0, 1)` check on columns already present, applied on demand
- not worth duplicating the full scored dataset on disk just to save a
comparison. Only the small aggregate counts are persisted.
"""
from pathlib import Path

import pandas as pd

from pipeline_utils import info, list_parquet_files, save_summary


def load_scored_language(step02_dir: Path, lang: str) -> pd.DataFrame:
    """Reads and concatenates the perspective_score/detoxify_score columns
    from every review_lang=<lang> file in step02's own output."""
    partition_dir = step02_dir / f"review_lang={lang}"
    files = list_parquet_files(partition_dir)
    frames = [pd.read_parquet(f, columns=["perspective_score", "detoxify_score"]) for f in files]
    return pd.concat(frames, ignore_index=True)


def apply_validity_mask(df: pd.DataFrame) -> pd.Series:
    """True where BOTH perspective_score and detoxify_score fall inside
    [0, 1] - False for Detoxify's -1.0 "failed to score" sentinel (or any
    other out-of-range value in either column)."""
    return df["perspective_score"].between(0, 1) & df["detoxify_score"].between(0, 1)


def summarize_validity(df: pd.DataFrame, lang: str) -> dict:
    """Counts valid vs. invalid rows for one language. rows_invalid_perspective
    and rows_invalid_detoxify can overlap (a row invalid in both columns
    counts toward both numbers) - they're reported separately so it's clear
    which model is responsible for how much of the invalid total."""
    perspective_valid = df["perspective_score"].between(0, 1)
    detoxify_valid = df["detoxify_score"].between(0, 1)
    valid = perspective_valid & detoxify_valid

    rows_total = len(df)
    rows_valid = int(valid.sum())
    rows_invalid_perspective = int((~perspective_valid).sum())
    rows_invalid_detoxify = int((~detoxify_valid).sum())
    rows_invalid_either = rows_total - rows_valid

    summary = {
        "language": lang,
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "rows_invalid_either": rows_invalid_either,
        "rows_invalid_perspective": rows_invalid_perspective,
        "rows_invalid_detoxify": rows_invalid_detoxify,
        "valid_pct": round(100 * rows_valid / rows_total, 2) if rows_total else 0.0,
    }
    info(
        f"[{lang}] {rows_valid} of {rows_total} rows valid ({summary['valid_pct']:.2f}%) - "
        f"invalid: {rows_invalid_either} "
        f"({rows_invalid_perspective} perspective, {rows_invalid_detoxify} detoxify)"
    )
    return summary


def save_validity_report(summaries: list, output_path: Path) -> Path:
    return save_summary({"languages": summaries}, output_path)
