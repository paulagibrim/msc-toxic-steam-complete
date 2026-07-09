"""Detector-agnostic helpers for cross-checking a review's declared
language against a language detector's own guess for its text - mask
application, mismatch breakdown, saving, and filtering.

Trimmed from dissertacao-steam/data_refactor/2-toxicity/language_revalidation.py:
that version's fastText-specific `validate_language_file` is dropped here -
this project uses langdetect_revalidation.py's
`validate_raw_reviews_langdetect` to produce the same-shaped validation
table instead (review_url, declared_language, detected_language,
detection_confidence). Everything below operates on that table regardless
of which detector produced it.
"""
from pathlib import Path

import pandas as pd

from pipeline_utils import info

# What a detector reports when it can't (or, for langdetect_revalidation.py,
# won't - see MIN_ALPHA_LENGTH there) confidently guess a language. Kept
# distinct from any real ISO language code so it's never confused with one.
UNDETERMINED_LABEL = "und"


def apply_language_mask(validation_df: pd.DataFrame) -> pd.Series:
    """A review is considered valid if the detector's guess matches the
    declared language, OR the detector couldn't confidently guess at all
    (UNDETERMINED_LABEL) - a low-confidence non-answer isn't evidence the
    declared language is wrong, so it isn't excluded. Only a confident guess
    of a *different* language counts as a mismatch."""
    return (validation_df["detected_language"] == validation_df["declared_language"]) | (
        validation_df["detected_language"] == UNDETERMINED_LABEL
    )


def filter_by_language_mask(df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    """Filters `df` (any dataframe with a `review_url` column) down to only
    the rows that pass apply_language_mask, via a `review_url` membership
    check. `validation_df` should be one language's validation table,
    matching `df`'s declared language."""
    valid_urls = set(validation_df.loc[apply_language_mask(validation_df), "review_url"])
    return df[df["review_url"].isin(valid_urls)]


def summarize_mismatches(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Breaks down the rows flagged invalid by apply_language_mask (a
    confident guess of a language other than declared) by which language
    was actually detected, sorted by count descending - answers "how many
    were misidentified, and as which languages" rather than just a single
    mismatch count.

    Returns a `(detected_language, count)` dataframe; also logs it via
    info()."""
    mismatches = validation_df[~apply_language_mask(validation_df)]
    breakdown = (
        mismatches["detected_language"]
        .value_counts()
        .rename_axis("detected_language")
        .reset_index(name="count")
    )

    declared = validation_df["declared_language"].iloc[0] if len(validation_df) else "?"
    info(f"[{declared}] {len(mismatches)} mismatched review(s) - breakdown by detected language:")
    for _, row in breakdown.iterrows():
        pct = 100 * row["count"] / len(mismatches) if len(mismatches) else 0.0
        info(f"  [{declared}] detected as '{row['detected_language']}': {row['count']} ({pct:.2f}%)")
    return breakdown


def save_language_validation(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    info(f"Saved language validation table to: {output_path}")
    return output_path


def save_mismatch_breakdown(breakdown: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(output_path, index=False)
    info(f"Saved mismatch breakdown to: {output_path}")
    return output_path
