"""Masks run_clean_reviews.py's output down to only the rows where
langdetect's `review_lang` AND Steam's own declared language
(`perspective_declared_language`, preserved from the raw `language` field, but
not used to decide review_lang's value - see clean_reviews.py) agree - a
stricter, higher-confidence subset than trusting either source alone.

Only pt and en are handled (per explicit user request) - other review_lang
values (es, fr, und, etc.) aren't touched by this module. review_lang is a
plain column (not a directory partition - see clean_reviews.py's module
docstring), so this module filters `review_lang == lang` itself, then
additionally requires `perspective_declared_language == lang` too, so only
rows both sources agree on survive.

The mask itself is a plain `==` on a column that's already there (no model
inference, no extra computation) - cheap enough to apply on demand every
time, so this module doesn't export a second copy of the *data* (the
multi-million-row partition), same reasoning as toxicity_mask.py/
language_revalidation.py's masks elsewhere in this project. The aggregate
*counts* (summarize_agreement/save_agreement_report below) are cheap by
comparison - a handful of numbers per language, not a copy of the rows -
so those are saved to disk as a small report.
"""
from pathlib import Path

import dask.dataframe as dd

from pipeline_utils import info, save_summary


def load_language_partition(reviews_cleaned_dir: Path, lang: str):
    """Reads run_clean_reviews.py's `reviews_cleaned.parquet` output and
    filters to review_lang == lang - review_lang is a plain column there,
    not a directory partition, so every file is read regardless of lang."""
    df = dd.read_parquet(str(reviews_cleaned_dir))
    return df[df["review_lang"] == lang]


def apply_agreement_mask(df, lang: str):
    """Keeps only rows where Steam's own declared_language also equals
    `lang` - `df` is assumed to already be filtered to review_lang == lang
    (see load_language_partition); this narrows it to rows Steam's raw
    `language` field agrees with too."""
    return df[df["perspective_declared_language"] == lang]


def summarize_agreement(df, lang: str) -> dict:
    """Computes agreement counts/percentage for one language partition,
    without keeping the filtered rows around - just the numbers."""
    rows_total = len(df)
    rows_agree = len(apply_agreement_mask(df, lang))
    rows_disagree = rows_total - rows_agree
    agree_pct = 100 * rows_agree / rows_total if rows_total else 0.0

    summary = {
        "language": lang,
        "rows_total": rows_total,
        "rows_agree": rows_agree,
        "rows_disagree": rows_disagree,
        "agree_pct": round(agree_pct, 2),
    }
    info(
        f"[{lang}] {rows_agree} of {rows_total} rows agree "
        f"(langdetect AND Steam both say '{lang}') ({agree_pct:.2f}%)"
    )
    return summary


def save_agreement_report(summaries: list, output_path: Path) -> Path:
    return save_summary({"languages": summaries}, output_path)
