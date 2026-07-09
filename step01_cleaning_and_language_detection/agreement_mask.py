"""Masks run_clean_reviews.py's language-partitioned output down to only
the rows where langdetect's `review_lang` AND Steam's own declared language
(`perspective_declared_language`, preserved from the raw `language` field, but no
longer used to decide the partition - see clean_reviews.py) agree - a
stricter, higher-confidence subset than trusting either source alone.

Only pt and en are handled (per explicit user request) - other partitions
(es, fr, und, etc.) aren't touched by this module. A review already sits in
`review_lang=<lang>` because langdetect assigned it that language; this
module additionally requires `perspective_declared_language == lang` too, so only
rows both sources agree on survive.

The mask itself is a plain `==` on a column that's already there (no model
inference, no extra computation) - cheap enough to apply on demand every
time, so this module doesn't export a second copy of the data, same
reasoning as toxicity_mask.py/language_revalidation.py's masks elsewhere in
this project: duplicating a multi-million-row partition on disk to save a
`==` comparison isn't a good trade.
"""
from pathlib import Path

import dask.dataframe as dd


def load_language_partition(reviews_cleaned_dir: Path, lang: str):
    """Reads one `review_lang=<lang>` partition from
    run_clean_reviews.py's `reviews_cleaned.parquet` output."""
    partition_path = reviews_cleaned_dir / f"review_lang={lang}"
    return dd.read_parquet(str(partition_path))


def apply_agreement_mask(df, lang: str):
    """Keeps only rows where Steam's own declared_language also equals
    `lang` - langdetect already agrees, since `df` is one `review_lang=<lang>`
    partition; this narrows it to rows Steam's raw `language` field agrees
    with too."""
    return df[df["perspective_declared_language"] == lang]
