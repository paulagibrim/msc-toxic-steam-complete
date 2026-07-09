"""Cleaning steps for the raw reviews dataset (steam-data/raw/reviews/*.parquet).

Ported from dissertacao-steam/data_refactor/0-cleaning/clean_reviews.py -
Dask-based, since the raw review volume (~97M rows) doesn't fit comfortably
in memory with plain pandas.

Two differences from where this was ported from:
  - `drop_duplicate_reviews` is now actually called (see run_clean_reviews.py)
    instead of being defined but skipped. In dissertacao-steam, deduplicating
    ~73M rows by `review_url` is a Dask shuffle (regroups rows across
    partitions by key, unlike every other step here, which stays within one
    partition at a time) that ran for 28+ minutes and then silently killed
    the kernel on a 24GB machine - deferred there for that reason. This
    project targets a many-core, more-RAM machine specifically to run this
    step; run_clean_reviews.py configures a dask.distributed.Client with
    explicit per-worker memory limits and disk-spilling so the shuffle has
    somewhere to go instead of exhausting RAM outright, rather than assuming
    a bigger machine alone fixes it.
  - `review_lang` (the column export_reviews partitions/filters by) is no
    longer Steam's own declared `language` field - per explicit user
    decision, that field is untrustworthy (see langdetect_revalidation.py's
    module docstring: manual inspection found genuinely English reviews
    filed under Steam's "pt" label) and is, for now, ignored entirely for
    this purpose. `detect_review_language` overwrites `review_lang` with
    langdetect's own guess instead, computed via `map_partitions` (so Dask's
    own worker pool parallelizes it - no separate multiprocessing layer, no
    join against a precomputed table). Steam's original field is kept under
    `perspective_declared_language`, for reference/comparison, but no longer
    drives which language folder a review ends up in.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from langdetect_revalidation import identify_language_langdetect
from pipeline_utils import info, read_parquet_dir, replace_sentinels, safe_astype

COLUMNS_TO_DROP = [
    "respostaDev",
    "linkDentroTexto",
    "title",
    "numComentarios",
    "util",
    "engracada",
    "premios",
]

RENAME_MAP = {
    "recomendado": "is_recommended",
    "dataPublicacao": "review_date",
    "texto": "review_text",
    "horasJogadas": "hours_played",
    "nomeUsuario": "user_url",
    "linkComentario": "review_url",
    "id": "game_id",
    "language": "perspective_declared_language",
    "toxicity": "perspective_score",
}

# Columns that must be present and valid for a review to be usable downstream.
CRITICAL_COLUMNS = ["game_id", "user_url", "review_url", "review_text"]

IS_RECOMMENDED_MAP = {1.0: True, 0.0: False, -1.0: np.nan}

TYPE_MAPPING = {
    "review_text": "string",
    "hours_played": "Float64",
    "user_url": "string",
    "review_url": "string",
    "perspective_declared_language": "string",
    "perspective_score": "Float64",
    "game_id": "int64",
}

# Portuguese month names, lowercased, as they appear in the raw review_date
# strings (e.g. "16 de maio de 2022"). Used instead of locale.setlocale so
# parsing doesn't depend on OS locale availability.
PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def load_raw_reviews(reviews_raw_dir: Path, blocksize=None):
    return read_parquet_dir(reviews_raw_dir, engine="dask", blocksize=blocksize)


def drop_irrelevant_columns(df):
    existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=existing)


def rename_columns(df):
    return df.rename(columns=RENAME_MAP)


def replace_missing_sentinels(df):
    return replace_sentinels(df)


def fix_game_id(df):
    """Parses game_id as numeric; unparseable values become NaN and are
    dropped later by drop_missing_critical (game_id is a critical column)."""
    import dask.dataframe as dd
    df["game_id"] = dd.to_numeric(df["game_id"], errors="coerce")
    return df


def drop_missing_critical(df):
    return df.dropna(subset=CRITICAL_COLUMNS)


def fix_is_recommended(df):
    df["is_recommended"] = df["is_recommended"].map_partitions(
        lambda s: s.map(IS_RECOMMENDED_MAP), meta=("is_recommended", "boolean")
    )
    return df


def _parse_pt_date_partition(series: pd.Series) -> pd.Series:
    lower = series.str.lower()
    parts = lower.str.extract(r"(\d{1,2}) de (\w+) de (\d{4})")
    day, month_name, year = parts[0], parts[1], parts[2]
    month_num = month_name.map(PT_MONTHS).astype("Int64")
    iso = year + "-" + month_num.astype("string").str.zfill(2) + "-" + day.str.zfill(2)
    return pd.to_datetime(iso, format="%Y-%m-%d", errors="coerce").dt.strftime("%Y-%m-%d")


def parse_review_date(df):
    df["review_date"] = df["review_date"].map_partitions(
        _parse_pt_date_partition, meta=("review_date", "string")
    )
    return df


def fix_types(df):
    return safe_astype(df, TYPE_MAPPING)


def drop_duplicate_reviews(df):
    """A Dask shuffle (regroups rows across partitions by review_url) - see
    module docstring for why this needs a tuned distributed Client rather
    than the default scheduler to run safely at this row count."""
    return df.drop_duplicates(subset=["review_url"])


def _detect_language_partition(partition: pd.DataFrame) -> pd.DataFrame:
    detections = partition["review_text"].apply(identify_language_langdetect)
    partition = partition.copy()
    partition["review_lang"] = detections.apply(lambda r: r[0]).astype("string")
    partition["detection_confidence"] = detections.apply(lambda r: r[1]).astype("Float64")
    return partition


def detect_review_language(df):
    """Adds `review_lang` (langdetect's own guess for review_text) and
    `detection_confidence`, computed per-partition - Dask's own worker pool
    parallelizes this across however many workers/threads the
    dask.distributed.Client was configured with, no separate
    multiprocessing layer needed (unlike langdetect_revalidation.py's
    ProcessPoolExecutor-based path, built for validating an already-labeled
    file outside of Dask).

    Rows langdetect can't confidently classify at all (too short/no real
    alphabetic content after cleaning - see identify_language_langdetect's
    min_alpha_length) get `review_lang="und"` - export_reviews doesn't
    filter these out, so they get their own `review_lang=und` folder like
    any other detected language, rather than being silently dropped.
    """
    meta = df._meta.assign(
        review_lang=pd.array([], dtype="string"),
        detection_confidence=pd.array([], dtype="Float64"),
    )
    return df.map_partitions(_detect_language_partition, meta=meta)


def export_reviews(df, processed_reviews_dir: Path):
    """Writes one folder per distinct `review_lang` value actually present
    in `df` (whatever langdetect detected - no fixed language list), each
    folder containing only reviews langdetect assigned to that language."""
    processed_reviews_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_reviews_dir / "reviews_cleaned.parquet"

    df.to_parquet(
        output_path,
        partition_on=["review_lang"],
        write_index=False,
        engine="pyarrow",
    )
    info(f"Exported cleaned reviews dataframe to: {output_path} (partitioned by review_lang)")
    return output_path
