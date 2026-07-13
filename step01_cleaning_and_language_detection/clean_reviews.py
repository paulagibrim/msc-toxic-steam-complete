"""Cleaning steps for the raw reviews dataset (steam-data/raw/reviews/*.parquet).

Ported from dissertacao-steam/data_refactor/0-cleaning/clean_reviews.py -
Dask-based, since the raw review volume (~97M rows) doesn't fit comfortably
in memory with plain pandas.

Split across two orchestrating scripts, not one, because the two expensive
steps here want opposite-shaped Dask clusters and mixing them in one Client
config was the source of a lot of instability (worker OOM-restarts, then
inter-worker connection timeouts, depending on which knob got tuned):

  - `drop_duplicate_reviews` (run_clean_reviews_dedup.py) is a Dask
    *shuffle* - every worker exchanges data with every other worker
    (~n_workers^2 connections during the transfer phase). This wants FEW
    workers with LOTS of memory each - fewer workers means less
    all-to-all connection overhead, and more memory per worker survives
    the shuffle's buffering needs. In dissertacao-steam, this step ran for
    28+ minutes and then silently killed the kernel on a 24GB machine -
    deferred there for that reason; this project targets a many-core,
    more-RAM machine specifically to run it, with an explicit
    dask.distributed.Client (real per-worker memory limits + disk-spilling)
    instead of assuming a bigger machine alone fixes it.
  - `detect_review_language` (run_detect_language.py) is pure-Python,
    CPU-bound, per-row work via `map_partitions` - no shuffle, no
    cross-worker communication at all. This wants MANY worker *processes*
    (langdetect holds the GIL, so real parallelism comes from process
    count, not threads) - and doesn't need much memory per worker, since
    there's no shuffle to buffer.

Splitting into two scripts also means the (expensive, multi-minute) dedup
shuffle only ever runs once, checkpointed to disk by
run_clean_reviews_dedup.py's export_deduped - if language detection needs
retuning or crashes, the dedup step never needs to be redone.

`review_lang` is not Steam's own declared `language` field - per explicit
user decision, that field is untrustworthy (see langdetect_revalidation.py's
module docstring: manual inspection found genuinely English reviews filed
under Steam's "pt" label) and is, for now, ignored entirely for this
purpose. Steam's original field is kept under `perspective_declared_language`,
for reference/comparison, but doesn't drive review_lang's value.

`review_lang` is written as a plain column, NOT as Hive-style directory
partitioning - see export_reviews's docstring for why (in short: so a
future re-run of language detection, e.g. after a new boilerplate pattern
is added, only ever changes review_lang's *value* for affected rows,
never which physical file they live in - avoiding a real incident where
downstream steps' filename-based resumability silently kept stale results
for reviews that had actually been reclassified).
"""
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
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
    than the default scheduler to run safely at this row count. In
    practice this shuffle proved unstable even after extensive tuning
    (worker memory-limit restarts poisoning the shuffle's global state,
    see run_clean_reviews_dedup.py) - scatter_to_buckets/dedup_buckets
    below is a shuffle-free alternative for the same result."""
    return df.drop_duplicates(subset=["review_url"])


N_BUCKETS = 200


def _review_url_bucket(url: str, n_buckets: int = N_BUCKETS) -> int:
    """Assigns a review_url to one of n_buckets buckets via a stable hash.
    Every row with the same review_url always lands in the same bucket
    regardless of which partition/worker processes it, so deduplicating
    each bucket independently is equivalent to deduplicating the whole
    dataset - no Dask shuffle needed to guarantee that."""
    return int(hashlib.md5(url.encode("utf-8")).hexdigest(), 16) % n_buckets


def _write_partition_to_buckets(partition: pd.DataFrame, buckets_dir: Path, partition_id: int, n_buckets: int) -> int:
    bucket_ids = partition["review_url"].map(lambda u: _review_url_bucket(u, n_buckets))
    for bucket_id, group in partition.groupby(bucket_ids):
        bucket_dir = buckets_dir / f"bucket={bucket_id}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        group.to_parquet(bucket_dir / f"part-{partition_id}.parquet", index=False, engine="pyarrow")
    return len(partition)


def scatter_to_buckets(df, buckets_dir: Path, n_buckets: int = N_BUCKETS):
    """Shuffle-free alternative to drop_duplicate_reviews (phase 1 of 2):
    splits every partition's rows into n_buckets groups by
    hash(review_url) and writes each group to its own bucket subfolder.
    Every partition writes only its own files, independently of every
    other partition/worker - no cross-worker data movement at all, so
    this can never trigger a Dask shuffle or be poisoned by another
    worker dying mid-transfer. Must be followed by dedup_buckets (outside
    of Dask) to actually remove duplicates within each bucket."""
    buckets_dir.mkdir(parents=True, exist_ok=True)

    def _write(partition, partition_info=None):
        partition_id = partition_info["number"] if partition_info else 0
        rows_written = _write_partition_to_buckets(partition, buckets_dir, partition_id, n_buckets)
        return pd.DataFrame({"rows_written": [rows_written]})

    meta = pd.DataFrame({"rows_written": pd.array([], dtype="int64")})
    return df.map_partitions(_write, meta=meta)


def _dedup_one_bucket(bucket_dir: Path, output_dir: Path) -> tuple:
    files = sorted(bucket_dir.glob("*.parquet"))
    combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows_before = len(combined)
    combined = combined.drop_duplicates(subset=["review_url"])
    rows_after = len(combined)
    combined.to_parquet(output_dir / f"{bucket_dir.name}.parquet", index=False, engine="pyarrow")
    return bucket_dir.name, rows_before, rows_after


def dedup_buckets(buckets_dir: Path, output_dir: Path, n_jobs: int = None) -> dict:
    """Shuffle-free alternative to drop_duplicate_reviews (phase 2 of 2):
    deduplicates each bucket independently, entirely outside of Dask
    (plain ProcessPoolExecutor, one process per bucket). Every row
    sharing a review_url is guaranteed to be inside the same bucket (see
    _review_url_bucket), so a single-process pandas drop_duplicates per
    bucket is equivalent to deduplicating the whole dataset - no
    cross-bucket coordination needed."""
    n_jobs = n_jobs or os.cpu_count()
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_dirs = sorted(buckets_dir.glob("bucket=*"))

    rows_before_total = 0
    rows_after_total = 0
    dedup_fn = partial(_dedup_one_bucket, output_dir=output_dir)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i, (name, before, after) in enumerate(executor.map(dedup_fn, bucket_dirs), start=1):
            rows_before_total += before
            rows_after_total += after
            info(f"[{i}/{len(bucket_dirs)}] {name}: {before} -> {after} row(s)")

    return {
        "buckets": len(bucket_dirs),
        "rows_before_dedup": rows_before_total,
        "rows_final": rows_after_total,
        "rows_dropped_duplicates": rows_before_total - rows_after_total,
    }


def export_deduped(df, output_path: Path) -> Path:
    """Stage 1's checkpoint: the cleaned+deduped reviews, written to disk
    before language detection ever runs. Whatever the dedup shuffle
    naturally settles on as the partition count is what gets written, one
    file per partition, same as any other Dask to_parquet call without
    partition_on."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, write_index=False, engine="pyarrow")
    info(f"Exported deduplicated checkpoint to: {output_path}")
    return output_path


def load_deduped_reviews(path: Path, blocksize=None):
    """Reads stage 1's checkpoint (run_clean_reviews_dedup.py's
    export_deduped output) - already cleaned and deduplicated, not yet
    language-tagged. A plain partitioned read, not a shuffle, so this is
    safe to load with many workers configured."""
    import dask.dataframe as dd
    return dd.read_parquet(str(path), blocksize=blocksize)


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
    filter these out, so they get the same `review_lang="und"` value as
    any other detected language, rather than being silently dropped.
    """
    meta = df._meta.assign(
        review_lang=pd.array([], dtype="string"),
        detection_confidence=pd.array([], dtype="Float64"),
    )
    return df.map_partitions(_detect_language_partition, meta=meta)


def export_reviews(df, processed_reviews_dir: Path):
    """Writes `review_lang` as a plain column, NOT as Hive-style directory
    partitioning (partition_on) - deliberately, so re-running language
    detection later (e.g. after a new boilerplate-stripping pattern is
    added to langdetect_revalidation.py) only ever changes the *value* of
    review_lang for affected rows, never which physical file/folder a row
    lives in. Partitioning by review_lang meant a review whose detected
    language changed on a later run physically moved to a different
    folder - every downstream consumer's "skip if this filename already
    has output" resumability logic then had no way to tell "this file's
    rows are unchanged" from "this file now holds a different review_lang
    mix after reclassification", causing reprocessed reviews to silently
    keep stale results (see this project's actual "produto reembolsado"
    incident). Consumers (step02_run_detoxify) now filter
    `df["review_lang"] == lang` themselves after reading, the same way
    they already filter `perspective_declared_language == lang`."""
    processed_reviews_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_reviews_dir / "reviews_cleaned.parquet"

    df.to_parquet(
        output_path,
        write_index=False,
        engine="pyarrow",
    )
    info(f"Exported cleaned reviews dataframe to: {output_path} (review_lang is a column, not a partition)")
    return output_path
