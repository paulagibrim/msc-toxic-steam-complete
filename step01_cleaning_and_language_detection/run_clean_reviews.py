"""CLI for clean_reviews.py - cleans steam-data's raw review files,
deduplicates by review_url (unlike dissertacao-steam, where this was
deferred - see clean_reviews.py's module docstring), and assigns each
review's language purely from langdetect - Steam's own declared `language`
field is ignored entirely for this (kept only as `perspective_declared_language`,
for reference), per explicit user decision: that field mislabels a
meaningful share of reviews (see langdetect_revalidation.py's module
docstring).

No fixed language list: output has one `review_lang=<code>` folder per
distinct language langdetect actually finds (including `und`, for reviews
too short/low-signal to classify confidently) - see
clean_reviews.export_reviews's docstring.

Usage:
    python run_clean_reviews.py \\
        --input ../../steam-data/raw/reviews \\
        --output-dir ../../steam-data/processed/reviews_by_lang \\
        --n-workers 8 --threads-per-worker 4 --memory-limit 4GB

Step order (through fix_types) matches
dissertacao-steam/data_refactor/0-cleaning/03_clean_reviews.ipynb.
drop_duplicate_reviews runs right after, before export.

WHY A TUNED dask.distributed.Client (this is the important part):
Deduplicating ~73M+ rows by review_url is a Dask *shuffle* (rows have to be
regrouped across partitions by key) - unlike every other cleaning step here,
which stays within one partition at a time. In dissertacao-steam, running
this with Dask's default (non-distributed) scheduler - which has no
per-worker memory limit and no disk-spilling - ran for 28+ minutes and then
silently killed the kernel on a 24GB machine. A bigger machine alone doesn't
fix that; the default scheduler still won't spill to disk when memory gets
tight, it'll just OOM later. Creating an explicit dask.distributed.Client
with a real memory_limit per worker gives Dask's distributed workers actual
memory awareness - they spill intermediate shuffle data to disk
automatically as usage approaches the limit (a distributed worker's default
thresholds: spill at 70%, pause new work at 80%, terminate at 95% of
memory_limit) instead of trying to hold everything in RAM until something
breaks.

Defaults below (8 workers x 4 threads x 4GB = 32GB) assume roughly a 40GB/
48-core machine - override via the CLI flags for whatever machine this
actually runs on (leave some RAM headroom for the OS/scheduler itself,
don't allocate all of it to workers).

Besides reviews_cleaned.parquet, writes (into --output-dir) everything the
original notebook printed as cell output or logged to MLflow, minus MLflow
itself - no null_summary_reviews.csv though, since the original notebook
never exported one for reviews (only games/users got that):
  - sample_reviews.csv (first 20 rows, matching the notebook's own head(20))
  - reviews_report.json (columns/dtypes/rows at each step)
"""
import argparse
from pathlib import Path

import clean_reviews as cr
from pipeline_utils import export_sample, info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleans and deduplicates steam-data's raw review files."
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory of raw reviews *.parquet files")
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write reviews_cleaned.parquet (partitioned by review_lang) to",
    )
    parser.add_argument("--n-workers", type=int, default=8, help="Number of Dask worker processes")
    parser.add_argument("--threads-per-worker", type=int, default=4, help="Threads per Dask worker")
    parser.add_argument(
        "--memory-limit", default="4GB",
        help="Memory limit per worker (e.g. '4GB') - workers spill to disk as they approach this",
    )
    parser.add_argument(
        "--local-directory", type=Path, default=Path("./dask-worker-space"),
        help="Directory Dask workers use to spill data to disk under memory pressure",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from dask.distributed import Client

    args.local_directory.mkdir(parents=True, exist_ok=True)
    client = Client(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        local_directory=str(args.local_directory),
    )
    info(f"Dask dashboard: {client.dashboard_link}")
    info(
        f"Cluster: {args.n_workers} worker(s) x {args.threads_per_worker} thread(s), "
        f"{args.memory_limit} each"
    )

    try:
        df = cr.load_raw_reviews(args.input)
        columns_loaded = df.columns.tolist()
        info(f"Columns: {columns_loaded}")

        df = cr.drop_irrelevant_columns(df)
        df = cr.rename_columns(df)
        info(f"Columns after rename: {df.columns.tolist()}")
        df = cr.replace_missing_sentinels(df)
        df = cr.fix_game_id(df)

        rows_loaded = len(df)
        df = cr.drop_missing_critical(df)
        rows_dropped_missing_critical = rows_loaded - len(df)
        info(f"Dropped {rows_dropped_missing_critical} row(s) missing a critical column")

        df = cr.fix_is_recommended(df)
        df = cr.parse_review_date(df)
        df = cr.fix_types(df)

        rows_before_dedup = len(df)
        df = cr.drop_duplicate_reviews(df)
        rows_dropped_duplicates = rows_before_dedup - len(df)
        info(f"Dropped {rows_dropped_duplicates} duplicate row(s) by review_url")

        df = cr.detect_review_language(df)
        info("Assigned review_lang from langdetect (Steam's declared language is ignored for this)")

        rows_final = len(df)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        language_counts = df["review_lang"].value_counts().compute().sort_values(ascending=False)
        info(f"Final: {rows_final} rows, {len(df.columns)} columns, {len(language_counts)} language(s) detected")
        for lang, count in language_counts.items():
            info(f"  {lang}: {count}")

        export_sample(df, args.output_dir / "sample_reviews.csv", n=20)
        save_summary(
            {
                "columns_loaded": columns_loaded,
                "rows_loaded": rows_loaded,
                "rows_dropped_missing_critical": rows_dropped_missing_critical,
                "rows_dropped_duplicates": rows_dropped_duplicates,
                "rows_final": rows_final,
                "columns_final": len(df.columns),
                "dtypes": dtypes,
                "languages_detected": int(len(language_counts)),
                "language_counts": language_counts.to_dict(),
            },
            args.output_dir / "reviews_report.json",
        )

        output_path = cr.export_reviews(df, args.output_dir)
        info(f"Done: {output_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
