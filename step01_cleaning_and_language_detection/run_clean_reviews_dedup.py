"""Stage 1 of 2 for cleaning steam-data's raw reviews: load, clean, and
deduplicate by review_url - writes a checkpoint to disk once done.

See clean_reviews.py's module docstring for why this is split from stage 2
(run_detect_language.py) - short version: deduplication is a Dask shuffle
(wants few workers, lots of memory each), language detection is CPU-bound
per-row work with no shuffle (wants many worker processes, little memory
each) - one Client configuration can't serve both well, and trying to was
the source of repeated worker OOM-restarts and inter-worker connection
timeouts.

Usage:
    python run_clean_reviews_dedup.py \\
        --input ../../steam-data/raw/reviews \\
        --output ../../steam-data/step01-output/reviews_deduped.parquet \\
        --n-workers 8 --threads-per-worker 4 --memory-limit 40GB

Step order (through fix_types) matches
dissertacao-steam/data_refactor/0-cleaning/03_clean_reviews.ipynb.
drop_duplicate_reviews runs right after, then the checkpoint is written -
no language detection happens in this script at all (see
run_detect_language.py).
"""
import argparse
from pathlib import Path

import clean_reviews as cr
from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: clean + deduplicate steam-data's raw reviews.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of raw reviews *.parquet files")
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the deduped checkpoint parquet to (e.g. .../reviews_deduped.parquet)",
    )
    parser.add_argument("--n-workers", type=int, default=8, help="Number of Dask worker processes")
    parser.add_argument("--threads-per-worker", type=int, default=4, help="Threads per Dask worker")
    parser.add_argument(
        "--memory-limit", default="40GB",
        help="Memory limit per worker - the dedup shuffle needs real headroom per worker "
        "(fewer, bigger workers, not many small ones - see clean_reviews.py's module docstring)",
    )
    parser.add_argument(
        "--local-directory", type=Path, default=Path("./dask-worker-space-dedup"),
        help="Directory Dask workers use to spill data to disk under memory pressure",
    )
    parser.add_argument(
        "--blocksize", default="256MB",
        help="Caps how much data Dask bundles into a single raw-file read task (e.g. '256MB'). "
        "Without this, Dask's own optimizer decides how many files to fuse into one read task, "
        "and can pick a fusion large enough to exceed --memory-limit outright. Pass 'none' to "
        "disable capping and let Dask decide (not recommended).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import dask
    from dask.distributed import Client, progress

    # Default is 30s - too tight when workers are busy under load; raising
    # it gives more slack before Dask decides a connection has failed.
    dask.config.set({
        "distributed.comm.timeouts.connect": "120s",
        "distributed.comm.timeouts.tcp": "120s",
    })

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

    blocksize = None if args.blocksize.lower() == "none" else args.blocksize
    info(f"Read blocksize cap: {blocksize}")

    try:
        df = cr.load_raw_reviews(args.input, blocksize=blocksize)
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

        info("Persisting the deduplicated dataframe (dedup shuffle) - progress:")
        df = df.persist()
        progress(df)
        print()  # progress() doesn't print a trailing newline
        info("Persist complete")

        rows_final = len(df)
        rows_dropped_duplicates = rows_before_dedup - rows_final
        info(f"Dropped {rows_dropped_duplicates} duplicate row(s) by review_url")
        info(f"Final: {rows_final} rows, {len(df.columns)} columns, {df.npartitions} partition(s)")

        output_path = cr.export_deduped(df, args.output)
        save_summary(
            {
                "columns_loaded": columns_loaded,
                "rows_loaded": rows_loaded,
                "rows_dropped_missing_critical": rows_dropped_missing_critical,
                "rows_dropped_duplicates": rows_dropped_duplicates,
                "rows_final": rows_final,
                "partitions_final": df.npartitions,
            },
            args.output.parent / "dedup_report.json",
        )
        info(f"Done: {output_path}")
        info("Next: run_detect_language.py --input " + str(output_path))
    finally:
        client.close()


if __name__ == "__main__":
    main()
