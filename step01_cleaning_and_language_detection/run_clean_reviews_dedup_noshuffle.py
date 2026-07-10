"""Shuffle-free alternative to run_clean_reviews_dedup.py: cleans and
deduplicates steam-data's raw reviews without ever invoking Dask's
distributed P2P shuffle.

The shuffle-based version (drop_duplicate_reviews) proved unstable in
practice even after extensive tuning (memory_limit 4GB->8GB->40GB->90GB,
blocksize 256MB->64MB, threads_per_worker 4->1, comm timeouts 30s->120s,
fewer/bigger workers): a single worker exceeding its own memory_limit
during shuffle buffering gets killed and restarted by the nanny, which
poisons the shuffle's global state (P2PConsistencyError) and forces a
full shuffle restart - sometimes recovering, sometimes cascading into a
total failure, observed as late as 99% complete.

This script sidesteps the shuffle mechanism entirely, in two phases:

  1. scatter_to_buckets - every partition hashes its own rows'
     review_url into one of N_BUCKETS buckets and writes each group to
     its own bucket subfolder. Every partition writes only its own
     files, independently of every other partition/worker - no
     cross-worker data movement, so this can never trigger a shuffle or
     be poisoned by another worker dying mid-transfer. Runs under Dask
     (many workers, modest memory each - same shape as language
     detection, since there's no shuffle here either).

  2. dedup_buckets - every row sharing a review_url is guaranteed (by
     the hash) to be inside the same bucket, wherever it came from - so
     each bucket can be deduplicated independently with a single-process
     pandas drop_duplicates. Runs entirely outside of Dask, via a plain
     ProcessPoolExecutor (one process per bucket, no cluster needed).

Usage:
    python run_clean_reviews_dedup_noshuffle.py \\
        --input ../../steam-data/raw/reviews \\
        --output ../../steam-data/step01-output/reviews_deduped.parquet \\
        --n-workers 16 --memory-limit 16GB

Output (--output) is a checkpoint DIRECTORY, same as
run_clean_reviews_dedup.py's - readable by
clean_reviews.load_deduped_reviews / run_detect_language.py unchanged.
"""
import argparse
import shutil
from pathlib import Path

import clean_reviews as cr
from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 (shuffle-free): clean + deduplicate steam-data's raw reviews via bucket hashing."
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory of raw reviews *.parquet files")
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the deduped checkpoint DIRECTORY to (e.g. .../reviews_deduped.parquet)",
    )
    parser.add_argument(
        "--buckets-dir", type=Path, default=None,
        help="Scratch directory for the intermediate per-bucket files, deleted after dedup unless "
        "--keep-buckets is passed. Defaults to '<output>_buckets' next to --output.",
    )
    parser.add_argument(
        "--keep-buckets", action="store_true",
        help="Don't delete the intermediate bucket scratch directory after dedup (useful for debugging)",
    )
    parser.add_argument("--n-buckets", type=int, default=cr.N_BUCKETS, help="Number of hash buckets")
    parser.add_argument(
        "--n-workers", type=int, default=16,
        help="Dask workers for the scatter phase - no shuffle here, so many workers with modest "
        "memory is fine (like language detection, not the old shuffle-based dedup)",
    )
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--local-directory", type=Path, default=Path("./dask-worker-space-scatter"),
        help="Directory Dask workers use to spill data to disk under memory pressure",
    )
    parser.add_argument(
        "--blocksize", default="64MB",
        help="Caps how much data Dask bundles into a single raw-file read task",
    )
    parser.add_argument(
        "--n-jobs-dedup", type=int, default=None,
        help="Process pool size for the per-bucket dedup phase (outside Dask). Defaults to os.cpu_count().",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import dask
    from dask.distributed import Client, progress

    dask.config.set({
        "distributed.comm.timeouts.connect": "120s",
        "distributed.comm.timeouts.tcp": "120s",
    })

    buckets_dir = args.buckets_dir or args.output.parent / f"{args.output.stem}_buckets"

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
        f"{args.memory_limit} each (scatter phase - no shuffle)"
    )

    blocksize = None if args.blocksize.lower() == "none" else args.blocksize

    try:
        df = cr.load_raw_reviews(args.input, blocksize=blocksize)
        columns_loaded = df.columns.tolist()
        info(f"Columns: {columns_loaded}")

        df = cr.drop_irrelevant_columns(df)
        df = cr.rename_columns(df)
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
        info(f"Scattering {rows_before_dedup} row(s) into {args.n_buckets} bucket(s) by hash(review_url) - progress:")
        written = cr.scatter_to_buckets(df, buckets_dir, n_buckets=args.n_buckets)
        written = written.persist()
        progress(written)
        print()  # progress() doesn't print a trailing newline
        rows_written = int(written["rows_written"].sum().compute())
        info(f"Scatter complete: {rows_written} row(s) written across bucket files")
    finally:
        client.close()

    info(f"Deduplicating {args.n_buckets} bucket(s) locally (outside Dask, {args.n_jobs_dedup or 'all cores'})...")
    dedup_stats = cr.dedup_buckets(buckets_dir, args.output, n_jobs=args.n_jobs_dedup)
    info(
        f"Dedup complete: {dedup_stats['rows_before_dedup']} -> {dedup_stats['rows_final']} row(s) "
        f"({dedup_stats['rows_dropped_duplicates']} duplicate(s) dropped) across {dedup_stats['buckets']} bucket(s)"
    )

    if not args.keep_buckets:
        info(f"Removing scratch bucket directory: {buckets_dir}")
        shutil.rmtree(buckets_dir)

    save_summary(
        {
            "columns_loaded": columns_loaded,
            "rows_loaded": rows_loaded,
            "rows_dropped_missing_critical": rows_dropped_missing_critical,
            **dedup_stats,
        },
        args.output.parent / "dedup_report.json",
    )
    info(f"Done: {args.output}")
    info("Next: run_detect_language.py --input " + str(args.output))


if __name__ == "__main__":
    main()
