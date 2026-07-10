"""Stage 2 of 2 for cleaning steam-data's raw reviews: reads stage 1's
deduped checkpoint (run_clean_reviews_dedup.py) and assigns each review's
language purely from langdetect - Steam/Perspective's own declared
`language` field is ignored entirely for this (kept only as
`perspective_declared_language`, for reference), per explicit user
decision.

See clean_reviews.py's module docstring for why this is a separate
script/cluster from stage 1's dedup: this step is pure-Python, CPU-bound,
per-row work via map_partitions - no shuffle, no cross-worker communication
at all - so it wants many worker *processes* (langdetect holds the GIL, so
real parallelism comes from process count, not threads) with comparatively
little memory each, the opposite shape from the dedup shuffle's ideal
cluster. Reading the checkpoint here is a plain partitioned read, not a
shuffle, so this script is far more tolerant of a high worker count than
stage 1 is.

Usage:
    python run_detect_language.py \\
        --input ../../steam-data/step01-output/reviews_deduped.parquet \\
        --output-dir ../../steam-data/step01-output/reviews_by_lang \\
        --n-workers 32 --memory-limit 8GB

No fixed language list: output has one `review_lang=<code>` folder per
distinct language langdetect actually finds (including `und`, for reviews
too short/low-signal to classify confidently) - see
clean_reviews.export_reviews's docstring. export_reviews's partitioned
write (`partition_on=["review_lang"]`) doesn't shuffle either - each
worker splits its own partition's rows into the right subfolder(s)
locally, no cross-worker data movement needed.
"""
import argparse
from pathlib import Path

import clean_reviews as cr
from pipeline_utils import export_sample, info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: assign review_lang via langdetect.")
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to stage 1's deduped checkpoint parquet (run_clean_reviews_dedup.py's --output)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write reviews_cleaned.parquet (partitioned by review_lang) to",
    )
    parser.add_argument(
        "--n-workers", type=int, default=32,
        help="Number of Dask worker processes - langdetect is pure Python and holds the GIL, so "
        "true parallelism here comes from process count, not threads (unlike stage 1's dedup)",
    )
    parser.add_argument(
        "--threads-per-worker", type=int, default=1,
        help="Threads per worker - 1 is usually right here (see --n-workers help)",
    )
    parser.add_argument(
        "--memory-limit", default="8GB",
        help="Memory limit per worker - this stage has no shuffle, so it needs much less "
        "headroom per worker than stage 1's dedup did",
    )
    parser.add_argument(
        "--local-directory", type=Path, default=Path("./dask-worker-space-langdetect"),
        help="Directory Dask workers use to spill data to disk under memory pressure",
    )
    parser.add_argument(
        "--blocksize", default="128MB",
        help="Caps how much data Dask bundles into a single checkpoint-file read task",
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

    try:
        df = cr.load_deduped_reviews(args.input, blocksize=blocksize)
        info(f"Loaded checkpoint: {df.npartitions} partition(s)")

        df = cr.detect_review_language(df)
        info("Assigned review_lang from langdetect (Steam/Perspective's declared language is ignored for this)")

        info("Persisting the language-detected dataframe (langdetect pass) - progress:")
        df = df.persist()
        progress(df)
        print()  # progress() doesn't print a trailing newline
        info("Persist complete")

        rows_final = len(df)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        language_counts = df["review_lang"].value_counts().compute().sort_values(ascending=False)
        info(f"Final: {rows_final} rows, {len(df.columns)} columns, {len(language_counts)} language(s) detected")
        for lang, count in language_counts.items():
            info(f"  {lang}: {count}")

        export_sample(df, args.output_dir / "sample_reviews.csv", n=20)
        save_summary(
            {
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
