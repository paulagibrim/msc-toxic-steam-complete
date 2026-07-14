"""Diagnostic (read-only, changes nothing): for every user, computes what
fraction of their total reviews (across every language) are in pt or en -
the two languages this project's toxic-user characterization/prediction
work (step06) covers.

Purpose: deciding a coverage threshold for which users are "represented
well enough" by pt/en to include in step06's analysis - e.g. a user with
100 reviews total but only 5 in pt/en would have their toxic-rate computed
from a small, unrepresentative slice of their real activity. Rather than
guessing a cutoff (e.g. "90%"), this reports the actual distribution
across the corpus - see this script's console/JSON output for how many
users clear 50%/70%/80%/90%/95%/100% pt/en coverage - so the threshold
used later can be chosen from real data.

Deliberately uses Stage 2's own language assignment (review_lang, a plain
column in reviews_by_lang/reviews_cleaned.parquet) - NOT step02's
agreement-mask-filtered output. This is a pure "how much of this user's
activity is in these languages" question, decoupled from whether
Perspective's declared language also agrees (a separate, stricter concern -
see step01's agreement_mask.py).

Usage:
    python run_language_coverage_diagnostic.py \\
        --deduped ../../steam-data/step01-output/reviews_deduped.parquet \\
        --reviews-by-lang ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output ../../steam-data/step06-output/language_coverage_report.json \\
        --n-workers 32 --memory-limit 8GB

Same cluster shape as step01's Stage 2 (many workers, modest memory each) -
groupby(...).size() is a tree-reduction aggregation, not the same kind of
full data shuffle drop_duplicates() needed, so it doesn't need Stage 1's
few-workers/lots-of-memory shape.
"""
import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import info, save_summary

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reports, per user, what fraction of their reviews (all languages) are pt/en."
    )
    parser.add_argument(
        "--deduped", required=True, type=Path,
        help="Path to step01's reviews_deduped.parquet checkpoint (Stage 1 output, every language)",
    )
    parser.add_argument(
        "--reviews-by-lang", required=True, type=Path,
        help="Path to step01's reviews_cleaned.parquet (Stage 2 output, partitioned by review_lang)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the coverage report JSON to")
    parser.add_argument("--n-workers", type=int, default=32)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--local-directory", type=Path, default=Path("./dask-worker-space-coverage"),
        help="Directory Dask workers use to spill data to disk under memory pressure",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import dask
    import dask.dataframe as dd
    from dask.distributed import Client

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

    try:
        info("Counting total reviews per user (every language)...")
        all_reviews = dd.read_parquet(str(args.deduped), columns=["user_url"])
        total_per_user = all_reviews.groupby("user_url").size().compute()
        info(f"{len(total_per_user)} unique user(s) found across all languages")

        info("Counting pt/en reviews per user...")
        # review_lang is a plain column in Stage 2's output, not a directory
        # partition - filter on it rather than reading per-language folders.
        by_lang = dd.read_parquet(str(args.reviews_by_lang), columns=["user_url", "review_lang"])
        pt_en = by_lang[by_lang["review_lang"].isin(["pt", "en"])]
        pt_en_per_user = pt_en.groupby("user_url").size().compute()
        info(f"{len(pt_en_per_user)} unique user(s) found with at least one pt/en review")
    finally:
        client.close()

    coverage = pt_en_per_user.reindex(total_per_user.index, fill_value=0) / total_per_user

    describe = coverage.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    info("Coverage distribution (fraction of a user's total reviews that are pt/en):")
    info(f"\n{describe}")

    threshold_counts = {}
    for t in THRESHOLDS:
        n = int((coverage >= t).sum())
        pct = 100 * n / len(coverage)
        threshold_counts[f">={int(t * 100)}%"] = {"n_users": n, "pct_of_all_users": round(pct, 2)}
        info(f"  >= {int(t * 100)}% pt/en coverage: {n} user(s) ({pct:.2f}% of all users)")

    report = {
        "total_users": int(len(coverage)),
        "coverage_describe": describe.to_dict(),
        "threshold_counts": threshold_counts,
    }
    save_summary(report, args.output)


if __name__ == "__main__":
    main()
