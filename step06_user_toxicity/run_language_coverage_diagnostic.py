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

NO DASK, DELIBERATELY: counting reviews per user is a groupby over
millions of distinct keys, which Dask implements as a full P2P shuffle -
the same mechanism that repeatedly OOM-killed workers during step01's
deduplication (see run_clean_reviews_dedup_noshuffle.py) and did so again
here on the first attempt. Instead, each file is counted independently
with a plain pandas value_counts() and the per-file counts (one row per
user seen in that file - far smaller than the file itself) are summed at
the end. No cross-worker coordination, no shuffle, bounded memory: peak
usage is one file plus the running per-user totals.

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
        --output ../../steam-data/step06-output/language_coverage_report.json
"""
import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import info, save_summary

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
LANGUAGES = ["pt", "en"]


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
        help="Path to step01's reviews_cleaned.parquet (Stage 2 output, review_lang is a plain column)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the coverage report JSON to")
    return parser.parse_args()


def count_per_user(directory: Path, label: str, languages: list = None) -> pd.Series:
    """Sums per-user review counts across every parquet file in `directory`,
    one file at a time. If `languages` is given, only rows whose review_lang
    is in that list are counted (the column must be present).

    Shuffle-free by construction: each file is reduced to its own per-user
    counts (much smaller than the file), and those are added into a running
    total - see module docstring for why this matters."""
    files = sorted(directory.glob("*.parquet"))
    info(f"[{label}] Counting across {len(files)} file(s)...")

    columns = ["user_url"] + (["review_lang"] if languages else [])
    totals = pd.Series(dtype="int64")

    for i, f in enumerate(files, start=1):
        df = pd.read_parquet(f, columns=columns)
        if languages:
            df = df[df["review_lang"].isin(languages)]

        counts = df["user_url"].value_counts()
        totals = totals.add(counts, fill_value=0)

        if i % 20 == 0 or i == len(files):
            info(f"[{label}] [{i}/{len(files)}] {len(totals)} unique user(s) so far")

    return totals.astype("int64")


def main():
    args = parse_args()

    total_per_user = count_per_user(args.deduped, label="all languages")
    info(f"{len(total_per_user)} unique user(s) found across all languages")

    pt_en_per_user = count_per_user(args.reviews_by_lang, label="pt/en", languages=LANGUAGES)
    info(f"{len(pt_en_per_user)} unique user(s) found with at least one pt/en review")

    # reindex(fill_value=0) so users with NO pt/en reviews at all still appear,
    # as 0% coverage, instead of dropping out of the distribution entirely.
    coverage = pt_en_per_user.reindex(total_per_user.index, fill_value=0) / total_per_user

    describe = coverage.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    info("Coverage distribution (fraction of a user's total reviews that are pt/en):")
    print(describe.to_string())

    threshold_counts = {}
    for t in THRESHOLDS:
        n = int((coverage >= t).sum())
        pct = 100 * n / len(coverage)
        threshold_counts[f">={int(t * 100)}%"] = {"n_users": n, "pct_of_all_users": round(pct, 2)}
        info(f"  >= {int(t * 100)}% pt/en coverage: {n} user(s) ({pct:.2f}% of all users)")

    report = {
        "total_users": int(len(coverage)),
        "users_with_any_pt_en": int(len(pt_en_per_user)),
        "coverage_describe": describe.to_dict(),
        "threshold_counts": threshold_counts,
    }
    save_summary(report, args.output)


if __name__ == "__main__":
    main()
