"""Diagnostic (read-only, changes nothing): for every user who actually
reaches step06's analysis (i.e. appears in step02's scored output), computes
what fraction of their TOTAL reviews - across every language, from step01's
pre-language-split checkpoint - are in pt or en.

Purpose: deciding a coverage threshold for which users are "represented
well enough" by pt/en to include in step06's analysis - e.g. a user with
100 reviews total but only 5 in pt/en would have their toxic-rate computed
from a small, unrepresentative slice of their real activity. Rather than
guessing a cutoff (e.g. "90%"), this reports the actual distribution - see
this script's console/JSON output for how many users clear 50%/70%/80%/
90%/95%/100% pt/en coverage - so the threshold used later can be chosen
from real data.

WHY THE TWO INPUTS ARE BOTH NEEDED: step02's output is already
language-filtered (pt/en only, agreement-masked), so it alone can't answer
"what fraction of this user's activity is pt/en" - by construction that
would always be 100%. The denominator (a user's reviews in EVERY language)
only exists in step01's reviews_deduped.parquet, before the language split.

Users are scoped to step02's roster first (a much smaller set than the full
corpus), and the step01 sweep only counts those users - so the running
totals stay proportional to "users step06 will actually analyze", not to
every user Steam ever had.

NO DASK, DELIBERATELY: counting reviews per user is a groupby over millions
of distinct keys, which Dask implements as a full P2P shuffle - the same
mechanism that repeatedly OOM-killed workers during step01's deduplication
(see run_clean_reviews_dedup_noshuffle.py) and did so again here on a first
attempt. Instead each file is counted independently with a plain pandas
value_counts() and the per-file counts are summed. No cross-worker
coordination, no shuffle, bounded memory.

Usage:
    python run_language_coverage_diagnostic.py \\
        --step02-dir ../../steam-data/step02-output \\
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
        description="Reports, for users in step02's output, what fraction of their reviews (all languages) are pt/en."
    )
    parser.add_argument(
        "--step02-dir", required=True, type=Path,
        help="Path to step02's output directory - defines WHICH users are in scope",
    )
    parser.add_argument(
        "--deduped", required=True, type=Path,
        help="Path to step01's reviews_deduped.parquet checkpoint (every language - the denominator)",
    )
    parser.add_argument(
        "--reviews-by-lang", required=True, type=Path,
        help="Path to step01's reviews_cleaned.parquet (Stage 2 output, review_lang is a plain column)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the coverage report JSON to")
    return parser.parse_args()


def load_users_in_scope(step02_dir: Path) -> set:
    """The users step06 will actually analyze: everyone appearing in step02's
    scored output. Used to scope both counting sweeps below, so memory stays
    proportional to this (much smaller) roster rather than the full corpus.

    Globs recursively - step02 writes one review_lang=<lang>/ subfolder per
    language, so the parquet files aren't directly under step02_dir."""
    files = sorted(step02_dir.rglob("*.parquet"))
    info(f"[scope] Reading user roster from {len(files)} step02 file(s)...")
    if not files:
        raise FileNotFoundError(
            f"No .parquet files found under {step02_dir} (searched recursively). "
            "Check the path - this should be step02's output directory."
        )

    users = set()
    for i, f in enumerate(files, start=1):
        users.update(pd.read_parquet(f, columns=["user_url"])["user_url"].dropna().unique())
        if i % 20 == 0 or i == len(files):
            info(f"[scope] [{i}/{len(files)}] {len(users)} unique user(s) so far")
    return users


def count_per_user(directory: Path, label: str, users_in_scope: set, languages: list = None) -> pd.Series:
    """Sums per-user review counts across every parquet file in `directory`,
    one file at a time, counting only users in `users_in_scope`. If
    `languages` is given, only rows whose review_lang is in that list count.

    Shuffle-free by construction: each file is reduced to its own per-user
    counts (much smaller than the file itself) - see module docstring for
    why avoiding Dask's distributed groupby matters here.

    The per-file counts are collected in a list and summed ONCE at the end,
    rather than folded into a running Series per file. Folding per file
    means realigning a Series that grows toward ~10M string-keyed entries,
    once per file - the alignment cost scales with the accumulated size, so
    it gets progressively slower as the sweep proceeds (measured: ~5 minutes
    for the first 20 of 200 files). Concatenating first and grouping once
    pays that alignment cost a single time."""
    files = sorted(directory.rglob("*.parquet"))
    info(f"[{label}] Counting across {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {directory} (searched recursively).")

    columns = ["user_url"] + (["review_lang"] if languages else [])
    partial_counts = []

    for i, f in enumerate(files, start=1):
        df = pd.read_parquet(f, columns=columns)
        df = df[df["user_url"].isin(users_in_scope)]
        if languages:
            df = df[df["review_lang"].isin(languages)]

        partial_counts.append(df["user_url"].value_counts())

        if i % 20 == 0 or i == len(files):
            info(f"[{label}] [{i}/{len(files)}] file(s) read")

    info(f"[{label}] Summing per-file counts...")
    totals = pd.concat(partial_counts).groupby(level=0).sum()
    info(f"[{label}] {len(totals)} of {len(users_in_scope)} in-scope user(s) have at least one matching review")
    return totals.astype("int64")


def main():
    args = parse_args()

    users_in_scope = load_users_in_scope(args.step02_dir)
    info(f"{len(users_in_scope)} user(s) in scope (present in step02's output)")

    total_per_user = count_per_user(args.deduped, "all languages", users_in_scope)
    info(f"Counted total (all-language) reviews for {len(total_per_user)} in-scope user(s)")
    if total_per_user.empty:
        raise SystemExit(
            "No in-scope user was found in --deduped. This usually means step02's user_url values "
            "don't match step01's (check that --deduped points at the same corpus step02 came from)."
        )

    pt_en_per_user = count_per_user(args.reviews_by_lang, "pt/en", users_in_scope, languages=LANGUAGES)
    info(f"Counted pt/en reviews for {len(pt_en_per_user)} in-scope user(s)")

    # reindex(fill_value=0) so an in-scope user with no pt/en reviews counted
    # here still appears (as 0% coverage) instead of dropping out entirely.
    coverage = pt_en_per_user.reindex(total_per_user.index, fill_value=0) / total_per_user

    describe = coverage.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    info("Coverage distribution (fraction of an in-scope user's total reviews that are pt/en):")
    print(describe.to_string())

    threshold_counts = {}
    for t in THRESHOLDS:
        n = int((coverage >= t).sum())
        pct = 100 * n / len(coverage)
        threshold_counts[f">={int(t * 100)}%"] = {"n_users": n, "pct_of_in_scope_users": round(pct, 2)}
        info(f"  >= {int(t * 100)}% pt/en coverage: {n} user(s) ({pct:.2f}% of in-scope users)")

    report = {
        "users_in_scope": int(len(users_in_scope)),
        "users_counted": int(len(coverage)),
        "coverage_describe": describe.to_dict(),
        "threshold_counts": threshold_counts,
    }
    save_summary(report, args.output)


if __name__ == "__main__":
    main()
