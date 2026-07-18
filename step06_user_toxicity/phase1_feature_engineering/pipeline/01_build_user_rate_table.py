"""Builds the per-user aggregate table the rest of step06 is built on top
of: for every user in step02's scope, how many reviews they have in total
(any language, from Stage 2's own output) and - of their AGREEMENT-MATCHED
pt/en reviews (step02's output, the same population every other analysis
in this project uses) - how many are toxic.

TWO sweeps, one over each source. Each is shuffle-free (see
run_language_coverage_diagnostic.py's module docstring for why - a groupby
over millions of distinct user_url keys is a full Dask P2P shuffle, the
same mechanism that OOM-killed workers during step01's deduplication - so
this uses plain per-file value_counts() plus a single final concat+groupby,
never Dask):
  1. reviews_by_lang/reviews_cleaned.parquet (Stage 2's own output, every
     language) -> n_total per user: EVERY review the user has, in ANY
     language (Spanish, German, whatever langdetect found) - not just
     pt/en. This is the true denominator for "what share of this user's
     overall Steam activity is pt/en" (e.g. 2 pt + 4 en + 5 es reviews ->
     pt's share is 2/11, en's is 4/11, their union is 6/11 - the other 5
     reviews, in a language nothing here ever scores, still count toward
     the total). No separate sweep over reviews_deduped.parquet: Stage 2
     is a plain map_partitions column-adding transform - it never drops or
     adds rows - so reviews_by_lang already has every row reviews_deduped
     does, just with review_lang/detection_confidence appended.
  2. step02's output (AFTER the agreement mask - perspective_declared_
     language == review_lang is already enforced there) -> n_pt, n_en
     (agreement-matched review counts - the SAME reviews step03/04/05 use)
     and n_pt_toxic, n_en_toxic (of those, how many meet this project's
     toxicity union rule: perspective_score >= 0.7 OR detoxify_score >=
     0.9, with invalid/sentinel scores excluded from both numerator and
     denominator - same rule as toxicity_mask.py/text_cleaning.py/
     tfidf_analysis.py).

n_pt/n_en (not a separate "raw" Stage-2-only count) are used for BOTH the
coverage numerator and the toxicity-rate denominator - by design, a single
source of truth: "does this user have a pt/en analysis at all" and "what
fraction of it is toxic" are both answered from the exact same
agreement-matched population every other step in this project already
uses. A user with n_pt == 0 simply has no pt-language analysis - that's a
population-membership fact, not a 0%-coverage data point to plot.

Output: one parquet file, one row per user_url in step02's scope:
    user_url, n_total, n_pt, n_pt_toxic, n_en, n_en_toxic

Usage:
    python 01_build_user_rate_table.py \\
        --step02-dir ../../../../steam-data/step02-output \\
        --reviews-by-lang ../../../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/user_rate_table.parquet
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline_utils import info

LANGUAGES = ["pt", "en"]
PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds the per-user aggregate table for step06's toxicity-rate analysis."
    )
    parser.add_argument(
        "--step02-dir", required=True, type=Path,
        help="Path to step02's output directory - defines WHICH users are in scope, and the source "
        "for agreement-matched pt/en + toxic counts",
    )
    parser.add_argument(
        "--reviews-by-lang", required=True, type=Path,
        help="Path to step01's reviews_cleaned.parquet (Stage 2 output, review_lang is a plain column) "
        "- source for n_total (every language, the true denominator)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the per-user table parquet to")
    return parser.parse_args()


def load_users_in_scope(step02_dir: Path) -> set:
    """The users step06 will actually analyze: everyone appearing in step02's
    scored output. Used to scope every sweep below, so memory/time stays
    proportional to this (much smaller) roster rather than the full corpus."""
    files = sorted(step02_dir.rglob("*.parquet"))
    info(f"[scope] Reading user roster from {len(files)} step02 file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step02_dir} (searched recursively).")

    users = set()
    bar = tqdm(files, desc="[scope]", unit="file")
    for f in bar:
        users.update(pd.read_parquet(f, columns=["user_url"])["user_url"].dropna().unique())
        bar.set_postfix(users=len(users))
    return users


def sweep_total(directory: Path, users_in_scope: set) -> pd.Series:
    """Single sweep over Stage 2's output computing n_total per user: EVERY
    review they have, in any language - no language filter at all (see
    module docstring for why this is the true cross-language denominator,
    and why it replaces a separate reviews_deduped.parquet sweep)."""
    files = sorted(directory.rglob("*.parquet"))
    info(f"[total] Counting across {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {directory} (searched recursively).")

    partials = []
    bar = tqdm(files, desc="[total]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=["user_url"])
        df = df[df["user_url"].isin(users_in_scope)]
        partials.append(df["user_url"].value_counts())

    info("[total] Summing per-file counts...")
    return pd.concat(partials).groupby(level=0).sum().astype("int64")


def sweep_step02(step02_dir: Path, users_in_scope: set) -> dict:
    """Single sweep over step02's output computing, per user and per
    language: agreement-matched review count AND toxic-review count (so
    step02's files are only read once, not once per language/metric)."""
    files = sorted(step02_dir.rglob("*.parquet"))
    info(f"[step02] Counting across {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step02_dir} (searched recursively).")

    partials_n = {lang: [] for lang in LANGUAGES}
    partials_toxic = {lang: [] for lang in LANGUAGES}

    columns = ["user_url", "review_lang", "perspective_declared_language", "perspective_score", "detoxify_score"]
    n_disagreement_total = 0
    bar = tqdm(files, desc="[step02]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=columns)
        df = df[df["user_url"].isin(users_in_scope)]

        # step02's own detoxify_scoring.py already applies this exact mask
        # before scoring, so every row here should already agree - this is
        # an explicit re-check (defense in depth), not a second filtering
        # pass, same as step03's text_cleaning.py / step05's
        # tfidf_analysis.py. Any disagreement found is unexpected.
        rows_before = len(df)
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        n_disagreement_total += rows_before - len(df)

        # Invalid/sentinel scores (Detoxify's -1.0 "failed to score") are
        # dropped BEFORE counting anything, so they're excluded from both
        # n_<lang> (the rate's denominator) and n_<lang>_toxic (its
        # numerator) - never counted as a (non-toxic) review either.
        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]

        is_toxic = (df["perspective_score"] >= PERSPECTIVE_THRESHOLD) | (df["detoxify_score"] >= DETOXIFY_THRESHOLD)

        for lang in LANGUAGES:
            lang_mask = df["review_lang"] == lang
            partials_n[lang].append(df.loc[lang_mask, "user_url"].value_counts())
            partials_toxic[lang].append(df.loc[lang_mask & is_toxic, "user_url"].value_counts())

    if n_disagreement_total:
        info(
            f"[step02] Excluded {n_disagreement_total} row(s) where review_lang != "
            f"perspective_declared_language (unexpected - step02 should have already filtered these)"
        )

    info("[step02] Summing per-file counts...")
    result = {}
    for lang in LANGUAGES:
        result[f"n_{lang}"] = pd.concat(partials_n[lang]).groupby(level=0).sum().astype("int64")
        result[f"n_{lang}_toxic"] = pd.concat(partials_toxic[lang]).groupby(level=0).sum().astype("int64")
        info(
            f"[step02] {lang}: {len(result[f'n_{lang}'])} user(s) with an agreement-matched review, "
            f"{len(result[f'n_{lang}_toxic'])} with at least one toxic one"
        )
    return result


def main():
    args = parse_args()

    users_in_scope = load_users_in_scope(args.step02_dir)
    info(f"{len(users_in_scope)} user(s) in scope (present in step02's output)")

    n_total = sweep_total(args.reviews_by_lang, users_in_scope)
    step02_counts = sweep_step02(args.step02_dir, users_in_scope)

    info("Assembling final table...")
    index = pd.Index(list(users_in_scope), name="user_url")
    table = pd.DataFrame(index=index)
    table["n_total"] = n_total.reindex(index, fill_value=0)
    for lang in LANGUAGES:
        table[f"n_{lang}"] = step02_counts[f"n_{lang}"].reindex(index, fill_value=0)
        table[f"n_{lang}_toxic"] = step02_counts[f"n_{lang}_toxic"].reindex(index, fill_value=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(args.output, index=False)
    info(f"Saved user rate table ({len(table)} users, {len(table.columns) + 1} columns) to: {args.output}")


if __name__ == "__main__":
    main()
