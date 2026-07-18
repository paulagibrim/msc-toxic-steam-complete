"""Joins step06's toxic-user label population against step01's cleaned
user-profile table (all_users.parquet), producing a profile-metadata table
restricted to users whose profile could actually be matched - the first
input to step06 Phase 1's feature table (the other being per-user text
embeddings, built separately since it's the expensive GPU step and should
only run over this same matched population).

WHY MOST LABELED USERS DON'T MATCH: all_users.parquet's join key
(`steam_url`) is always the numeric `/profiles/<steam_id>` form, with no
trailing slash. The labels' `user_url` (the same key used everywhere else
in this project, from step01's review cleaning onward) has a trailing
slash AND includes vanity `/id/<name>/` URLs for users who set a custom
profile name - those can never match `steam_url` without resolving the
vanity name to a steam_id via the Steam API (out of scope here, offline).
Measured on the real data: 33.35% of labeled users are vanity-style (no
match possible at all), and even among numeric-style URLs the match rate
is only 63.47% (some accounts aren't in the raw profile dump - deleted,
never scraped, etc.) - 42.31% overall. This was an explicit, confirmed
decision: accept the loss, keep only users with a real profile match,
rather than half-filling the feature table with nulls for the rest.

SENTINEL CLEANUP: most numeric profile columns use -1 as a "profile
private/unavailable" sentinel rather than NaN (e.g. 98.2% of `guides`,
94.6% of `days_since_last_ban`, 87.0% of `awards` are -1) - left as -1,
a model would read this as a real ordinal value ("library of -1 games")
instead of "unknown". Converted to NaN here, same treatment as the -1.0
Detoxify sentinel elsewhere in this project - imputation (if any) is a
Phase 2 modeling decision, not this script's.

GEO EXCLUDED: city/state/region/country are dropped entirely (not just
nulled) - an explicit decision, not a bug. ~51% null, and using
country/region as a predictor of toxicity risks encoding national/regional
stereotypes into the model rather than a real behavioral signal; the
reference paper (AIIDE26) doesn't use geolocation as a feature either.

Output: one parquet row per user_url that both (a) appears in
--labels and (b) has a matching profile in --users:
    user_url, profile_level, has_ban, days_since_last_ban, ban_reason,
    awards, insignias, library_size, screenshots, workshop_items, guides,
    arts, groups, friends_count, profile_description

Usage:
    python build_user_profile_metadata.py \\
        --labels ../../../steam-data/step06-output/toxic_user_labels.parquet \\
        --users ../../../steam-data/step01-output/users/all_users.parquet \\
        --output ../../../steam-data/step06-output/user_profile_metadata.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import info, save_summary

GEO_COLUMNS = ["city", "state/region", "country"]
SENTINEL_COLUMNS = [
    "profile_level", "days_since_last_ban", "awards", "insignias", "library_size",
    "screenshots", "workshop_items", "guides", "arts", "groups", "friends_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joins toxic_user_labels.parquet against all_users.parquet, keeping only matched users."
    )
    parser.add_argument("--labels", required=True, type=Path, help="Path to toxic_user_labels.parquet")
    parser.add_argument("--users", required=True, type=Path, help="Path to step01's all_users.parquet")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the matched profile-metadata parquet to")
    return parser.parse_args()


def main():
    args = parse_args()

    labels = pd.read_parquet(args.labels, columns=["user_url"])
    info(f"Loaded {len(labels):,} labeled user(s) from {args.labels}")

    users = pd.read_parquet(args.users)
    info(f"Loaded {len(users):,} user profile(s) from {args.users}")

    # Normalize the join key: all_users.parquet's steam_url has no trailing
    # slash and is always numeric-ID form; labels' user_url has a trailing
    # slash and may be vanity-style (never matchable). Stripping the slash
    # is the only normalization possible here - vanity URLs simply won't
    # match, by design (see module docstring).
    labels = labels.assign(_join_key=labels["user_url"].str.rstrip("/"))
    users = users.rename(columns={"steam_url": "_join_key"})

    is_vanity = labels["_join_key"].str.contains("/id/")
    info(f"{int(is_vanity.sum()):,} of {len(labels):,} labeled user(s) ({100 * is_vanity.mean():.2f}%) are vanity-style URLs - can never match")

    merged = labels.merge(users, on="_join_key", how="inner")
    info(f"Matched {len(merged):,} of {len(labels):,} labeled user(s) ({100 * len(merged) / len(labels):.2f}%) to a profile in --users")

    merged = merged.drop(columns=["_join_key", "steam_id", "reviews_langs"] + GEO_COLUMNS, errors="ignore")

    for col in SENTINEL_COLUMNS:
        n_sentinel = int((merged[col] == -1).sum())
        merged.loc[merged[col] == -1, col] = pd.NA
        info(f"  [{col}] {n_sentinel:,} sentinel (-1) value(s) converted to NaN ({100 * n_sentinel / len(merged):.1f}%)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    info(f"Saved user profile metadata ({len(merged)} users, {len(merged.columns)} columns) to: {args.output}")

    save_summary(
        {
            "n_labeled_users": int(len(labels)),
            "n_vanity_url_users": int(is_vanity.sum()),
            "n_matched": int(len(merged)),
            "pct_matched": round(100 * len(merged) / len(labels), 4),
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
