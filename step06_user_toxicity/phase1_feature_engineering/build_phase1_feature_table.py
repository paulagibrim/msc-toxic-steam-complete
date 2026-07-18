"""Assembles step06 Phase 1's final feature table for the classical-ML
models planned in later phases: joins profile metadata, per-user text
embeddings, and toxic-user labels, all keyed by user_url.

BASE POPULATION: user_profile_metadata.parquet - by construction (it was
built as an inner join between toxic_user_labels.parquet and step01's
user-profile table), every row already has both a Steam profile AND a
toxic-user label. That's the ~42.31% of toxic_user_labels.parquet's
eligible population whose profile could actually be matched (see that
script's docstring for why the rest can't be - vanity-URL profiles and a
handful of accounts missing from the raw profile dump).

LEFT-JOINED WITH:
  - toxic_user_labels.parquet: should match every row 1:1 (profile
    metadata's population is a strict subset of the labels' population),
    but joined explicitly rather than assumed - any row that fails to
    match is a real problem worth surfacing loudly, not silently
    producing a NaN label.
  - user_text_embeddings.parquet: may have FEWER rows than the profile
    metadata table. A small number of matched users' reviews clean to
    empty text after boilerplate/URL stripping (e.g. a review that was
    ONLY a boilerplate phrase or a bare URL) and are therefore never
    embedded at all. Those users keep their profile + label columns but
    get NaN embedding columns - a real data-completeness fact for Phase 2
    to decide how to handle (e.g. drop rows lacking embeddings for models
    that require them, or fall back to profile-only features), not
    something to paper over or impute here.

Usage:
    python build_phase1_feature_table.py \\
        --labels ../../../steam-data/step06-output/toxic_user_labels.parquet \\
        --profile-metadata ../../../steam-data/step06-output/user_profile_metadata.parquet \\
        --embeddings ../../../steam-data/step06-output/user_text_embeddings.parquet \\
        --output ../../../steam-data/step06-output/phase1_feature_table.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joins profile metadata + per-user text embeddings + toxic labels into step06's Phase 1 feature table."
    )
    parser.add_argument("--labels", required=True, type=Path, help="Path to toxic_user_labels.parquet")
    parser.add_argument("--profile-metadata", required=True, type=Path, help="Path to build_user_profile_metadata.py's output")
    parser.add_argument("--embeddings", required=True, type=Path, help="Path to build_user_text_embeddings.py's output")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the final Phase 1 feature table parquet to")
    return parser.parse_args()


def main():
    args = parse_args()

    profile = pd.read_parquet(args.profile_metadata)
    info(f"Loaded profile metadata: {len(profile):,} user(s) from {args.profile_metadata}")

    labels = pd.read_parquet(args.labels)
    info(f"Loaded toxic user labels: {len(labels):,} user(s) from {args.labels}")

    embeddings = pd.read_parquet(args.embeddings)
    info(f"Loaded text embeddings: {len(embeddings):,} user(s) from {args.embeddings}")

    table = profile.merge(labels, on="user_url", how="left", validate="one_to_one")
    # eligible_* (not is_toxic_*) is the right column to check for a failed
    # merge: eligible_pt/en/union is always True/False for every real row in
    # --labels, never NaN - unlike is_toxic_*, which IS legitimately NaN
    # whenever a user isn't eligible in that population (no p-value/label
    # computed for them there). Using is_toxic_* here would misreport every
    # non-eligible user as a merge failure.
    eligibility_col = next(c for c in table.columns if c.startswith("eligible_"))
    n_missing_label = int(table[eligibility_col].isna().sum())
    if n_missing_label:
        info(
            f"WARNING: {n_missing_label:,} profile-matched user(s) have no row in --labels - "
            "this should never happen (profile metadata was built as an inner join against the labels), check inputs"
        )

    table = table.merge(embeddings, on="user_url", how="left", validate="one_to_one")
    n_missing_emb = int(table["n_union_embedded"].isna().sum()) if "n_union_embedded" in table.columns else len(table)
    info(
        f"{n_missing_emb:,} of {len(table):,} user(s) ({100 * n_missing_emb / len(table):.2f}%) have no embeddable "
        "review (all their reviews cleaned to empty text) - their embedding columns are NaN"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    info(f"Saved Phase 1 feature table ({len(table)} users, {len(table.columns)} columns) to: {args.output}")

    save_summary(
        {
            "n_users": int(len(table)),
            "n_columns": int(len(table.columns)),
            "n_missing_label": n_missing_label,
            "n_missing_embedding": n_missing_emb,
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
