"""Diagnostic (read-only, changes nothing): a SECONDARY, more conservative
companion to 03_validate_labels_with_sentiment.py's comparison - NOT a
correction or a "fairer" replacement of it. That script's all-reviews
comparison remains the primary validation and is not circular the way
phase2_classical_ml's ML control needed fixing for: the toxic-user label
comes from Perspective/Detoxify (offensiveness), sentiment_score comes
from an independent model measuring a different construct (tone) - so
there's no structural reason averaging in a user's own toxic review(s)
rigs that comparison to succeed. This script is modelled on
phase2_classical_ml's leave-toxic-out control (02_build_leave_toxic_out_
embeddings.py / 01_phase2_nested_cv.py --leave-toxic-out-embeddings) only
in mechanics, not in the reason for existing.

WHAT THIS ADDS: a narrower, additional question - does a sentiment
difference survive when a toxic user's own flagged review(s) are excluded
from their average? Compares:
  - toxic users' average sentiment computed from ONLY their NON-toxic
    reviews (07_build_leave_toxic_out_sentiment.py's output, the "_lto"
    columns)
  - against non-toxic users' average sentiment computed the normal way
    (06_build_user_sentiment.py's output - unaffected, since non-toxic
    users have no flagged review to exclude in the first place)
A difference surviving here is additional evidence the pattern extends
beyond the specifically flagged content - it is not required for
03_validate_labels_with_sentiment.py's result to be valid.

REAL COST, weigh against the extra rigor: ~30% of toxic users (799 of
2,669) have EVERY review counted as toxic and are entirely absent from
this comparison (see 07_build_leave_toxic_out_sentiment.py's docstring) -
this script's population is a strict, smaller subset of
03_validate_labels_with_sentiment.py's, which includes every toxic user
with a sentiment score at all.

Same statistical test as 03_validate_labels_with_sentiment.py: one-sided
Mann-Whitney U (alternative='less', toxic < non-toxic) plus rank-biserial
effect size (sample-size-independent, unlike the p-value).

Usage:
    python 04_validate_labels_leave_toxic_out.py \\
        --sentiment ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/user_sentiment_table.parquet \\
        --leave-toxic-out-sentiment ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/leave_toxic_out_sentiment_table.parquet \\
        --labels ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/diagnostics/sentiment_validation_leave_toxic_out_report.json
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

from pipeline_utils import info, save_summary

UNION_KEY = "pt+en"
POPULATIONS = ["pt", "en", UNION_KEY]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-toxic-out validation of toxic_user_labels.parquet against sentiment_score."
    )
    parser.add_argument("--sentiment", required=True, type=Path, help="Path to 06_build_user_sentiment.py's output")
    parser.add_argument(
        "--leave-toxic-out-sentiment", required=True, type=Path,
        help="Path to 07_build_leave_toxic_out_sentiment.py's output",
    )
    parser.add_argument("--labels", required=True, type=Path, help="Path to toxic_user_labels.parquet")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the validation report JSON to")
    return parser.parse_args()


def validate_population(labels: pd.DataFrame, sentiment: pd.DataFrame, lto_sentiment: pd.DataFrame, population: str) -> dict:
    sentiment_label = "union" if population == UNION_KEY else population
    label_suffix = population

    eligible = labels[f"eligible_{label_suffix}"].fillna(False)
    subset = labels.loc[eligible, ["user_url", f"is_toxic_{label_suffix}"]].copy()
    is_toxic = subset[f"is_toxic_{label_suffix}"].fillna(False)

    toxic_users = subset.loc[is_toxic, "user_url"]
    nontoxic_users = subset.loc[~is_toxic, "user_url"]

    lto_col = f"avg_sentiment_{sentiment_label}_lto"
    toxic_lto = lto_sentiment.set_index("user_url")[lto_col].reindex(toxic_users).dropna()

    sentiment_col = f"avg_sentiment_{sentiment_label}"
    nontoxic_full = sentiment.set_index("user_url")[sentiment_col].reindex(nontoxic_users).dropna()

    n_toxic_no_nontoxic_review = int(is_toxic.sum()) - len(toxic_lto)
    info(
        f"[{population}] {len(toxic_lto):,} toxic user(s) have >=1 non-toxic review to average "
        f"({n_toxic_no_nontoxic_review:,} excluded - 100% of their content is toxic), "
        f"{len(nontoxic_full):,} non-toxic user(s) with a sentiment score"
    )

    if len(toxic_lto) < 2 or len(nontoxic_full) < 2:
        info(f"[{population}] Too few user(s) for a stable test - skipping")
        return {"n_toxic_with_nontoxic_review": int(len(toxic_lto)), "n_nontoxic": int(len(nontoxic_full)), "skipped": True}

    u_stat, p_value = mannwhitneyu(toxic_lto, nontoxic_full, alternative="less")
    n1, n2 = len(toxic_lto), len(nontoxic_full)
    rank_biserial = 2 * u_stat / (n1 * n2) - 1

    result = {
        "n_toxic_with_nontoxic_review": n1,
        "n_toxic_excluded_all_content_toxic": n_toxic_no_nontoxic_review,
        "n_nontoxic": n2,
        "mean_sentiment_toxic_leave_toxic_out": round(float(toxic_lto.mean()), 4),
        "mean_sentiment_nontoxic": round(float(nontoxic_full.mean()), 4),
        "median_sentiment_toxic_leave_toxic_out": round(float(toxic_lto.median()), 4),
        "median_sentiment_nontoxic": round(float(nontoxic_full.median()), 4),
        "mannwhitney_u": float(u_stat),
        "p_value_one_sided": float(p_value),
        "rank_biserial_effect_size": round(float(rank_biserial), 4),
        "skipped": False,
    }
    info(
        f"[{population}] mean sentiment (leave-toxic-out): toxic={result['mean_sentiment_toxic_leave_toxic_out']:.3f}, "
        f"non-toxic={result['mean_sentiment_nontoxic']:.3f} | p={p_value:.2e} | "
        f"effect size (rank-biserial)={rank_biserial:.4f}"
    )
    return result


def main():
    args = parse_args()

    labels = pd.read_parquet(args.labels)
    info(f"Loaded toxic user labels: {len(labels):,} user(s) from {args.labels}")

    sentiment = pd.read_parquet(args.sentiment)
    info(f"Loaded (full) sentiment table: {len(sentiment):,} user(s) from {args.sentiment}")

    lto_sentiment = pd.read_parquet(args.leave_toxic_out_sentiment)
    info(f"Loaded leave-toxic-out sentiment table: {len(lto_sentiment):,} toxic user(s) from {args.leave_toxic_out_sentiment}")

    report = {pop: validate_population(labels, sentiment, lto_sentiment, pop) for pop in POPULATIONS}
    save_summary(report, args.output)


if __name__ == "__main__":
    main()
