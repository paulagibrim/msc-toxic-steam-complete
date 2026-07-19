"""Diagnostic (read-only, changes nothing): external validation of
toxic_user_labels.parquet using sentiment_score - a signal NEVER used to
build those labels (which come purely from Perspective/Detoxify toxicity
scores via 02_build_toxic_user_labels.py's binomial test). If users
labeled "toxic" also write systematically more negative reviews (lower
sentiment_score, nlptown's 1.0-5.0 expected-star-rating) than non-toxic
users in the same population, that's convergent evidence the label
captures a real behavioral difference - not an artifact of the specific
T_rate/alpha/T_coverage choices made when designing the labeling scheme.

THIS IS THE PRIMARY VALIDATION - it uses ALL of a user's reviews,
including whichever were individually toxic, and that is the correct
choice here (not a shortcut needing a "fix"). Averaging in the toxic
review(s) is not circular the way it would be for an ML classifier: the
toxic-user label comes from Perspective/Detoxify (offensiveness), while
sentiment_score comes from a completely different model (nlptown)
measuring a different construct (positive/negative tone) - the two aren't
measuring the same thing just twice, so there's no structural reason this
comparison is rigged to succeed. A companion script,
04_validate_labels_leave_toxic_out.py (built on
07_build_leave_toxic_out_sentiment.py), additionally checks whether the
difference survives when a toxic user's own flagged review(s) are
excluded from their average - a secondary, more conservative question
("is there still a difference outside the flagged content specifically"),
NOT a correction of this script's result. That secondary check also has a
real cost worth weighing against its extra rigor: ~30% of toxic users
(799 of 2,669) have EVERY review counted as toxic, so excluding toxic
reviews leaves nothing to average for them - they are dropped from that
analysis entirely, whereas every toxic user with a sentiment score is
included here.

Per population (pt/en/union), among users ELIGIBLE in that population
(matching every other per-population comparison in this project - see
02_build_toxic_user_labels.py):
  - compares avg_sentiment_<pop> between is_toxic_<pop>=True vs False
  - Mann-Whitney U test (one-sided: toxic users' sentiment is LOWER),
    non-parametric because sentiment_score's distribution is not assumed
    normal and the two groups are wildly different sizes (dozens vs
    millions)
  - rank-biserial correlation as an effect-size measure (Mann-Whitney's
    U statistic normalised to [-1, 1] - independent of sample size,
    unlike the p-value, which is nearly guaranteed significant at this
    scale even for a tiny real difference)

Usage:
    python 03_validate_labels_with_sentiment.py \\
        --labels ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --sentiment ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/user_sentiment_table.parquet \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/diagnostics/sentiment_validation_report.json
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
        description="Validates toxic_user_labels.parquet against sentiment_score - an independent signal."
    )
    parser.add_argument("--labels", required=True, type=Path, help="Path to toxic_user_labels.parquet")
    parser.add_argument("--sentiment", required=True, type=Path, help="Path to 06_build_user_sentiment.py's output")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the validation report JSON to")
    return parser.parse_args()


def validate_population(table: pd.DataFrame, population: str) -> dict:
    sentiment_label = "union" if population == UNION_KEY else population
    label_suffix = population  # matches toxic_user_labels.parquet's own "pt+en" suffix

    eligible = table[f"eligible_{label_suffix}"].fillna(False)
    subset = table.loc[eligible]
    is_toxic = subset[f"is_toxic_{label_suffix}"].fillna(False)

    sentiment_col = f"avg_sentiment_{sentiment_label}"
    has_sentiment = subset[sentiment_col].notna()
    subset = subset.loc[has_sentiment]
    is_toxic = is_toxic.loc[has_sentiment]

    toxic_sentiment = subset.loc[is_toxic, sentiment_col]
    nontoxic_sentiment = subset.loc[~is_toxic, sentiment_col]

    info(
        f"[{population}] {len(toxic_sentiment):,} toxic user(s), {len(nontoxic_sentiment):,} non-toxic user(s) "
        f"with a valid sentiment score"
    )

    if len(toxic_sentiment) < 2 or len(nontoxic_sentiment) < 2:
        info(f"[{population}] Too few user(s) with both a label and a sentiment score - skipping test")
        return {
            "n_toxic": int(len(toxic_sentiment)),
            "n_nontoxic": int(len(nontoxic_sentiment)),
            "skipped": True,
        }

    # One-sided: toxic users' sentiment is expected to be LOWER (alternative='less').
    u_stat, p_value = mannwhitneyu(toxic_sentiment, nontoxic_sentiment, alternative="less")
    n1, n2 = len(toxic_sentiment), len(nontoxic_sentiment)
    # Rank-biserial correlation: 2*U/(n1*n2) - 1, in [-1, 1] - a sample-size-independent
    # effect size, since the p-value alone is nearly always significant at this scale.
    rank_biserial = 2 * u_stat / (n1 * n2) - 1

    result = {
        "n_toxic": n1,
        "n_nontoxic": n2,
        "mean_sentiment_toxic": round(float(toxic_sentiment.mean()), 4),
        "mean_sentiment_nontoxic": round(float(nontoxic_sentiment.mean()), 4),
        "median_sentiment_toxic": round(float(toxic_sentiment.median()), 4),
        "median_sentiment_nontoxic": round(float(nontoxic_sentiment.median()), 4),
        "mannwhitney_u": float(u_stat),
        "p_value_one_sided": float(p_value),
        "rank_biserial_effect_size": round(float(rank_biserial), 4),
        "skipped": False,
    }
    info(
        f"[{population}] mean sentiment: toxic={result['mean_sentiment_toxic']:.3f}, "
        f"non-toxic={result['mean_sentiment_nontoxic']:.3f} | p={p_value:.2e} | "
        f"effect size (rank-biserial)={rank_biserial:.4f}"
    )
    return result


def main():
    args = parse_args()

    labels = pd.read_parquet(args.labels)
    info(f"Loaded toxic user labels: {len(labels):,} user(s) from {args.labels}")

    sentiment = pd.read_parquet(args.sentiment)
    info(f"Loaded sentiment table: {len(sentiment):,} user(s) from {args.sentiment}")

    table = labels.merge(sentiment, on="user_url", how="left")

    report = {pop: validate_population(table, pop) for pop in POPULATIONS}
    save_summary(report, args.output)


if __name__ == "__main__":
    main()
