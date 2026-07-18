"""Builds the final toxic-user label table from build_user_rate_table.py's
per-user counts, using the labeling scheme validated in
run_trate_diagnostic.py: coverage-filtered eligibility + a one-sided
binomial test against each population's own baseline toxic rate.

Population (per pt/en/pt+en union, independently): users with >=1
agreement-matched review in that population AND coverage (their reviews in
this population / their TOTAL reviews, every language) >= --t-coverage.
Default 0.90 - below that, achievable coverage values get too coarse for a
fixed cutoff to behave sensibly (e.g. a user with 9 of 10 reviews in a
language sits at exactly 90%; a 95%+ cutoff would reject them as if they
were impure, when they only have a single review elsewhere). See
run_language_coverage_diagnostic.py's report and this conversation's
comparison for the data behind that choice.

Label: within each population, a one-sided binomial test - is this user's
toxic-review count significantly higher than what the population's own
baseline rate would predict for their sample size? p = P(X >= n_toxic |
n = n_<lang>, p = baseline_rate), baseline computed as
sum(n_<lang>_toxic)/sum(n_<lang>) over that population's eligible users.
Flagged toxic if p < --alpha. Default 0.01 - the point where a lone toxic
review among a tiny sample (e.g. 1 toxic out of 1 total, p == the baseline
rate itself, ~1.5-2%) stops passing the test on its own, without needing a
separate min-review-count cutoff (see run_trate_diagnostic.py's edge-case
numbers). Both --t-coverage and --alpha are exposed as CLI flags so this
can be re-run at a different setting without editing code.

Output: one parquet row per user_url that is eligible in AT LEAST ONE
population, with (per population) an eligibility flag, the binomial
p-value, and the resulting toxic label - independent per population, e.g.
a user can be eligible+toxic in en while not eligible in pt at all:
    user_url,
    eligible_pt, p_value_pt, is_toxic_pt,
    eligible_en, p_value_en, is_toxic_en,
    eligible_union, p_value_union, is_toxic_union

Usage:
    python 02_build_toxic_user_labels.py \\
        --table ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/user_rate_table.parquet \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --t-coverage 0.90 --alpha 0.01
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binom

from pipeline_utils import info, save_summary

UNION_KEY = "pt+en"
POPULATIONS = ["pt", "en", UNION_KEY]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds the final toxic-user label table (coverage-filtered eligibility + binomial-test label)."
    )
    parser.add_argument("--table", required=True, type=Path, help="Path to build_user_rate_table.py's output parquet")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the label table parquet to")
    parser.add_argument(
        "--t-coverage", type=float, default=0.90,
        help="Minimum share of a user's total reviews that must be in a population for them to be eligible (default 0.90)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="Binomial-test significance level - a user is labeled toxic if p < alpha (default 0.01)",
    )
    return parser.parse_args()


def population_counts(table: pd.DataFrame, population: str) -> tuple:
    if population == UNION_KEY:
        n = table["n_pt"] + table["n_en"]
        t = table["n_pt_toxic"] + table["n_en_toxic"]
    else:
        n = table[f"n_{population}"]
        t = table[f"n_{population}_toxic"]
    return n, t


def label_population(table: pd.DataFrame, population: str, t_coverage: float, alpha: float) -> pd.DataFrame:
    n, n_toxic = population_counts(table, population)
    coverage = n / table["n_total"]
    eligible = (n > 0) & (coverage >= t_coverage)

    n_elig = n[eligible]
    t_elig = n_toxic[eligible]
    baseline_rate = t_elig.sum() / n_elig.sum()

    p_value = pd.Series(index=table.index, dtype="float64")
    p_value.loc[eligible] = binom.sf(t_elig.to_numpy() - 1, n_elig.to_numpy(), baseline_rate)

    is_toxic = pd.Series(index=table.index, dtype="boolean")
    is_toxic.loc[eligible] = p_value.loc[eligible] < alpha

    n_flagged = int(is_toxic.sum())
    info(
        f"[{population}] {int(eligible.sum()):,} eligible user(s) (coverage>={t_coverage:.0%}), "
        f"baseline rate={100 * baseline_rate:.3f}%, {n_flagged:,} labeled toxic "
        f"({100 * n_flagged / eligible.sum():.3f}% of eligible, alpha={alpha})"
    )

    return pd.DataFrame(
        {
            f"eligible_{population}": eligible,
            f"p_value_{population}": p_value,
            f"is_toxic_{population}": is_toxic,
        }
    ), baseline_rate, n_flagged, int(eligible.sum())


def main():
    args = parse_args()
    table = pd.read_parquet(args.table).set_index("user_url")
    info(f"Loaded user rate table: {len(table)} user(s) from {args.table}")

    per_population = {}
    summary_populations = {}
    for pop in POPULATIONS:
        cols, baseline_rate, n_flagged, n_eligible = label_population(table, pop, args.t_coverage, args.alpha)
        per_population[pop] = cols
        summary_populations[pop] = {
            "n_eligible": n_eligible,
            "baseline_toxic_rate": round(baseline_rate, 6),
            "n_toxic": n_flagged,
            "pct_toxic_of_eligible": round(100 * n_flagged / n_eligible, 4) if n_eligible else 0.0,
        }

    result = pd.concat(per_population.values(), axis=1)
    any_eligible = result[[f"eligible_{pop}" for pop in POPULATIONS]].any(axis=1)
    result = result.loc[any_eligible].reset_index()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    info(f"Saved toxic user label table ({len(result)} users, {len(result.columns)} columns) to: {args.output}")

    save_summary(
        {
            "t_coverage": args.t_coverage,
            "alpha": args.alpha,
            "n_users_in_output": int(len(result)),
            "populations": summary_populations,
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
