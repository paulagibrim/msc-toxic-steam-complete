"""Diagnostic (read-only, changes nothing): compares candidate ways of
labeling a user "toxic" from build_user_rate_table.py's per-user counts,
so T_rate is picked from real numbers instead of a guess.

Population: for each of pt/en/pt+en(union), users with n_<lang> > 0 AND
coverage (n_<lang> / n_total) >= T_COVERAGE - the 90% purity cutoff decided
from run_language_coverage_diagnostic.py's output (below 90% the achievable
coverage values are too coarse for a fixed threshold to behave sensibly -
e.g. at n_total=10, 9/10=90% is the only value between 80% and 100%, so a
95%+ cutoff would reject "9 good + 1 stray" users as if they were impure).

TWO labeling approaches are compared, at several settings each:
  - simple: flag a user toxic if n_<lang> >= min_n AND rate >= T_rate. Easy
    to explain, but for small n_<lang> (the median is 1) "rate" only takes a
    few coarse discrete values (0%, 50%, 100% at n=2), so a fixed T_rate
    either lets single-bad-review users dominate the label (if min_n is
    small) or discards most of the population (if min_n is large enough to
    make rate meaningful).
  - binomial: flag a user toxic if a one-sided binomial test rejects "this
    user's toxic reviews are just the corpus's baseline rate, applied to
    their sample size" - p = P(X >= n_toxic | n = n_<lang>, p = baseline
    rate), baseline computed as sum(n_<lang>_toxic)/sum(n_<lang>) over the
    SAME eligible (coverage-filtered) population being tested. This adapts
    to sample size automatically instead of needing a separate min_n: a
    single toxic review out of one total has p == the baseline rate itself
    (~1.5-2%), which clears alpha=0.05 but not alpha=0.01 - so alpha alone
    controls how much a lone bad review can trigger the label, without an
    arbitrary min_n cutoff. Tested at a spread of alpha (uncorrected) plus
    a Bonferroni-corrected 0.05 (dividing by the number of users tested, to
    account for running one test per user).

Usage:
    python 02_run_trate_diagnostic.py \\
        --table ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/user_rate_table.parquet \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/diagnostics/trate_diagnostic_report.json
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binom

from pipeline_utils import info, save_summary

T_COVERAGE = 0.90
UNION_KEY = "pt+en"
POPULATIONS = ["pt", "en", UNION_KEY]

SIMPLE_SETTINGS = [
    {"min_n": 1, "t_rate": 0.20},
    {"min_n": 3, "t_rate": 0.20},
    {"min_n": 5, "t_rate": 0.20},
    {"min_n": 3, "t_rate": 0.25},
    {"min_n": 5, "t_rate": 0.25},
    {"min_n": 5, "t_rate": 0.30},
]

ALPHAS = [0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compares simple-threshold vs binomial-test T_rate labeling on build_user_rate_table.py's output."
    )
    parser.add_argument("--table", required=True, type=Path, help="Path to build_user_rate_table.py's output parquet")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the comparison report JSON to")
    return parser.parse_args()


def eligible_population(table: pd.DataFrame, population: str) -> pd.DataFrame:
    """Users with >=1 review in `population` AND coverage >= T_COVERAGE
    (see module docstring for why 90%, not 95% or 100%)."""
    if population == UNION_KEY:
        n = table["n_pt"] + table["n_en"]
        t = table["n_pt_toxic"] + table["n_en_toxic"]
    else:
        n = table[f"n_{population}"]
        t = table[f"n_{population}_toxic"]
    coverage = n / table["n_total"]
    mask = (n > 0) & (coverage >= T_COVERAGE)
    return pd.DataFrame({"n": n[mask], "n_toxic": t[mask]})


def evaluate_simple(pop: pd.DataFrame, min_n: int, t_rate: float) -> dict:
    eligible = pop["n"] >= min_n
    rate = pop["n_toxic"] / pop["n"]
    flagged = eligible & (rate >= t_rate)
    n_pop = len(pop)
    return {
        "min_n": min_n,
        "t_rate": t_rate,
        "n_flagged": int(flagged.sum()),
        "pct_of_eligible_population": round(100 * flagged.sum() / n_pop, 4) if n_pop else 0.0,
        "pct_of_min_n_subset": round(100 * flagged.sum() / eligible.sum(), 4) if eligible.sum() else 0.0,
    }


def evaluate_binomial(pop: pd.DataFrame, baseline_rate: float, alpha: float, label: str) -> dict:
    pvals = binom.sf(pop["n_toxic"].to_numpy() - 1, pop["n"].to_numpy(), baseline_rate)
    flagged = pvals < alpha
    n_pop = len(pop)
    return {
        "alpha": alpha,
        "alpha_label": label,
        "n_flagged": int(flagged.sum()),
        "pct_of_eligible_population": round(100 * flagged.sum() / n_pop, 4) if n_pop else 0.0,
    }


def summarize_population(table: pd.DataFrame, population: str) -> dict:
    pop = eligible_population(table, population)
    n_pop = len(pop)
    baseline_rate = pop["n_toxic"].sum() / pop["n"].sum()
    info(f"--- {population}: eligible population (coverage>={T_COVERAGE:.0%}) = {n_pop:,} user(s), baseline rate = {100 * baseline_rate:.3f}% ---")

    simple_results = []
    for setting in SIMPLE_SETTINGS:
        result = evaluate_simple(pop, setting["min_n"], setting["t_rate"])
        simple_results.append(result)
        info(f"  [simple] n>={result['min_n']}, rate>={result['t_rate']:.0%} -> {result['n_flagged']:,} flagged ({result['pct_of_eligible_population']:.2f}% of eligible pop)")

    binomial_results = []
    bonferroni_alpha = 0.05 / n_pop if n_pop else 0.0
    alphas_to_test = ALPHAS + [bonferroni_alpha]
    alpha_labels = [f"{a:.2e} (uncorrected)" for a in ALPHAS] + [f"{bonferroni_alpha:.2e} (Bonferroni, 0.05/{n_pop:,})"]
    for alpha, label in zip(alphas_to_test, alpha_labels):
        result = evaluate_binomial(pop, baseline_rate, alpha, label)
        binomial_results.append(result)
        info(f"  [binomial] alpha={label} -> {result['n_flagged']:,} flagged ({result['pct_of_eligible_population']:.2f}% of eligible pop)")

    single_toxic = pop[(pop["n"] == 1) & (pop["n_toxic"] == 1)]
    edge_case = None
    if len(single_toxic):
        p_single = float(binom.sf(0, 1, baseline_rate))
        edge_case = {
            "description": "users with exactly 1 review, 1 toxic (rate=100%, the smallest-sample case)",
            "n_users": int(len(single_toxic)),
            "p_value": round(p_single, 6),
            "flagged_at_alpha_0.05": p_single < 0.05,
            "flagged_at_alpha_0.01": p_single < 0.01,
        }
        info(f"  [edge case] n=1,toxic=1 users ({len(single_toxic):,} of them) -> p-value = {p_single:.4f}")

    return {
        "n_eligible_population": int(n_pop),
        "t_coverage": T_COVERAGE,
        "baseline_toxic_rate": round(baseline_rate, 6),
        "simple_threshold": simple_results,
        "binomial_test": binomial_results,
        "edge_case_single_review": edge_case,
    }


def main():
    args = parse_args()
    table = pd.read_parquet(args.table)
    info(f"Loaded user rate table: {len(table)} user(s) from {args.table}")

    report = {
        "t_coverage": T_COVERAGE,
        "populations": {pop: summarize_population(table, pop) for pop in POPULATIONS},
    }
    save_summary(report, args.output)


if __name__ == "__main__":
    main()
