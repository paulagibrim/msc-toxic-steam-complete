"""Computes per-user average sentiment score (pt/en/union) from step04's
output - an INDEPENDENT signal (never used anywhere in building
toxic_user_labels.parquet, which is built purely from Perspective/Detoxify
toxicity scores) meant to validate that labeling scheme externally: if
users labeled "toxic" also write systematically more negative reviews than
non-toxic users, that's evidence the label captures a real behavioral
difference rather than an artifact of the T_rate/alpha/T_coverage choices.
See diagnostics/03_validate_labels_with_sentiment.py for the actual
comparison this table feeds.

SAME POPULATION DEFINITION as 01_build_user_rate_table.py's n_pt/n_en:
agreement-matched rows only (review_lang == perspective_declared_language)
with valid perspective/detoxify scores - not because sentiment_score
itself depends on those, but so the population being averaged here is
EXACTLY the same set of reviews already established elsewhere in this
project as "this user's pt/en reviews", not a subtly different one.

SHUFFLE-FREE (same pattern as 01_build_user_rate_table.py): each step04
file is reduced to its own per-user partial SUM(sentiment_score) + COUNT,
never folded into a running total - all partials collected in a list,
combined with a single final concat+groupby().sum() at the end.

Output: one parquet file, one row per user_url with >=1 valid review in
ANY of the three populations:
    user_url,
    avg_sentiment_pt, n_pt_sentiment,
    avg_sentiment_en, n_en_sentiment,
    avg_sentiment_union, n_union_sentiment

Usage:
    python 06_build_user_sentiment.py \\
        --step04-dir ../../../../steam-data/step04-output \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/user_sentiment_table.parquet
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline_utils import info, save_summary

LANGUAGES = ["pt", "en"]
UNION_KEY = "pt+en"
POPULATIONS = ["pt", "en", UNION_KEY]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds per-user average sentiment score (pt/en/union) from step04's output."
    )
    parser.add_argument("--step04-dir", required=True, type=Path, help="Path to step04's output directory")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the per-user sentiment table parquet to")
    return parser.parse_args()


def sweep_sentiment(step04_dir: Path) -> dict:
    files = sorted(step04_dir.rglob("*.parquet"))
    info(f"[sentiment] Sweeping {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step04_dir} (searched recursively).")

    sum_partials = {pop: [] for pop in POPULATIONS}
    count_partials = {pop: [] for pop in POPULATIONS}

    columns = ["user_url", "review_lang", "perspective_declared_language", "perspective_score", "detoxify_score", "sentiment_score"]
    n_disagreement_total = 0
    bar = tqdm(files, desc="[sentiment]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=columns)

        rows_before = len(df)
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        n_disagreement_total += rows_before - len(df)

        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]
        df = df[df["sentiment_score"].notna()]
        if df.empty:
            continue

        for pop in POPULATIONS:
            sub = df if pop == UNION_KEY else df[df["review_lang"] == pop]
            if sub.empty:
                continue
            sum_partials[pop].append(sub.groupby("user_url")["sentiment_score"].sum())
            count_partials[pop].append(sub.groupby("user_url").size())

        bar.set_postfix(rows=len(df))

    if n_disagreement_total:
        info(f"[sentiment] Excluded {n_disagreement_total:,} row(s) where review_lang != perspective_declared_language")

    info("[sentiment] Summing per-file partials...")
    means = {}
    counts = {}
    for pop in POPULATIONS:
        total_sum = pd.concat(sum_partials[pop]).groupby(level=0).sum()
        total_count = pd.concat(count_partials[pop]).groupby(level=0).sum()
        means[pop] = total_sum.div(total_count)
        counts[pop] = total_count
        info(f"[sentiment] [{pop}] {len(total_sum):,} user(s) with a valid sentiment score")

    return means, counts


def main():
    args = parse_args()
    means, counts = sweep_sentiment(args.step04_dir)

    info("Assembling final table...")
    all_users = sorted(set().union(*[means[pop].index for pop in POPULATIONS]))
    index = pd.Index(all_users, name="user_url")

    table = pd.DataFrame(index=index)
    for pop in POPULATIONS:
        pop_label = "union" if pop == UNION_KEY else pop
        table[f"avg_sentiment_{pop_label}"] = means[pop].reindex(index)
        table[f"n_{pop_label}_sentiment"] = counts[pop].reindex(index, fill_value=0).astype("int64")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(args.output, index=False)
    info(f"Saved user sentiment table ({len(table)} users, {len(table.columns) + 1} columns) to: {args.output}")

    save_summary({"n_users": int(len(table))}, args.output.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
