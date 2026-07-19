"""Recomputes per-user average sentiment score for TOXIC users ONLY,
excluding their own individually-toxic reviews - a SECONDARY, more
conservative check that complements 03_validate_labels_with_sentiment.py's
comparison (which uses ALL of a user's reviews and remains the primary
validation - see that script's docstring for why including toxic reviews
there is not circular and does not need this "fix"). Named/structured
after phase2_classical_ml/02_build_leave_toxic_out_embeddings.py's control
because the mechanics are the same, but the motivation differs: that
script addresses a REAL circularity risk (the ML classifier's embedding
feature is built from the same text the toxicity detectors read, so it
could just be re-detecting toxic language); sentiment_score comes from an
independent model measuring a different construct (tone, not
offensiveness), so there is no equivalent structural circularity here to
fix - this script instead asks a narrower, additional question.

WHAT THIS ANSWERS: is there still a sentiment difference between toxic
and non-toxic users even when a toxic user's own flagged review(s) are
excluded from their average? 06_build_user_sentiment.py's avg_sentiment_
<pop> mixes in whichever review(s) were toxic and caused the label -
this script recomputes the average using ONLY each toxic user's NON-toxic
reviews (same union rule as everywhere else in this project: toxic if
perspective_score >= 0.7 OR detoxify_score >= 0.9, invalid/sentinel scores
excluded from consideration entirely), so 04_validate_labels_leave_toxic_
out.py can check whether a difference survives outside the specifically
flagged content.

REAL COST OF THIS STRICTER VIEW, weigh against the extra rigor: ~30% of
toxic users (799 of 2,669 - see n_no_nontoxic_review_at_all below) have
EVERY review counted as toxic, so there is nothing non-toxic left to
average - they are silently absent from this output and therefore from
04_validate_labels_leave_toxic_out.py's comparison entirely, unlike
03_validate_labels_with_sentiment.py, which includes every toxic user who
has a sentiment score at all. This script's result is a supplementary data
point, not a replacement for the primary validation.

SCOPE: only toxic users (75 pt / 1,225 en / 1,316 union, per
toxic_user_labels.parquet), not the full population - non-toxic users'
sentiment average is untouched (already computed in
06_build_user_sentiment.py), since this control only concerns whether
TOXIC users' own flagged reviews were doing the work in the primary,
all-reviews comparison.

Reads step04-output directly (not step02) - it already carries
perspective_score/detoxify_score alongside sentiment_score in the same
row, so no separate join against step02 is needed to know which reviews
are individually toxic.

Output: one parquet file, one row per toxic user_url with >=1 valid
NON-toxic review in at least one population:
    user_url,
    avg_sentiment_pt_lto, n_pt_sentiment_lto,
    avg_sentiment_en_lto, n_en_sentiment_lto,
    avg_sentiment_union_lto, n_union_sentiment_lto
(_lto suffix distinguishes these from 06_build_user_sentiment.py's
all-reviews columns when the two tables are joined together.)

Usage:
    python 07_build_leave_toxic_out_sentiment.py \\
        --step04-dir ../../../../steam-data/step04-output \\
        --labels ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/leave_toxic_out_sentiment_table.parquet
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline_utils import info, save_summary

UNION_KEY = "pt+en"
POPULATIONS = ["pt", "en", UNION_KEY]
PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recomputes toxic users' average sentiment from their NON-toxic reviews only (leave-toxic-out control)."
    )
    parser.add_argument("--step04-dir", required=True, type=Path, help="Path to step04's output directory")
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="Path to build_toxic_user_labels.py's output - defines which users are toxic (is_toxic_pt/en/pt+en)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the leave-toxic-out sentiment table parquet to")
    return parser.parse_args()


def load_toxic_users(labels_path: Path) -> set:
    df = pd.read_parquet(labels_path, columns=["user_url", "is_toxic_pt", "is_toxic_en", "is_toxic_pt+en"])
    is_toxic_any = df["is_toxic_pt"].fillna(False) | df["is_toxic_en"].fillna(False) | df["is_toxic_pt+en"].fillna(False)
    users = set(df.loc[is_toxic_any, "user_url"])
    info(f"[scope] {len(users):,} user(s) toxic in at least one population - sentiment recomputed for these only")
    return users


def sweep_sentiment(step04_dir: Path, toxic_users: set) -> tuple:
    files = sorted(step04_dir.rglob("*.parquet"))
    info(f"[sentiment] Sweeping {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step04_dir} (searched recursively).")

    sum_partials = {pop: [] for pop in POPULATIONS}
    count_partials = {pop: [] for pop in POPULATIONS}
    n_excluded_toxic_review = 0

    columns = ["user_url", "review_lang", "perspective_declared_language", "perspective_score", "detoxify_score", "sentiment_score"]
    bar = tqdm(files, desc="[sentiment]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=columns)
        df = df[df["user_url"].isin(toxic_users)]
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        if df.empty:
            continue

        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]
        df = df[df["sentiment_score"].notna()]
        if df.empty:
            continue

        # The whole point of this control: exclude each user's OWN
        # individually-toxic reviews (same union rule as everywhere else),
        # keeping only their non-toxic ones for the average.
        is_toxic_review = (df["perspective_score"] >= PERSPECTIVE_THRESHOLD) | (df["detoxify_score"] >= DETOXIFY_THRESHOLD)
        n_excluded_toxic_review += int(is_toxic_review.sum())
        df = df[~is_toxic_review]
        if df.empty:
            continue

        for pop in POPULATIONS:
            sub = df if pop == UNION_KEY else df[df["review_lang"] == pop]
            if sub.empty:
                continue
            sum_partials[pop].append(sub.groupby("user_url")["sentiment_score"].sum())
            count_partials[pop].append(sub.groupby("user_url").size())

        bar.set_postfix(rows=len(df))

    info(f"[sentiment] Excluded {n_excluded_toxic_review:,} individually-toxic review(s) across all toxic users (the control's whole point)")

    info("[sentiment] Summing per-file partials...")
    means = {}
    counts = {}
    for pop in POPULATIONS:
        if not sum_partials[pop]:
            means[pop] = pd.Series(dtype="float64")
            counts[pop] = pd.Series(dtype="int64")
            info(f"[sentiment] [{pop}] 0 toxic user(s) have a non-toxic review to average")
            continue
        total_sum = pd.concat(sum_partials[pop]).groupby(level=0).sum()
        total_count = pd.concat(count_partials[pop]).groupby(level=0).sum()
        means[pop] = total_sum.div(total_count)
        counts[pop] = total_count
        info(f"[sentiment] [{pop}] {len(total_sum):,} toxic user(s) have >=1 non-toxic review to average")

    return means, counts


def main():
    args = parse_args()
    toxic_users = load_toxic_users(args.labels)
    means, counts = sweep_sentiment(args.step04_dir, toxic_users)

    info("Assembling final table...")
    populated_users = set().union(*[means[pop].index for pop in POPULATIONS if len(means[pop])])
    all_users = sorted(populated_users)
    index = pd.Index(all_users, name="user_url")

    table = pd.DataFrame(index=index)
    for pop in POPULATIONS:
        pop_label = "union" if pop == UNION_KEY else pop
        table[f"avg_sentiment_{pop_label}_lto"] = means[pop].reindex(index)
        table[f"n_{pop_label}_sentiment_lto"] = counts[pop].reindex(index, fill_value=0).astype("int64")

    n_toxic_total = len(toxic_users)
    n_no_nontoxic_at_all = n_toxic_total - len(table)
    info(
        f"{n_no_nontoxic_at_all:,} of {n_toxic_total:,} toxic user(s) have NO non-toxic review in ANY population "
        "(100% of their agreement-matched content is toxic) - entirely absent from this output"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(args.output, index=False)
    info(f"Saved leave-toxic-out sentiment table ({len(table)} user(s), {len(table.columns) + 1} columns) to: {args.output}")

    save_summary(
        {
            "n_toxic_users_total": n_toxic_total,
            "n_users_output": int(len(table)),
            "n_no_nontoxic_review_at_all": n_no_nontoxic_at_all,
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
