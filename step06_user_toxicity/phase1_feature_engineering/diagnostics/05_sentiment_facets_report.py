"""Diagnostic (read-only, changes nothing): a REVIEW-level companion to
03_validate_labels_with_sentiment.py's USER-level validation. That script
asks "do toxic-labeled USERS write more negatively overall" - this script
asks three narrower questions about individual REVIEWS' sentiment_score,
which the user-level rollup can't answer on its own:

  (i) SENTIMENT BY TOXICITY, PER LANGUAGE: the distribution of
      sentiment_score for toxic vs non-toxic REVIEWS (not users), separately
      for pt and en - the same union rule as everywhere else in this
      project (perspective_score >= 0.7 OR detoxify_score >= 0.9).

  (ii) SENTIMENT BY VENN REGION: reviews split into the toxicity Venn's four
      regions - perspective_only, detoxify_only, both, neither - each with
      its own sentiment_score distribution. Perspective and Detoxify are
      different models trained on different data; if one's "only" region
      has a distinctly different sentiment profile than the other's, that's
      evidence the two tools are sensitive to different FACETS of toxic
      language (e.g. one catching profanity/slurs the other misses, or vice
      versa) - not just noisier/stricter versions of the same detector.

  (iii) CROSSED WITH is_recommended: whether a review's sentiment/toxicity
      relationship interacts with whether the user recommended the game -
      e.g. a toxic AND not-recommended review ("this game is broken AND I
      hate everyone in it") may differ from a toxic BUT recommended one
      ("great game, [insult directed at other players]"). Reported as a
      count/percentage cross-tab (toxic x is_recommended) plus
      sentiment_score distributions for each of the resulting four cells,
      per language. is_recommended can be None (missing) in the raw data -
      kept as its own bucket, never merged into False.

SAME FILTERS as every other review-level analysis in this project:
agreement-matched rows only (review_lang == perspective_declared_language),
valid perspective/detoxify scores (Detoxify's -1.0 "failed to score"
sentinel and anything outside [0, 1] excluded entirely - neither toxic nor
non-toxic), and a valid (non-null) sentiment_score.

Distributions are reported as full describe() stats (count/mean/std/min/
25/50/75/max), not just the mean - computed from actual review-level
values collected across all of step04's files (concatenated once at the
end, the same shuffle-free "collect partials, single final combine"
pattern as this project's other diagnostics, just holding raw values
instead of pre-aggregated sums/counts, since percentiles cannot be
computed by summing per-file percentiles).

Usage:
    python 05_sentiment_facets_report.py \\
        --step04-dir ../../../../steam-data/step04-output \\
        --output ../../../../steam-data/step06-output/phase1_feature_engineering/diagnostics/sentiment_facets_report.json
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_utils import info, save_summary

LANGUAGES = ["pt", "en"]
PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9
VENN_REGIONS = ["perspective_only", "detoxify_only", "both", "neither"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review-level sentiment facets: by toxicity, by Venn region, crossed with is_recommended."
    )
    parser.add_argument("--step04-dir", required=True, type=Path, help="Path to step04's output directory")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the facets report JSON to")
    return parser.parse_args()


def venn_region(perspective_flag: pd.Series, detoxify_flag: pd.Series) -> pd.Series:
    region = pd.Series("neither", index=perspective_flag.index, dtype="object")
    region[perspective_flag & ~detoxify_flag] = "perspective_only"
    region[~perspective_flag & detoxify_flag] = "detoxify_only"
    region[perspective_flag & detoxify_flag] = "both"
    return region


def sweep(step04_dir: Path) -> pd.DataFrame:
    """Single pass over step04's files, returning ONE concatenated
    DataFrame of the columns needed downstream (language, is_toxic,
    venn_region, is_recommended, sentiment_score) - filtered to
    agreement-matched, valid-score, valid-sentiment rows only. Small
    enough to hold in memory at once (a handful of columns x the corpus's
    ~27M agreement-matched pt/en reviews), unlike the 384-dim embedding
    case elsewhere in this project."""
    files = sorted(step04_dir.rglob("*.parquet"))
    info(f"[sweep] Reading {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step04_dir} (searched recursively).")

    columns = [
        "review_lang", "perspective_declared_language", "perspective_score",
        "detoxify_score", "sentiment_score", "is_recommended",
    ]
    partials = []
    n_disagreement_total = 0
    bar = tqdm(files, desc="[sweep]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=columns)

        rows_before = len(df)
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        n_disagreement_total += rows_before - len(df)

        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]
        df = df[df["sentiment_score"].notna()]
        df = df[df["review_lang"].isin(LANGUAGES)]
        if df.empty:
            continue

        p_flag = df["perspective_score"] >= PERSPECTIVE_THRESHOLD
        d_flag = df["detoxify_score"] >= DETOXIFY_THRESHOLD
        df = df.assign(
            is_toxic=p_flag | d_flag,
            venn_region=venn_region(p_flag, d_flag),
        )
        partials.append(df[["review_lang", "is_toxic", "venn_region", "is_recommended", "sentiment_score"]])
        bar.set_postfix(rows=len(df))

    if n_disagreement_total:
        info(f"[sweep] Excluded {n_disagreement_total:,} row(s) where review_lang != perspective_declared_language")

    info("[sweep] Concatenating partials...")
    combined = pd.concat(partials, ignore_index=True)
    info(f"[sweep] {len(combined):,} agreement-matched, valid-score, valid-sentiment review(s) across {LANGUAGES}")
    return combined


def describe_stats(values: pd.Series) -> dict:
    if len(values) == 0:
        return {"count": 0}
    desc = values.describe(percentiles=[0.25, 0.5, 0.75])
    return {
        "count": int(desc["count"]),
        "mean": round(float(desc["mean"]), 4),
        "std": round(float(desc["std"]), 4) if len(values) > 1 else None,
        "min": round(float(desc["min"]), 4),
        "p25": round(float(desc["25%"]), 4),
        "median": round(float(desc["50%"]), 4),
        "p75": round(float(desc["75%"]), 4),
        "max": round(float(desc["max"]), 4),
    }


def build_sentiment_by_toxicity(df: pd.DataFrame) -> dict:
    """(i) Distribution of sentiment_score for toxic vs non-toxic REVIEWS,
    per language."""
    report = {}
    for lang in LANGUAGES + ["all"]:
        sub = df if lang == "all" else df[df["review_lang"] == lang]
        report[lang] = {
            "toxic": describe_stats(sub.loc[sub["is_toxic"], "sentiment_score"]),
            "non_toxic": describe_stats(sub.loc[~sub["is_toxic"], "sentiment_score"]),
        }
        info(
            f"[i] [{lang}] toxic mean={report[lang]['toxic'].get('mean')}, "
            f"non_toxic mean={report[lang]['non_toxic'].get('mean')}"
        )
    return report


def build_sentiment_by_venn_region(df: pd.DataFrame) -> dict:
    """(ii) Distribution of sentiment_score per Venn region (perspective_only/
    detoxify_only/both/neither), per language - what facet each tool
    captures."""
    report = {}
    for lang in LANGUAGES + ["all"]:
        sub = df if lang == "all" else df[df["review_lang"] == lang]
        report[lang] = {}
        for region in VENN_REGIONS:
            values = sub.loc[sub["venn_region"] == region, "sentiment_score"]
            report[lang][region] = describe_stats(values)
            info(f"[ii] [{lang}] [{region}] n={len(values):,}, mean={report[lang][region].get('mean')}")
    return report


def build_crossed_with_recommendation(df: pd.DataFrame) -> dict:
    """(iii) Cross-tab of toxic-status x is_recommended (counts/%) plus
    sentiment_score distribution for each of the resulting cells, per
    language. is_recommended's None (missing) values are kept as their own
    bucket, never merged into False."""
    report = {}
    for lang in LANGUAGES + ["all"]:
        sub = df if lang == "all" else df[df["review_lang"] == lang]
        rec_label = sub["is_recommended"].map({True: "recommended", False: "not_recommended"}).fillna("missing")

        crosstab_counts = pd.crosstab(sub["is_toxic"], rec_label)
        n_total = len(sub)
        crosstab = {
            ("toxic" if is_toxic else "non_toxic"): {
                rec_cat: {
                    "count": int(crosstab_counts.loc[is_toxic, rec_cat]) if rec_cat in crosstab_counts.columns else 0,
                    "pct_of_total": round(
                        100 * (crosstab_counts.loc[is_toxic, rec_cat] if rec_cat in crosstab_counts.columns else 0) / n_total, 4
                    ) if n_total else 0.0,
                }
                for rec_cat in ["recommended", "not_recommended", "missing"]
            }
            for is_toxic in [True, False]
        }

        sentiment_by_cell = {}
        for is_toxic, toxic_label in [(True, "toxic"), (False, "non_toxic")]:
            sentiment_by_cell[toxic_label] = {}
            for rec_cat in ["recommended", "not_recommended", "missing"]:
                mask = (sub["is_toxic"] == is_toxic) & (rec_label == rec_cat)
                sentiment_by_cell[toxic_label][rec_cat] = describe_stats(sub.loc[mask, "sentiment_score"])

        report[lang] = {"crosstab_counts": crosstab, "sentiment_by_cell": sentiment_by_cell}
        info(f"[iii] [{lang}] crosstab: {crosstab}")
    return report


def main():
    args = parse_args()
    df = sweep(args.step04_dir)

    report = {
        "sentiment_by_toxicity": build_sentiment_by_toxicity(df),
        "sentiment_by_venn_region": build_sentiment_by_venn_region(df),
        "sentiment_crossed_with_recommendation": build_crossed_with_recommendation(df),
    }
    save_summary(report, args.output)


if __name__ == "__main__":
    main()
