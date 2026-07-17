"""Measures how far Perspective and Detoxify agree, per language, and
saves the result as JSON.

Both scores already live on the same row of step02's own output (it keeps
every column step01 wrote and adds `detoxify_score`), so this is a
read-and-correlate, no join against step01 needed.

WHAT IS AND ISN'T IN SCOPE:
Only the languages step02 actually scored (pt/en by default) can appear
here - Detoxify never ran on the other 54 languages step01 detected. The
population is also narrower than "all pt/en reviews": step02 scores only
rows passing step01's agreement mask (review_lang == lang AND
perspective_declared_language == lang), so these correlations describe the
subset both language sources agree on, not the full language partition.

Rows where either score falls outside [0, 1] are excluded via step02's own
`validity_mask.apply_validity_mask` - Detoxify writes -1.0 when a batch
fails to score, and a sentinel dragged into a correlation would bias it
badly (a -1.0 sits far outside the real range, so it acts as a
high-leverage outlier rather than merely adding noise). They are excluded,
never coerced to 0.

BOTH PEARSON AND SPEARMAN ARE REPORTED, and the pair matters more than
either alone. Pearson asks whether the two scores track each other on the
same linear scale; Spearman asks only whether they rank reviews in the
same order.

Measured, Spearman comes out *below* Pearson in both languages (en: 0.72
vs 0.77; pt: 0.60 vs 0.76). That is the reverse of the usual expectation
for two differently-calibrated models, and it is a consequence of the
score distribution rather than of the models tracking each other linearly:
both scores pile up near zero (mean perspective_score 0.13 for en, 0.08
for pt), and within that mass the two models order reviews close to
arbitrarily. Spearman weights that dense, noisy low-toxicity region the
same as everything else, while Pearson is carried by the genuinely toxic
tail where the two models do move together. Reading Pearson alone would
therefore overstate how much the models agree on any single review.

Usage:
    python run_score_correlation.py \\
        --input ../../steam-data/step02-output \\
        --output ../../steam-data/step02-output/score_correlation_report.json
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from pipeline_utils import info, save_summary
from validity_mask import apply_validity_mask

SCORE_COLUMNS = ["perspective_score", "detoxify_score", "review_lang"]

# The project's toxicity thresholds, the same pair used everywhere else
# (review_examples/show_review_examples.py labels a review toxic on
# perspective_score >= 0.7 OR detoxify_score >= 0.9). Repeated rather than
# imported: review_examples is a sibling tool, not a dependency of this
# step, and this file must stay runnable on the GPU machine where only
# step02's own folder is copied.
PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9


def discover_languages(step02_dir: Path) -> list:
    """Lists the languages present, supporting both of step02's output
    layouts: Hive-style `review_lang=xx/` subfolders (what is on disk), and
    the flat one-folder-of-parquet form the rest of step02's modules now
    expect (see validity_mask.load_scored_language). Reading the flat form
    means opening the files, so the partitioned form is checked first."""
    partitions = sorted(step02_dir.glob("review_lang=*"))
    if partitions:
        return [p.name.split("=", 1)[1] for p in partitions]

    files = sorted(step02_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"No parquet files and no review_lang=*/ subfolders under {step02_dir} - "
            f"is this step02's output directory?"
        )
    langs = set()
    for f in files:
        langs.update(pd.read_parquet(f, columns=["review_lang"])["review_lang"].dropna().unique())
    return sorted(langs)


def load_scores(step02_dir: Path, lang: str) -> pd.DataFrame:
    """Reads only the two score columns for one language, from whichever
    layout is on disk. `review_lang` is stored inside the files as well as
    in the partition path, so it is filtered on explicitly either way
    rather than trusting the folder name."""
    partition = step02_dir / f"review_lang={lang}"
    source = partition if partition.is_dir() else step02_dir
    df = pd.read_parquet(source, columns=SCORE_COLUMNS)
    return df[df["review_lang"] == lang]


def threshold_agreement(perspective: pd.Series, detoxify: pd.Series) -> dict:
    """Counts how often the two models agree on the *toxic label*, not on
    the score - i.e. how much of what gets called toxic depends on which
    model was believed.

    This answers a different question from the correlations above, and the
    answer is far less reassuring. A correlation describes the whole score
    range at once, and both models spend most of it agreeing that ordinary
    reviews are not toxic - which is easy, and which dominates the
    statistic. What downstream analysis actually consumes is the label, and
    the label is decided at a single point where the two models are free to
    disagree completely. Reporting only r would hide that.
    """
    perspective_toxic = perspective >= PERSPECTIVE_THRESHOLD
    detoxify_toxic = detoxify >= DETOXIFY_THRESHOLD

    both = int((perspective_toxic & detoxify_toxic).sum())
    perspective_only = int((perspective_toxic & ~detoxify_toxic).sum())
    detoxify_only = int((~perspective_toxic & detoxify_toxic).sum())
    either = both + perspective_only + detoxify_only

    return {
        "perspective_threshold": PERSPECTIVE_THRESHOLD,
        "detoxify_threshold": DETOXIFY_THRESHOLD,
        "toxic_either": either,
        "toxic_both": both,
        "toxic_perspective_only": perspective_only,
        "toxic_detoxify_only": detoxify_only,
        # Of everything either model calls toxic, the share both agree on.
        # This is the number that matters: 1 - it is the share of the toxic
        # set that rests on a single model's say-so.
        "both_pct_of_either": round(100 * both / either, 2) if either else 0.0,
    }


def correlate_language(df: pd.DataFrame, lang: str) -> dict:
    rows_total = len(df)
    valid = df[apply_validity_mask(df)]
    rows_valid = len(valid)

    summary = {
        "language": lang,
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "rows_excluded_invalid": rows_total - rows_valid,
        "valid_pct": round(100 * rows_valid / rows_total, 2) if rows_total else 0.0,
    }

    # Correlation is undefined for a constant input, and scipy returns NaN
    # with a warning rather than raising - caught here so the report says
    # why the number is missing instead of carrying a bare null.
    if rows_valid < 2:
        summary["error"] = "fewer than 2 valid rows - correlation undefined"
        return summary

    p, d = valid["perspective_score"].astype("float64"), valid["detoxify_score"].astype("float64")
    if p.nunique() < 2 or d.nunique() < 2:
        summary["error"] = "a score column is constant - correlation undefined"
        return summary

    pearson = stats.pearsonr(p, d)
    spearman = stats.spearmanr(p, d)

    summary.update({
        "pearson_r": round(float(pearson.statistic), 4),
        "spearman_rho": round(float(spearman.statistic), 4),
        "perspective_mean": round(float(p.mean()), 4),
        "perspective_std": round(float(p.std()), 4),
        "detoxify_mean": round(float(d.mean()), 4),
        "detoxify_std": round(float(d.std()), 4),
        "threshold_agreement": threshold_agreement(p, d),
    })
    ta = summary["threshold_agreement"]
    info(
        f"[{lang}] n={rows_valid} | Pearson r={summary['pearson_r']} | "
        f"Spearman rho={summary['spearman_rho']} | "
        f"excluded {summary['rows_excluded_invalid']} invalid row(s)"
    )
    info(
        f"[{lang}] toxic by either model: {ta['toxic_either']} | both agree: {ta['toxic_both']} "
        f"({ta['both_pct_of_either']}%) | perspective only: {ta['toxic_perspective_only']} | "
        f"detoxify only: {ta['toxic_detoxify_only']}"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlates perspective_score against detoxify_score, per language."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to step02's output directory")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the correlation report JSON to")
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language to correlate (repeat for multiple). Defaults to every language found.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages or discover_languages(args.input)
    info(f"Correlating {len(languages)} language(s): {', '.join(languages)}")

    summaries = []
    for lang in languages:
        df = load_scores(args.input, lang)
        summaries.append(correlate_language(df, lang))

    save_summary(
        {
            "note": (
                "Only languages step02 scored appear here; each population is step01's "
                "agreement-mask subset (review_lang == lang AND "
                "perspective_declared_language == lang), not the full language partition."
            ),
            "languages": summaries,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
