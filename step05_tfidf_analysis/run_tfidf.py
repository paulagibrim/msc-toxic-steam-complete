"""CLI for tfidf_analysis.py - labels toxicity (same union rule/thresholds
used throughout this project), cleans text, and compares toxic vs.
non-toxic TF-IDF term weights across step02's pt/en reviews.

Usage:
    python run_tfidf.py \\
        --input ../../steam-data/step02-output \\
        --output-dir ../../steam-data/step05-output

Defaults to both pt and en (pass --lang to override, repeatable).
`--input` is step02's output directory - review_lang is a plain column
there, not a subfolder (see detoxify_scoring.py's module docstring) -
unlike step03/step04, this reads EVERY review (toxic and non-toxic),
since the whole point is comparing the two groups' term weights.
"""
import argparse
from pathlib import Path

import tfidf_analysis as ta
from pipeline_utils import info, save_summary

LANG_NLTK_NAME = {"pt": "portuguese", "en": "english"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-IDF toxic vs. non-toxic term-weight comparison for step02's pt/en reviews."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to step02's output directory",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write outputs to")
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language to process (repeat for multiple). Defaults to both pt and en.",
    )
    return parser.parse_args()


def run_one(input_dir: Path, output_dir: Path, lang: str) -> None:
    info(f"[{lang}] Loading step02 output...")
    df = ta.load_reviews_for_language(input_dir, lang)
    rows_loaded = len(df)
    info(f"[{lang}] Loaded {rows_loaded} reviews")

    df = ta.label_toxicity(df)
    rows_excluded_invalid = rows_loaded - len(df)
    rows_toxic = int(df["is_toxic"].sum())
    rows_non_toxic = len(df) - rows_toxic
    info(
        f"[{lang}] Valid: {len(df)} ({rows_excluded_invalid} excluded, invalid score) - "
        f"Toxic: {rows_toxic} ({100 * rows_toxic / len(df):.2f}%), Non-toxic: {rows_non_toxic}"
    )

    df["review_text_clean"] = ta.clean_text_for_tfidf(df["review_text"])
    ta.export_cleaned_labeled(df, output_dir, lang)

    stop_words = ta.build_stop_words(LANG_NLTK_NAME[lang], ta.STOPWORD_EXTRAS[lang])
    vectorizer = ta.fit_vocabulary(df["review_text_clean"], stop_words)
    vocab_size = len(vectorizer.get_feature_names_out())
    info(f"[{lang}] Vocabulary size: {vocab_size}")

    means = ta.compute_group_means(df, vectorizer)

    table = ta.build_lexicon_table(vectorizer, means)
    output_path = ta.export_lexicon_table(table, output_dir, lang)

    info(f"[{lang}] Top 15 most toxic-associated terms:")
    print(table.head(15).to_string(index=False))

    save_summary(
        {
            "language": lang,
            "rows_loaded": rows_loaded,
            "rows_excluded_invalid": rows_excluded_invalid,
            "rows_toxic": rows_toxic,
            "rows_non_toxic": rows_non_toxic,
            "vocabulary_size": vocab_size,
            "count_toxic": means["count_toxic"],
            "count_neutral": means["count_neutral"],
            "lexicon_path": str(output_path),
        },
        output_dir / f"tfidf_report_{lang}.json",
    )


def main():
    args = parse_args()
    for lang in args.languages or ["pt", "en"]:
        run_one(args.input, args.output_dir, lang)


if __name__ == "__main__":
    main()
