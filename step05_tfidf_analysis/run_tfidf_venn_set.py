"""CLI for tfidf_venn_set_analysis.py - TF-IDF term-weight comparison
across the three mutually-exclusive Venn regions (Perspective only,
Detoxify only, Both) instead of step05's original toxic-vs-non-toxic
split. See that module's docstring for why step04 (not step02) is the
data source, and how the three-way diferenca/log_ratio is computed.

Usage:
    python run_tfidf_venn_set.py \\
        --input ../../steam-data/step04-output \\
        --output-dir ../../steam-data/step05-output-venn

Defaults to both pt and en (pass --lang to override, repeatable).
"""
import argparse
from pathlib import Path

import tfidf_analysis as ta
import tfidf_venn_set_analysis as tv
from pipeline_utils import info, save_summary

LANG_NLTK_NAME = {"pt": "portuguese", "en": "english"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-IDF term-weight comparison across Perspective-only/Detoxify-only/Both Venn regions."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to step04's output directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write outputs to")
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language to process (repeat for multiple). Defaults to both pt and en.",
    )
    return parser.parse_args()


def run_one(input_dir: Path, output_dir: Path, lang: str) -> None:
    info(f"[{lang}] Loading step04 output...")
    df = tv.load_flagged_reviews_for_language(input_dir, lang)

    counts_by_set = df["venn_set"].value_counts().to_dict()
    info(f"[{lang}] Venn-set breakdown: {counts_by_set}")

    df["review_text_clean"] = ta.clean_text_for_tfidf(df["review_text"])

    stop_words = ta.build_stop_words(LANG_NLTK_NAME[lang], ta.STOPWORD_EXTRAS[lang])
    vectorizer = ta.fit_vocabulary(df["review_text_clean"], stop_words)
    vocab_size = len(vectorizer.get_feature_names_out())
    info(f"[{lang}] Vocabulary size: {vocab_size}")

    result = tv.compute_set_means(df, vectorizer)
    table = tv.build_lexicon_table(vectorizer, result)
    output_path = tv.export_lexicon_table(table, output_dir, lang)

    for venn_set in tv.VENN_SETS:
        info(f"[{lang}] Top 10 terms most distinctive of {venn_set}:")
        top = table.sort_values(f"diferenca_{venn_set}", ascending=False).head(10)
        print(top[["termo", f"tfidf_{venn_set}", f"diferenca_{venn_set}"]].to_string(index=False))

    save_summary(
        {
            "language": lang,
            "vocabulary_size": vocab_size,
            "counts_by_venn_set": {k: int(v) for k, v in result["counts"].items()},
            "lexicon_path": str(output_path),
        },
        output_dir / f"tfidf_venn_report_{lang}.json",
    )


def main():
    args = parse_args()
    for lang in args.languages or ["pt", "en"]:
        run_one(args.input, args.output_dir, lang)


if __name__ == "__main__":
    main()
