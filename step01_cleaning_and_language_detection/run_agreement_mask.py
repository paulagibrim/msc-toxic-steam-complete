"""CLI for agreement_mask.py - reports and saves, per language, how many
reviews in `review_lang=<lang>` also have Steam's own declared language
agreeing (perspective_declared_language == lang). Doesn't save a filtered
copy of the underlying data - the mask itself is a plain `==` on an
already-present column, cheap enough to apply on demand wherever it's
needed (see agreement_mask.py's module docstring) - but the small,
aggregate per-language counts/percentages ARE saved, as a report.

Usage:
    python run_agreement_mask.py \\
        --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output ../../steam-data/step01-output/agreement_report.json

Defaults to checking pt and en; pass --lang (repeatable) to override.

`--input` is run_detect_language.py's output directory (the
`reviews_cleaned.parquet` folder itself, containing the `review_lang=*`
partitions) - not the raw reviews.
"""
import argparse
from pathlib import Path

import agreement_mask as am


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reports and saves how many reviews per language have langdetect and Steam's declared language agreeing."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to run_detect_language.py's reviews_cleaned.parquet directory",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the agreement report JSON to",
    )
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language to check (repeat for multiple). Defaults to pt and en.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages or ["pt", "en"]

    summaries = []
    for lang in languages:
        df = am.load_language_partition(args.input, lang)
        summaries.append(am.summarize_agreement(df, lang))

    am.save_agreement_report(summaries, args.output)


if __name__ == "__main__":
    main()
