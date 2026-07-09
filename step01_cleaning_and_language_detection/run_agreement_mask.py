"""CLI for agreement_mask.py - reports how many reviews in each
review_lang=<lang> partition also have Steam's own declared language
agreeing (perspective_declared_language == lang). Doesn't save a filtered
copy of the data - the mask is a plain `==` on an already-present column,
cheap enough to apply on demand wherever it's needed (see
agreement_mask.py's module docstring) rather than duplicating a
multi-million-row partition on disk.

Usage:
    python run_agreement_mask.py \\
        --input /Users/gibrim/Documents/dev/steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --lang pt --lang en

`--input` is run_clean_reviews.py's output directory (the
`reviews_cleaned.parquet` folder itself, containing the `review_lang=*`
partitions) - not the raw reviews.
"""
import argparse
from pathlib import Path

import agreement_mask as am
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reports how many reviews per language have langdetect and Steam's declared language agreeing."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to run_clean_reviews.py's reviews_cleaned.parquet directory",
    )
    parser.add_argument(
        "--lang", action="append", required=True, dest="languages",
        help="Language to check (repeat for multiple, e.g. --lang pt --lang en)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for lang in args.languages:
        df = am.load_language_partition(args.input, lang)
        rows_before = len(df)

        mask = am.apply_agreement_mask(df, lang)
        rows_agree = len(mask)

        if rows_before:
            info(
                f"[{lang}] {rows_agree} of {rows_before} rows agree "
                f"(langdetect AND Steam both say '{lang}') ({100*rows_agree/rows_before:.2f}%)"
            )
        else:
            info(f"[{lang}] 0 rows in partition")


if __name__ == "__main__":
    main()
