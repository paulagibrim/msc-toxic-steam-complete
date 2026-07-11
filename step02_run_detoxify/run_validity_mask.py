"""CLI for validity_mask.py - reports and saves how many rows in step02's
own scored output have a valid perspective_score/detoxify_score (both
inside [0, 1]) vs. an invalid/sentinel value (Detoxify's -1.0 "failed to
score" being the main case).

Usage:
    python run_validity_mask.py \\
        --input ../../steam-data/step02-output \\
        --output ../../steam-data/step02-output/validity_report.json

Defaults to checking both pt and en; pass --lang (repeatable) to override.
`--input` is this step's own output directory (contains review_lang=<lang>
subfolders) - the scored data run_detoxify.py already produced.
"""
import argparse
from pathlib import Path

import validity_mask as vm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reports how many rows in step02's output have a valid perspective_score/detoxify_score."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to step02's own output directory (contains review_lang=<lang> subfolders)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the validity report JSON to",
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
        df = vm.load_scored_language(args.input, lang)
        summaries.append(vm.summarize_validity(df, lang))

    vm.save_validity_report(summaries, args.output)


if __name__ == "__main__":
    main()
