"""Diagnostic (read-only, changes nothing): measures how many reviews
currently sitting in a review_lang=<lang> partition (Stage 2's output)
contain the Steam boilerplate phrases that clean_for_detection was missing
before the fix, and how many of those would get a DIFFERENT review_lang if
re-detected with the fix applied.

Only runs the (slow) langdetect model on rows that actually contain the
boilerplate - a cheap substring scan first narrows down to that subset, so
this doesn't need to touch the full corpus with the model.

Usage:
    python run_boilerplate_diagnostic.py \\
        --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output ../../steam-data/step01-output/boilerplate_impact_report.json

Defaults to scanning every review_lang=* partition found; pass --lang
(repeatable) to restrict to specific ones, e.g. --lang pt --lang en.
"""
import argparse
from pathlib import Path

import pandas as pd

from langdetect_revalidation import BOILERPLATE_PATTERNS, identify_language_langdetect
from pipeline_utils import info, save_summary

BOILERPLATE_REGEX_STR = "|".join(BOILERPLATE_PATTERNS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measures how many reviews would get a different review_lang once boilerplate stripping is fixed."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to reviews_cleaned.parquet (contains review_lang=* subfolders)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the impact report JSON to",
    )
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language partition(s) to scan (repeat for multiple). Defaults to every "
        "review_lang=* partition found under --input.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.languages:
        lang_dirs = [args.input / f"review_lang={lang}" for lang in args.languages]
        lang_dirs = [d for d in lang_dirs if d.is_dir()]
    else:
        lang_dirs = sorted(d for d in args.input.iterdir() if d.is_dir() and d.name.startswith("review_lang="))
    info(f"Scanning {len(lang_dirs)} language partition(s): {[d.name for d in lang_dirs]}")

    total_rows = 0
    total_with_boilerplate = 0
    unchanged = 0
    flips: dict = {}

    for lang_dir in lang_dirs:
        old_lang = lang_dir.name.split("=", 1)[1]
        files = sorted(lang_dir.glob("*.parquet"))
        info(f"[{old_lang}] Scanning {len(files)} file(s)...")

        lang_rows = 0
        lang_boilerplate = 0
        for i, f in enumerate(files, start=1):
            df = pd.read_parquet(f, columns=["review_text"])
            lang_rows += len(df)

            mask = df["review_text"].str.contains(BOILERPLATE_REGEX_STR, case=False, na=False, regex=True)
            affected = df.loc[mask, "review_text"]
            lang_boilerplate += len(affected)

            for text in affected:
                new_lang, _ = identify_language_langdetect(text)
                if new_lang == old_lang:
                    unchanged += 1
                else:
                    key = f"{old_lang}->{new_lang}"
                    flips[key] = flips.get(key, 0) + 1

            if i % 20 == 0 or i == len(files):
                info(f"[{old_lang}] [{i}/{len(files)}] {lang_boilerplate} boilerplate row(s) found so far")

        total_rows += lang_rows
        total_with_boilerplate += lang_boilerplate
        info(f"[{old_lang}] Done: {lang_rows} row(s), {lang_boilerplate} with boilerplate")

    changed = total_with_boilerplate - unchanged
    report = {
        "total_rows_scanned": total_rows,
        "total_rows_with_boilerplate": total_with_boilerplate,
        "unchanged_after_fix": unchanged,
        "changed_after_fix": changed,
        "changed_pct_of_boilerplate_rows": (
            round(100 * changed / total_with_boilerplate, 2) if total_with_boilerplate else 0.0
        ),
        "changed_pct_of_all_rows": round(100 * changed / total_rows, 4) if total_rows else 0.0,
        "flip_breakdown": flips,
    }

    info(
        f"Total rows: {total_rows} | with boilerplate: {total_with_boilerplate} | "
        f"would change language: {changed} ({report['changed_pct_of_boilerplate_rows']:.2f}% of boilerplate rows, "
        f"{report['changed_pct_of_all_rows']:.4f}% of all rows)"
    )
    info(f"Flip breakdown: {flips}")

    save_summary(report, args.output)


if __name__ == "__main__":
    main()
