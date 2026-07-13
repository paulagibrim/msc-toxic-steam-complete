"""Diagnostic (read-only, changes nothing): measures how many reviews
currently assigned a given review_lang (Stage 2's output) contain the
Steam boilerplate phrases that clean_for_detection was missing before the
fix, and how many of those would get a DIFFERENT review_lang if
re-detected with the fix applied.

Only runs the (slow) langdetect model on rows that actually contain the
boilerplate - a cheap substring scan first narrows down to that subset, so
this doesn't need to touch the full corpus with the model.

review_lang is a plain column here (see clean_reviews.export_reviews's
docstring for why it's not Hive-style directory partitioning), so this
reads every file under --input and groups by the review_lang value found
in each row, rather than by folder.

Usage:
    python run_boilerplate_diagnostic.py \\
        --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output ../../steam-data/step01-output/boilerplate_impact_report.json

Defaults to scanning every review_lang value found; pass --lang (repeatable)
to restrict to specific ones, e.g. --lang pt --lang en.
"""
import argparse
from pathlib import Path

import pandas as pd

from langdetect_revalidation import BOILERPLATE_PATTERNS, identify_language_langdetect
from pipeline_utils import info, list_parquet_files, save_summary

BOILERPLATE_REGEX_STR = "|".join(BOILERPLATE_PATTERNS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measures how many reviews would get a different review_lang once boilerplate stripping is fixed."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to reviews_cleaned.parquet (review_lang is a column, not a subfolder)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the impact report JSON to",
    )
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="review_lang value(s) to scan (repeat for multiple). Defaults to every value found.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = list_parquet_files(args.input)

    total_rows = 0
    total_with_boilerplate = 0
    unchanged = 0
    flips: dict = {}
    per_lang_rows: dict = {}
    per_lang_boilerplate: dict = {}

    for i, f in enumerate(files, start=1):
        df = pd.read_parquet(f, columns=["review_text", "review_lang"])
        if args.languages:
            df = df[df["review_lang"].isin(args.languages)]
        if df.empty:
            continue

        total_rows += len(df)
        for lang, count in df["review_lang"].value_counts().items():
            per_lang_rows[lang] = per_lang_rows.get(lang, 0) + int(count)

        mask = df["review_text"].str.contains(BOILERPLATE_REGEX_STR, case=False, na=False, regex=True)
        affected = df.loc[mask, ["review_text", "review_lang"]]
        total_with_boilerplate += len(affected)

        for text, old_lang in zip(affected["review_text"], affected["review_lang"]):
            per_lang_boilerplate[old_lang] = per_lang_boilerplate.get(old_lang, 0) + 1
            new_lang, _ = identify_language_langdetect(text)
            if new_lang == old_lang:
                unchanged += 1
            else:
                key = f"{old_lang}->{new_lang}"
                flips[key] = flips.get(key, 0) + 1

        if i % 20 == 0 or i == len(files):
            info(f"[{i}/{len(files)}] {total_with_boilerplate} boilerplate row(s) found so far")

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
        "rows_per_lang": per_lang_rows,
        "boilerplate_rows_per_lang": per_lang_boilerplate,
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
