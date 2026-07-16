"""Diagnostic (read-only, changes nothing): decomposes the `und`
(undetermined) label produced by run_detect_language.py into the distinct
conditions that all collapse into it.

`identify_language_langdetect` returns ("und", 0.0) from three separate
paths, indistinguishable in the output since the label AND the confidence
are identical in every case:

  1. GATED - the cleaned text has fewer than MIN_ALPHA_LENGTH alphabetic
     characters, so langdetect is never called at all. This is a deliberate
     abstention, not a model failure.
  2. EXCEPTION - langdetect was called and raised LangDetectException.
  3. EMPTY - langdetect was called, returned normally, but produced no
     candidate at all.

The distinction matters for reporting: "we declined to ask the model" and
"we asked and the model failed" are different methodological claims, and
the und rate can only be defended as principled abstention to the extent
that path 1 dominates.

Cheap to run despite the corpus size (~1 minute on 12 cores): separating
path 1 from the rest needs no model call at all, just clean_for_detection
(regex) plus a length check, and the model only has to run on the handful
of rows that pass the gate. The cost is reading `review_text`, which is
~82% of the corpus on disk - so files are processed in parallel, each
worker reading only the two columns it needs.

A fourth counter, `und_inconsistent`, catches rows that pass the gate AND
re-detect cleanly - which should be impossible for a row already labelled
und (DetectorFactory.seed is fixed, so detection is reproducible). It is
reported rather than silently folded into another bucket: a nonzero value
means the labels on disk do not reproduce from the current code, and the
whole breakdown should be distrusted until that is explained.

Usage:
    python run_und_breakdown.py \\
        --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output ../../steam-data/step01-output/und_breakdown_report.json
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from langdetect import LangDetectException, detect_langs

from langdetect_revalidation import MAX_CHARS, MIN_ALPHA_LENGTH, clean_for_detection
from language_revalidation import UNDETERMINED_LABEL
from pipeline_utils import info, list_parquet_files, save_summary


def _scan_file(file: Path) -> Counter:
    """Counts each und path for one parquet file. Returns a Counter so the
    parent can just sum them - keeps only counts in the returned object, no
    review text, so nothing large crosses the process boundary."""
    df = pd.read_parquet(file, columns=["review_text", "review_lang"])
    counts = Counter(rows_scanned=len(df))

    und = df.loc[df["review_lang"] == UNDETERMINED_LABEL, "review_text"]
    counts["und_total"] = len(und)

    for text in und:
        cleaned = clean_for_detection(text[:MAX_CHARS] if isinstance(text, str) else text)
        if len(cleaned) < MIN_ALPHA_LENGTH:
            counts["und_gated"] += 1
            # Reviews that clean away to nothing at all - no letter in any
            # script survives. A strict subset of und_gated, reported
            # separately because it is the unambiguous floor: no threshold
            # choice, however calibrated, could rescue these.
            if not cleaned:
                counts["und_gated_empty_after_cleaning"] += 1
            continue

        try:
            guesses = detect_langs(cleaned)
        except LangDetectException:
            counts["und_langdetect_exception"] += 1
            continue

        if not guesses:
            counts["und_empty_guesses"] += 1
        else:
            counts["und_inconsistent"] += 1

    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decomposes the und label into gated / exception / empty-guess counts."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to run_detect_language.py's reviews_cleaned.parquet directory",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the breakdown report JSON to",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Worker processes (defaults to every core on the machine)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = list_parquet_files(args.input)

    totals = Counter()
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        for i, counts in enumerate(executor.map(_scan_file, files), start=1):
            totals.update(counts)
            if i % 40 == 0 or i == len(files):
                info(f"[{i}/{len(files)}] {totals['und_total']} und row(s) classified so far")

    und_total = totals["und_total"]
    rows_scanned = totals["rows_scanned"]

    def pct_of_und(n):
        return round(100 * n / und_total, 4) if und_total else 0.0

    # Every percentage names its own denominator, so no field's meaning
    # depends on where it sits in the tree - `empty_after_cleaning` is
    # nested under the gated bucket and is reported as a share of it, which
    # a bare "pct" (or a second "pct_of_und") next to the parent's own
    # percentage reads as ambiguous. The two denominators happen to give
    # near-identical values here (the gate is 99.96% of und), so the
    # ambiguity is invisible in the numbers themselves.
    def pct_of_gated(n):
        gated = totals["und_gated"]
        return round(100 * n / gated, 4) if gated else 0.0

    report = {
        "files_scanned": len(files),
        "rows_scanned": rows_scanned,
        "und_total": und_total,
        "und_pct_of_corpus": round(100 * und_total / rows_scanned, 2) if rows_scanned else 0.0,
        "min_alpha_length": MIN_ALPHA_LENGTH,
        "max_chars": MAX_CHARS,
        "paths": {
            "gated_model_never_called": {
                "count": totals["und_gated"],
                "pct_of_und": pct_of_und(totals["und_gated"]),
                "empty_after_cleaning": {
                    "count": totals["und_gated_empty_after_cleaning"],
                    "pct_of_gated": pct_of_gated(totals["und_gated_empty_after_cleaning"]),
                },
            },
            "langdetect_exception": {
                "count": totals["und_langdetect_exception"],
                "pct_of_und": pct_of_und(totals["und_langdetect_exception"]),
            },
            "empty_guesses": {
                "count": totals["und_empty_guesses"],
                "pct_of_und": pct_of_und(totals["und_empty_guesses"]),
            },
        },
        "und_inconsistent": totals["und_inconsistent"],
    }

    info(
        f"und: {und_total} of {rows_scanned} rows ({report['und_pct_of_corpus']:.2f}% of corpus) | "
        f"gated: {totals['und_gated']} ({pct_of_und(totals['und_gated']):.2f}%) | "
        f"exception: {totals['und_langdetect_exception']} | "
        f"empty guesses: {totals['und_empty_guesses']}"
    )
    if totals["und_inconsistent"]:
        info(
            f"WARNING: {totals['und_inconsistent']} und row(s) re-detect cleanly from the "
            f"current code - the labels on disk do not reproduce, distrust this breakdown"
        )

    save_summary(report, args.output)


if __name__ == "__main__":
    main()
