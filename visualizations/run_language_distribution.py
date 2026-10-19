"""CLI for language_distribution.py - counts reviews per review_lang
(langdetect's own classification) across step01's Stage 2 output, and
plots the top N by volume as a bar chart (PNG + PDF).

Unlike the other steps, this script is meant to run LOCALLY, not on the
remote GPU machine - it needs the "Heading Now" font, which is only
installed on this machine (falls back to DejaVu Sans elsewhere, but the
whole point of running it here is to get the real font). Copy just the
small JSON/CSV files this needs down into visualizations/data/ (gitignored -
see .gitignore - never commit real data there, it's a personal dropbox for
whatever this machine needs that the remote one produced). That folder is
for *input* only; see below for where the chart is written.

Output goes under steam-data/, next to every other artefact this project
produces and alongside run_toxicity_venn.py's diagrams - not into
visualizations/data/, which is only an inbox for files copied down from the
remote machine.

Usage (fast path - reuses Stage 2's own counts, no re-scan):
    python run_language_distribution.py \\
        --report data/reviews_report.json \\
        --output-dir ../../steam-data/visualizations/languages

Usage (slow path - scans the corpus fresh, e.g. reviews_report.json isn't
available or you want counts for a different/filtered slice - only
practical if steam-data/ is also reachable from this machine):
    python run_language_distribution.py \\
        --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output-dir ../../steam-data/visualizations/languages

Exactly one of --report / --input is required.

Writes:
    <output-dir>/language_counts.csv         - review_lang, review_count
    <output-dir>/top_languages.png           - chart (raster)
    <output-dir>/top_languages.pdf           - chart (vector, for print)

--top-n defaults to 15 (per explicit request); pass a different number, or
0 for every language found.
"""
import argparse
from pathlib import Path

import language_distribution as ld
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plots the top N languages found by langdetect, by review volume."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--report", type=Path,
        help="Path to Stage 2's reviews_report.json - reuses its language_counts directly, no re-scan.",
    )
    source.add_argument(
        "--input", type=Path,
        help="Path to step01's reviews_cleaned.parquet directory - scans it fresh (slower).",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write language_counts.csv and the chart files to",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Number of top languages to plot (default: 15). Pass 0 to plot every language found.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n = None if args.top_n == 0 else args.top_n

    if args.report:
        counts = ld.load_counts_from_report(args.report)
    else:
        counts = ld.count_reviews_per_language(args.input)
        info(f"Found {len(counts)} distinct review_lang value(s), {int(counts.sum())} review(s) total")

    ld.save_language_counts(counts, args.output_dir / "language_counts.csv")
    ld.plot_top_languages(counts, args.output_dir / "top_languages.png", n=n)


if __name__ == "__main__":
    main()
