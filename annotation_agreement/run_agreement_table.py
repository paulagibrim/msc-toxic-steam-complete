"""CLI for agreement_table.py - rebuilds the inter-annotator agreement
table for every language found and saves it as JSON.

Reads the raw annotation spreadsheets only (steam-data/raw/annotations/
<lang>/*.xlsx) - no pipeline step's output is needed, and nothing here
depends on a step folder being present.

Each language contributes two spreadsheets: one sampled evenly across
Perspective's bins, one across Detoxify's. Which is which is detected from
the data, not from the filename - see agreement_table.detect_stratification
for why the filenames cannot be trusted.

Usage:
    python run_agreement_table.py \\
        --input ../../steam-data/raw/annotations \\
        --output ../../steam-data/annotation-output/agreement_table.json

    # reproduce the previously published table exactly (see --help)
    python run_agreement_table.py --fill-missing-with-majority ...
"""
import argparse
from pathlib import Path

import agreement_table as at
from pipeline_utils import info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuilds the per-bin inter-annotator agreement table from the annotation spreadsheets."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the annotations directory (contains one subfolder per language)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the JSON report to")
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language subfolder to process (repeat for multiple). Defaults to every one found.",
    )
    parser.add_argument(
        "--fill-missing-with-majority", action="store_true",
        help=(
            "Fill a blank annotation with the majority vote of the annotators who did label that "
            "review, instead of dropping the review. Required to reproduce the published table (four "
            "of P's en labels are missing from the spreadsheet), but it is reconstruction, not data, "
            "and it can only ever raise agreement - leave it off for a defensible figure."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages or sorted(p.name for p in args.input.iterdir() if p.is_dir())
    if not languages:
        raise SystemExit(f"No language subfolders under {args.input}")
    info(f"Found {len(languages)} language(s): {', '.join(languages)}")

    if args.fill_missing_with_majority:
        info(
            "--fill-missing-with-majority is ON: blank annotations are filled from the other "
            "annotators' majority, so agreement is biased upward"
        )

    results = []
    for lang in languages:
        sheets = sorted((args.input / lang).glob("*.xlsx"))
        if not sheets:
            info(f"[{lang}] no .xlsx files - skipping")
            continue

        halves = [at.agreement_table(sheet, args.fill_missing_with_majority) for sheet in sheets]
        entry = {"language": lang}
        for half in halves:
            # Keyed by the model whose bins the sheet was sampled across, so
            # a consumer never has to know which filename held which model.
            entry[half["model"]] = half
        results.append(entry)

        missing = sum(b["n_rows_missing_an_annotation"] for h in halves for b in h["bins"])
        if missing:
            info(f"[{lang}] {missing} review(s) have at least one blank annotation")

    save_summary(
        {
            "kappa_statistic": "Randolph's free-marginal kappa (statsmodels fleiss_kappa method='rand')",
            "majority_rule": "per-review 2-of-3 vote, then majority of those per-review verdicts",
            "missing_annotations": (
                "filled with the majority vote of the annotators who did label the review"
                if args.fill_missing_with_majority
                else "reviews with a blank annotation are dropped from their bin"
            ),
            "note": (
                "Annotators are three people labelling reviews toxic/non-toxic; the model only "
                "determines which bin a review falls in. Kappa here measures agreement among the "
                "humans within a bin, not between the two models."
            ),
            "languages": results,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
