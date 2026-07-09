"""CLI for langdetect_revalidation.py - checks steam-data's raw review
files against langdetect's own language guess, in parallel across CPU
cores, and saves a slim validation table + mismatch breakdown.

Usage:
    python run_langdetect_revalidation.py \\
        --input ../../steam-data/raw/reviews \\
        --lang pt --output-dir ./output --n-jobs 48

Run once per language (pt, then en) - each is a separate process, so there's
no reason to couple them into one invocation. `--input` is a directory (the
raw files aren't split by language - see langdetect_revalidation.py).
"""
import argparse
from pathlib import Path

import language_revalidation as lr
from langdetect_revalidation import validate_raw_reviews_langdetect
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-checks steam-data's raw review files' declared language "
        "against langdetect's own guess, in parallel across CPU cores."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Directory containing the raw *.parquet review files (e.g. steam-data/raw/reviews)",
    )
    parser.add_argument("--lang", required=True, help="Declared language to check, e.g. 'pt' or 'en'")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write results to")
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Worker processes (default: os.cpu_count() - every core on this machine)",
    )
    parser.add_argument("--batch-size", type=int, default=500_000, help="Rows read per batch (bounds memory)")
    parser.add_argument(
        "--min-alpha-length", type=int, default=20,
        help="Minimum real alphabetic characters (after stripping emoji/ASCII-art/digits/punctuation) "
        "before trusting a language guess at all - shorter reviews are marked undetermined",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    validation = validate_raw_reviews_langdetect(
        args.input,
        args.lang,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        min_alpha_length=args.min_alpha_length,
    )

    breakdown = lr.summarize_mismatches(validation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lr.save_language_validation(validation, args.output_dir / f"language_validation_{args.lang}.parquet")
    lr.save_mismatch_breakdown(breakdown, args.output_dir / f"language_mismatch_breakdown_{args.lang}.csv")

    mask = lr.apply_language_mask(validation)
    info(
        f"[{args.lang}] DONE - {int(mask.sum())} of {len(validation)} pass the language mask "
        f"({100*mask.sum()/len(validation):.2f}%)"
    )


if __name__ == "__main__":
    main()
