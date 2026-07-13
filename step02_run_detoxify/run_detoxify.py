"""CLI for detoxify_scoring.py - scores step01's cleaned reviews (pt and en
by default) with Detoxify, keeping only `toxicity` (renamed
`detoxify_score`).

Usage:
    python run_detoxify.py \\
        --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output-dir ../../steam-data/step02-output

`--input` is step01's reviews_cleaned.parquet directory - review_lang is a
plain column there, not a subfolder (see clean_reviews.py's module
docstring). Every target language (pt+en by default; pass --lang to
override, repeatable) is scored together in ONE pass over each file - this
step's own output is flat too (review_lang stays a column, no
`review_lang=<lang>/` subfolders), for consistency with step01 and every
other step in this project. Downstream steps filter `review_lang == lang`
themselves after reading. Output goes to --output-dir/ directly (same
filenames as the input files, so each is independently resumable).

Device auto-detects cuda > mps > cpu; pass --device to force one (e.g.
--device cuda:0 to pin a specific GPU on a multi-GPU machine).

--cache-from: optional path to an already-scored step02 output directory
(flat or the older per-language-folder layout - see
detoxify_scoring.load_score_cache). Reuses each review's already-computed
detoxify_score (matched by review_url) instead of re-running the model on
it - Detoxify scores a review's text, not its file/partition location, so
a review that only moved between files/languages doesn't need rescoring.
Only reviews with no cached score (genuinely new to this output) go
through the model.

--cache-exclude-pattern: optional regex - cached rows whose review_text
matches it are excluded from the cache (forced to re-score) instead of
reusing a stale value. Use after adding a new boilerplate pattern to
BOILERPLATE_PATTERNS, so only the rows that pattern actually affects get
re-scored, not the whole cache.

--fix-pattern: optional regex - patches already-scored files IN PLACE,
directly against the SAME --output-dir (no --cache-from, no moving
anything aside first). Checks each existing output file for rows whose
review_text matches the pattern; if none match, the file is skipped as
usual; if some do, only those rows are re-scored and the file is
overwritten - everything else in it stays untouched. Simplest way to apply
a newly-added BOILERPLATE_PATTERNS entry to already-scored data.
"""
import argparse
from pathlib import Path

import detoxify_scoring as ds
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scores step01's cleaned reviews with Detoxify (toxicity only, as detoxify_score)."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to step01's reviews_cleaned.parquet directory",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write scored output to")
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language to score (repeat for multiple). Defaults to both pt and en, scored together in one pass.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force a device (e.g. 'cuda', 'cuda:0', 'mps', 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--cache-from", type=Path, default=None,
        help="Path to an already-scored step02 output directory - reuses cached "
        "detoxify_score by review_url instead of re-running the model on reviews "
        "already scored there. Optional.",
    )
    parser.add_argument(
        "--cache-exclude-pattern", default=None,
        help="Regex - cached rows whose review_text matches this are excluded from "
        "the cache and re-scored, instead of reusing a stale value. Optional.",
    )
    parser.add_argument(
        "--fix-pattern", default=None,
        help="Regex - patches already-scored files in place (same --output-dir): "
        "rows whose review_text matches this are re-scored, everything else in the "
        "file is left untouched. Optional.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages or ["pt", "en"]

    device = None
    if args.device is not None:
        import torch
        device = torch.device(args.device)

    ds.run_detoxify(
        args.input, args.output_dir, languages, device=device,
        cache_from=args.cache_from, cache_exclude_pattern=args.cache_exclude_pattern,
        fix_pattern=args.fix_pattern,
    )
    info(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
