"""CLI for detoxify_scoring.py - scores step01's cleaned reviews (pt and en)
with Detoxify, keeping only `toxicity` (renamed `detoxify_score`).

Usage:
    python run_detoxify.py \\
        --input /Users/gibrim/Documents/dev/steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output-dir /Users/gibrim/Documents/dev/steam-data/step02-output

Defaults to scoring both pt and en (pass --lang to override, repeatable).
`--input` is step01's reviews_cleaned.parquet directory (contains the
review_lang=<lang> subfolders). Output goes to
--output-dir/review_lang=<lang>/ (same filenames as the input files, so
each is independently resumable).

Device auto-detects cuda > mps > cpu; pass --device to force one (e.g.
--device cuda:0 to pin a specific GPU on a multi-GPU machine).
"""
import argparse
from pathlib import Path

import detoxify_scoring as ds
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scores step01's cleaned pt/en reviews with Detoxify (toxicity only, as detoxify_score)."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to step01's reviews_cleaned.parquet directory",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write scored output to")
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language to score (repeat for multiple). Defaults to both pt and en.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force a device (e.g. 'cuda', 'cuda:0', 'mps', 'cpu'). Default: auto-detect.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    languages = args.languages or ["pt", "en"]

    device = None
    if args.device is not None:
        import torch
        device = torch.device(args.device)

    for lang in languages:
        output_dir = args.output_dir / f"review_lang={lang}"
        ds.run_detoxify_for_language(args.input, output_dir, lang, device=device)
        info(f"[{lang}] done -> {output_dir}")


if __name__ == "__main__":
    main()
