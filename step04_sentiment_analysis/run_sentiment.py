"""CLI for sentiment_scoring.py - scores step02's pt/en reviews with
nlptown/bert-base-multilingual-uncased-sentiment, adding a continuous
`sentiment_score` (expected star rating, 1.0-5.0).

Usage:
    python run_sentiment.py \\
        --input ../../steam-data/step02-output \\
        --output-dir ../../steam-data/step04-output

Defaults to scoring both pt and en (pass --lang to override, repeatable).
`--input` is step02's output directory (contains the review_lang=<lang>
subfolders). Output goes to --output-dir/review_lang=<lang>/ (same
filenames as the input files, so each is independently resumable).

Device auto-detects cuda > mps > cpu; pass --device to force one (e.g.
--device cuda:0 to pin a specific GPU on a multi-GPU machine).

--cache-from: optional path to an already-scored step04 output directory
(e.g. the old ../../steam-data/step04-output, before a step01
language-detection fix). Reuses each review's already-computed
sentiment_score (matched by review_url) instead of re-running the model on
it. Only reviews with no cached score go through the model.

--reverse: process files last-to-first. Run a second process with this
flag (pinned to the other GPU via --device) at the same time as a normal
forward run, to work through the same file list from both ends at once -
e.g.:
    python run_sentiment.py --lang en --device cuda:0 ...            # forward
    python run_sentiment.py --lang en --device cuda:1 --reverse ...  # backward
"""
import argparse
from pathlib import Path

import sentiment_scoring as ss
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scores step02's pt/en reviews with a multilingual sentiment model "
        "(continuous sentiment_score, 1.0-5.0)."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to step02's output directory (contains review_lang=<lang> subfolders)",
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
    parser.add_argument(
        "--cache-from", type=Path, default=None,
        help="Path to an already-scored step04 output directory - reuses cached "
        "sentiment_score by review_url instead of re-running the model on reviews "
        "already scored there. Optional.",
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="Process files last-to-first - run alongside a normal forward run "
        "(ideally on a different --device) to work through the file list from both ends.",
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
        ss.run_sentiment_for_language(
            args.input, output_dir, lang, device=device, cache_from=args.cache_from, reverse=args.reverse
        )
        info(f"[{lang}] done -> {output_dir}")


if __name__ == "__main__":
    main()
