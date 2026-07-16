"""Finds reviews made up of ASCII/Unicode art and writes their links to CSV.

These are the reviews that motivate this step's MIN_ALPHA_LENGTH gate (see
langdetect_revalidation.py's module docstring): text that is visually
elaborate but linguistically empty, which any language detector will
happily assign a confident, meaningless language to. This script locates
them so they can be inspected by hand.

Reads step01's reviews_cleaned.parquet only - no model, no other step's
output needed.

WHY A CHARACTER *FRACTION*, NOT A COUNT:
An absolute count of art characters does not separate art from prose. Two
corrections were needed to make this work at all, both found by inspecting
what a naive version actually matched:

  1. The Halfwidth/Fullwidth Forms block (U+FF00-FFEF) is NOT art - it is
     ordinary CJK punctuation (U+FF0C FULLWIDTH COMMA, etc.). Including it
     made the top of the results a list of perfectly normal Chinese
     reviews. It is deliberately absent from ART_PATTERN below.
  2. Even with the right blocks, a long Chinese review picks up dozens of
     art-block characters incidentally, out-scoring a short pure-art
     review on absolute count. The discriminator is the *share* of the
     text that is art: real art measures 0.58-0.97, ordinary CJK prose
     0.04-0.07 - two well-separated populations, which is why the total is
     insensitive to where in between the threshold is placed.

Emoji-mosaic art (rows of coloured squares) is a distinct style from
Braille/box-drawing art and lives in different Unicode blocks; both are
included, since both are linguistically empty in the same way.

Usage:
    python run_find_ascii_art.py \\
        --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \\
        --output ../../steam-data/step01-output/ascii_art_reviews.csv

    # loosen to catch reviews that merely *contain* art, and cap the output
    python run_find_ascii_art.py --min-frac 0.1 --limit 500 ...
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from pipeline_utils import info, list_parquet_files

# Unicode blocks used to draw pictures, not to write words.
#   U+2500-257F  Box Drawing        U+2580-259F  Block Elements
#   U+25A0-25FF  Geometric Shapes   U+2800-28FF  Braille Patterns
#   U+2B00-2BFF  Misc Symbols/Arrows (incl. U+2B1B/2B1C large squares)
#   U+1F780-1F7FF Geometric Shapes Extended (incl. U+1F7E8 coloured squares)
# Deliberately NOT U+FF00-FFEF - that is CJK punctuation, see docstring.
ART_PATTERN = re.compile("[─-╿▀-▟■-◿⠀-⣿⬀-⯿\U0001f780-\U0001f7ff]")

PREVIEW_CHARS = 80


def _scan_file(file: Path, min_frac: float, min_art_chars: int) -> pd.DataFrame:
    df = pd.read_parquet(file, columns=["review_url", "review_text", "review_lang", "game_id"])
    text = df["review_text"].fillna("")

    art_chars = text.map(lambda t: len(ART_PATTERN.findall(t)))
    # clip(lower=1) so an empty review can't divide by zero - it scores 0
    # art characters anyway, so it fails min_art_chars regardless.
    art_frac = art_chars / text.str.len().clip(lower=1)

    hit = (art_frac >= min_frac) & (art_chars >= min_art_chars)
    out = df.loc[hit, ["review_url", "review_lang", "game_id"]].copy()
    out["art_chars"] = art_chars[hit]
    out["art_frac"] = art_frac[hit].round(4)
    out["text_length"] = text.str.len()[hit]
    # Newlines flattened so one review stays on one CSV row - art is drawn
    # line by line, so an unescaped preview would break the file.
    out["preview"] = text[hit].str[:PREVIEW_CHARS].str.replace(r"\s+", " ", regex=True)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Finds ASCII/Unicode art reviews and writes their links to CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Path to reviews_cleaned.parquet")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the CSV to")
    parser.add_argument(
        "--min-frac", type=float, default=0.5,
        help="Minimum share of the review's characters that must be art (default 0.5 - art-dominated)",
    )
    parser.add_argument(
        "--min-art-chars", type=int, default=20,
        help="Minimum absolute art characters, to exclude incidental decoration (default 20)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Keep only the top N by art_frac (default: keep all)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Worker processes (default: every core)")
    return parser.parse_args()


def main():
    args = parse_args()
    files = list_parquet_files(args.input)

    scan = partial(_scan_file, min_frac=args.min_frac, min_art_chars=args.min_art_chars)
    frames = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        for i, frame in enumerate(executor.map(scan, files), start=1):
            frames.append(frame)
            if i % 40 == 0 or i == len(files):
                info(f"[{i}/{len(files)}] {sum(len(f) for f in frames)} art review(s) found so far")

    result = pd.concat(frames, ignore_index=True).sort_values("art_frac", ascending=False)
    total_found = len(result)
    if args.limit:
        result = result.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    info(
        f"{total_found} art review(s) matched (min_frac={args.min_frac}, "
        f"min_art_chars={args.min_art_chars}); wrote {len(result)} row(s) to {args.output}"
    )
    if total_found:
        info(f"review_lang breakdown of matches:\n{result['review_lang'].value_counts().head(8).to_string()}")


if __name__ == "__main__":
    main()
