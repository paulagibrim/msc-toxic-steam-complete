"""CLI for show_review_examples.py - samples example reviews matching a
language + toxicity + (optionally) a text term / game tag, joining in
whatever perspective/detoxify/sentiment/topic data is available, and saves
the result as CSV.

Usage:
    python run_show_examples.py \\
        --lang pt --toxic --n 50 \\
        --games ../../steam-data/step01-output/games/games.parquet \\
        --step02-dir ../../steam-data/step02-output \\
        --step04-dir ../../steam-data/step04-output \\
        --step03-results ../../steam-data/step03-output/pt/results/classified_toxic.parquet \\
        --contains "trash" --game-tag "Action" --seed 42 \\
        --output ../../steam-data/examples/toxic_pt_trash.csv

--step04-dir and --step03-results are optional - omit them if that step
hasn't run yet for this language; the corresponding columns come back
empty rather than erroring (see show_review_examples.py's module docstring).
"""
import argparse
from pathlib import Path

import show_review_examples as sre
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Samples example reviews (language + toxicity + optional term/tag filters) for manual inspection."
    )
    parser.add_argument("--lang", required=True, help="Language code, e.g. 'pt' or 'en'")
    toxic_group = parser.add_mutually_exclusive_group(required=True)
    toxic_group.add_argument("--toxic", dest="toxic", action="store_true", help="Sample toxic reviews")
    toxic_group.add_argument("--non-toxic", dest="toxic", action="store_false", help="Sample non-toxic reviews")
    parser.add_argument("--n", required=True, type=int, help="Number of examples to sample")
    parser.add_argument("--games", required=True, type=Path, help="Path to step01's games.parquet")
    parser.add_argument("--step02-dir", required=True, type=Path, help="Path to step02's output directory")
    parser.add_argument(
        "--step04-dir", type=Path, default=None,
        help="Path to step04's output directory (optional - omit if not run yet for this language)",
    )
    parser.add_argument(
        "--step03-results", type=Path, default=None,
        help="Path to step03's classified_toxic.parquet for this language (optional - omit if Stage 7 hasn't run)",
    )
    parser.add_argument("--contains", default=None, help="Substring the review text must contain (case-insensitive)")
    parser.add_argument("--game-tag", default=None, help="Game must have this popular_tag (case-insensitive)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the sampled examples CSV to")
    return parser.parse_args()


def main():
    args = parse_args()

    examples = sre.get_review_examples(
        lang=args.lang,
        toxic=args.toxic,
        n=args.n,
        games_path=args.games,
        step02_dir=args.step02_dir,
        step04_dir=args.step04_dir,
        step03_results_path=args.step03_results,
        contains=args.contains,
        game_tag=args.game_tag,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(args.output, index=False)
    info(f"Saved {len(examples)} example(s) to: {args.output}")


if __name__ == "__main__":
    main()
