"""CLI for show_review_examples.py - samples example reviews matching a
language + toxicity + (optionally) a text term / game tag, joining in
whatever perspective/detoxify/sentiment/topic data is available, and saves
the result as CSV.

Usage (bare minimum - every path defaults to this project's established
--lang <lang> conventions, see _default_paths below):
    python run_show_examples.py --lang pt --toxic --n 50

Usage (overriding a default, e.g. a different step02 run):
    python run_show_examples.py \\
        --lang pt --toxic --n 50 \\
        --step02-dir ../../steam-data/step02-output-v3/review_lang=pt \\
        --contains "trash" --game-tag "Action" --seed 42 \\
        --output ../../steam-data/examples/toxic_pt_trash.csv

--step04-dir and --step03-results still resolve to a default path even
though that step is optional - if the path doesn't exist yet (step hasn't
run for this language), the corresponding column just comes back empty
rather than erroring (see show_review_examples.py's module docstring).
"""
import argparse
from pathlib import Path

import show_review_examples as sre
from pipeline_utils import info

# This project's established data-layout conventions (see step02_run_detoxify/
# detoxify_scoring.py, step03_bertopic/config_<lang>.yaml, and this session's
# step02-output/step04-output-v2/review_lang=<lang> reorganisation) - every
# path below is derived from --lang alone, so a normal run needs no path
# flags at all. Override any individual one with its own flag when pointing
# at a non-standard location (e.g. a different step02 run for comparison).
GAMES_PATH = Path("../../steam-data/step01-output/games/games.parquet")


def _default_paths(lang: str) -> dict:
    return {
        "games": GAMES_PATH,
        "step02_dir": Path(f"../../steam-data/step02-output/review_lang={lang}"),
        "step04_dir": Path(f"../../steam-data/step04-output-v2/review_lang={lang}"),
        "step03_results": Path(f"../../steam-data/step03-output/{lang}/results/classified_toxic.parquet"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Samples example reviews (language + toxicity + optional term/tag filters) for manual inspection."
    )
    parser.add_argument("--lang", required=True, help="Language code, e.g. 'pt' or 'en'")
    toxic_group = parser.add_mutually_exclusive_group(required=True)
    toxic_group.add_argument("--toxic", dest="toxic", action="store_true", default=None, help="Sample toxic reviews")
    toxic_group.add_argument("--non-toxic", dest="toxic", action="store_false", help="Sample non-toxic reviews")
    toxic_group.add_argument(
        "--venn-set", dest="venn_set", choices=["perspective_only", "detoxify_only", "both"],
        help="Sample from one of the three mutually-exclusive Venn regions instead of toxic/non-toxic "
        "(same regions as run_toxicity_venn.py / tfidf_venn_set_analysis.py)",
    )
    parser.add_argument("--n", required=True, type=int, help="Number of examples to sample")
    parser.add_argument(
        "--games", type=Path, default=None,
        help=f"Path to step01's games.parquet (default: {GAMES_PATH})",
    )
    parser.add_argument(
        "--step02-dir", type=Path, default=None,
        help="Path to step02's output directory (default: step02-output/review_lang=<lang>)",
    )
    parser.add_argument(
        "--step04-dir", type=Path, default=None,
        help="Path to step04's output directory (default: step04-output-v2/review_lang=<lang>; "
        "empty sentiment_score column if that path doesn't exist)",
    )
    parser.add_argument(
        "--step03-results", type=Path, default=None,
        help="Path to step03's classified_toxic.parquet for this language "
        "(default: step03-output/<lang>/results/classified_toxic.parquet; "
        "empty topic column if that path doesn't exist)",
    )
    parser.add_argument("--contains", default=None, help="Substring the review text must contain (case-insensitive)")
    parser.add_argument("--game-tag", default=None, help="Game must have this popular_tag (case-insensitive)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--light-mode", action="store_true", help="Filter data while reading to drastically reduce RAM usage")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write the sampled examples CSV to "
        "(default: ../../steam-data/examples/<toxic|non_toxic>_<lang>.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = _default_paths(args.lang)

    games_path = args.games or defaults["games"]
    step02_dir = args.step02_dir or defaults["step02_dir"]
    step04_dir = args.step04_dir or defaults["step04_dir"]
    step03_results = args.step03_results or defaults["step03_results"]
    selector_label = args.venn_set or ("toxic" if args.toxic else "non_toxic")
    output = args.output or Path(f"../../steam-data/examples/{selector_label}_{args.lang}.csv")

    if not step04_dir.exists():
        info(f"No step04 output at {step04_dir} - sentiment_score will be empty (pass --step04-dir to override).")
        step04_dir = None
    if not step03_results.exists():
        info(f"No step03 results at {step03_results} - topic will be empty (pass --step03-results to override).")
        step03_results = None

    examples = sre.get_review_examples(
        lang=args.lang,
        toxic=args.toxic if args.venn_set is None else None,
        venn_set=args.venn_set,
        n=args.n,
        games_path=games_path,
        step02_dir=step02_dir,
        step04_dir=step04_dir,
        step03_results_path=step03_results,
        contains=args.contains,
        game_tag=args.game_tag,
        seed=args.seed,
        light_mode=args.light_mode,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output, index=False)
    info(f"Saved {len(examples)} example(s) to: {output}")


if __name__ == "__main__":
    main()
