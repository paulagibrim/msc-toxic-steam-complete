#!/usr/bin/env python3
"""
Stage 1 — Text Cleaning
=======================
Reads step02_run_detoxify's output parquet files for one or more languages,
applies toxicity labels, cleans review text, and writes one output parquet
per input file to cleaned_data_dir.

Usage:
    python run/01_clean.py                 # both en and pt
    python run/01_clean.py --lang en       # just en
    python run/01_clean.py --lang en --resume

Expected inputs (config_<lang>.yaml → paths.detoxify_data_dir, e.g.
../../steam-data/step02-output/review_lang=en):
    *.parquet files with columns:
        review_text                    (str)
        perspective_score              (float)
        detoxify_score                 (float)
        perspective_declared_language  (str)
        review_url                     (str)
        game_id                        (str)

Outputs (config_<lang>.yaml → paths.cleaned_data_dir):
    *.parquet files with columns:
        review_text_clean  (str)
        is_toxic           (bool)
        review_url         (str)
        game_id            (str)
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable from the run/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import Settings
from src.text_cleaning import run_cleaning
from src.utils import setup_logging, timer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: batch text cleaning of step02's detoxify-scored reviews."
    )
    p.add_argument(
        "--lang",
        action="append",
        dest="languages",
        default=None,
        help="Language(s) to run (repeatable: --lang en --lang pt). Defaults to both.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip input files that already have a matching cleaned output "
             "(default: True; use --no-resume to reprocess everything).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Reprocess all files even if cleaned output already exists.",
    )
    return p.parse_args()


def run_one(lang: str, resume: bool) -> None:
    settings = Settings.from_lang(lang)
    settings.create_directories()

    log_file = settings.results_dir / "logs" / "01_clean.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 1 — Text Cleaning ===")
    print(f"Input  : {settings.detoxify_data_dir}")
    print(f"Output : {settings.cleaned_data_dir}")
    print(f"Resume : {resume}")
    print()

    with timer(f"[{lang}] Stage 1 — full run"):
        stats = run_cleaning(settings, resume=resume)

    ok   = sum(1 for s in stats if s["status"] == "ok")
    skip = sum(1 for s in stats if s["status"] == "skipped")
    err  = sum(1 for s in stats if s["status"] == "error")
    print(f"\n[{lang}] Done — processed: {ok} | skipped: {skip} | errors: {err}")


def main() -> None:
    args = parse_args()
    for lang in args.languages or ["en", "pt"]:
        run_one(lang, args.resume)


if __name__ == "__main__":
    main()
