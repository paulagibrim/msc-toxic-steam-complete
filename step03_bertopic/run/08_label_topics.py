#!/usr/bin/env python3
"""
Stage 8 — LLM-Assisted Topic Labeling
========================================
Samples representative example documents from each topic (from the Stage 7
classified corpus) and asks Claude to produce a short label, a one-sentence
description, a fixed-taxonomy category, and a copypasta suspicion flag.

Usage:
    python run/08_label_topics.py --lang en --api-key sk-ant-...
    python run/08_label_topics.py --lang pt --n-examples 20 --api-key sk-ant-...
    python run/08_label_topics.py --api-key sk-ant-...              # both en and pt

API key: pass --api-key directly, or set it once per terminal session and
omit the flag on every call:
    Windows (PowerShell): $env:ANTHROPIC_API_KEY = "sk-ant-..."
    Windows (cmd):         set ANTHROPIC_API_KEY=sk-ant-...
    macOS/Linux:            export ANTHROPIC_API_KEY=sk-ant-...

Note: --api-key is convenient but less private than the environment
variable - command-line arguments can be visible to other processes/users
on the same machine (e.g. via Task Manager's command-line column, or `ps`
on macOS/Linux) and get written to your shell's command history file.

Expected inputs:
    results/classified_toxic.parquet       (from Stage 7)
    results/topic_info_real_counts.csv     (from Stage 7)

Output:
    results/topic_labels.csv
        Columns: Topic, Count_Real, Name, Representation, label,
                 description, category, copypasta_suspected,
                 copypasta_evidence
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import Settings
from src.topic_labeling import label_all_topics
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 8: LLM-assisted topic labeling."
    )
    p.add_argument(
        "--lang",
        action="append",
        dest="languages",
        default=None,
        help="Language(s) to run (repeatable: --lang en --lang pt). Defaults to both.",
    )
    p.add_argument(
        "--model",
        default="claude-sonnet-5",
        help="Claude model id to use for labeling (default: claude-sonnet-5).",
    )
    p.add_argument(
        "--n-examples",
        type=int,
        default=15,
        help="Number of example documents sampled per topic (default: 15).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key. If omitted, reads the ANTHROPIC_API_KEY "
             "environment variable instead.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Output token cap per call (default: 4096). Models that always "
             "think (e.g. claude-fable-5) spend part of this on internal "
             "reasoning before the JSON answer - too low a cap can leave no "
             "room for the response. This is a ceiling, not a target: "
             "raising it doesn't increase cost unless the model actually "
             "generates that many tokens.",
    )
    return p.parse_args()


def run_one(lang: str, model: str, n_examples: int, api_key: str | None, max_tokens: int) -> None:
    settings = Settings.from_lang(lang)
    settings.create_directories()

    log_file = settings.results_dir / "logs" / "08_label_topics.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 8 — Topic Labeling ===")
    print(f"Model      : {model}")
    print(f"N examples : {n_examples}")
    print(f"Max tokens : {max_tokens}")
    print(f"Input      : {settings.results_dir}")

    df_labels = label_all_topics(
        settings, model=model, n_examples=n_examples, api_key=api_key, max_tokens=max_tokens
    )

    print(f"\n[{lang}] Labeled {len(df_labels)} topics.")
    print(df_labels[["Topic", "Count_Real", "label", "category"]].to_string(index=False))
    print(f"[{lang}] Saved to: {settings.results_dir / 'topic_labels.csv'}")


def main() -> None:
    args = parse_args()
    languages = args.languages or ["en", "pt"]
    for lang in languages:
        run_one(lang, args.model, args.n_examples, args.api_key, args.max_tokens)


if __name__ == "__main__":
    main()
