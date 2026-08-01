#!/usr/bin/env python3
"""
Stage 8b — Export Topic Prompts for Manual/Fresh-Session Labeling
=====================================================================
Writes one ready-to-use prompt file per topic (same example-selection logic
as 08_label_topics.py: random sample with a fixed seed, fixed count per
topic - default 20) WITHOUT calling the Anthropic API. No cost, no API key
needed.

Use this when you want a genuinely fresh, context-free LLM session (or a
human) to do the labeling - e.g. to avoid carrying over judgments/bias from
whatever session ran the rest of this pipeline.

Usage:
    python run/08b_export_prompts_for_manual_labeling.py --lang pt
    python run/08b_export_prompts_for_manual_labeling.py            # both en and pt

Output (per language):
    results/llm_prompts/
        _system_prompt.txt        - read this first
        _instructions.txt         - what to do with the rest of the folder
        _manifest.csv             - one row per topic (Topic, Count_Real,
                                     Name, Representation, n_examples,
                                     prompt_file)
        topic_XXX_prompt.txt      - one file per non-outlier topic

Then, in a fresh chat/session, point it at this folder and follow
_instructions.txt.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import Settings
from src.topic_labeling import export_prompts_for_manual_labeling
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 8b: export per-topic prompts for manual/fresh-session labeling."
    )
    p.add_argument(
        "--lang",
        action="append",
        dest="languages",
        default=None,
        help="Language(s) to run (repeatable: --lang en --lang pt). Defaults to both.",
    )
    p.add_argument(
        "--n-examples",
        type=int,
        default=20,
        help="Fixed number of example documents per topic (default: 20, "
             "same count for every topic regardless of size).",
    )
    return p.parse_args()


def run_one(lang: str, n_examples: int) -> None:
    settings = Settings.from_lang(lang)
    settings.create_directories()

    log_file = settings.results_dir / "logs" / "08b_export_prompts.log"
    setup_logging(log_file)

    print(f"\n=== [{lang}] Stage 8b — Export Prompts for Manual Labeling ===")
    print(f"N examples : {n_examples} (fixed)")

    out_dir = export_prompts_for_manual_labeling(settings, n_examples=n_examples)

    print(f"[{lang}] Exported to: {out_dir}")
    print(f"[{lang}] Start a fresh chat, point it at that folder, and follow _instructions.txt.")


def main() -> None:
    args = parse_args()
    languages = args.languages or ["en", "pt"]
    for lang in languages:
        run_one(lang, args.n_examples)


if __name__ == "__main__":
    main()
