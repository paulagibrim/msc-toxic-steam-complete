"""CLI for venn_region_sentiment_diagnostics.py - generates boxplot and
violin charts of sentiment_score per toxicity Venn region (perspective_only/
detoxify_only/both/neither), side by side per language.

Run LOCALLY (needs the "Heading Now" font, same as the other charts here).
Reads step04-output directly (raw per-review scores, not the summary-stats
JSON from 05_sentiment_facets_report.py) - can take a few minutes to sweep
the full corpus.

Usage:
    python run_venn_region_sentiment_diagnostics.py \\
        --step04-dir ../../steam-data/step04-output \\
        --output-dir ../../steam-data/step06-output/phase1_feature_engineering/diagnostics/charts
"""
import argparse
from pathlib import Path

import venn_region_sentiment_diagnostics as diag
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates boxplot/violin charts of sentiment by toxicity Venn region, per language."
    )
    parser.add_argument("--step04-dir", required=True, type=Path, help="Path to step04's output directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write charts to")
    return parser.parse_args()


def main():
    args = parse_args()
    data = diag.load_sampled_sentiment_by_region(args.step04_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diag.plot_venn_region_boxplot(data, args.output_dir / "venn_region_sentiment_boxplot")
    diag.plot_venn_region_violin(data, args.output_dir / "venn_region_sentiment_violin")

    info(f"Done. Charts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
