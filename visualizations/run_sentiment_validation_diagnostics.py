"""CLI for sentiment_validation_diagnostics.py - generates the boxplot,
violin, and CDF charts comparing toxic vs non-toxic users' average
sentiment (see that module's docstring for why this all-reviews comparison
is the primary, valid validation of toxic_user_labels.parquet).

Run LOCALLY (needs the "Heading Now" font, same as the other charts here).

Usage:
    python run_sentiment_validation_diagnostics.py \\
        --labels ../step06_user_toxicity/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --sentiment ../step06_user_toxicity/phase1_feature_engineering/pipeline/user_sentiment_table.parquet \\
        --output-dir ../../steam-data/step06-output/phase1_feature_engineering/diagnostics/charts
"""
import argparse
from pathlib import Path

import sentiment_validation_diagnostics as diag
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates boxplot/violin/CDF charts comparing toxic vs non-toxic users' sentiment."
    )
    parser.add_argument("--labels", required=True, type=Path, help="Path to toxic_user_labels.parquet")
    parser.add_argument("--sentiment", required=True, type=Path, help="Path to 06_build_user_sentiment.py's output")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write charts to")
    return parser.parse_args()


def main():
    args = parse_args()
    table = diag.load_data(args.labels, args.sentiment)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diag.plot_sentiment_boxplot(table, args.output_dir / "sentiment_boxplot")
    diag.plot_sentiment_violin(table, args.output_dir / "sentiment_violin")
    diag.plot_sentiment_cdf(table, args.output_dir / "sentiment_cdf")

    info(f"Done. Charts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
