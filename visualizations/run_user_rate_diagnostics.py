"""CLI for user_toxicity_rate_diagnostics.py - generates the three charts
for deciding step06's T_rate threshold, for the pt, en, and pt+en (union)
populations (see that module's docstring for exactly how each population
and its coverage/rate are defined).

Run LOCALLY (needs the "Heading Now" font, same as run_language_distribution.py).
Copy build_user_rate_table.py's output down from the server first - it's a
small, already-aggregated per-user table, not the raw corpus.

Usage:
    python run_user_rate_diagnostics.py \\
        --table ../../steam-data/step06-output/user_rate_table.parquet \\
        --output-dir ../../steam-data/step06-output/charts
"""
import argparse
from pathlib import Path

import user_toxicity_rate_diagnostics as diag
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates coverage/toxic-rate diagnostic charts for step06's T_rate decision."
    )
    parser.add_argument(
        "--table", required=True, type=Path,
        help="Path to build_user_rate_table.py's output parquet",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write charts to")
    return parser.parse_args()


def main():
    args = parse_args()
    table = diag.load_user_rate_table(args.table)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diag.plot_coverage_cdf(table, args.output_dir / "coverage_cdf")
    diag.plot_rate_distribution(table, args.output_dir / "rate_distribution")
    diag.plot_prevalence_vs_threshold(table, args.output_dir / "prevalence_vs_threshold")

    info(f"Done. Charts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
