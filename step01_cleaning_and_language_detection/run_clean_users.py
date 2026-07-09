"""CLI for clean_users.py - cleans steam-data's raw users files.

Usage:
    python run_clean_users.py \\
        --input /Users/gibrim/Documents/dev/steam-data/raw/users \\
        --output /Users/gibrim/Documents/dev/steam-data/processed/users/all_users.parquet

Step order matches dissertacao-steam/data_refactor/0-cleaning/02_clean_users.ipynb -
note replace_missing_sentinels is defined in clean_users.py but was never
actually called there, so it's skipped here too (preserving that behavior,
not an oversight).

Besides the cleaned all_users.parquet, writes (next to --output) everything
that notebook printed/exported as cell output or MLflow artifacts, minus
MLflow itself:
  - null_summary_users.csv (after cleaning - matches the notebook's export)
  - sample_users.csv (first 5 rows)
  - users_report.json (shapes/dtypes/null counts before+after/drop counts)
"""
import argparse
from pathlib import Path

import clean_users as cu
from pipeline_utils import compute_null_summary, export_null_summary, export_sample, info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Cleans steam-data's raw users files.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of raw users *.parquet files")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the cleaned all_users.parquet")
    return parser.parse_args()


def main():
    args = parse_args()
    report_dir = args.output.parent

    df = cu.load_raw_users(args.input)
    rows_loaded, columns_loaded = df.shape
    info(f"Loaded shape: {df.shape}")

    null_counts_before = compute_null_summary(df)

    df = cu.rename_columns(df)
    df = cu.drop_untrusted_columns(df)
    df = cu.fix_types(df)

    rows_before = len(df)
    df = cu.drop_duplicate_users(df)
    rows_dropped_duplicates = rows_before - len(df)
    info(f"Dropped {rows_dropped_duplicates} duplicate row(s) by steam_id")

    info(f"Final: {len(df)} rows, {len(df.columns)} columns")

    null_summary_after = compute_null_summary(df)
    export_null_summary(null_summary_after, report_dir / "null_summary_users.csv")
    export_sample(df, report_dir / "sample_users.csv", n=5)
    save_summary(
        {
            "rows_loaded": rows_loaded,
            "columns_loaded": columns_loaded,
            "null_counts_before_cleaning": null_counts_before.to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "rows_dropped_duplicates": rows_dropped_duplicates,
            "rows_final": len(df),
            "columns_final": len(df.columns),
        },
        report_dir / "users_report.json",
    )

    cu.export_users(df, args.output)


if __name__ == "__main__":
    main()
