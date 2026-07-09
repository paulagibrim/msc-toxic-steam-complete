"""CLI for clean_games.py - cleans steam-data's raw games file.

Usage:
    python run_clean_games.py \\
        --input ../../steam-data/raw/games/todos_jogos.json \\
        --output ../../steam-data/processed/games/games.parquet

Step order matches dissertacao-steam/data_refactor/0-cleaning/01_clean_games.ipynb.
Besides the cleaned games.parquet, writes (next to --output, i.e. in its
parent directory) everything that notebook printed/exported as cell output
or MLflow artifacts, minus MLflow itself:
  - null_summary_games.csv
  - sample_games.csv (first 5 rows)
  - games_report.json (shapes/dtypes/describe() stats/drop counts at each step)
"""
import argparse
from pathlib import Path

import clean_games as cg
from pipeline_utils import compute_null_summary, export_null_summary, export_sample, info, save_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Cleans steam-data's raw games file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to todos_jogos.json")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the cleaned games.parquet")
    return parser.parse_args()


def main():
    args = parse_args()
    report_dir = args.output.parent

    df = cg.load_raw_games(args.input)
    rows_loaded, columns_loaded = df.shape
    info(f"Loaded shape: {df.shape}")

    df = cg.rename_columns(df)
    df = cg.drop_low_value_columns(df)
    df = cg.replace_missing_sentinels(df)
    df = cg.fix_dates(df)
    df = cg.fix_price(df)
    df = cg.fix_categorical_columns(df)

    numeric_describe = df.describe(include=["int64", "float64"]).to_dict()

    rows_before = len(df)
    df = cg.drop_missing_titles(df)
    rows_dropped_missing_title = rows_before - len(df)
    info(f"Dropped {rows_dropped_missing_title} row(s) with missing title/game_id")

    rows_before = len(df)
    df = cg.drop_duplicate_games(df)
    rows_dropped_duplicates = rows_before - len(df)
    info(f"Dropped {rows_dropped_duplicates} duplicate row(s) by game_id")

    info(f"Final: {len(df)} rows, {len(df.columns)} columns")

    null_summary = compute_null_summary(df)
    export_null_summary(null_summary, report_dir / "null_summary_games.csv")
    export_sample(df, report_dir / "sample_games.csv", n=5)
    save_summary(
        {
            "rows_loaded": rows_loaded,
            "columns_loaded": columns_loaded,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "describe_numeric": numeric_describe,
            "rows_dropped_missing_title": rows_dropped_missing_title,
            "rows_dropped_duplicates": rows_dropped_duplicates,
            "rows_final": len(df),
            "columns_final": len(df.columns),
        },
        report_dir / "games_report.json",
    )

    cg.export_games(df, args.output)


if __name__ == "__main__":
    main()
