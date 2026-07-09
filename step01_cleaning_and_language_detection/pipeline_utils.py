"""Logging/IO helpers, ported from dissertacao-steam's
data_refactor/shared/pipeline_utils.py - trimmed of the mlflow-based step()
context manager (this project doesn't use mlflow), but keeping
replace_sentinels/safe_astype/read_parquet_dir since clean_games.py,
clean_users.py, and clean_reviews.py need them."""
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"[INFO {_now()}] {msg}")


def error(msg: str) -> None:
    print(f"[ERR {_now()}] {msg}")


def warn_if_not_materialized(path: Path) -> None:
    """Warns if `path` looks like an un-downloaded cloud-storage placeholder
    (reports a real size but 0 disk blocks allocated) - reading it would
    block until the content is fetched. Only prints a heads-up."""
    try:
        st = os.stat(path)
        if st.st_size > 0 and st.st_blocks == 0:
            info(
                f"'{path.name}' parece nao estar baixado localmente (iCloud?) "
                f"— a leitura pode demorar bastante enquanto o conteudo e baixado."
            )
    except (OSError, AttributeError):
        pass  # st_blocks isn't available on every platform; skip silently.


def list_parquet_files(directory: Path) -> list:
    """Lists every *.parquet file in `directory`, warning about any other
    file types found (ignored) and any file that looks like an un-downloaded
    cloud-storage placeholder."""
    directory = Path(directory)
    parquet_files = sorted(directory.glob("*.parquet"))
    other_files = [f for f in directory.iterdir() if f.is_file() and f.suffix != ".parquet"]

    info(f"Found {len(parquet_files)} parquet file(s) in {directory}")
    if other_files:
        info(f"Ignoring {len(other_files)} non-parquet file(s), suffixes: "
             f"{sorted({f.suffix for f in other_files})}")
    for f in parquet_files:
        warn_if_not_materialized(f)

    return parquet_files


def read_parquet_dir(directory: Path, engine: str = "pandas"):
    """Reads every *.parquet file in `directory` into a single dataframe.

    engine="pandas": loads eagerly (used for users, which fits in memory).
    engine="dask": builds a lazy Dask dataframe (used for reviews, too large
    to load eagerly).
    """
    parquet_files = list_parquet_files(directory)

    if engine == "dask":
        import dask.dataframe as dd
        return dd.read_parquet([str(f) for f in parquet_files])

    import pandas as pd
    dfs = []
    for i, f in enumerate(parquet_files, start=1):
        file_start = time.perf_counter()
        dfs.append(pd.read_parquet(f))
        info(f"[{i}/{len(parquet_files)}] read {f.name} in {time.perf_counter() - file_start:.1f}s")
    return pd.concat(dfs, ignore_index=True)


def replace_sentinels(df):
    """Replaces known 'missing value' sentinels (empty string, '-1', -1) with NaN.

    Works on both pandas and Dask dataframes (no `inplace=True`, since Dask
    doesn't support it — callers must reassign the result).
    """
    return df.replace(["", "-1", -1], np.nan)


def safe_astype(df, type_mapping: dict):
    """Casts only the columns in `type_mapping` that actually exist in `df`."""
    existing = {col: dtype for col, dtype in type_mapping.items() if col in df.columns}
    missing = set(type_mapping) - set(existing)
    if missing:
        info(f"Skipping type cast for missing columns: {sorted(missing)}")
    return df.astype(existing)


def compute_null_summary(df):
    """Null count per column, descending - the same
    `df.isna().sum().sort_values(ascending=False)` every cleaning notebook
    printed/exported. Works on pandas or (already-computed) Dask results;
    for a Dask dataframe, pass `df.compute()` or call `.compute()` on the
    result of `.isna().sum()` yourself if you want to avoid materializing
    the whole frame first."""
    return df.isna().sum().sort_values(ascending=False)


def export_null_summary(null_summary, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    null_summary.to_csv(output_path, header=["null_count"])
    info(f"Saved null summary to: {output_path}")
    return output_path


def export_sample(df, output_path: Path, n: int = 5) -> Path:
    """Saves the first `n` rows as CSV - same as the cleaning notebooks'
    `sample_<name>.csv` artifacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(n).to_csv(output_path, index=False)
    info(f"Saved {n}-row sample to: {output_path}")
    return output_path


def save_summary(summary: dict, output_path: Path) -> Path:
    """Saves a run's step-by-step summary (row counts, dtypes, describe()
    stats, etc.) as JSON - replaces what used to only be visible as
    printed notebook cell output / MLflow params+metrics, now that these
    scripts run from a terminal instead."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    info(f"Saved run summary to: {output_path}")
    return output_path
