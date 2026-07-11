"""Minimal logging/IO helpers - same as the other steps', copied here so
this folder stays independently copyable to whatever machine runs this
analysis (CPU-only is fine - no GPU/model inference in this step)."""
import json
import os
from datetime import datetime
from pathlib import Path


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
    """Lists every *.parquet file in `directory`, warning about any file
    that looks like an un-downloaded cloud-storage placeholder."""
    directory = Path(directory)
    parquet_files = sorted(directory.glob("*.parquet"))
    info(f"Found {len(parquet_files)} parquet file(s) in {directory}")
    for f in parquet_files:
        warn_if_not_materialized(f)
    return parquet_files


def save_summary(summary: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    info(f"Saved run summary to: {output_path}")
    return output_path
