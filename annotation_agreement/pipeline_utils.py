"""Minimal logging/IO helpers - same as the other steps', copied here so
this folder stays independently copyable and doesn't depend on any step
folder being present (it only reads raw annotation spreadsheets)."""
import json
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"[INFO {_now()}] {msg}")


def save_summary(summary: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    info(f"Saved run summary to: {output_path}")
    return output_path
