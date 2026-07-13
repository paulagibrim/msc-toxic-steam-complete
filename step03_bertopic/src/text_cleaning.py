"""
text_cleaning.py — Stage 1: batch text cleaning over step02's detoxify parquet files.

Responsibilities:
  - Apply the is_toxic label (Perspective OR Detoxify threshold - the same
    union rule and the same thresholds as toxicity_mask.py in dissertacao-steam).
  - Clean raw review text: lowercase, strip URLs, normalise unicode, remove
    non-alpha characters, collapse whitespace.
  - Preserve review_url and game_id for downstream joining.
  - Write one cleaned parquet per input file, keeping the same filename.
  - Support resuming an interrupted run (--resume flag in the entrypoint).

Ported from dissertacao-steam/bertopic_pipeline/src/text_cleaning.py, with
three changes to match this project's conventions:
  - Detoxify's score column is named `detoxify_score` here, not `toxicity`
    (see step02_run_detoxify's README - toxicity was renamed on export).
  - Rows where either score is outside [0, 1] (Detoxify's -1.0 "failed to
    score" sentinel - see step02's detoxify_scoring.py) are EXCLUDED before
    labelling, not treated as non-toxic via fillna(0). Silently counting a
    scoring failure as "definitely not toxic" was found to be a real bug
    when auditing tfidf_analysis.py/tag_toxicity.py earlier in this project
    and is deliberately not repeated here.
  - Re-applies step01/step02's language-agreement mask (perspective_declared_
    language == settings.lang_code) before anything else. step02's
    detoxify_scoring.py already applies this same mask unconditionally
    before scoring, so every row read here should already agree - this is
    a cheap, explicit double-check (a `==` on a column already present),
    not a second data-processing pass, kept in case this stage is ever
    pointed at un-filtered input by mistake.

Why file-by-file?
  Processing one file at a time keeps the memory footprint proportional to
  the largest single file, rather than requiring the whole language's
  dataset in RAM for a purely CPU-bound text-manipulation task.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd
from pandarallel import pandarallel

from .settings import Settings
from .utils import get_pandarallel_workers, timer

logger = logging.getLogger(__name__)

# Columns that must exist in every input file (step02_run_detoxify's output).
_REQUIRED_INPUT_COLUMNS = {"review_text", "perspective_score", "detoxify_score", "perspective_declared_language"}

# Columns written to each cleaned parquet file.
_OUTPUT_COLUMNS = ["review_text_clean", "is_toxic", "review_url", "game_id"]

# Boilerplate phrases stripped before cleaning - same patterns step02's
# detoxify_scoring.py strips before scoring (Steam's own early-access/refund
# notices, injected as plain text in some reviews).
_BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]


# ── Text cleaning ──────────────────────────────────────────────────────────────


def clean_text(text: object) -> str:
    """Normalise a single review string for BERTopic ingestion.

    Steps applied in order:
      1. Coerce non-strings to empty string.
      2. Lowercase.
      3. Remove URLs (http/https/www).
      4. Remove known Steam boilerplate phrases (early-access/refund notices).
      5. Strip accents via unicode NFKD decomposition.
      6. Remove all characters that are not ASCII letters or whitespace.
      7. Collapse multiple spaces and strip leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for pattern in _BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Per-file processing ────────────────────────────────────────────────────────

def _process_file(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    resume: bool,
) -> dict:
    """Clean one parquet file and write the result to output_path.

    Returns a dict with per-file statistics for logging.
    """
    if resume and output_path.exists():
        logger.info("SKIP (already cleaned): %s", input_path.name)
        return {"file": input_path.name, "status": "skipped"}

    try:
        df = pd.read_parquet(input_path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", input_path.name, exc)
        return {"file": input_path.name, "status": "error", "error": str(exc)}

    if df.empty:
        logger.warning("Empty file, skipping: %s", input_path.name)
        return {"file": input_path.name, "status": "empty"}

    missing = _REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        logger.error("Missing required columns %s in %s", missing, input_path.name)
        return {"file": input_path.name, "status": "error", "error": f"missing columns {missing}"}

    # ── Language-agreement double-check ────────────────────────────────────────
    # step02's detoxify_scoring.py already applies this exact mask before
    # scoring, so this should be a no-op in the normal flow - kept explicit
    # here in case this stage is ever run against un-filtered input.
    rows_before_agreement = len(df)
    df = df[df["perspective_declared_language"] == settings.lang_code]
    n_excluded_disagreement = rows_before_agreement - len(df)
    if n_excluded_disagreement:
        logger.warning(
            "%s: excluding %d row(s) where perspective_declared_language != '%s' "
            "(unexpected here - step02 should have already filtered these)",
            input_path.name, n_excluded_disagreement, settings.lang_code,
        )

    # ── Exclude invalid/sentinel scores ────────────────────────────────────────
    # Detoxify writes -1.0 when scoring a row failed (see step02's
    # detoxify_scoring.py) - that's "no score", not "low score", so those
    # rows are dropped before labelling rather than silently voting
    # "non-toxic" via fillna(0).
    perspective_valid = df["perspective_score"].between(0, 1)
    detoxify_valid = df["detoxify_score"].between(0, 1)
    valid = perspective_valid & detoxify_valid
    n_excluded_invalid = int((~valid).sum())
    if n_excluded_invalid:
        logger.info(
            "%s: excluding %d row(s) with an invalid/sentinel score",
            input_path.name, n_excluded_invalid,
        )
    df = df[valid]

    # ── Toxicity label ─────────────────────────────────────────────────────────
    # Union rule: a review is toxic if EITHER classifier exceeds its threshold
    # (same rule and thresholds as toxicity_mask.py in dissertacao-steam).
    mask_perspective = df["perspective_score"] >= settings.perspective_threshold
    mask_detoxify    = df["detoxify_score"]    >= settings.detoxify_threshold
    df["is_toxic"]   = mask_perspective | mask_detoxify

    # ── Parallel text cleaning ─────────────────────────────────────────────────
    df["review_text_clean"] = df["review_text"].parallel_apply(clean_text)

    # ── Select output columns ──────────────────────────────────────────────────
    # Only keep columns that actually exist (review_url / game_id may be absent
    # in edge-case partition files).
    available = [c for c in _OUTPUT_COLUMNS if c in df.columns]
    df_out = df[available]

    df_out.to_parquet(output_path, index=False)

    stats = {
        "file": input_path.name,
        "status": "ok",
        "total_rows": len(df),
        "excluded_disagreement": n_excluded_disagreement,
        "excluded_invalid_score": n_excluded_invalid,
        "toxic_rows": int(df["is_toxic"].sum()),
        "empty_text": int((df["review_text_clean"] == "").sum()),
    }
    logger.info(
        "Cleaned %s — %d rows, %d toxic (%.1f%%)",
        input_path.name,
        stats["total_rows"],
        stats["toxic_rows"],
        100 * stats["toxic_rows"] / max(stats["total_rows"], 1),
    )
    return stats


# ── Public entry point ─────────────────────────────────────────────────────────

def run_cleaning(settings: Settings, resume: bool = True) -> List[dict]:
    """Process all parquet files in settings.detoxify_data_dir.

    Args:
        settings: pipeline configuration object.
        resume:   if True, files that already have a matching output are skipped,
                  allowing the stage to be restarted after an interruption.

    Returns:
        List of per-file statistics dicts.
    """
    hw_workers = get_pandarallel_workers(
        {"cpu_count": __import__("os").cpu_count() or 1}
    )
    pandarallel.initialize(progress_bar=False, nb_workers=hw_workers, verbose=0)
    logger.info("pandarallel initialised with %d workers.", hw_workers)

    input_files = sorted(settings.detoxify_data_dir.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(
            f"No .parquet files found in {settings.detoxify_data_dir}. "
            "Verify that step02_run_detoxify has been completed for this language."
        )

    logger.info(
        "Stage 1 — Text Cleaning | %d files | input: %s | output: %s",
        len(input_files),
        settings.detoxify_data_dir,
        settings.cleaned_data_dir,
    )
    settings.cleaned_data_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    with timer("Stage 1 — Text Cleaning"):
        for f in input_files:
            out = settings.cleaned_data_dir / f.name
            stats = _process_file(f, out, settings, resume)
            all_stats.append(stats)

    ok_count   = sum(1 for s in all_stats if s["status"] == "ok")
    skip_count = sum(1 for s in all_stats if s["status"] == "skipped")
    err_count  = sum(1 for s in all_stats if s["status"] == "error")
    logger.info(
        "Stage 1 complete — processed: %d | skipped: %d | errors: %d",
        ok_count, skip_count, err_count,
    )

    return all_stats
