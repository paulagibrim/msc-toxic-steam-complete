"""Runs Detoxify on step01's cleaned+language-partitioned reviews
(reviews_cleaned.parquet/review_lang=<lang>/*.parquet), one file at a time,
after applying step01's agreement mask (langdetect AND the
Perspective-scrape-declared language must agree - see
step01_cleaning_and_language_detection/agreement_mask.py).

Per explicit user request, only Detoxify's `toxicity` output is kept - not
the other six sub-scores (severe_toxicity, obscene, identity_attack,
insult, threat, sexual_explicit) Detoxify also computes in the same call -
and it's renamed to `detoxify_score`.

Unlike toxicity_mask.py/language_revalidation.py/agreement_mask.py's slim
companion tables, the output here keeps every column already in step01's
data (review_text, game_id, perspective_score, etc.), not just review_url -
Detoxify already needs review_text in memory to run inference, so there's
no extra read/join cost to keeping the rest of the row too, and unlike
those other masks, virtually every downstream analysis needs review_text/
game_id (and eventually user data) right alongside the score anyway -
merging them back together later would just be a recurring cost for no
benefit.
"""
import re
from pathlib import Path

import pandas as pd

from pipeline_utils import error, info, list_parquet_files, warn_if_not_materialized

BATCH_SIZE = 32
MAX_CHARS = 1200

# Steam early-access/refund boilerplate notices, stripped from review text
# before scoring - applied to both languages (this boilerplate is inserted
# in Portuguese by Steam's own interface regardless of the review's actual
# language, per prior observation in this project's earlier iteration).
BOILERPLATE_PATTERNS = [
    r"AN[AÁ]LISE DE ACESSO ANTECIPADO",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]


def clean_review_text(text):
    """Strips known boilerplate phrases from review text (case-insensitive).
    Non-string input (e.g. NaN) passes through unchanged."""
    if not isinstance(text, str):
        return text
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text.strip()


def apply_agreement_mask(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Same check as step01's agreement_mask.py: keeps only rows where the
    Perspective-scrape-declared language also equals `lang` - langdetect
    already agrees, since `df` comes from one `review_lang=<lang>` file."""
    return df[df["perspective_declared_language"] == lang]


def load_score_cache(old_output_dir: Path, lang: str) -> dict:
    """Builds a review_url -> detoxify_score lookup from an already-scored
    step02 output directory (e.g. before an upstream language-detection fix
    that changes which file/partition a review lands in, but not its
    score - Detoxify scores a review's text, not its file location).
    Passed into score_file/run_detoxify_for_language to skip re-running the
    model on reviews already scored, however this re-run's file layout
    happens to be split up. Returns {} if the directory doesn't exist."""
    partition_dir = old_output_dir / f"review_lang={lang}"
    if not partition_dir.exists():
        info(f"No cache directory found at {partition_dir} - starting with an empty cache")
        return {}
    files = list_parquet_files(partition_dir)
    if not files:
        return {}
    frames = [pd.read_parquet(f, columns=["review_url", "detoxify_score"]) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    cache = dict(zip(combined["review_url"], combined["detoxify_score"]))
    info(f"[{lang}] Loaded {len(cache)} cached score(s) from {old_output_dir}")
    return cache


def load_detoxify_model(device=None):
    import torch
    from detoxify import Detoxify

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return Detoxify("multilingual", device=device), device


def score_file(
    file_path: Path,
    output_dir: Path,
    lang: str,
    model,
    device,
    batch_size: int = BATCH_SIZE,
    max_chars: int = MAX_CHARS,
    cache: dict = None,
) -> Path:
    """Scores one review_lang=<lang> file with Detoxify, writing every
    column already in the file (review_text, game_id, perspective_score,
    etc.) plus the new `detoxify_score` to `output_dir` under the same
    filename. Skips (resumes) if the output file already exists.

    If `cache` (a review_url -> detoxify_score dict, see load_score_cache)
    is given, rows whose review_url is already in it reuse that score
    instead of running the model again - useful when this file's layout
    came from a re-run of an earlier stage (e.g. a language-detection fix)
    that doesn't change individual reviews' scores, only which file/
    partition they land in."""
    output_path = output_dir / file_path.name
    if output_path.exists():
        info(f"Skipping {file_path.name} (already scored)")
        return output_path

    warn_if_not_materialized(file_path)
    df = pd.read_parquet(file_path)
    df = apply_agreement_mask(df, lang).copy().reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        info(f"Nothing to score in {file_path.name} after the agreement mask - writing an empty result.")
        df["detoxify_score"] = pd.Series(dtype="float64")
        df.to_parquet(output_path, engine="pyarrow", index=False)
        return output_path

    cache = cache or {}
    cached_mask = df["review_url"].isin(cache)
    to_score_idx = df.index[~cached_mask]
    final_scores = [cache.get(url) for url in df["review_url"]]

    if cached_mask.any():
        info(f"{file_path.name}: reusing {int(cached_mask.sum())} cached score(s), scoring {len(to_score_idx)} new row(s)")

    if len(to_score_idx) > 0:
        # Boilerplate stripped only for scoring - review_text is saved as-is,
        # untouched, same as this project's other cleaning-vs-detection split
        # (e.g. langdetect_revalidation.py's clean_for_detection).
        texts = (
            df.loc[to_score_idx, "review_text"]
            .apply(clean_review_text).fillna("").astype(str).str.slice(0, max_chars).tolist()
        )
        new_scores = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                predictions = model.predict(batch)
                new_scores.extend(predictions["toxicity"])
            except Exception as e:
                error(f"Batch starting at row {i} failed: {e}")
                new_scores.extend([-1.0] * len(batch))
            if device.type == "mps":
                import torch
                torch.mps.empty_cache()
            elif device.type == "cuda":
                import torch
                torch.cuda.empty_cache()

        for pos, idx in enumerate(to_score_idx):
            final_scores[idx] = new_scores[pos]

    df["detoxify_score"] = final_scores
    df.to_parquet(output_path, engine="pyarrow", index=False)
    info(f"Scored {file_path.name} ({len(to_score_idx)} new, {len(df) - len(to_score_idx)} cached) -> {output_path}")

    import gc
    del df, final_scores
    gc.collect()
    return output_path


def run_detoxify_for_language(
    reviews_cleaned_dir: Path, output_dir: Path, lang: str, device=None, cache_from: Path = None
) -> list:
    """Scores every file in review_lang=<lang>, resuming per-file. A
    failure on one file is logged and skipped rather than aborting the run.

    cache_from: optional path to an already-scored step02 output directory
    (see load_score_cache) - reused so re-scoring after an upstream fix
    doesn't re-run the model on reviews it already scored."""
    partition_dir = reviews_cleaned_dir / f"review_lang={lang}"
    files = list_parquet_files(partition_dir)

    cache = load_score_cache(cache_from, lang) if cache_from else {}

    model, device = load_detoxify_model(device)
    info(f"[{lang}] Detoxify model loaded on device: {device}")

    output_paths = []
    for i, f in enumerate(files, start=1):
        info(f"[{lang}] [{i}/{len(files)}] {f.name}")
        try:
            output_paths.append(score_file(f, output_dir, lang, model, device, cache=cache))
        except Exception as e:
            error(f"[{lang}] Fatal error on {f.name}: {e} - skipping to next file")
            continue
    return output_paths
