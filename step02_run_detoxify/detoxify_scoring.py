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
) -> Path:
    """Scores one review_lang=<lang> file with Detoxify, writing every
    column already in the file (review_text, game_id, perspective_score,
    etc.) plus the new `detoxify_score` to `output_dir` under the same
    filename. Skips (resumes) if the output file already exists."""
    output_path = output_dir / file_path.name
    if output_path.exists():
        info(f"Skipping {file_path.name} (already scored)")
        return output_path

    warn_if_not_materialized(file_path)
    df = pd.read_parquet(file_path)
    df = apply_agreement_mask(df, lang).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        info(f"Nothing to score in {file_path.name} after the agreement mask - writing an empty result.")
        df["detoxify_score"] = pd.Series(dtype="float64")
        df.to_parquet(output_path, engine="pyarrow", index=False)
        return output_path

    # Boilerplate stripped only for scoring - review_text is saved as-is,
    # untouched, same as this project's other cleaning-vs-detection split
    # (e.g. langdetect_revalidation.py's clean_for_detection).
    texts = df["review_text"].apply(clean_review_text).fillna("").astype(str).str.slice(0, max_chars).tolist()
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            predictions = model.predict(batch)
            scores.extend(predictions["toxicity"])
        except Exception as e:
            error(f"Batch starting at row {i} failed: {e}")
            scores.extend([-1.0] * len(batch))
        if device.type == "mps":
            import torch
            torch.mps.empty_cache()
        elif device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    df["detoxify_score"] = scores
    df.to_parquet(output_path, engine="pyarrow", index=False)
    info(f"Scored {len(texts)} review(s) in {file_path.name} -> {output_path}")

    import gc
    del df, texts, scores
    gc.collect()
    return output_path


def run_detoxify_for_language(reviews_cleaned_dir: Path, output_dir: Path, lang: str, device=None) -> list:
    """Scores every file in review_lang=<lang>, resuming per-file. A
    failure on one file is logged and skipped rather than aborting the run."""
    partition_dir = reviews_cleaned_dir / f"review_lang={lang}"
    files = list_parquet_files(partition_dir)

    model, device = load_detoxify_model(device)
    info(f"[{lang}] Detoxify model loaded on device: {device}")

    output_paths = []
    for i, f in enumerate(files, start=1):
        info(f"[{lang}] [{i}/{len(files)}] {f.name}")
        try:
            output_paths.append(score_file(f, output_dir, lang, model, device))
        except Exception as e:
            error(f"[{lang}] Fatal error on {f.name}: {e} - skipping to next file")
            continue
    return output_paths
