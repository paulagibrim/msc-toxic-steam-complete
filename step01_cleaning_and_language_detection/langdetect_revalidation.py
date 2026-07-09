"""Cross-checks the declared language of steam-data's raw review files
(steam-data/raw/reviews/*.parquet) against langdetect's own guess for the
actual review text, in parallel across CPU cores.

Adapted from dissertacao-steam/data_refactor/2-toxicity/langdetect_revalidation.py
for this project's data shape, which differs in two ways:
  - Raw, pre-rename column names (`linkComentario` for review_url, `texto`
    for review_text, `language` for the declared language - matches
    dissertacao-steam's clean_reviews.py RENAME_MAP, before it's applied).
  - Not split into one file per language yet: `language` is a per-row
    column inside every file, and every file mixes every language together
    - `validate_raw_reviews_langdetect` filters each batch down to
    `declared_lang` before running detection on it, and loops over every
    *.parquet file in the given directory.

WHY LANGDETECT NEEDS TEXT-QUALITY GATING FIRST (see MIN_ALPHA_LENGTH):
A first pass at this exact check using fastText on the pre-split final
dataset (a different, related project) found a ~37% "mismatch" rate for PT
that turned out to be almost entirely a false-positive storm, not real
contamination: short reviews ("meh.", "gud game") and reviews made mostly of
ASCII-art/emoji (Braille/box-drawing Unicode block characters) give almost
any language-ID model too little real signal, and it guesses some unrelated
language with deceptively high confidence. This isn't fastText-specific -
langdetect suffers the identical problem on the identical inputs - so this
module strips non-linguistic noise and requires a minimum amount of real
alphabetic content before trusting *any* detector's guess.

WHY MULTIPROCESSING:
langdetect is pure Python (no compiled fast path) - roughly 1,000-3,000
texts/second on one core by published benchmarks. Parallelizing across
cores with ProcessPoolExecutor is what makes checking tens of millions of
reviews practical. `n_jobs` defaults to every core on whatever machine this
runs on (os.cpu_count()).
"""
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect_langs

from language_revalidation import UNDETERMINED_LABEL
from pipeline_utils import info, warn_if_not_materialized

# langdetect's underlying algorithm draws random samples internally, so the
# same text can get slightly different probability estimates across runs
# unless seeded - fixed for reproducibility.
DetectorFactory.seed = 0

# A review needs at least this many real alphabetic characters (after
# stripping URLs/digits/punctuation/emoji/ASCII-art) before we trust any
# language guess for it - below this, guesses from any detector amount to
# noise (see module docstring). Below the threshold, the row is labeled
# UNDETERMINED_LABEL without even calling the model.
MIN_ALPHA_LENGTH = 20

# Rows per batch when reading each raw file - keeps only one batch of
# review text in memory at a time (these raw files can be large).
BATCH_SIZE = 500_000

# This project's raw column names, pre-rename (matches
# dissertacao-steam/data_refactor/0-cleaning/clean_reviews.py's RENAME_MAP,
# before it's applied).
RAW_URL_COLUMN = "linkComentario"
RAW_TEXT_COLUMN = "texto"
RAW_LANGUAGE_COLUMN = "language"


def clean_for_detection(text) -> str:
    """Strips URLs, digits, punctuation, and symbol/emoji/ASCII-art
    characters (Unicode Braille/box-drawing blocks are common in Steam
    review "art" and have no linguistic content), keeping actual letters
    (any script) and spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[0-9_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def identify_language_langdetect(text, min_alpha_length: int = MIN_ALPHA_LENGTH):
    """Detects the language of a single piece of text via langdetect.
    Returns a (language_code, confidence) tuple - UNDETERMINED_LABEL if
    there isn't enough real alphabetic content to trust a guess, or if
    langdetect itself can't find enough signal (LangDetectException)."""
    cleaned = clean_for_detection(text)
    if len(cleaned) < min_alpha_length:
        return UNDETERMINED_LABEL, 0.0

    try:
        guesses = detect_langs(cleaned)
    except LangDetectException:
        return UNDETERMINED_LABEL, 0.0

    if not guesses:
        return UNDETERMINED_LABEL, 0.0

    best = guesses[0]
    return best.lang, best.prob


def _count_rows_for_language(raw_dir: Path, declared_lang: str) -> int:
    """Quick pre-scan (just the `language` column, across every file) so
    progress logging has a real denominator - cheap compared to the actual
    detection pass."""
    import pyarrow.parquet as pq

    total = 0
    for file in sorted(raw_dir.glob("*.parquet")):
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[RAW_LANGUAGE_COLUMN]):
            total += int((batch.to_pandas()[RAW_LANGUAGE_COLUMN] == declared_lang).sum())
    return total


def validate_raw_reviews_langdetect(
    raw_dir: Path,
    declared_lang: str,
    n_jobs: int = None,
    batch_size: int = BATCH_SIZE,
    min_alpha_length: int = MIN_ALPHA_LENGTH,
) -> pd.DataFrame:
    """Runs langdetect over every row whose own `language` column equals
    `declared_lang`, across every *.parquet file in `raw_dir` (rows in any
    other language are skipped - not relevant to this check).

    Returns a dataframe: review_url, declared_language, detected_language,
    detection_confidence - one row per matching input row, same shape
    language_revalidation.py's functions expect regardless of detector.
    """
    import os

    import pyarrow.parquet as pq

    n_jobs = n_jobs or os.cpu_count()
    info(f"[{declared_lang}] Using {n_jobs} worker process(es) for langdetect")

    files = sorted(raw_dir.glob("*.parquet"))
    info(f"[{declared_lang}] Found {len(files)} raw file(s) in {raw_dir}")

    total_rows = _count_rows_for_language(raw_dir, declared_lang)
    info(f"[{declared_lang}] {total_rows} row(s) declared as '{declared_lang}' across all files")

    detect_fn = partial(identify_language_langdetect, min_alpha_length=min_alpha_length)

    review_urls = []
    detected_languages = []
    detection_confidences = []
    rows_done = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i, file in enumerate(files, start=1):
            warn_if_not_materialized(file)
            parquet_file = pq.ParquetFile(file)
            info(f"[{declared_lang}] [{i}/{len(files)}] {file.name}")

            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=[RAW_URL_COLUMN, RAW_TEXT_COLUMN, RAW_LANGUAGE_COLUMN]
            ):
                batch_df = batch.to_pandas()
                batch_df = batch_df[batch_df[RAW_LANGUAGE_COLUMN] == declared_lang]
                if batch_df.empty:
                    continue

                review_urls.extend(batch_df[RAW_URL_COLUMN].tolist())
                for language_code, confidence in executor.map(
                    detect_fn, batch_df[RAW_TEXT_COLUMN], chunksize=1000
                ):
                    detected_languages.append(language_code)
                    detection_confidences.append(confidence)

                rows_done += len(batch_df)
                info(f"[{declared_lang}] {rows_done}/{total_rows} reviews checked")

    result = pd.DataFrame({
        "review_url": review_urls,
        "declared_language": declared_lang,
        "detected_language": pd.array(detected_languages, dtype="string"),
        "detection_confidence": pd.array(detection_confidences, dtype="Float64"),
    })

    n_match = int((result["detected_language"] == declared_lang).sum())
    n_undetermined = int((result["detected_language"] == UNDETERMINED_LABEL).sum())
    n_mismatch = len(result) - n_match - n_undetermined
    info(
        f"[{declared_lang}] {len(result)} reviews checked - "
        f"match: {n_match} ({100*n_match/len(result):.2f}%), "
        f"mismatch: {n_mismatch} ({100*n_mismatch/len(result):.2f}%), "
        f"undetermined: {n_undetermined} ({100*n_undetermined/len(result):.2f}%)"
    )
    return result
