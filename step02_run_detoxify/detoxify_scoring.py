"""Runs Detoxify on step01's cleaned reviews (reviews_cleaned.parquet/
*.parquet), one file at a time, keeping only rows that pass BOTH: langdetect
says `review_lang` is one of the target languages, AND Perspective's own
declared language agrees with that SAME value
(`perspective_declared_language == review_lang`) - see
step01_cleaning_and_language_detection/agreement_mask.py.

review_lang is a plain column in step01's output, not a directory
partition (see clean_reviews.export_reviews's docstring) - and, per
explicit user decision, this step's OWN output is flat too, for the same
reason and for consistency across the whole pipeline: every file is read
ONCE (not once per language - a straight efficiency win over an earlier
per-language-loop version of this module) and scores every target
language's matching rows together in the same pass, writing one flat
output file per source file. Downstream steps (step03/04/05,
review_examples) filter `review_lang == lang` themselves after reading,
the same way they already filter `perspective_declared_language == lang` -
see this project's actual "produto reembolsado" incident, where Hive-style
partitioning by review_lang caused reprocessed reviews to silently keep
stale results under file-based resumability.

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
# before scoring - applied to every language (this boilerplate is inserted
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


def apply_language_mask(df: pd.DataFrame, langs: list) -> pd.DataFrame:
    """Keeps rows where review_lang is one of `langs` AND
    perspective_declared_language agrees with that SAME value (not just
    membership in `langs` - both signals must point to the identical
    language). Since review_lang is a plain column (not a directory
    partition - see clean_reviews.py's module docstring), a source file
    can hold any mix of languages, so this check is necessary regardless
    of which file `df` came from."""
    return df[df["review_lang"].isin(langs) & (df["review_lang"] == df["perspective_declared_language"])]


def load_score_cache(old_output_dir: Path, langs: list, exclude_pattern: str = None) -> dict:
    """Builds a review_url -> detoxify_score lookup from an already-scored
    step02 output directory. Passed into score_file/run_detoxify to skip
    re-running the model on reviews already scored, however this re-run's
    file layout happens to be split up. Returns {} if the directory
    doesn't exist.

    Supports both this module's current flat output layout (read directly)
    and the older per-language `review_lang=<lang>/` folder layout still
    sitting around from before this project standardized on flat+mask
    everywhere - tries flat first, falls back to per-language folders per
    language in `langs` if no flat files are found.

    exclude_pattern: optional regex (case-insensitive) - review_urls whose
    review_text matches this are EXCLUDED from the cache, forcing them to
    be re-scored instead of reusing a stale value. Use this when the old
    output was scored before a BOILERPLATE_PATTERNS fix (e.g. a newly
    added pattern), so only the actually-affected rows get re-scored and
    everything else still reuses the cache."""
    if not old_output_dir.exists():
        info(f"No cache directory found at {old_output_dir} - starting with an empty cache")
        return {}

    columns = ["review_url", "detoxify_score", "review_text"] if exclude_pattern else ["review_url", "detoxify_score"]

    files = list_parquet_files(old_output_dir)
    if not files:
        # Fall back to the older review_lang=<lang>/ folder layout.
        files = []
        for lang in langs:
            lang_dir = old_output_dir / f"review_lang={lang}"
            if lang_dir.is_dir():
                files.extend(list_parquet_files(lang_dir))
    if not files:
        return {}

    frames = [pd.read_parquet(f, columns=columns) for f in files]
    combined = pd.concat(frames, ignore_index=True)

    if exclude_pattern:
        affected = combined["review_text"].str.contains(exclude_pattern, case=False, na=False, regex=True)
        n_affected = int(affected.sum())
        if n_affected:
            info(f"Excluding {n_affected} cached score(s) matching exclude_pattern (will be re-scored)")
        combined = combined[~affected]

    cache = dict(zip(combined["review_url"], combined["detoxify_score"]))
    info(f"Loaded {len(cache)} cached score(s) from {old_output_dir}")
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


def _score_texts(texts: list, model, device, batch_size: int) -> list:
    """Runs Detoxify in batches over already-cleaned texts, returning one
    score per text (-1.0 on batch failure)."""
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
    return scores


def score_file(
    file_path: Path,
    output_dir: Path,
    langs: list,
    model,
    device,
    batch_size: int = BATCH_SIZE,
    max_chars: int = MAX_CHARS,
    cache: dict = None,
    fix_pattern: str = None,
) -> Path:
    """Scores one file with Detoxify - every target language's matching
    rows in the same pass - writing every column already in the file
    (review_text, game_id, perspective_score, review_lang, etc.) plus the
    new `detoxify_score` to `output_dir` under the same filename. Skips
    (resumes) if the output file already exists.

    If `cache` (a review_url -> detoxify_score dict, see load_score_cache)
    is given, rows whose review_url is already in it reuse that score
    instead of running the model again - useful when this file's layout
    came from a re-run of an earlier stage (e.g. a language-detection fix)
    that doesn't change individual reviews' scores, only which file they
    land in.

    If `fix_pattern` is given AND the output file already exists, checks
    whether any of ITS rows have review_text matching that pattern
    (meaning they were scored before a BOILERPLATE_PATTERNS entry was
    added). If none match, the file is skipped as usual. If some do, ONLY
    those rows are re-scored and the file is overwritten in place -
    everything else in it is untouched. Lets a boilerplate-pattern fix be
    applied directly against the same --output-dir, without moving
    anything aside first."""
    output_path = output_dir / file_path.name

    if output_path.exists():
        if not fix_pattern:
            info(f"Skipping {file_path.name} (already scored)")
            return output_path

        existing = pd.read_parquet(output_path)
        affected_mask = existing["review_text"].str.contains(fix_pattern, case=False, na=False, regex=True)
        n_affected = int(affected_mask.sum())
        if n_affected == 0:
            info(f"Skipping {file_path.name} (already scored, no rows match fix_pattern)")
            return output_path

        info(f"{file_path.name}: {n_affected} already-scored row(s) match fix_pattern - re-scoring in place")
        texts = (
            existing.loc[affected_mask, "review_text"]
            .apply(clean_review_text).fillna("").astype(str).str.slice(0, max_chars).tolist()
        )
        new_scores = _score_texts(texts, model, device, batch_size)
        existing.loc[affected_mask, "detoxify_score"] = new_scores
        existing.to_parquet(output_path, engine="pyarrow", index=False)
        info(f"Patched {n_affected} row(s) in {file_path.name} -> {output_path}")
        return output_path

    warn_if_not_materialized(file_path)
    df = pd.read_parquet(file_path)
    df = apply_language_mask(df, langs).copy().reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        info(f"Nothing to score in {file_path.name} after the language mask - writing an empty result.")
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
        new_scores = _score_texts(texts, model, device, batch_size)
        for pos, idx in enumerate(to_score_idx):
            final_scores[idx] = new_scores[pos]

    df["detoxify_score"] = final_scores
    df.to_parquet(output_path, engine="pyarrow", index=False)
    counts_by_lang = df["review_lang"].value_counts().to_dict()
    info(
        f"Scored {file_path.name} ({len(to_score_idx)} new, {len(df) - len(to_score_idx)} cached, "
        f"by language: {counts_by_lang}) -> {output_path}"
    )

    import gc
    del df, final_scores
    gc.collect()
    return output_path


def run_detoxify(
    reviews_cleaned_dir: Path, output_dir: Path, langs: list, device=None, cache_from: Path = None,
    cache_exclude_pattern: str = None, fix_pattern: str = None,
) -> list:
    """Scores every file under reviews_cleaned_dir ONCE, resuming per-file.
    A failure on one file is logged and skipped rather than aborting the
    run.

    langs: every target language is scored together in the same pass over
    each file (apply_language_mask filters review_lang.isin(langs) inside
    score_file) - unlike an earlier per-language-loop version of this
    module, a file is never read twice for two different languages.

    cache_from: optional path to an already-scored step02 output directory
    (see load_score_cache) - reused so re-scoring after an upstream fix
    doesn't re-run the model on reviews it already scored.

    cache_exclude_pattern: optional regex - cached rows whose review_text
    matches it are excluded from the cache (forced to re-score). Use after
    adding a new BOILERPLATE_PATTERNS entry, to invalidate just the rows
    that pattern affects instead of the whole cache.

    fix_pattern: optional regex - patches already-scored output files IN
    PLACE (same --output-dir, no cache_from/moving anything aside needed).
    See score_file's docstring."""
    files = list_parquet_files(reviews_cleaned_dir)

    cache = load_score_cache(cache_from, langs, exclude_pattern=cache_exclude_pattern) if cache_from else {}

    model, device = load_detoxify_model(device)
    info(f"Detoxify model loaded on device: {device} (languages: {langs})")

    output_paths = []
    for i, f in enumerate(files, start=1):
        info(f"[{i}/{len(files)}] {f.name}")
        try:
            output_paths.append(score_file(f, output_dir, langs, model, device, cache=cache, fix_pattern=fix_pattern))
        except Exception as e:
            error(f"Fatal error on {f.name}: {e} - skipping to next file")
            continue
    return output_paths
