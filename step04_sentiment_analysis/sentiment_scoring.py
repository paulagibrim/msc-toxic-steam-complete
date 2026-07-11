"""Runs sentiment analysis on step02's detoxify-scored reviews
(review_lang=<lang>/*.parquet), one file at a time, using
nlptown/bert-base-multilingual-uncased-sentiment - a 5-star multilingual
classifier trained on product reviews, the closest available domain match
to Steam reviews, and multilingual enough to cover pt and en with a single
model (no need to split by language like BERTopic's embedding step).

Rather than keeping just the argmax star label (a coarse 1-5 integer), this
keeps a continuous `sentiment_score` - the expected value over the model's
5-class softmax distribution (sum(p_i * i) for i=1..5) - so two reviews
that both land on "3 stars" but with different underlying probabilities
aren't collapsed to the same value. This keeps sentiment_score on the same
continuous footing as perspective_score/detoxify_score for downstream
correlation analysis, per explicit user request for a fine-grained
intensity signal rather than a discrete class.

Every column already in step02's output is kept (review_text, game_id,
perspective_score, detoxify_score, etc.) - same rationale as
detoxify_scoring.py: the model already needs review_text in memory, so
there's no extra cost to keeping the rest of the row, and downstream
analysis needs text/game/score together anyway.
"""
import re
from pathlib import Path

import pandas as pd

from pipeline_utils import error, info, list_parquet_files, warn_if_not_materialized

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
BATCH_SIZE = 32
MAX_CHARS = 1200

# Steam early-access/refund boilerplate notices, stripped from review text
# before scoring - same patterns step02's detoxify_scoring.py strips, so
# sentiment isn't skewed by non-review text Steam injects into the body.
BOILERPLATE_PATTERNS = [
    r"AN[AÁ]LISE DE ACESSO ANTECIPADO",
    r"produto recebido de gra[cç]a",
]

# Sentinel written when a batch fails to score - outside the valid [1, 5]
# range so it can't be mistaken for a real (low) score, same convention as
# detoxify_scoring.py's -1.0 for its [0, 1] range.
FAILED_SCORE_SENTINEL = -1.0


def clean_review_text(text):
    """Strips known boilerplate phrases from review text (case-insensitive).
    Non-string input (e.g. NaN) passes through unchanged."""
    if not isinstance(text, str):
        return text
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text.strip()


def load_sentiment_model(device=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _score_batch(texts, tokenizer, model, device) -> list:
    """Returns one continuous sentiment_score per text: the expected star
    rating (1.0-5.0) over the model's 5-class softmax distribution, not
    just the argmax label - see module docstring."""
    import torch

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        stars = torch.arange(1, probs.shape[-1] + 1, device=device, dtype=probs.dtype)
        scores = (probs * stars).sum(dim=-1)
    return scores.cpu().tolist()


def score_file(
    file_path: Path,
    output_dir: Path,
    tokenizer,
    model,
    device,
    batch_size: int = BATCH_SIZE,
    max_chars: int = MAX_CHARS,
) -> Path:
    """Scores one review_lang=<lang> file, writing every column already in
    the file plus the new `sentiment_score` to `output_dir` under the same
    filename. Skips (resumes) if the output file already exists."""
    output_path = output_dir / file_path.name
    if output_path.exists():
        info(f"Skipping {file_path.name} (already scored)")
        return output_path

    warn_if_not_materialized(file_path)
    df = pd.read_parquet(file_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        info(f"Nothing to score in {file_path.name} - writing an empty result.")
        df["sentiment_score"] = pd.Series(dtype="float64")
        df.to_parquet(output_path, engine="pyarrow", index=False)
        return output_path

    texts = df["review_text"].apply(clean_review_text).fillna("").astype(str).str.slice(0, max_chars).tolist()
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            scores.extend(_score_batch(batch, tokenizer, model, device))
        except Exception as e:
            error(f"Batch starting at row {i} failed: {e}")
            scores.extend([FAILED_SCORE_SENTINEL] * len(batch))
        if device.type == "mps":
            import torch
            torch.mps.empty_cache()
        elif device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    df["sentiment_score"] = scores
    df.to_parquet(output_path, engine="pyarrow", index=False)
    info(f"Scored {len(texts)} review(s) in {file_path.name} -> {output_path}")

    import gc
    del df, texts, scores
    gc.collect()
    return output_path


def run_sentiment_for_language(reviews_dir: Path, output_dir: Path, lang: str, device=None) -> list:
    """Scores every file in review_lang=<lang>, resuming per-file. A
    failure on one file is logged and skipped rather than aborting the run."""
    partition_dir = reviews_dir / f"review_lang={lang}"
    files = list_parquet_files(partition_dir)

    tokenizer, model, device = load_sentiment_model(device)
    info(f"[{lang}] Sentiment model loaded on device: {device}")

    output_paths = []
    for i, f in enumerate(files, start=1):
        info(f"[{lang}] [{i}/{len(files)}] {f.name}")
        try:
            output_paths.append(score_file(f, output_dir, tokenizer, model, device))
        except Exception as e:
            error(f"[{lang}] Fatal error on {f.name}: {e} - skipping to next file")
            continue
    return output_paths
