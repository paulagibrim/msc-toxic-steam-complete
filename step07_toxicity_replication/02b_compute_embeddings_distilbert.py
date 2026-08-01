"""Step 7, Step 2b: DistilBERT variant of 02_compute_embeddings.py - one
embedding per review (not pooled), same review-level granularity, but with
the frozen encoder used for phase3's fine-tuning and phase4's frozen-encoder
LOO experiment, instead of MiniLM. Lets pooling_ablation.py (phase4) compare
MiniLM vs DistilBERT as the base representation, holding pooling strategy
and downstream classifier fixed.

MODEL: distilbert-base-multilingual-cased, 768-dim. There is no
sentence-transformer tuned for this checkpoint, so the raw encoder is
wrapped with a mean-pooling head over token embeddings - the same pattern
used by phase4_text_only_fullpop/01_build_distilbert_embeddings.py (which
does per-USER pooling; this script stops one level earlier, at per-REVIEW).

STORAGE: same convention as 02_compute_embeddings.py - a flat float32 .npy
array (row order matches the input parquet's row order), separate from the
existing review_dataset_<lang>.parquet (label/user_url/text stay there
unchanged, this only adds a second, 768-dim embeddings file alongside the
384-dim MiniLM one - review_embeddings_<lang>.npy is never overwritten).

RESUMABLE: identical checkpoint pattern to 02_compute_embeddings.py.

Usage:
    python 02b_compute_embeddings_distilbert.py \\
        --review-dataset ../../steam-data/step07-output/review_dataset_pt.parquet \\
        --output ../../steam-data/step07-output/review_embeddings_distilbert_pt.npy \\
        --batch-size 256
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_utils import info

MODEL_NAME = "distilbert-base-multilingual-cased"
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 128  # matches 01_build_distilbert_embeddings.py, for parity with phase4's per-user DistilBERT vectors
ENCODE_CHUNK = 50_000  # rows per encode() call - bounds peak memory regardless of dataset size, not a scientific threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Computes one DistilBERT embedding per review (step07).")
    parser.add_argument("--review-dataset", required=True, type=Path, help="Path to 01_build_review_dataset.py's output")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the .npy embedding array to")
    parser.add_argument("--batch-size", type=int, default=256, help="SentenceTransformer encode() batch size")
    return parser.parse_args()


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    df = pd.read_parquet(args.review_dataset, columns=["review_text_clean"])
    n_total = len(df)
    info(f"Loaded {n_total:,} review(s) from {args.review_dataset}")

    device = get_device()
    info(f"Loading {MODEL_NAME} on device={device}...")
    # Frozen base DistilBERT + mean-pooling (there is no sentence-transformer
    # tuned for distilbert-base-multilingual-cased, so we wrap the raw
    # encoder with a mean-pooling head, same as phase4's per-user builder).
    from sentence_transformers import SentenceTransformer, models
    _word = models.Transformer(MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH)
    _pool = models.Pooling(_word.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[_word, _pool], device=device)

    checkpoint_meta = args.output.with_suffix(".progress.json")
    n_done = 0
    if args.output.exists() and checkpoint_meta.exists():
        n_done = json.loads(checkpoint_meta.read_text())["n_done"]
        info(f"[checkpoint] Resuming: {n_done:,} of {n_total:,} already embedded")

    out = np.lib.format.open_memmap(
        args.output, mode="r+" if n_done else "w+", dtype="float32", shape=(n_total, EMBEDDING_DIM)
    )

    texts = df["review_text_clean"].tolist()
    bar = tqdm(total=n_total, initial=n_done, desc="[embed]", unit="review")
    for start in range(n_done, n_total, ENCODE_CHUNK):
        end = min(start + ENCODE_CHUNK, n_total)
        batch_vecs = model.encode(
            texts[start:end], batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True,
        )
        out[start:end] = batch_vecs.astype("float32")
        out.flush()
        checkpoint_meta.write_text(json.dumps({"n_done": end}))
        bar.update(end - start)
    bar.close()

    info(f"Saved {n_total:,} embeddings ({EMBEDDING_DIM}-dim) to: {args.output}")
    checkpoint_meta.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
