"""Step 7, Step 2: computes ONE embedding per review (not pooled) for
01_build_review_dataset.py's output - the input classical models (and the
PCA-vs-no-PCA diagnostic) need.

MODEL: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, 384-dim
- reused verbatim from step06 Phase 1 / step03's BERTopic, not retrained.
No aggregation here (unlike Phase 1's mean-pooling into a per-user vector):
this script's whole point is one vector PER REVIEW, since the classifier
being tested operates at review granularity.

STORAGE: embeddings are saved as a float32 .npy array (row order matches
the input parquet's row order after resetting the index), separate from a
small parquet carrying is_toxic/user_url/review_url - NOT as 384 extra
parquet columns. At en's scale (~25.3M reviews x 384 dims), a wide parquet
would be ~39GB and slower to read/write than a flat memmap-able .npy array
that numpy/sklearn can consume directly without deserialising a DataFrame.

RESUMABLE: encodes in chunks, checkpointing progress (how many rows of the
input have been embedded so far) so an interrupted run doesn't restart from
zero - mirrors the checkpoint pattern used throughout this project's
long-running scripts.

Usage:
    python 02_compute_embeddings.py \\
        --review-dataset ../../steam-data/step07-output/review_dataset_pt.parquet \\
        --output ../../steam-data/step07-output/review_embeddings_pt.npy \\
        --batch-size 256
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_utils import info

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
ENCODE_CHUNK = 50_000  # rows per encode() call - bounds peak memory regardless of dataset size, not a scientific threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Computes one embedding per review (step07).")
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
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)

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
