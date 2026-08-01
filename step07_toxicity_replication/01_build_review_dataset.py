"""Step 7, Step 1: builds the review-level dataset for testing how well a
model (classical or fine-tuned) can approximate step02's own toxicity
classification (Perspective OR Detoxify union rule), instead of running
both scorers.

SCOPE, DELIBERATELY NOT SUBSAMPLED: full corpus, real prevalence - every
valid-scored pt/en review, not a matched/balanced sample. This mirrors
step06 Phase 2's own choice (real prevalence, no negative subsampling) and
avoids the "balanced sample inflates AUC-PR, not directly comparable"
caveat that had to be attached to step06 Phase 6's results. Nothing here
is a new threshold invented for this script - every filter below reuses an
existing, already-justified decision from elsewhere in the project:

  - AGREEMENT MASK (review_lang == perspective_declared_language): same
    filter step06 Phase 1's 04_build_user_text_embeddings.py applies when
    scoping a review to one language population.
  - VALIDITY MASK (scores in [0, 1]): reuses step02's own
    validity_mask.apply_validity_mask (Detoxify's -1.0 failed-batch
    sentinel, or any other out-of-range value, excluded).
  - TOXICITY MASK (perspective>=0.7 OR detoxify>=0.9): reuses step02's own
    toxicity_mask.apply_toxicity_mask verbatim - this IS the label being
    approximated, so it must be computed identically, not re-derived.
  - TEXT CLEANING: step06 Phase 1's light_clean (boilerplate/URL stripping
    only, case/accent/punctuation preserved) - copied verbatim rather than
    re-implemented, and empty-after-cleaning rows are dropped, exactly as
    that script does.

Output: one parquet per language (pt/en) with review_text_clean,
is_toxic (the union label), user_url, review_url - no embedding yet
(02_compute_embeddings.py does that separately, so this step's output is
reusable regardless of which embedding/model approach follows).

Usage:
    python 01_build_review_dataset.py \\
        --step02-dir ../../steam-data/step02-output \\
        --language pt \\
        --output ../../steam-data/step07-output/review_dataset_pt.parquet
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline_utils import info, save_summary

# Verbatim copy of step06 Phase 1's boilerplate patterns
# (04_build_user_text_embeddings.py) - see module docstring.
BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]
PERSPECTIVE_THRESHOLD = 0.7  # step02's toxicity_mask.py default, reused verbatim
DETOXIFY_THRESHOLD = 0.9     # step02's toxicity_mask.py default, reused verbatim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds the review-level (text, union-toxicity-label) dataset for step07."
    )
    parser.add_argument("--step02-dir", required=True, type=Path, help="Path to step02's output directory")
    parser.add_argument("--language", required=True, choices=["pt", "en"], help="Which language partition to build")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the review-level parquet to")
    return parser.parse_args()


def light_clean(text: object) -> str:
    """Verbatim copy of step06 Phase 1's light_clean - see module docstring."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def main():
    args = parse_args()

    lang_dir = args.step02_dir / f"review_lang={args.language}"
    if not lang_dir.exists():
        sys.exit(f"Expected step02 output partition not found: {lang_dir}")

    files = sorted(lang_dir.glob("*.parquet"))
    info(f"[{args.language}] Reading {len(files)} file(s) from {lang_dir}...")

    frames = []
    n_before_agreement = 0
    n_before_validity = 0
    for f in tqdm(files, desc=f"[{args.language}]", unit="file"):
        df = pd.read_parquet(
            f,
            columns=[
                "user_url", "review_url", "review_text", "review_lang",
                "perspective_declared_language", "perspective_score", "detoxify_score",
            ],
        )
        n_before_agreement += len(df)
        df = df[df["review_lang"] == df["perspective_declared_language"]]

        n_before_validity += len(df)
        valid = df["perspective_score"].between(0, 1) & df["detoxify_score"].between(0, 1)
        df = df[valid]
        if df.empty:
            continue

        df["review_text_clean"] = df["review_text"].apply(light_clean)
        df = df[df["review_text_clean"].str.len() > 0]
        if df.empty:
            continue

        df["is_toxic"] = (df["perspective_score"] >= PERSPECTIVE_THRESHOLD) | (df["detoxify_score"] >= DETOXIFY_THRESHOLD)
        frames.append(df[["user_url", "review_url", "review_text_clean", "is_toxic"]])

    dataset = pd.concat(frames, ignore_index=True)
    n_toxic = int(dataset["is_toxic"].sum())
    prevalence = n_toxic / len(dataset)

    info(
        f"[{args.language}] {n_before_agreement:,} row(s) read -> {n_before_validity:,} after agreement mask "
        f"-> {len(dataset):,} after validity + non-empty-text filters"
    )
    info(f"[{args.language}] {n_toxic:,} toxic ({prevalence:.4%}), {len(dataset) - n_toxic:,} non-toxic")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)
    info(f"Saved review-level dataset ({len(dataset):,} rows) to: {args.output}")

    save_summary(
        {
            "language": args.language,
            "n_rows_read": n_before_agreement,
            "n_rows_after_agreement_mask": n_before_validity,
            "n_rows_final": int(len(dataset)),
            "n_toxic": n_toxic,
            "prevalence": round(prevalence, 6),
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
