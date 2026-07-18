"""Builds per-user text embeddings from step02's agreement-matched pt/en
reviews, restricted to the population already matched to a profile in
build_user_profile_metadata.py's output - the other Phase 1 input (profile
metadata) is scoped to the same population, so the final feature table
(assembled separately) doesn't need a second exclusion pass. Running the
GPU-bound encoding step only over this ~42%-of-eligible subset, instead of
every eligible user in toxic_user_labels.parquet, was an explicit decision
made when planning this phase.

CLEANING: light, NOT step03's aggressive clean_text. Only boilerplate
phrases and URLs are stripped here - case, accents, and punctuation are
left as-is. Rationale (from this project's step06 planning discussion):
step03's aggressive cleaning exists because its cleaned text also feeds
BERTopic's c-TF-IDF term extraction, which needs a normalised vocabulary
(otherwise "não"/"nao"/"NÃO" inflate the vocabulary as three distinct
tokens); that cleaned text is then reused for its embedding step too, out
of convenience, not because the embedding itself benefits from stripping.
Here there is no bag-of-words consumer downstream - only the sentence
embedding - so there's no equivalent reason to discard case/punctuation/
accents, which a sentence-transformer can otherwise still make use of
(tone, emphasis, shouting-via-caps). Preserving them costs nothing; the
aggressive cleaning would only guarantee that information is gone.

MODEL: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (same
model as step03's BERTopic embeddings, 384-dim) - reused for consistency
with the rest of this project, not retrained or fine-tuned here.

AGGREGATION: two-stage mean pooling, the same scheme (adapted from a
BERTimbau/Twitter-toxicity reference paper consulted while planning this
phase) as: word embeddings -> mean into one post's vector -> mean into one
user's vector. The model's own .encode() already performs the first stage
(a sentence-transformer's forward pass IS mean/attention pooling over
token embeddings into one vector per text) - this script performs only the
second stage, mean-pooling a user's own per-review vectors into one vector
per user. Done independently per population (pt/en/union), matching every
other per-population computation in step06 (n_pt/n_en, is_toxic_pt/en/
union, etc.): a user's pt vector is the mean of ONLY their pt reviews,
their union vector the mean of ALL their pt+en reviews - all three reuse
the SAME per-review embeddings (each review is encoded exactly once, never
three times).

SHUFFLE-FREE AGGREGATION (same pattern as build_user_rate_table.py and
run_language_coverage_diagnostic.py, extended to vectors instead of
scalars): each step02 file is embedded and reduced to its own small
per-user PARTIAL SUM + COUNT (a groupby().sum() within that file, at most
that file's number of unique users - never the full corpus at once, so
peak memory stays proportional to one file's embeddings, not all of
them). All partials are collected in a list and combined with a SINGLE
final concat+groupby().sum() at the end (mean = sum / count) - never
folded into a running total per file, which was measured elsewhere in this
project to get progressively slower as the accumulated size grows.

Output: one parquet file, one row per user_url with >=1 embeddable review
in this population, columns:
    user_url,
    emb_pt_0..emb_pt_383, n_pt_embedded,
    emb_en_0..emb_en_383, n_en_embedded,
    emb_union_0..emb_union_383, n_union_embedded
(NaN in a population's emb_* columns for a user with no reviews in it -
n_<pop>_embedded is how many reviews the mean was computed over, which can
be less than n_<pop> in user_rate_table.parquet if some reviews cleaned to
empty text, e.g. a review that was ONLY a boilerplate phrase or a URL).

Usage:
    python build_user_text_embeddings.py \\
        --step02-dir ../../../steam-data/step02-output \\
        --profile-metadata ../../../steam-data/step06-output/user_profile_metadata.parquet \\
        --output ../../../steam-data/step06-output/user_text_embeddings.parquet \\
        --batch-size 256
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_utils import info, save_summary

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
LANGUAGES = ["pt", "en"]
UNION_KEY = "pt+en"
POPULATIONS = ["pt", "en", UNION_KEY]

# Same phrases step02/step03/step05 already strip - Steam's own injected
# early-access/refund boilerplate, not part of the user's actual writing.
BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds per-user (pt/en/union) mean-pooled text embeddings for step06's Phase 1 feature table."
    )
    parser.add_argument("--step02-dir", required=True, type=Path, help="Path to step02's output directory")
    parser.add_argument(
        "--profile-metadata", required=True, type=Path,
        help="Path to build_user_profile_metadata.py's output - defines which users to embed",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the per-user embeddings parquet to")
    parser.add_argument("--batch-size", type=int, default=256, help="SentenceTransformer encoding batch size (default 256)")
    return parser.parse_args()


def _save_checkpoint(checkpoint_dir: Path, sum_partials: dict, count_partials: dict, emb_cols: list) -> None:
    """Save intermediate aggregation state to disk so a resumed run can
    pick up where it left off without re-processing files."""
    for pop in POPULATIONS:
        if sum_partials[pop]:
            partial_sum = pd.concat(sum_partials[pop]).groupby(level=0).sum()
            partial_count = pd.concat(count_partials[pop]).groupby(level=0).sum()
            checkpoint_file = checkpoint_dir / f"checkpoint_{pop}.parquet"
            partial_sum.to_parquet(checkpoint_file)
            (checkpoint_dir / f"checkpoint_{pop}_count.npy").unlink(missing_ok=True)
            np.save(checkpoint_dir / f"checkpoint_{pop}_count.npy", partial_count.to_numpy())


def _load_checkpoint(checkpoint_dir: Path, emb_cols: list) -> tuple:
    """Load checkpoint if it exists, returning (sum_partials, count_partials)
    ready to continue accumulation. Returns (empty, empty) if no checkpoint."""
    sum_partials = {pop: [] for pop in POPULATIONS}
    count_partials = {pop: [] for pop in POPULATIONS}

    for pop in POPULATIONS:
        checkpoint_file = checkpoint_dir / f"checkpoint_{pop}.parquet"
        count_file = checkpoint_dir / f"checkpoint_{pop}_count.npy"
        if checkpoint_file.exists() and count_file.exists():
            partial_sum = pd.read_parquet(checkpoint_file)
            partial_count = pd.Series(np.load(count_file), index=partial_sum.index)
            sum_partials[pop] = [partial_sum]
            count_partials[pop] = [partial_count]
            info(f"[checkpoint] Loaded {pop} checkpoint: {len(partial_sum):,} user(s)")

    return sum_partials, count_partials


def light_clean(text: object) -> str:
    """Strips only boilerplate phrases and URLs - case, accents, and
    punctuation are preserved (see module docstring for why)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_matched_users(profile_metadata_path: Path) -> set:
    users = pd.read_parquet(profile_metadata_path, columns=["user_url"])["user_url"]
    info(f"[scope] {len(users):,} user(s) with matched profile metadata - embedding is restricted to these")
    return set(users)


def embed_and_accumulate(step02_dir: Path, matched_users: set, model, batch_size: int, checkpoint_dir: Path = None) -> dict:
    """Single sweep over step02's files: for each file, load+clean+filter
    its rows for `matched_users`, encode the surviving text once, then
    reduce to a per-user partial sum + count per population (pt/en/union).
    Partials are collected per population and combined once at the end.

    If checkpoint_dir is provided, periodically saves intermediate results
    so the run can resume from the last checkpoint if interrupted. Files
    already processed (tracked via .processed markers under checkpoint_dir)
    are skipped on resume."""
    files = sorted(step02_dir.rglob("*.parquet"))
    info(f"[embed] Encoding across {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step02_dir} (searched recursively).")

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        processed_marker_dir = checkpoint_dir / "processed"
        processed_marker_dir.mkdir(parents=True, exist_ok=True)
    else:
        processed_marker_dir = None

    already_processed = set()
    if processed_marker_dir:
        already_processed = {f.stem for f in processed_marker_dir.glob("*.done")}
        if already_processed:
            info(f"[checkpoint] Resuming: {len(already_processed):,} file(s) already processed")
            files = [f for f in files if f.stem not in already_processed]
            info(f"[checkpoint] {len(files):,} file(s) remaining to process")

    emb_cols = [f"e{i}" for i in range(EMBEDDING_DIM)]

    # Try to load checkpoint if resuming
    sum_partials, count_partials = _load_checkpoint(checkpoint_dir, emb_cols) if checkpoint_dir else ({pop: [] for pop in POPULATIONS}, {pop: [] for pop in POPULATIONS})

    n_reviews_encoded = 0
    users_seen = set()
    files_processed = 0
    CHECKPOINT_EVERY = 10  # Save partial results every N files

    bar = tqdm(files, desc="[embed]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=["user_url", "review_lang", "perspective_declared_language", "review_text"])
        df = df[df["user_url"].isin(matched_users)]
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        if df.empty:
            (processed_marker_dir / f"{f.stem}.done").touch() if processed_marker_dir else None
            bar.set_postfix(reviews=n_reviews_encoded, users=len(users_seen))
            continue

        df["review_text_clean"] = df["review_text"].apply(light_clean)
        df = df[df["review_text_clean"].str.len() > 0]
        if df.empty:
            (processed_marker_dir / f"{f.stem}.done").touch() if processed_marker_dir else None
            bar.set_postfix(reviews=n_reviews_encoded, users=len(users_seen))
            continue

        embeddings = model.encode(
            df["review_text_clean"].tolist(), batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        combined = pd.concat([df[["user_url", "review_lang"]], emb_df], axis=1)

        for pop in POPULATIONS:
            sub = combined if pop == UNION_KEY else combined[combined["review_lang"] == pop]
            if sub.empty:
                continue
            grouped = sub.groupby("user_url")[emb_cols]
            sum_partials[pop].append(grouped.sum())
            count_partials[pop].append(sub.groupby("user_url").size())

        n_reviews_encoded += len(combined)
        users_seen.update(combined["user_url"].unique())
        files_processed += 1

        # Mark file as done
        if processed_marker_dir:
            (processed_marker_dir / f"{f.stem}.done").touch()

        # Save checkpoint every N files
        if processed_marker_dir and files_processed % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_dir, sum_partials, count_partials, emb_cols)
            info(f"[checkpoint] Saved checkpoint after {files_processed} file(s)")

        bar.set_postfix(reviews=n_reviews_encoded, users=len(users_seen))

    info("[embed] Aggregating per-file partials (incremental)...")
    emb_cols = [f"e{i}" for i in range(EMBEDDING_DIM)]
    means = {}
    counts = {}

    for pop in POPULATIONS:
        if not sum_partials[pop]:
            # No file contributed any row for this population within scope
            means[pop] = pd.DataFrame(columns=emb_cols)
            counts[pop] = pd.Series(dtype="int64")
            info(f"[embed] [{pop}] 0 user(s) with an embeddable review")
            continue

        # Aggregate incrementally in small batches instead of all at once
        # to avoid memory spike from concat'ing gigantic DataFrames
        info(f"[embed] [{pop}] Aggregating {len(sum_partials[pop])} partial(s) incrementally...")
        batch_size = 10
        total_sum = None
        total_count = None

        for i in range(0, len(sum_partials[pop]), batch_size):
            batch_sum = sum_partials[pop][i : i + batch_size]
            batch_count = count_partials[pop][i : i + batch_size]

            # Aggregate this batch
            batch_sum_agg = pd.concat(batch_sum).groupby(level=0).sum()
            batch_count_agg = pd.concat(batch_count).groupby(level=0).sum()

            # Merge into running total
            if total_sum is None:
                total_sum = batch_sum_agg
                total_count = batch_count_agg
            else:
                # Combine: reindex + add to align indices
                total_sum = total_sum.add(batch_sum_agg.reindex(total_sum.index, fill_value=0), fill_value=0)
                total_count = total_count.add(batch_count_agg.reindex(total_count.index, fill_value=0), fill_value=0)
                # Also catch users that are in batch but not in total yet
                new_users = batch_sum_agg.index.difference(total_sum.index)
                if len(new_users):
                    total_sum = pd.concat([total_sum, batch_sum_agg.loc[new_users]])
                    total_count = pd.concat([total_count, batch_count_agg.loc[new_users]])

        means[pop] = total_sum.div(total_count, axis=0)
        counts[pop] = total_count
        info(f"[embed] [{pop}] {len(total_sum):,} user(s) with an embeddable review")

    return means, counts


def main():
    args = parse_args()

    matched_users = load_matched_users(args.profile_metadata)

    device = get_device()
    info(f"Loading {MODEL_NAME} on device={device}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)

    checkpoint_dir = args.output.parent / f".{args.output.stem}_checkpoint"
    means, counts = embed_and_accumulate(args.step02_dir, matched_users, model, args.batch_size, checkpoint_dir)

    info("Assembling final table...")
    all_users = sorted(set().union(*[means[pop].index for pop in POPULATIONS]))
    index = pd.Index(all_users, name="user_url")

    table = pd.DataFrame(index=index)
    for pop in POPULATIONS:
        pop_label = "union" if pop == UNION_KEY else pop
        emb = means[pop].reindex(index)
        emb.columns = [f"emb_{pop_label}_{i}" for i in range(EMBEDDING_DIM)]
        table = table.join(emb)
        table[f"n_{pop_label}_embedded"] = counts[pop].reindex(index, fill_value=0).astype("int64")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(args.output, index=False)
    info(f"Saved user text embeddings ({len(table)} users, {len(table.columns) + 1} columns) to: {args.output}")

    # Clean up checkpoint directory on successful completion
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        info(f"Cleaned up checkpoint directory: {checkpoint_dir}")

    save_summary(
        {
            "model_name": MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "device": device,
            "n_users_output": int(len(table)),
            "n_users_per_population": {
                ("union" if pop == UNION_KEY else pop): int(len(means[pop])) for pop in POPULATIONS
            },
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
