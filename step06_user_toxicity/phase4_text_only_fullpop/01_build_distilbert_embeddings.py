"""Phase 4, Step 1: per-user text embeddings for the FULL eligible
population of ONE language, plus that language's labels, written as a slim
table that 01_phase2_nested_cv.py --text-only can consume directly.

WHY THIS EXISTS (experiment B of the text-only ablation): Phase 1's
embeddings were deliberately scoped to the 42.31% of eligible users whose
Steam profile could be matched (see 03_build_user_profile_metadata.py -
the other 57.69% are unmatchable vanity URLs or accounts missing from the
profile dump). That was the right call while profile metadata was a
feature. Experiment A then measured what profile metadata actually
contributes and found it undetectable (pt: AUC-PR 0.1362 with profile vs
0.1343 without; paired-bootstrap 95% CI on the difference [-0.0171,
+0.0213], comfortably spanning zero) - unsurprising once you notice 8 of
the 12 profile columns are majority-null, so median imputation flattens
them to near-constants. Dropping profile features therefore costs nothing
measurable and lifts the population cap, roughly DOUBLING the positive
class: pt 75 -> 160, en 1,225 -> 2,479, union 1,316 -> 2,668. For a class
this rare, that is the single biggest available gain in statistical power.

ONE POPULATION PER RUN, deliberately: a combined three-population table at
full scale would be ~21GB, awkward to ship to a cluster and pointless
since a given LOO run reads exactly one population's embedding block. Each
run here produces one self-contained, directly-runnable slim table.

REUSES PHASE 1'S EMBEDDINGS (--existing-embeddings): the ~2.37M
profile-matched users already have vectors computed under the identical
model and cleaning, so re-encoding them would waste ~42% of the GPU
budget. Users found there are dropped from the sweep scope and their rows
are copied into the output verbatim. Passing a file computed under
DIFFERENT settings would silently mix incompatible vectors - so only pass
Phase 1's own output, or omit the flag to compute everything fresh.

MEMORY: accumulation is a preallocated (n_users x 384) float32 sum array
plus an int64 count array, scatter-added per file. Phase 1 instead
concatenated per-file pandas partials and grouped once, which OOM-killed
that run at 2.37M users and needed an incremental-batching rescue; at
5.1M users that approach is not viable at all. Per-file groupby already
yields unique users, so the scatter-add is a plain `sums[idx] += vals`
(no np.add.at needed) and peak memory stays at the preallocated arrays
(~7.9GB for en) plus one file. Output is streamed to parquet in row-group
chunks rather than materialised as one DataFrame, for the same reason.

CLEANING/MODEL/POOLING: identical to phase1_feature_engineering/pipeline/
04_build_user_text_embeddings.py - light cleaning (boilerplate + URLs
only, case/accents/punctuation preserved), paraphrase-multilingual-
MiniLM-L12-v2, mean pooling over a user's reviews. The constants below are
duplicated rather than imported: this folder follows the project's
convention of being independently copyable to the cluster, and that module
cannot be imported normally anyway (its filename starts with a digit).
CHANGING EITHER COPY WITHOUT THE OTHER INVALIDATES --existing-embeddings.

Output (one parquet, one row per eligible user of --population):
    user_url, eligible_<pop>, is_toxic_<pop>,
    emb_<pop>_0..emb_<pop>_383, n_<pop>_embedded
Users with no embeddable review (all text empty after cleaning) keep NaN
embeddings and n_<pop>_embedded == 0; they are reported, not dropped, so
the downstream run decides what to do with them.

Usage:
    python 01_build_fullpop_embeddings.py \\
        --step02-dir ../../../steam-data/step02-output \\
        --labels ../../../steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --population pt \\
        --existing-embeddings ../../../steam-data/step06-output/phase1_feature_engineering/pipeline/user_text_embeddings.parquet \\
        --output ../../../steam-data/step06-output/phase4_text_only_fullpop/fullpop_pt.parquet
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from pipeline_utils import info, save_summary

# --- MUST stay in sync with phase1's 04_build_user_text_embeddings.py ---
MODEL_NAME = "distilbert-base-multilingual-cased"
EMBEDDING_DIM = 768
BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]
# -----------------------------------------------------------------------

WRITE_CHUNK_ROWS = 250_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-population per-user text embeddings + labels for one language (text-only ablation, experiment B)."
    )
    parser.add_argument("--step02-dir", required=True, type=Path, help="Path to step02's output directory")
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="Path to build_toxic_user_labels.py's output - defines the eligible population AND supplies the labels",
    )
    parser.add_argument("--population", required=True, choices=["pt", "en", "union"], help="Which population to build")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the slim per-population table to")
    parser.add_argument(
        "--existing-embeddings", type=Path, default=None,
        help="Optional: Phase 1's user_text_embeddings.parquet - its users are reused verbatim instead of "
        "re-encoded (see module docstring; only valid if computed under the same model/cleaning)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="SentenceTransformer encoding batch size (default 256)")
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Save the accumulator every N step02 files so an interrupted run resumes instead of restarting "
        "(default 50). Each save writes the full sums array (~7.9GB for en), so very small values cost real I/O.",
    )
    return parser.parse_args()


def _checkpoint_signature(population: str, n_rows: int, reused: int) -> dict:
    """What a checkpoint must agree with to be safe to resume from. Phase 3
    hit a real incident where a checkpoint was resumed under different
    hyperparameters and silently mixed incompatible progress - here a
    mismatch restarts the sweep instead."""
    return {"population": population, "n_rows": int(n_rows), "n_reused": int(reused)}


def save_sweep_checkpoint(checkpoint_dir: Path, sums: np.ndarray, counts: np.ndarray,
                          processed: set, signature: dict) -> None:
    """Written to temp files then atomically renamed: a job killed mid-write
    would otherwise leave a truncated array that resumes as silent garbage."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("sums", sums), ("counts", counts)):
        # The temp name has to END in .npy: np.save appends the extension
        # itself when it's missing, so "sums.npy.tmp" would be written as
        # "sums.npy.tmp.npy" and the rename below would fail on a file that
        # was never created.
        tmp = checkpoint_dir / f"{name}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, checkpoint_dir / f"{name}.npy")
    tmp = checkpoint_dir / "state.json.tmp"
    tmp.write_text(json.dumps({"signature": signature, "processed": sorted(processed)}))
    os.replace(tmp, checkpoint_dir / "state.json")


def load_sweep_checkpoint(checkpoint_dir: Path, signature: dict) -> tuple:
    """Returns (sums, counts, processed_files) or (None, None, set()) if
    there's no checkpoint or it doesn't match `signature`."""
    state_file = checkpoint_dir / "state.json"
    sums_file, counts_file = checkpoint_dir / "sums.npy", checkpoint_dir / "counts.npy"
    if not (state_file.exists() and sums_file.exists() and counts_file.exists()):
        return None, None, set()

    state = json.loads(state_file.read_text())
    if state.get("signature") != signature:
        info(
            f"[checkpoint] Found a checkpoint for {state.get('signature')} but this run is {signature} - "
            "ignoring it and starting the sweep from scratch"
        )
        return None, None, set()

    sums, counts = np.load(sums_file), np.load(counts_file)
    processed = set(state.get("processed", []))
    info(f"[checkpoint] Resuming: {len(processed)} file(s) already swept")
    return sums, counts, processed


def light_clean(text: object) -> str:
    """Strips only boilerplate phrases and URLs - case, accents and
    punctuation are preserved (see module docstring)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def get_device() -> str:
    """cuda (cluster) > mps (Apple Silicon) > cpu. Phase 1's equivalent
    only ever checked cuda, so its run fell back to cpu on this project's
    Mac - checking mps here makes a local run of this script materially
    faster without changing any numerics."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def load_population_labels(labels_path: Path, population: str) -> pd.DataFrame:
    """Eligible users of `population`, with their label. Returns columns
    user_url / eligible_<pop> / is_toxic_<pop>, in a stable sorted order
    (the row order of the output table, so a resumed/compared run lines
    up)."""
    label_suffix = "pt+en" if population == "union" else population
    cols = ["user_url", f"eligible_{label_suffix}", f"is_toxic_{label_suffix}"]
    df = pd.read_parquet(labels_path, columns=cols)
    df = df[df[f"eligible_{label_suffix}"].fillna(False)].copy()
    df[f"is_toxic_{label_suffix}"] = df[f"is_toxic_{label_suffix}"].fillna(False).astype(bool)
    df = df.sort_values("user_url").reset_index(drop=True)
    info(f"[{population}] {len(df):,} eligible user(s), {int(df[f'is_toxic_{label_suffix}'].sum()):,} positive (toxic)")
    return df


def load_existing(existing_path: Path, population: str, wanted: pd.Index) -> pd.DataFrame:
    """Rows of Phase 1's embeddings that are (a) in scope and (b) actually
    have this population's vector (n_<pop>_embedded > 0 - Phase 1 leaves
    NaN for a user with no review in that language).

    Read in row-group batches rather than one pd.read_parquet call: Phase
    1's table is ~9GB on disk and a single read materialises every selected
    row before any filtering, whereas filtering per batch keeps only the
    in-scope rows and gives the read a progress bar instead of a silent
    multi-minute stall."""
    emb_cols = [f"emb_{population}_{i}" for i in range(EMBEDDING_DIM)]
    count_col = f"n_{population}_embedded"
    wanted_set = set(wanted)

    pf = pq.ParquetFile(existing_path)
    kept = []
    n_scanned = 0
    bar = tqdm(total=pf.metadata.num_rows, desc="[reuse]", unit="row")
    for batch in pf.iter_batches(batch_size=100_000, columns=["user_url"] + emb_cols + [count_col]):
        df = batch.to_pandas()
        n_scanned += len(df)
        df = df[df["user_url"].isin(wanted_set)]
        df = df[df[count_col].fillna(0) > 0]
        if len(df):
            kept.append(df)
        bar.update(len(batch))
        bar.set_postfix(kept=sum(len(k) for k in kept))
    bar.close()

    result = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["user_url"] + emb_cols + [count_col])
    info(f"[reuse] {len(result):,} of {n_scanned:,} user(s) reused from {existing_path.name} - skipped in the sweep")
    return result.set_index("user_url")


def sweep_and_accumulate(step02_dir: Path, users_index: dict, n_rows: int,
                          population: str, model, batch_size: int,
                          checkpoint_dir: Path = None, checkpoint_every: int = 50,
                          signature: dict = None) -> tuple:
    """One pass over step02: encode each in-scope review once, scatter-add
    its vector into the per-user running sum.

    `users_index` maps user_url -> that user's ROW POSITION IN THE OUTPUT
    TABLE (not a position within the swept subset), and the accumulators
    are allocated at full output length (`n_rows`) to match. Sizing them to
    the subset instead would make every index out of range or, worse,
    silently land on the wrong row - and it would force a re-expansion step
    afterwards that this avoids entirely."""
    if population != "union" and (step02_dir / f"review_lang={population}").is_dir():
        files = sorted((step02_dir / f"review_lang={population}").rglob("*.parquet"))
    else:
        files = sorted(step02_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step02_dir} (searched recursively).")
    info(f"[embed] Encoding across {len(files)} file(s) on scope of {len(users_index):,} user(s)...")

    sums = counts = None
    processed = set()
    if checkpoint_dir is not None:
        sums, counts, processed = load_sweep_checkpoint(checkpoint_dir, signature)
    if sums is None:
        sums = np.zeros((n_rows, EMBEDDING_DIM), dtype=np.float32)
        counts = np.zeros(n_rows, dtype=np.int64)

    scope = set(users_index)
    n_encoded = 0
    n_since_checkpoint = 0
    bar = tqdm(files, desc=f"[{population}]", unit="file")
    for f in bar:
        # Keyed by relative POSIX path, not bare filename (since review_lang=pt/
        # and review_lang=en/ both contain part.0.parquet, part.1.parquet, etc.),
        # while still surviving a resume from a different root directory.
        rel_key = f.relative_to(step02_dir).as_posix()
        if rel_key in processed:
            bar.set_postfix(encoded=n_encoded, skipped=len(processed))
            continue

        df = pd.read_parquet(f, columns=["user_url", "review_lang", "perspective_declared_language", "review_text"])
        df = df[df["user_url"].isin(scope)]
        # Same agreement mask every other consumer of step02 re-applies.
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        if population != "union":
            df = df[df["review_lang"] == population]

        if not df.empty:
            df = df.assign(review_text_clean=df["review_text"].map(light_clean))
            df = df[df["review_text_clean"].str.len() > 0]

        # A file can legitimately contribute nothing (no in-scope user, or
        # every text empty after cleaning) - it still counts as swept, or a
        # resume would redo it forever.
        if not df.empty:
            vectors = model.encode(
                df["review_text_clean"].tolist(), batch_size=batch_size,
                show_progress_bar=False, convert_to_numpy=True,
            ).astype(np.float32)

            # Reduce THIS FILE to one row per user first (cheap, file-sized),
            # so the scatter below has unique indices and a plain += is right.
            per_user = pd.DataFrame(vectors).groupby(df["user_url"].to_numpy())
            file_sums = per_user.sum()
            file_counts = per_user.size()
            idx = np.fromiter((users_index[u] for u in file_sums.index), dtype=np.int64, count=len(file_sums))
            sums[idx] += file_sums.to_numpy(dtype=np.float32)
            counts[idx] += file_counts.to_numpy(dtype=np.int64)
            n_encoded += len(df)

        processed.add(rel_key)
        n_since_checkpoint += 1
        if checkpoint_dir is not None and n_since_checkpoint >= checkpoint_every:
            save_sweep_checkpoint(checkpoint_dir, sums, counts, processed, signature)
            n_since_checkpoint = 0
            bar.write(f"[checkpoint] saved after {len(processed)}/{len(files)} file(s)")
        bar.set_postfix(encoded=n_encoded, done=len(processed))

    if checkpoint_dir is not None and n_since_checkpoint:
        save_sweep_checkpoint(checkpoint_dir, sums, counts, processed, signature)

    info(f"[embed] Encoded {n_encoded:,} review(s); {int((counts > 0).sum()):,} user(s) got at least one vector")
    return sums, counts


def write_table(output: Path, labels: pd.DataFrame, population: str,
                sums: np.ndarray, counts: np.ndarray, existing: pd.DataFrame) -> int:
    """Streams the output parquet in row-group chunks (never materialising
    the whole ~8GB table), filling each user's row from either the sweep's
    accumulator or the reused Phase 1 rows."""
    label_suffix = "pt+en" if population == "union" else population
    emb_cols = [f"emb_{population}_{i}" for i in range(EMBEDDING_DIM)]
    count_col = f"n_{population}_embedded"

    existing_pos = {}
    if existing is not None and len(existing):
        existing_pos = {u: i for i, u in enumerate(existing.index)}
        existing_emb = existing[[f"emb_{population}_{i}" for i in range(EMBEDDING_DIM)]].to_numpy(dtype=np.float32)
        existing_cnt = existing[count_col].to_numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_no_embedding = 0
    try:
        for start in tqdm(range(0, len(labels), WRITE_CHUNK_ROWS), desc="[write]", unit="chunk"):
            block = labels.iloc[start:start + WRITE_CHUNK_ROWS]
            n = len(block)
            emb = np.full((n, EMBEDDING_DIM), np.nan, dtype=np.float32)
            cnt = np.zeros(n, dtype=np.int64)

            # Vectorised, not a per-row Python loop: at 5.6M rows the loop
            # version dominates runtime for work numpy does in one shot.
            from_existing = block["user_url"].map(existing_pos).to_numpy() if existing_pos else np.full(n, np.nan)
            has_existing = ~pd.isna(from_existing)
            if has_existing.any():
                pos = from_existing[has_existing].astype(np.int64)
                emb[has_existing] = existing_emb[pos]
                cnt[has_existing] = existing_cnt[pos]

            fresh = ~has_existing
            abs_idx = np.arange(start, start + n)[fresh]
            got_vector = counts[abs_idx] > 0
            target_rows = np.flatnonzero(fresh)[got_vector]
            source_rows = abs_idx[got_vector]
            emb[target_rows] = sums[source_rows] / counts[source_rows][:, None]
            cnt[target_rows] = counts[source_rows]

            n_no_embedding += int((cnt == 0).sum())

            out = pd.DataFrame({
                "user_url": block["user_url"].to_numpy(),
                f"eligible_{label_suffix}": block[f"eligible_{label_suffix}"].to_numpy(),
                f"is_toxic_{label_suffix}": block[f"is_toxic_{label_suffix}"].to_numpy(),
            })
            out = pd.concat([out, pd.DataFrame(emb, columns=emb_cols, index=out.index)], axis=1)
            out[count_col] = cnt

            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return n_no_embedding


def main():
    args = parse_args()

    labels = load_population_labels(args.labels, args.population)
    all_users = pd.Index(labels["user_url"])

    existing = None
    if args.existing_embeddings:
        existing = load_existing(args.existing_embeddings, args.population, all_users)

    to_sweep = all_users.difference(existing.index) if existing is not None else all_users
    info(f"[scope] {len(to_sweep):,} user(s) still need encoding (of {len(all_users):,} eligible)")

    # Positions are into `labels` (the output's row order), so the sweep can
    # scatter straight into the row each user will occupy.
    # The set is built ONCE here, not inside the comprehension's condition:
    # `if u in set(to_sweep)` would rebuild a 0.5-5.6M-element set on every
    # iteration, making this O(n^2) - measured as "never finishes" on pt.
    sweep_set = set(to_sweep)
    users_index = {
        u: i
        for i, u in enumerate(tqdm(labels["user_url"], desc="[scope]", unit="user"))
        if u in sweep_set
    }

    checkpoint_dir = args.output.parent / f".{args.output.stem}_checkpoint"
    signature = _checkpoint_signature(args.population, len(labels), len(existing) if existing is not None else 0)

    if users_index:
        device = get_device()
        info(f"Loading {MODEL_NAME} on device={device}...")
        # Frozen base DistilBERT + mean-pooling (there is no sentence-transformer
        # tuned for distilbert-base-multilingual-cased, so we wrap the raw encoder
        # with a mean-pooling head - the same aggregation MiniLM uses internally).
        from sentence_transformers import SentenceTransformer, models
        _word = models.Transformer(MODEL_NAME, max_seq_length=128)
        _pool = models.Pooling(_word.get_word_embedding_dimension(), pooling_mode="mean")
        model = SentenceTransformer(modules=[_word, _pool], device=device)
        sums, counts = sweep_and_accumulate(
            args.step02_dir, users_index, len(labels), args.population, model, args.batch_size,
            checkpoint_dir=checkpoint_dir, checkpoint_every=args.checkpoint_every, signature=signature,
        )
    else:
        info("[embed] Nothing left to encode - every eligible user came from --existing-embeddings")
        device = "n/a"
        sums = np.zeros((len(labels), EMBEDDING_DIM), dtype=np.float32)
        counts = np.zeros(len(labels), dtype=np.int64)

    n_no_embedding = write_table(args.output, labels, args.population, sums, counts, existing)
    info(f"Saved slim table ({len(labels):,} users) to: {args.output}")

    # Only after the output is safely written - a checkpoint removed any
    # earlier would strand a run that then failed during the write.
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        info(f"Cleaned up checkpoint directory: {checkpoint_dir}")
    if n_no_embedding:
        info(f"{n_no_embedding:,} user(s) have NO embeddable review (NaN embedding, n_embedded=0) - kept, not dropped")

    label_suffix = "pt+en" if args.population == "union" else args.population
    save_summary(
        {
            "population": args.population,
            "model_name": MODEL_NAME,
            "device": device,
            "n_eligible": int(len(labels)),
            "n_positive": int(labels[f"is_toxic_{label_suffix}"].sum()),
            "n_reused_from_existing": int(len(existing)) if existing is not None else 0,
            "n_encoded_fresh": int(len(users_index)),
            "n_without_embedding": int(n_no_embedding),
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
