"""Recomputes per-user text embeddings for TOXIC users ONLY, excluding
their own toxic reviews - the "leave-toxic-out" control for step06's
classical-ML classifiers (01_phase2_nested_cv.py).

WHY: a toxic user's normal embedding (phase1_feature_engineering/pipeline/
04_build_user_text_embeddings.py) is the mean of ALL their agreement-matched
reviews, including whichever review(s) were toxic and caused their label in
the first place. A classifier using that embedding could just be detecting
the flagged review's own language ("this text is offensive" -> "this user
is toxic"), not a genuine behavioral signal beyond the content that already
defines the label - a circularity risk flagged when this project's step06
plan was first discussed. This script recomputes each toxic user's
embedding using ONLY their NON-toxic reviews (same union rule as
everywhere else in this project: toxic if perspective_score >= 0.7 OR
detoxify_score >= 0.9, invalid/sentinel scores excluded from consideration
entirely - never counted as either toxic or non-toxic). If
01_phase2_nested_cv.py, re-run with THESE embeddings substituted in for
positive users, still discriminates toxic from non-toxic users, that is
evidence of a real behavioral pattern beyond the flagged content; if
performance collapses toward chance, the original signal was mostly
circular.

SCOPE: only toxic users (75 pt / 1,225 en / 1,316 union, per
toxic_user_labels.parquet - a tiny fraction of the full corpus), NOT the
full population - negative (non-toxic) users' embeddings are untouched by
this control, since the question is only whether TOXIC users' own flagged
content was doing all the work.

USERS WITH ZERO NON-TOXIC REVIEWS: some toxic users may have EVERY review
in a language counted as toxic - there's nothing to average for them in
that population. They're reported (not silently dropped) via
n_<pop>_embedded == 0, and 01_phase2_nested_cv.py's leave-toxic-out mode
excludes them from that population's evaluation (there is no
"leave-toxic-out" embedding to substitute for someone whose entire
footprint IS toxic).

CLEANING/MODEL/AGGREGATION: identical to
04_build_user_text_embeddings.py - light cleaning (boilerplate+URL only),
paraphrase-multilingual-MiniLM-L12-v2, two-stage mean pooling - see that
script's docstring for the rationale. Only the review-selection filter
differs (excludes toxic reviews here, keeps everything there).

Output: one parquet file, one row per toxic user_url with >=1 embeddable
NON-toxic review in at least one population:
    user_url,
    emb_pt_0..emb_pt_383, n_pt_embedded,
    emb_en_0..emb_en_383, n_en_embedded,
    emb_union_0..emb_union_383, n_union_embedded
(same schema as 04_build_user_text_embeddings.py, so
01_phase2_nested_cv.py's --leave-toxic-out-embeddings flag can substitute
these in directly for the corresponding user_url rows.)

Usage:
    python 02_build_leave_toxic_out_embeddings.py \\
        --step02-dir ../../../../steam-data/step02-output \\
        --labels ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --output ../../../../steam-data/step06-output/phase2_classical_ml/leave_toxic_out_embeddings.parquet \\
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
PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9

BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recomputes toxic users' embeddings from their NON-toxic reviews only (leave-toxic-out control)."
    )
    parser.add_argument("--step02-dir", required=True, type=Path, help="Path to step02's output directory")
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="Path to build_toxic_user_labels.py's output - defines which users are toxic (is_toxic_pt/en/pt+en)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the leave-toxic-out embeddings parquet to")
    parser.add_argument("--batch-size", type=int, default=256, help="SentenceTransformer encoding batch size (default 256)")
    return parser.parse_args()


def light_clean(text: object) -> str:
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


def load_toxic_users(labels_path: Path) -> set:
    """Union of every user flagged toxic in ANY of the three populations -
    the full set this control needs embeddings recomputed for."""
    df = pd.read_parquet(labels_path, columns=["user_url", "is_toxic_pt", "is_toxic_en", "is_toxic_pt+en"])
    is_toxic_any = df["is_toxic_pt"].fillna(False) | df["is_toxic_en"].fillna(False) | df["is_toxic_pt+en"].fillna(False)
    users = set(df.loc[is_toxic_any, "user_url"])
    info(f"[scope] {len(users):,} user(s) toxic in at least one population - embedding recomputed for these only")
    return users


def embed_and_accumulate(step02_dir: Path, toxic_users: set, model, batch_size: int) -> tuple:
    """Sweeps step02's files, keeping only rows for toxic_users AND
    excluding individually-toxic reviews (the whole point of this
    control), then mean-pools the SURVIVING (non-toxic) reviews per user
    per population - same shuffle-free partial-sum pattern as
    04_build_user_text_embeddings.py."""
    files = sorted(step02_dir.rglob("*.parquet"))
    info(f"[embed] Encoding across {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step02_dir} (searched recursively).")

    emb_cols = [f"e{i}" for i in range(EMBEDDING_DIM)]
    sum_partials = {pop: [] for pop in POPULATIONS}
    count_partials = {pop: [] for pop in POPULATIONS}
    n_reviews_encoded = 0
    users_seen = set()
    n_excluded_toxic_review = 0

    bar = tqdm(files, desc="[embed]", unit="file")
    for f in bar:
        df = pd.read_parquet(
            f, columns=[
                "user_url", "review_lang", "perspective_declared_language",
                "review_text", "perspective_score", "detoxify_score",
            ],
        )
        df = df[df["user_url"].isin(toxic_users)]
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        if df.empty:
            bar.set_postfix(reviews=n_reviews_encoded, users=len(users_seen))
            continue

        # Same union toxicity rule as everywhere else in this project -
        # invalid/sentinel scores are excluded entirely (neither toxic nor
        # non-toxic), not silently treated as non-toxic.
        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]
        is_toxic_review = (df["perspective_score"] >= PERSPECTIVE_THRESHOLD) | (df["detoxify_score"] >= DETOXIFY_THRESHOLD)
        n_excluded_toxic_review += int(is_toxic_review.sum())
        df = df[~is_toxic_review]
        if df.empty:
            bar.set_postfix(reviews=n_reviews_encoded, users=len(users_seen))
            continue

        df["review_text_clean"] = df["review_text"].apply(light_clean)
        df = df[df["review_text_clean"].str.len() > 0]
        if df.empty:
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
        bar.set_postfix(reviews=n_reviews_encoded, users=len(users_seen))

    info(f"[embed] Excluded {n_excluded_toxic_review:,} individually-toxic review(s) across all toxic users (the control's whole point)")

    info("[embed] Aggregating per-file partials...")
    means = {}
    counts = {}
    for pop in POPULATIONS:
        if not sum_partials[pop]:
            means[pop] = pd.DataFrame(columns=emb_cols)
            counts[pop] = pd.Series(dtype="int64")
            info(f"[embed] [{pop}] 0 user(s) with an embeddable non-toxic review")
            continue
        total_sum = pd.concat(sum_partials[pop]).groupby(level=0).sum()
        total_count = pd.concat(count_partials[pop]).groupby(level=0).sum()
        means[pop] = total_sum.div(total_count, axis=0)
        counts[pop] = total_count
        info(f"[embed] [{pop}] {len(total_sum):,} toxic user(s) have >=1 non-toxic review to embed")

    return means, counts


def main():
    args = parse_args()

    toxic_users = load_toxic_users(args.labels)

    device = get_device()
    info(f"Loading {MODEL_NAME} on device={device}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)

    means, counts = embed_and_accumulate(args.step02_dir, toxic_users, model, args.batch_size)

    info("Assembling final table...")
    populated_users = set().union(*[means[pop].index for pop in POPULATIONS if len(means[pop])])
    all_users = sorted(populated_users)
    index = pd.Index(all_users, name="user_url")

    table = pd.DataFrame(index=index)
    for pop in POPULATIONS:
        pop_label = "union" if pop == UNION_KEY else pop
        emb = means[pop].reindex(index)
        emb.columns = [f"emb_{pop_label}_{i}" for i in range(EMBEDDING_DIM)]
        table = table.join(emb)
        table[f"n_{pop_label}_embedded"] = counts[pop].reindex(index, fill_value=0).astype("int64")

    # Users with zero non-toxic reviews in a population - report, don't hide.
    n_toxic_total = len(toxic_users)
    n_zero_pt = int((table["n_pt_embedded"] == 0).sum()) if len(table) else 0
    n_zero_en = int((table["n_en_embedded"] == 0).sum()) if len(table) else 0
    n_zero_union = int((table["n_union_embedded"] == 0).sum()) if len(table) else 0
    n_no_embedding_at_all = n_toxic_total - len(table)
    info(
        f"{n_no_embedding_at_all:,} of {n_toxic_total:,} toxic user(s) have NO non-toxic review in ANY population "
        f"(100% of their agreement-matched content is toxic) - entirely absent from this output"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(args.output, index=False)
    info(f"Saved leave-toxic-out embeddings ({len(table)} user(s), {len(table.columns) + 1} columns) to: {args.output}")

    save_summary(
        {
            "model_name": MODEL_NAME,
            "n_toxic_users_total": n_toxic_total,
            "n_users_output": int(len(table)),
            "n_no_nontoxic_review_at_all": n_no_embedding_at_all,
            "n_zero_nontoxic_reviews_per_population": {"pt": n_zero_pt, "en": n_zero_en, "union": n_zero_union},
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
