"""Phase 3, Step 1: gathers per-review text for step06's DistilBERT
Multiple Instance Learning (MIL) classifier - Track B (per-user), where
each user is a "bag" of review "instances" fine-tuned end-to-end (see this
project's step06 planning conversation for why MIL, not text
concatenation: a user's reviews vary from 1 to ~12,780, so concatenating
and truncating to a fixed token budget would let a handful of a prolific
user's earliest reviews stand in for their entire footprint - MIL
processes every review and only pools the resulting vectors, so no
review is silently dropped by a token limit).

SCOPE: the SAME profile-matched population as phase1_feature_table.parquet
(2,375,760 users) - not the full eligible population - so Track B's
results are directly comparable to Phase 2's classical-ML results (same
users, same labels; only the modeling approach differs: frozen embeddings
+ profile features there, fine-tuned text-only here).

WHY TEXT-ONLY (no profile features, unlike Phase 2): Track B isolates the
question "how good is a fine-tuned text encoder alone", as a distinct
comparison point from Phase 2's "frozen text embeddings + profile
features combined" - mixing profile features back in here would muddy
that comparison.

CLEANING: same light cleaning as phase1_feature_engineering/pipeline/
04_build_user_text_embeddings.py (boilerplate + URL stripping only,
case/accents/punctuation preserved) - for the same reason: no bag-of-words
consumer here, and DistilBERT (like the sentence-transformer used in
Phase 1/2) can make use of tone/emphasis signals that aggressive cleaning
would destroy.

PER-REVIEW TOXICITY FLAG: this output also carries is_toxic_review (the
same union rule as everywhere else in this project: perspective_score >=
0.7 OR detoxify_score >= 0.9, invalid/sentinel scores excluded) - needed
by 02_train_mil_distilbert.py's --leave-toxic-out mode, which drops a
positive user's individually-toxic reviews from their bag before pooling,
mirroring Phase 2's leave-toxic-out control for this MIL architecture.

Output: one parquet row per surviving (non-empty-after-cleaning,
agreement-matched) review:
    user_url, review_lang, review_text_clean, is_toxic_review

Usage:
    python 01_prepare_review_texts.py \\
        --step02-dir ../../../../steam-data/step02-output \\
        --feature-table ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/phase1_feature_table.parquet \\
        --output ../../../../steam-data/step06-output/phase3_distilbert/review_texts.parquet
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline_utils import info, save_summary

PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9

BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gathers per-review cleaned text for step06 Phase 3's MIL DistilBERT classifier."
    )
    parser.add_argument("--step02-dir", required=True, type=Path, help="Path to step02's output directory")
    parser.add_argument(
        "--feature-table", required=True, type=Path,
        help="Path to phase1_feature_table.parquet - defines the population in scope (profile-matched users)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the per-review text parquet to")
    return parser.parse_args()


def light_clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_scoped_users(feature_table_path: Path) -> set:
    users = pd.read_parquet(feature_table_path, columns=["user_url"])["user_url"]
    info(f"[scope] {len(users):,} user(s) from phase1_feature_table.parquet - text is gathered for these only")
    return set(users)


def sweep_reviews(step02_dir: Path, scoped_users: set) -> pd.DataFrame:
    files = sorted(step02_dir.rglob("*.parquet"))
    info(f"[reviews] Sweeping across {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step02_dir} (searched recursively).")

    columns = [
        "user_url", "review_lang", "perspective_declared_language",
        "review_text", "perspective_score", "detoxify_score",
    ]
    partials = []
    n_reviews_kept = 0
    users_seen = set()

    bar = tqdm(files, desc="[reviews]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=columns)
        df = df[df["user_url"].isin(scoped_users)]
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        if df.empty:
            bar.set_postfix(reviews=n_reviews_kept, users=len(users_seen))
            continue

        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]
        if df.empty:
            bar.set_postfix(reviews=n_reviews_kept, users=len(users_seen))
            continue

        df["is_toxic_review"] = (df["perspective_score"] >= PERSPECTIVE_THRESHOLD) | (df["detoxify_score"] >= DETOXIFY_THRESHOLD)
        df["review_text_clean"] = df["review_text"].apply(light_clean)
        df = df[df["review_text_clean"].str.len() > 0]
        if df.empty:
            bar.set_postfix(reviews=n_reviews_kept, users=len(users_seen))
            continue

        kept = df[["user_url", "review_lang", "review_text_clean", "is_toxic_review"]]
        partials.append(kept)
        n_reviews_kept += len(kept)
        users_seen.update(kept["user_url"].unique())
        bar.set_postfix(reviews=n_reviews_kept, users=len(users_seen))

    info("[reviews] Concatenating partials...")
    result = pd.concat(partials, ignore_index=True)
    info(f"[reviews] {len(result):,} review(s) kept, {result['user_url'].nunique():,} unique user(s)")
    return result


def main():
    args = parse_args()

    scoped_users = load_scoped_users(args.feature_table)
    reviews = sweep_reviews(args.step02_dir, scoped_users)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_parquet(args.output, index=False)
    info(f"Saved per-review text ({len(reviews)} rows) to: {args.output}")

    n_toxic_review = int(reviews["is_toxic_review"].sum())
    save_summary(
        {
            "n_reviews": int(len(reviews)),
            "n_users": int(reviews["user_url"].nunique()),
            "n_toxic_reviews": n_toxic_review,
            "pct_toxic_reviews": round(100 * n_toxic_review / len(reviews), 4),
            "n_reviews_by_lang": reviews["review_lang"].value_counts().to_dict(),
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
