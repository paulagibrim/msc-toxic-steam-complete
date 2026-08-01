"""Cross-lingual transfer: train the frozen-encoder + classical-head
toxicity classifier on ONE language's FULL population, evaluate on the
OTHER language's FULL population - no k-fold, no held-out split inside
either population, since the two languages are already disjoint train/test
sets by construction (a user is virtually never coverage-eligible in both
pt and en at once, given the >=90% single-language coverage filter).

WHY THIS IS A MEANINGFUL TEST: DistilBERT-multilingual (like MiniLM-
multilingual) projects pt and en text into a SHARED embedding space. If the
"toxicity direction" pooling_ablation.py's classifiers learn in that space
is language-agnostic (a property of tone/aggression the encoder captures
regardless of language) rather than language-specific lexical cues, a
classifier trained on one language's users should transfer to the other's
with a real, above-chance ROC-AUC - not necessarily as strong as within-
language CV (some drop is expected), but well above 0.5.

A REAL CAVEAT, not a bug: is_toxic_<lang> is defined by a ONE-SIDED
BINOMIAL TEST AGAINST EACH POPULATION'S OWN BASELINE RATE (see
phase1_feature_engineering/pipeline/02_build_toxic_user_labels.py). So
"toxic in en" and "toxic in pt" are both relative labels, not calibrated to
an identical absolute standard - this experiment tests whether the LEARNED
DECISION BOUNDARY generalizes across languages, not whether the two
languages' labeling criteria are identical (they aren't, by design).

Reuses pooling_ablation.py's feature building, model fitting and pooling
assembly UNCHANGED (same module, imported directly - both files live in
this folder) - only the train/test split logic differs: instead of
5-fold CV within one language, SOURCE language = train (with the same
hard-negative-style downsampling pooling_ablation.py uses), TARGET
language = test, evaluated whole, at its own real prevalence.

Circular vs honest (leave-toxic-out) is still supported and applied
INDEPENDENTLY to source and target: it drops a positive user's
individually-toxic reviews before pooling, in whichever language they're
being pooled in.

Top-k pooling's prototype is built ONLY from the SOURCE'S positive users'
reviews (never the target's - that would leak the target label into
scoring), then applied to score BOTH source's and target's reviews - the
prototype is who travels across languages, not any label information.

Usage:
    python cross_lingual_transfer.py \\
        --embeddings-pt ../../../steam-data/step07-output/review_embeddings_distilbert_pt.npy \\
        --embeddings-en ../../../steam-data/step07-output/review_embeddings_distilbert_en.npy \\
        --review-dataset-pt ../../../steam-data/step07-output/review_dataset_pt.parquet \\
        --review-dataset-en ../../../steam-data/step07-output/review_dataset_en.parquet \\
        --labels ../../../steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet \\
        --output ../../../steam-data/step06-output/phase3_distilbert/cross_lingual_transfer.json
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from pooling_ablation import (
    EMBEDDING_DIM, HAS_XGB, RANDOM_STATE,
    build_user_features, assemble, oof_scores,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-pt", required=True, type=Path, help="review_embeddings_<encoder>_pt.npy")
    p.add_argument("--embeddings-en", required=True, type=Path, help="review_embeddings_<encoder>_en.npy")
    p.add_argument("--review-dataset-pt", required=True, type=Path)
    p.add_argument("--review-dataset-en", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path,
                   help="toxic_user_labels.parquet - has eligible_pt/is_toxic_pt AND eligible_en/is_toxic_en, "
                        "so one file covers both languages")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--poolings", default="mean,mean+max,mean+max+topk")
    p.add_argument("--models", default="logreg,svm_linear,svm_rbf,xgboost")
    p.add_argument("--neg-ratio", type=int, default=30,
                   help="downsample the SOURCE (train) negatives only; the TARGET (test) always stays at "
                        "its own full, real prevalence. 0 = use every source negative (slow at scale).")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def lang_args(args, lang: str) -> SimpleNamespace:
    """build_user_features() only reads a few attributes off `args` - a
    plain namespace stands in for argparse's Namespace so the SAME function
    can be called once per language without pooling_ablation.py's own
    --language/--embeddings/... CLI flags getting in the way."""
    return SimpleNamespace(
        language=lang,
        embeddings=args.embeddings_pt if lang == "pt" else args.embeddings_en,
        review_dataset=args.review_dataset_pt if lang == "pt" else args.review_dataset_en,
        labels=args.labels,
        with_metadata=False,
        feature_table=None,
    )


def build_prototype(feats, pos_users_global):
    """Mean of the given users' review embeddings, L2-normalized - the first
    half of pooling_ablation.topk_features_for_fold, split out so the SAME
    prototype (built once, from SOURCE positives only) can be applied to
    both source's and target's reviews."""
    rev_emb, rev_user = feats["rev_emb"], feats["rev_user"]
    is_pos_review = np.isin(rev_user, pos_users_global)
    if is_pos_review.sum() == 0:
        return None
    proto = rev_emb[is_pos_review].mean(0)
    return proto / (np.linalg.norm(proto) + 1e-8)


def score_topk(feats, proto, k):
    """Second half of topk_features_for_fold: scores THIS feats' own
    reviews against an externally-supplied prototype and mean-pools each
    user's top-k. Works unchanged whether `feats` is the source's or the
    target's - the prototype is what was built once from source positives."""
    nU = len(feats["yuser"])
    rev_emb, rev_user = feats["rev_emb"], feats["rev_user"]
    if proto is None or len(rev_user) == 0:
        return np.zeros((nU, EMBEDDING_DIM), np.float32)
    norms = np.linalg.norm(rev_emb, axis=1, keepdims=True) + 1e-8
    scores = (rev_emb / norms) @ proto
    order = np.lexsort((-scores, rev_user))
    su = rev_user[order]
    se = rev_emb[order]
    change = np.ones(len(su), bool)
    change[1:] = su[1:] != su[:-1]
    group_start = np.flatnonzero(change)
    rank = np.arange(len(su)) - np.repeat(group_start, np.diff(np.append(group_start, len(su))))
    keepmask = rank < k
    sel_u, sel_e = su[keepmask], se[keepmask]
    out = np.zeros((nU, EMBEDDING_DIM), np.float32)
    cnt = np.zeros(nU, np.int64)
    np.add.at(out, sel_u, sel_e)
    np.add.at(cnt, sel_u, 1)
    return out / np.maximum(cnt[:, None], 1)


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    poolings = [p.strip() for p in args.poolings.split(",") if p.strip()]
    need_topk = any("topk" in p for p in poolings)

    results = {"xgboost_installed": HAS_XGB, "topk": args.topk, "directions": {}}
    if args.output.exists():
        # Resumable at (direction, condition) granularity - same pattern as
        # pooling_ablation.py: a crash partway through doesn't throw away
        # whatever combinations already finished and were checkpointed.
        prior = json.loads(args.output.read_text())
        results["directions"] = prior.get("directions", {})
        if results["directions"]:
            done = {d: list(c) for d, c in results["directions"].items()}
            print(f"[resume] {args.output} already has: {done}", flush=True)

    for src, tgt in [("pt", "en"), ("en", "pt")]:
        dir_key = f"{src}_to_{tgt}"
        results["directions"].setdefault(dir_key, {})

        for cond, excl in [("circular", False), ("leave_toxic_out", True)]:
            if cond in results["directions"][dir_key]:
                print(f"\n### {dir_key}/{cond} - already done, skipping ###", flush=True)
                continue
            t0 = time.time()

            src_feats = build_user_features(lang_args(args, src), exclude_toxic_from_pos=excl, need_topk=need_topk)
            tgt_feats = build_user_features(lang_args(args, tgt), exclude_toxic_from_pos=excl, need_topk=need_topk)

            src_keep, tgt_keep = src_feats["keep"], tgt_feats["keep"]
            y_src = src_feats["yuser"][src_keep]
            y_tgt = tgt_feats["yuser"][tgt_keep]
            print(f"\n### {dir_key}/{cond} | train({src}) N={src_keep.sum():,} pos={int(y_src.sum())} "
                  f"| test({tgt}) N={tgt_keep.sum():,} pos={int(y_tgt.sum())} ###", flush=True)

            # Prototype built ONLY from source positives (global indices, same
            # convention as pooling_ablation.topk_features_for_fold), then
            # applied to score both source's own reviews and the target's.
            src_pos_global = np.flatnonzero(src_keep)[y_src == 1]
            proto = build_prototype(src_feats, src_pos_global) if need_topk else None
            zeros_full = lambda feats: np.zeros((len(feats["yuser"]), EMBEDDING_DIM), np.float32)
            topk_src_full = score_topk(src_feats, proto, args.topk) if need_topk else zeros_full(src_feats)
            topk_tgt_full = score_topk(tgt_feats, proto, args.topk) if need_topk else zeros_full(tgt_feats)

            src_k = dict(mean=src_feats["mean"][src_keep], maxes=src_feats["maxes"][src_keep], meta=None)
            tgt_k = dict(mean=tgt_feats["mean"][tgt_keep], maxes=tgt_feats["maxes"][tgt_keep], meta=None)
            topk_src, topk_tgt = topk_src_full[src_keep], topk_tgt_full[tgt_keep]

            # Hard-negative-style downsampling of the SOURCE (train) only -
            # mirrors pooling_ablation.py's per-fold downsampling, but here
            # there is only one "fold" (the whole source population) and the
            # target (test) always stays at its own full, real prevalence.
            idx = np.arange(len(y_src))
            pos_tr, neg_tr = idx[y_src == 1], idx[y_src == 0]
            if args.neg_ratio > 0 and len(neg_tr) > args.neg_ratio * len(pos_tr):
                rng = np.random.default_rng(RANDOM_STATE)
                neg_tr = rng.choice(neg_tr, size=args.neg_ratio * len(pos_tr), replace=False)
            tr_sub = np.concatenate([pos_tr, neg_tr])

            cond_res = {}
            for po in poolings:
                Xsrc = assemble(po, src_k, topk_src)
                Xtgt = assemble(po, tgt_k, topk_tgt)
                cond_res[po] = {}
                for mo in models:
                    s = oof_scores(mo, Xsrc[tr_sub], y_src[tr_sub], Xtgt)
                    cond_res[po][mo] = {"roc_auc": round(float(roc_auc_score(y_tgt, s)), 4),
                                        "auc_pr": round(float(average_precision_score(y_tgt, s)), 5)}
                    print(f"  {po:<16} {mo:<12} ROC-AUC={cond_res[po][mo]['roc_auc']:.3f} "
                          f"AUC-PR={cond_res[po][mo]['auc_pr']:.4f}", flush=True)

            results["directions"][dir_key][cond] = {
                "n_train_users": int(src_keep.sum()), "n_train_positive": int(y_src.sum()),
                "n_test_users": int(tgt_keep.sum()), "n_test_positive": int(y_tgt.sum()),
                "seconds": round(time.time() - t0), "results": cond_res,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2))
            print(f"[checkpoint] Saved {dir_key}/{cond} to {args.output}", flush=True)

    print(f"\nSalvo em {args.output}")


if __name__ == "__main__":
    main()
