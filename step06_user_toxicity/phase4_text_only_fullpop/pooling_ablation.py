"""Pooling ablation for user-level toxicity from review-text embeddings.

Motivation: mean-pooling a user's review embeddings dilutes the rare toxic
signal (a positive user may have 1-2 toxic reviews among dozens of neutral
ones). This script compares aggregation strategies - mean, mean+max, and
mean+max+top-k (top-k selected per fold by cosine similarity to a POSITIVE
PROTOTYPE built ONLY on the training fold, so it never leaks the test label) -
across several classical models, with and without the profile/metadata
feature block, and under both the circular and the leave-toxic-out (honest)
conditions.

Reuses the per-review embeddings already produced by step07
(review_embeddings_<lang>.npy + review_dataset_<lang>.parquet, which carry
user_url and the review-level toxicity flag), so nothing is re-encoded.
Works with any encoder's output - set EMBEDDING_DIM to match the .npy's
column count (default 384/MiniLM; DistilBERT is 768, see
EMBEDDING_DIM=768 in step07/02b_compute_embeddings_distilbert.py's output).

Circular vs honest:
  - circular   : pool ALL of a user's reviews.
  - honest/LTO : drop a positive user's individually-toxic reviews before
                 pooling (mirrors phase3's --leave-toxic-out); positives left
                 with zero reviews are dropped, same as phase2/phase3.

Evaluation: stratified 5-fold, out-of-fold ROC-AUC and AUC-PR. ROC-AUC is the
headline metric (prevalence-invariant, comparable across phases).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))  # override for non-MiniLM encoders, e.g. DistilBERT=768
RANDOM_STATE = 42
N_FOLDS = 5
CHUNK = 100_000

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=["pt", "en"])
    p.add_argument("--embeddings", required=True, type=Path, help="review_embeddings_<lang>.npy (per-review, row-aligned)")
    p.add_argument("--review-dataset", required=True, type=Path, help="review_dataset_<lang>.parquet (user_url, is_toxic)")
    p.add_argument("--labels", required=True, type=Path, help="labels_slim.parquet (eligible_<lang>, is_toxic_<lang>)")
    p.add_argument("--feature-table", type=Path, default=None, help="phase1 feature table, for the --with-metadata variant")
    p.add_argument("--with-metadata", action="store_true", help="concatenate the profile/rate/sentiment metadata block")
    p.add_argument("--topk", type=int, default=3, help="k for the top-k pooling (default 3)")
    p.add_argument("--poolings", default="mean,mean+max,mean+max+topk",
                   help="comma-separated pooling variants to evaluate (mean, mean+max, mean+max+topk)")
    p.add_argument("--models", default="logreg,svm_linear,svm_rbf,xgboost", help="comma-separated: logreg,svm_linear,svm_rbf,xgboost")
    p.add_argument("--neg-ratio", type=int, default=30, help="train on all positives + this many negatives per positive "
                   "(hard-negative-style downsampling; evaluation is still on the FULL test fold at real prevalence). "
                   "0 = use every negative (slow at scale).")
    p.add_argument("--output", required=True, type=Path, help="results JSON")
    return p.parse_args()


# ---------------------------------------------------------------- models
def oof_scores(model_name, Xtr, ytr, Xte):
    """Fit `model_name` on (Xtr,ytr), return a continuous score for Xte."""
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    pos_weight = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    if model_name == "logreg":
        m = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if model_name == "svm_linear":
        m = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
        m.fit(Xtr, ytr)
        return m.decision_function(Xte)
    if model_name == "svm_rbf":  # RBF via Nystroem kernel approximation (scales to millions)
        nys = Nystroem(kernel="rbf", n_components=300, random_state=RANDOM_STATE)
        Ztr = nys.fit_transform(Xtr)
        Zte = nys.transform(Xte)
        m = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
        m.fit(Ztr, ytr)
        return m.decision_function(Zte)
    if model_name == "xgboost":
        if HAS_XGB:
            m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                              eval_metric="aucpr", n_jobs=-1, tree_method="hist")
        else:  # fall back to sklearn's gradient boosting if xgboost isn't installed
            m = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    raise ValueError(model_name)


# ---------------------------------------------------------------- feature building
def build_user_features(args, exclude_toxic_from_pos, need_topk=True):
    """Returns per-eligible-user: MEAN, MAX (pooled review embeddings), the
    retained per-review embeddings + user index (for per-fold top-k), and the
    user-level label. `exclude_toxic_from_pos` toggles the leave-toxic-out
    condition."""
    lang = args.language
    lab = pd.read_parquet(args.labels, columns=["user_url", f"eligible_{lang}", f"is_toxic_{lang}"])
    lab = lab[lab[f"eligible_{lang}"].fillna(False)].reset_index(drop=True)
    yuser = lab[f"is_toxic_{lang}"].fillna(False).astype(int).to_numpy()
    uidx = {u: i for i, u in enumerate(lab["user_url"])}
    nU = len(lab)

    rds = pd.read_parquet(args.review_dataset, columns=["user_url", "is_toxic"])
    ru = rds["user_url"].map(uidx).to_numpy()
    rtox = rds["is_toxic"].to_numpy().astype(bool)
    emb = np.load(args.embeddings, mmap_mode="r")
    assert emb.shape[1] == EMBEDDING_DIM, (
        f"{args.embeddings} has {emb.shape[1]} dims but EMBEDDING_DIM={EMBEDDING_DIM} "
        "(set the EMBEDDING_DIM env var to match, e.g. 768 for DistilBERT)"
    )

    sums = np.zeros((nU, EMBEDDING_DIM), np.float64)
    counts = np.zeros(nU, np.int64)
    maxes = np.full((nU, EMBEDDING_DIM), -1e30, np.float32)
    kept_rev_emb, kept_rev_user = [], []  # for per-fold top-k
    for s in range(0, len(rds), CHUNK):
        u = ru[s:s + CHUNK]; t = rtox[s:s + CHUNK]
        m = ~np.isnan(u)
        ui = np.where(m, u, 0).astype(np.int64)
        if exclude_toxic_from_pos:
            m = m & ~(t & yuser[ui].astype(bool))
        sel = np.where(m)[0]
        if len(sel) == 0:
            continue
        v = np.asarray(emb[s:s + CHUNK][sel], np.float32)
        idx = ui[sel]
        np.add.at(sums, idx, v.astype(np.float64))
        np.add.at(counts, idx, 1)
        np.maximum.at(maxes, idx, v)
        if need_topk:
            kept_rev_emb.append(v)
            kept_rev_user.append(idx)

    mean = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0).astype(np.float32)
    maxes[counts == 0] = 0.0
    rev_emb = np.concatenate(kept_rev_emb) if kept_rev_emb else np.zeros((0, EMBEDDING_DIM), np.float32)
    rev_user = np.concatenate(kept_rev_user) if kept_rev_user else np.zeros(0, np.int64)

    meta = None
    if args.with_metadata:
        meta = load_metadata(args.feature_table, lang, lab["user_url"].to_numpy())

    keep = counts > 0
    return dict(mean=mean, maxes=maxes, rev_emb=rev_emb, rev_user=rev_user,
                yuser=yuser, meta=meta, keep=keep)


def load_metadata(feature_table, lang, user_urls):
    """The non-embedding feature block (profile/rate/sentiment) from the phase1
    feature table, aligned to `user_urls`. Everything that is NOT emb_* / a
    label / p_value_* (the binomial-test p-value the label is thresholded
    from - not a legitimate predictor) / user_url is treated as a metadata
    feature."""
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(feature_table).schema_arrow
    allc = [f.name for f in schema]
    drop_prefixes = ("emb_",)
    drop_exact = {"user_url"} | {c for c in allc if c.startswith(("eligible_", "is_toxic_", "n_", "p_value_"))}
    meta_cols = [c for c in allc if c not in drop_exact and not c.startswith(drop_prefixes)]
    ft = pd.read_parquet(feature_table, columns=["user_url"] + meta_cols).set_index("user_url")
    ft = ft.reindex(user_urls)
    return ft.select_dtypes("number").fillna(0.0).to_numpy(np.float32)


def topk_features_for_fold(feats, train_pos_users, k):
    """Per fold: prototype = mean of the training-fold POSITIVE users' review
    embeddings; score every review by cosine sim to it; per user, mean-pool
    the k highest-scoring reviews. Built only from training positives -> no
    test-label leakage."""
    rev_emb, rev_user = feats["rev_emb"], feats["rev_user"]
    nU = len(feats["yuser"])
    is_train_pos_review = np.isin(rev_user, train_pos_users)
    if is_train_pos_review.sum() == 0:
        return np.zeros((nU, EMBEDDING_DIM), np.float32)
    proto = rev_emb[is_train_pos_review].mean(0)
    proto /= (np.linalg.norm(proto) + 1e-8)
    norms = np.linalg.norm(rev_emb, axis=1, keepdims=True) + 1e-8
    scores = (rev_emb / norms) @ proto  # cosine sim, one per review

    # group by user, best score first within user; keep the first k of each group
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


def assemble(pooling, feats, topk_feat):
    parts = [feats["mean"]]
    if "max" in pooling:
        parts.append(feats["maxes"])
    if "topk" in pooling:
        parts.append(topk_feat)
    X = np.hstack(parts)
    if feats["meta"] is not None:
        X = np.hstack([X, feats["meta"]])
    return X


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    poolings = [p.strip() for p in args.poolings.split(",") if p.strip()]
    need_topk = any("topk" in p for p in poolings)
    results = {"language": args.language, "with_metadata": args.with_metadata,
               "xgboost_installed": HAS_XGB, "topk": args.topk, "conditions": {}}
    if args.output.exists():
        # Resumable at CONDITION granularity (circular / leave_toxic_out are
        # each ~half the runtime and independent) - a condition already
        # present in a prior partial output is trusted verbatim and skipped,
        # not recomputed, so a crash mid-leave_toxic_out doesn't throw away
        # an already-finished circular run.
        prior = json.loads(args.output.read_text())
        results["conditions"] = prior.get("conditions", {})
        if results["conditions"]:
            print(f"[resume] {args.output} already has: {list(results['conditions'])}", flush=True)

    for cond, excl in [("circular", False), ("leave_toxic_out", True)]:
        if cond in results["conditions"]:
            print(f"\n### {cond}{' + metadata' if args.with_metadata else ''} - already done, skipping ###", flush=True)
            continue
        t0 = time.time()
        feats = build_user_features(args, exclude_toxic_from_pos=excl, need_topk=need_topk)
        keep = feats["keep"]
        y = feats["yuser"][keep]
        print(f"\n### {cond}{' + metadata' if args.with_metadata else ''} "
              f"| N={keep.sum():,} positivos={int(y.sum())} ###", flush=True)

        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(skf.split(np.zeros(keep.sum()), y))
        keep_pos = np.flatnonzero(keep)  # map compact -> global user index

        oof = {(mo, po): np.zeros(keep.sum()) for mo in models for po in poolings}
        for fi, (tr, te) in enumerate(splits):
            train_pos_global = keep_pos[tr[y[tr] == 1]]
            topk_feat_full = (topk_features_for_fold(feats, train_pos_global, args.topk)
                              if need_topk else np.zeros((len(feats["yuser"]), EMBEDDING_DIM), np.float32))
            # restrict every feature block to kept users, in compact order
            kfeats = dict(mean=feats["mean"][keep], maxes=feats["maxes"][keep],
                          meta=None if feats["meta"] is None else feats["meta"][keep])
            topk_c = topk_feat_full[keep]
            # hard-negative-style downsampling of the TRAIN fold only (prototypes above
            # still use all train positives; the TEST fold `te` stays at full prevalence)
            pos_tr = tr[y[tr] == 1]
            neg_tr = tr[y[tr] == 0]
            if args.neg_ratio > 0 and len(neg_tr) > args.neg_ratio * len(pos_tr):
                rng = np.random.default_rng(RANDOM_STATE + fi)
                neg_tr = rng.choice(neg_tr, size=args.neg_ratio * len(pos_tr), replace=False)
            tr_sub = np.concatenate([pos_tr, neg_tr])
            for po in poolings:
                X = assemble(po, kfeats, topk_c)
                for mo in models:
                    oof[(mo, po)][te] = oof_scores(mo, X[tr_sub], y[tr_sub], X[te])

        cond_res = {}
        for po in poolings:
            cond_res[po] = {}
            for mo in models:
                s = oof[(mo, po)]
                cond_res[po][mo] = {"roc_auc": round(float(roc_auc_score(y, s)), 4),
                                    "auc_pr": round(float(average_precision_score(y, s)), 5)}
                print(f"  {po:<16} {mo:<12} ROC-AUC={cond_res[po][mo]['roc_auc']:.3f} "
                      f"AUC-PR={cond_res[po][mo]['auc_pr']:.4f}", flush=True)
        results["conditions"][cond] = {"n_users": int(keep.sum()), "n_positive": int(y.sum()),
                                       "seconds": round(time.time() - t0), "results": cond_res}

        # Written after EACH condition, not just at the end - a later crash
        # (e.g. mid leave_toxic_out) leaves this condition's result on disk
        # instead of losing it, and a resumed run picks it up above.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"[checkpoint] Saved {cond} to {args.output}", flush=True)

    print(f"\nSalvo em {args.output}")


if __name__ == "__main__":
    main()
