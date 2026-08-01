"""Step 7 diagnostic (not part of the main pipeline): empirically checks
whether reducing embeddings via PCA (50 components, step06 Phase 2's
already-established choice) helps or hurts here, before locking it into
02's full k-fold training run.

WHY THIS NEEDS CHECKING (per this project's own rule: no threshold gets
reused into a new context without validating it there first): Phase 2's
PCA-50 choice was made under Nested LEAVE-ONE-OUT CV with only 75-1,316
positives - dimensionality reduction guards against overfitting when a
model is refit up to ~1,316 times on tiny training folds. Step07 has
33,685-480,729 positives and millions of negatives, evaluated under plain
k-fold, not LOO - the overfitting pressure PCA was solving for in Phase 2
may not apply the same way here. Rather than assume either "keep PCA-50
for consistency" or "skip PCA because there's now enough data" is correct,
this script runs one quick train/validation comparison and reports the
answer instead of guessing.

METHOD: single stratified 80/20 train/validation split (not full k-fold -
this is a preliminary check to inform a design choice, not the reported
result) on pt (already computed, smaller). LogisticRegression
(class_weight="balanced", same as Phase 2) fit twice: once on standardised
raw 384-dim embeddings, once on standardised embeddings reduced to 50
components via PCA (Phase 2's exact number, reused unchanged if kept at
all). AUC-PR on the held-out validation slice decides it.

Usage:
    python 03_validate_pca_decision.py \\
        --review-dataset ../../steam-data/step07-output/review_dataset_pt.parquet \\
        --embeddings ../../steam-data/step07-output/review_embeddings_pt.npy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline_utils import info

RANDOM_STATE = 42
PCA_COMPONENTS = 50  # step06 Phase 2's already-established value, reused as-is if PCA is kept at all
VALIDATION_FRACTION = 0.20  # standard train/val split fraction, not a scientific threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Empirically checks whether PCA-50 helps or hurts on step07's data.")
    parser.add_argument("--review-dataset", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.review_dataset, columns=["is_toxic"])
    y = df["is_toxic"].astype(int).to_numpy()
    X = np.load(args.embeddings, mmap_mode="r")
    info(f"Loaded {len(y):,} row(s), {y.sum():,} toxic ({y.mean():.4%})")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, stratify=y
    )
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    info(f"Train: {len(y_train):,} ({y_train.sum():,} toxic) / Val: {len(y_val):,} ({y_val.sum():,} toxic)")

    # --- Raw (no PCA) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model_raw = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1)
    model_raw.fit(X_train_scaled, y_train)
    auc_raw = average_precision_score(y_val, model_raw.predict_proba(X_val_scaled)[:, 1])
    info(f"[no PCA, 384-dim] AUC-PR = {auc_raw:.4f}")

    # --- PCA-50 ---
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    model_pca = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1)
    model_pca.fit(X_train_pca, y_train)
    auc_pca = average_precision_score(y_val, model_pca.predict_proba(X_val_pca)[:, 1])
    info(f"[PCA-{PCA_COMPONENTS}] AUC-PR = {auc_pca:.4f} (explained variance: {pca.explained_variance_ratio_.sum():.4%})")

    info(f"DECISION: {'keep PCA-' + str(PCA_COMPONENTS) if auc_pca >= auc_raw else 'skip PCA, use raw 384-dim'} "
         f"(diff = {auc_pca - auc_raw:+.4f})")


if __name__ == "__main__":
    main()
