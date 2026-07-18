"""Phase 2, Step 1: nested cross-validation for step06's classical-ML
toxic-user classifiers, using the same procedure for all three populations
(pt/en/union) so results stay comparable - no population gets an easier or
harder validation scheme than another.

WHY THIS DESIGN (see this project's step06 planning conversation): every
population has a tiny positive (toxic) class relative to its population -
75/179,042 in pt, 1,225/2,191,286 in en, 1,316/2,375,760 in union. A plain
k-fold would leave too few positives per test fold (pt: ~15 at k=5) for a
stable estimate, so the OUTER loop is Leave-One-Out over the positive class
specifically - each fold holds out exactly one toxic user, training on
every other positive plus (almost) the entire negative population.

NEGATIVES ARE NOT SUBSAMPLED PER FOLD (an explicit decision - the
population size matters to the user running this project): a single,
FIXED split of negatives into train (80%) and test (20%) is made ONCE per
population, not per fold. The 80% train-negatives appear in every single
outer fold's training set unchanged; the 20% test-negatives are never
trained on in any fold, so they stay genuinely held-out throughout, and are
scored by averaging predictions across every outer fold's model (each of
which is equally "out-of-fold" for them, since none of them were trained on
any test-negative).

INNER LOOP (hyperparameter tuning): LogisticRegressionCV's built-in
cross-validated regularization search (average-precision-scored), run
separately inside each outer fold's training data - never sees the
held-out positive or any test-negative, so tuning cannot leak into the
outer evaluation.

PREPROCESSING (fit on training data only, per outer fold - never on the
full dataset, to avoid leakage): profile features are median-imputed
(NaN -> per-fold training median) then standardised; the 384-dim embedding
block is standardised then PCA-reduced to 50 dims (same reduction BERTopic
uses in step03) - profile features pass through PCA untouched, only the
embedding block is compressed.

OUTPUT: one parquet row per positive user (their out-of-fold predicted
probability) plus one row per test-negative (their averaged out-of-fold
probability), enough to compute an AUC-PR / precision-recall curve for the
population - and a summary JSON with the resulting AUC-PR.

Usage:
    python phase2_nested_cv.py \\
        --feature-table ../../steam-data/step06-output/phase1_feature_table.parquet \\
        --population pt \\
        --output ../../steam-data/step06-output/phase2_pt_oof_predictions.parquet \\
        --n-jobs -1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline_utils import info, save_summary

PROFILE_FEATURES = [
    "profile_level", "has_ban", "days_since_last_ban", "awards", "insignias",
    "library_size", "screenshots", "workshop_items", "guides", "arts", "groups", "friends_count",
]
EMBEDDING_DIM = 384
PCA_COMPONENTS = 50
TEST_NEGATIVE_FRACTION = 0.20
INNER_CV_FOLDS = 3
C_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0]
RANDOM_STATE = 42
CHECKPOINT_EVERY = 50  # outer folds, per batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested Leave-One-Out CV for one population's toxic-user classifier."
    )
    parser.add_argument("--feature-table", required=True, type=Path, help="Path to phase1_feature_table.parquet")
    parser.add_argument("--population", required=True, choices=["pt", "en", "union"], help="Which population to run")
    parser.add_argument("--output", required=True, type=Path, help="Path to write out-of-fold predictions parquet to")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs across outer LOO folds (default: all cores)")
    parser.add_argument(
        "--limit-positives", type=int, default=None,
        help="VALIDATION ONLY: cap the number of outer LOO folds actually run, instead of all positives",
    )
    return parser.parse_args()


def load_population_data(table: pd.DataFrame, population: str) -> tuple:
    """Returns (X, y, user_urls) for one population's eligible users.
    population is 'pt', 'en', or 'union' - the label column suffix for
    union is 'pt+en' (build_toxic_user_labels.py's naming) while the
    embedding column prefix is 'union' (build_user_text_embeddings.py's
    naming) - both are handled here so callers just say 'union'."""
    label_suffix = "pt+en" if population == "union" else population
    emb_prefix = population  # already "union" for that case

    eligible = table[f"eligible_{label_suffix}"].fillna(False)
    subset = table.loc[eligible]

    emb_cols = [f"emb_{emb_prefix}_{i}" for i in range(EMBEDDING_DIM)]

    # Plain float32 numpy arrays, not a pandas DataFrame - joblib's loky
    # backend automatically memory-maps large numpy arrays passed through
    # delayed() (shared read-only across worker processes via a temp file),
    # instead of pickling/copying the whole thing into every worker.
    # Passing a DataFrame instead defeats this - it gets pickled fresh per
    # task, which is what blew up RAM (~40GB) on this project's machine
    # before this fix. float32 (not float64) also halves the footprint.
    X_profile = subset[PROFILE_FEATURES].copy()
    X_profile["has_ban"] = X_profile["has_ban"].astype("float32")
    for col in PROFILE_FEATURES:
        if col != "has_ban":
            X_profile[col] = X_profile[col].astype("float32")
    X_profile = X_profile.to_numpy(dtype="float32")
    X_emb = subset[emb_cols].to_numpy(dtype="float32")

    y = subset[f"is_toxic_{label_suffix}"].fillna(False).astype(int).to_numpy()
    user_urls = subset["user_url"].to_numpy()

    info(f"[{population}] {len(subset):,} eligible user(s), {int(y.sum()):,} positive (toxic)")
    return X_profile, X_emb, y, user_urls


def preprocess_fold(train_profile_raw: np.ndarray, train_emb_raw: np.ndarray,
                     test_profile_raw: np.ndarray, test_emb_raw: np.ndarray) -> tuple:
    """Fits imputation/scaling/PCA on the TRAIN arrays ONLY, applies the
    fitted transforms to both train and test - never lets test data
    influence any fitted parameter."""
    medians = np.nanmedian(train_profile_raw, axis=0)
    nan_mask_train = np.isnan(train_profile_raw)
    train_profile = np.where(nan_mask_train, medians, train_profile_raw)
    nan_mask_test = np.isnan(test_profile_raw)
    test_profile = np.where(nan_mask_test, medians, test_profile_raw)

    profile_scaler = StandardScaler()
    train_profile_scaled = profile_scaler.fit_transform(train_profile)
    test_profile_scaled = profile_scaler.transform(test_profile)

    emb_scaler = StandardScaler()
    train_emb_scaled = emb_scaler.fit_transform(train_emb_raw)
    test_emb_scaled = emb_scaler.transform(test_emb_raw)

    n_components = min(PCA_COMPONENTS, train_emb_scaled.shape[0] - 1, train_emb_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    train_emb_pca = pca.fit_transform(train_emb_scaled)
    test_emb_pca = pca.transform(test_emb_scaled)

    X_train_final = np.hstack([train_profile_scaled, train_emb_pca]).astype("float32")
    X_test_final = np.hstack([test_profile_scaled, test_emb_pca]).astype("float32")
    return X_train_final, X_test_final


def run_outer_fold(held_out_pos_pos: int, all_pos_idx: np.ndarray, train_neg_idx: np.ndarray,
                    test_neg_idx: np.ndarray, X_profile: np.ndarray, X_emb: np.ndarray, y: np.ndarray) -> tuple:
    """One LOO fold: hold out exactly one positive (by its position within
    all_pos_idx), train on every other positive + the fixed train-negatives,
    then score BOTH the held-out positive and every test-negative with this
    fold's model (test-negatives are never trained on in any fold, so every
    fold's score for them is equally valid out-of-fold - averaged later).

    X_profile/X_emb arrive here as joblib-memmapped read-only numpy arrays
    (see load_population_data) - fancy-indexing them (X_profile[idx]) makes
    a small, ordinary in-memory copy of just this fold's rows, not the
    whole array, so this stays cheap regardless of population size."""
    held_out_idx = all_pos_idx[held_out_pos_pos]
    fold_train_pos_idx = np.delete(all_pos_idx, held_out_pos_pos)
    fold_train_idx = np.concatenate([fold_train_pos_idx, train_neg_idx])

    y_train = y[fold_train_idx]

    eval_idx = np.concatenate([[held_out_idx], test_neg_idx])

    X_train_final, X_eval_final = preprocess_fold(
        X_profile[fold_train_idx], X_emb[fold_train_idx],
        X_profile[eval_idx], X_emb[eval_idx],
    )

    n_inner_splits = max(2, min(INNER_CV_FOLDS, int(y_train.sum())))
    model = LogisticRegressionCV(
        Cs=C_CANDIDATES, cv=n_inner_splits, scoring="average_precision",
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, n_jobs=1,
    )
    model.fit(X_train_final, y_train)

    scores = model.predict_proba(X_eval_final)[:, 1]
    pos_score = scores[0]
    neg_scores = scores[1:]
    return held_out_pos_pos, held_out_idx, pos_score, neg_scores


def save_checkpoint(checkpoint_dir: Path, pos_predictions: dict, neg_score_sum: np.ndarray, n_folds_done: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    positions = list(pos_predictions.keys())
    held_out_indices = [v[0] for v in pos_predictions.values()]
    pos_scores = [v[1] for v in pos_predictions.values()]
    pd.DataFrame({"position": positions, "held_out_idx": held_out_indices, "pos_score": pos_scores}).to_parquet(
        checkpoint_dir / "pos_predictions.parquet", index=False
    )
    np.save(checkpoint_dir / "neg_score_sum.npy", neg_score_sum)
    (checkpoint_dir / "n_folds_done.txt").write_text(str(n_folds_done))


def load_checkpoint(checkpoint_dir: Path, n_test_negatives: int) -> tuple:
    pred_file = checkpoint_dir / "pos_predictions.parquet"
    neg_file = checkpoint_dir / "neg_score_sum.npy"
    count_file = checkpoint_dir / "n_folds_done.txt"

    pos_predictions = {}
    neg_score_sum = np.zeros(n_test_negatives)
    n_folds_done = 0

    if pred_file.exists() and neg_file.exists() and count_file.exists():
        df = pd.read_parquet(pred_file)
        pos_predictions = {
            int(row.position): (int(row.held_out_idx), float(row.pos_score)) for row in df.itertuples()
        }
        neg_score_sum = np.load(neg_file)
        n_folds_done = int(count_file.read_text())
        info(f"[checkpoint] Resuming: {n_folds_done} outer fold(s) already completed")

    return pos_predictions, neg_score_sum, n_folds_done


def main():
    args = parse_args()

    table = pd.read_parquet(args.feature_table)
    info(f"Loaded feature table: {len(table):,} user(s) from {args.feature_table}")

    X_profile, X_emb, y, user_urls = load_population_data(table, args.population)
    del table  # the full 1179-column table is no longer needed - only this population's slice is

    all_pos_idx = np.where(y == 1)[0]
    all_neg_idx = np.where(y == 0)[0]

    train_neg_idx, test_neg_idx = train_test_split(
        all_neg_idx, test_size=TEST_NEGATIVE_FRACTION, random_state=RANDOM_STATE
    )
    info(
        f"[{args.population}] Fixed negative split: {len(train_neg_idx):,} train-negatives "
        f"(used in every fold), {len(test_neg_idx):,} test-negatives (never trained on, held out throughout)"
    )

    n_folds_to_run = len(all_pos_idx)
    if args.limit_positives:
        n_folds_to_run = min(args.limit_positives, n_folds_to_run)
        info(f"[VALIDATION MODE] Limiting to {n_folds_to_run} of {len(all_pos_idx)} outer fold(s)")

    checkpoint_dir = args.output.parent / f".{args.output.stem}_checkpoint"
    pos_predictions, neg_score_sum, n_folds_done = load_checkpoint(checkpoint_dir, len(test_neg_idx))
    remaining_positions = [i for i in range(n_folds_to_run) if i not in pos_predictions]
    if pos_predictions:
        info(f"[{args.population}] {len(remaining_positions)} outer fold(s) remaining (of {n_folds_to_run})")

    # max_nbytes (default) lets joblib memory-map X_profile/X_emb once, as
    # read-only shared memory across worker processes, instead of pickling
    # a full copy into every one of n_jobs workers - critical for en/union
    # where X_emb alone is ~2.2M x 384 floats.
    info(f"[{args.population}] Running {len(remaining_positions)} outer LOO fold(s) across n_jobs={args.n_jobs}...")
    for batch_start in range(0, len(remaining_positions), CHECKPOINT_EVERY):
        batch = remaining_positions[batch_start : batch_start + CHECKPOINT_EVERY]
        results = Parallel(n_jobs=args.n_jobs, verbose=5, max_nbytes="10M")(
            delayed(run_outer_fold)(i, all_pos_idx, train_neg_idx, test_neg_idx, X_profile, X_emb, y)
            for i in batch
        )
        for position, held_out_idx, pos_score, neg_scores in results:
            pos_predictions[position] = (held_out_idx, pos_score)
            neg_score_sum += neg_scores
        n_folds_done += len(batch)
        save_checkpoint(checkpoint_dir, pos_predictions, neg_score_sum, n_folds_done)
        info(f"[checkpoint] Saved after {n_folds_done}/{n_folds_to_run} outer fold(s)")

    neg_predictions_avg = neg_score_sum / n_folds_done

    held_out_indices = [v[0] for v in pos_predictions.values()]
    pos_scores = [v[1] for v in pos_predictions.values()]
    pos_rows = pd.DataFrame({
        "user_url": user_urls[held_out_indices],
        "y_true": 1,
        "y_score": pos_scores,
    })
    neg_rows = pd.DataFrame({
        "user_url": user_urls[test_neg_idx],
        "y_true": 0,
        "y_score": neg_predictions_avg,
    })
    oof = pd.concat([pos_rows, neg_rows], ignore_index=True)

    auc_pr = average_precision_score(oof["y_true"], oof["y_score"])
    info(f"[{args.population}] AUC-PR (out-of-fold): {auc_pr:.4f} (based on {len(pos_rows)} positive fold(s) run)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.output, index=False)
    info(f"Saved out-of-fold predictions ({len(oof)} rows) to: {args.output}")

    if not args.limit_positives and checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        info(f"Cleaned up checkpoint directory: {checkpoint_dir}")

    save_summary(
        {
            "population": args.population,
            "n_eligible": int(len(y)),
            "n_positive_total": int(len(all_pos_idx)),
            "n_outer_folds_run": n_folds_to_run,
            "n_train_negatives": int(len(train_neg_idx)),
            "n_test_negatives": int(len(test_neg_idx)),
            "auc_pr": round(float(auc_pr), 6),
            "validation_mode": args.limit_positives is not None,
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
