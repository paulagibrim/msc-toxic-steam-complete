"""Step 7, Step 4: k-fold CV training for classical models (Logistic
Regression + HistGB) trying to approximate step02's union toxicity label
from review text embeddings alone.

WHY PLAIN K-FOLD, NOT LEAVE-ONE-OUT (step06 Phase 2's design): Phase 2 used
LOO specifically because its positive class was tiny and degenerate
(75-1,316 users) - a plain k-fold would leave too few positives per test
fold for a stable estimate. Here the positive class is abundant (33,685 pt
/ 480,729 en), so a standard stratified k-fold is the appropriate,
standard choice - the same sample-size reasoning that justified LOO in
Phase 2 justifies NOT using it here.

WHY k=5: reuses step06 Phase 3 Track B's already-established k-fold count
(that script's docstring: "Validação por k-fold (k=5), não Leave-One-Out
- fine-tuning de transformer é caro demais pra repetir uma vez por
positivo"), not a new number invented for this script.

HYPERPARAMETER TUNING: reuses Phase 2's exact nested-CV pattern verbatim -
LogisticRegressionCV (Cs=[0.001, 0.01, 0.1, 1.0, 10.0], the same grid) and
GridSearchCV over HistGradientBoostingClassifier (learning_rate=[0.05, 0.1,
0.2], max_leaf_nodes=[15, 31], the same grid just added to Phase 2) - both
tuned INSIDE each outer fold's training data only, never seeing that
fold's held-out data. Reused rather than re-derived so this experiment's
models are tuned exactly as Phase 2's were, not a fresh, undiscussed
search.

PCA: controlled by --use-pca, decided by 03_validate_pca_decision.py's
empirical result, not assumed either way.

OUTPUT: out-of-fold predictions parquet (review_url, y_true, y_score per
model) + summary JSON with AUC-PR per model - same shape as Phase 2's
output, for a consistent significance-testing step afterward (Phase 5's
script pattern applies unchanged).

Usage:
    python 04_train_classical_models.py \\
        --review-dataset ../../steam-data/step07-output/review_dataset_pt.parquet \\
        --embeddings ../../steam-data/step07-output/review_embeddings_pt.npy \\
        --language pt \\
        --use-pca no \\
        --output ../../steam-data/step07-output/review_classical_pt_oof.parquet \\
        --n-jobs -1
"""
import argparse
import contextlib
from pathlib import Path

import joblib.parallel
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pipeline_utils import info, save_summary

N_FOLDS = 5  # reused from step06 Phase 3 Track B's already-established choice, see module docstring
INNER_CV_FOLDS = 3  # same inner-tuning fold count Phase 2 uses
C_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0]  # verbatim from step06 Phase 2
HISTGB_GRID = {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31]}  # verbatim from step06 Phase 2's HistGB addition
PCA_COMPONENTS = 50  # verbatim from step06 Phase 2, used only if --use-pca yes
RANDOM_STATE = 42
CHECKPOINT_EVERY = 1  # outer folds - recycle worker processes each fold, same rationale as Phase 2


def parse_args():
    parser = argparse.ArgumentParser(description="k-fold CV: classical models approximating step02's union label from review embeddings.")
    parser.add_argument("--review-dataset", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--language", required=True, choices=["pt", "en"])
    parser.add_argument("--use-pca", required=True, choices=["yes", "no"], help="Decided by 03_validate_pca_decision.py's empirical result")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def build_models(n_inner_splits: int, n_jobs: int) -> dict:
    """n_jobs parallelizes each model's OWN inner-CV hyperparameter search
    (across the small C/grid candidates), not across outer folds - outer
    folds stay sequential (see main()) to keep peak memory bounded on very
    large en, but the inner search is cheap to parallelize since it shares
    the same already-in-memory fold data."""
    return {
        "logreg": LogisticRegressionCV(
            Cs=C_CANDIDATES, cv=n_inner_splits, scoring="average_precision",
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, n_jobs=n_jobs,
        ),
        "histgb": GridSearchCV(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            HISTGB_GRID, cv=n_inner_splits, scoring="average_precision", n_jobs=n_jobs,
        ),
    }


def preprocess_fold(X_train_raw: np.ndarray, X_test_raw: np.ndarray, use_pca: bool) -> tuple:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    if use_pca:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train).astype("float32")
        X_test = pca.transform(X_test).astype("float32")
    return X_train, X_test


@contextlib.contextmanager
def tqdm_joblib(desc: str):
    """Advance a tqdm bar once per joblib task completed inside the block - i.e.
    per inner-CV candidate fit of the hyperparameter search - so the progress
    bar moves DURING a long model.fit(), not just once per outer fold. Defensive:
    if joblib's internals differ, it silently skips the bar rather than break the fit."""
    pbar = tqdm(desc=desc, leave=False, unit="fit")
    original = joblib.parallel.BatchCompletionCallBack
    try:
        class TqdmBatchCallBack(original):
            def __call__(self, *args, **kwargs):
                try:
                    pbar.update(self.batch_size)
                except Exception:
                    pass
                return super().__call__(*args, **kwargs)
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCallBack
    except Exception:
        pass
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = original
        pbar.close()


def run_fold(fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray, X: np.ndarray, y: np.ndarray, use_pca: bool, n_jobs: int) -> dict:
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train, X_test = preprocess_fold(X_train_raw, X_test_raw, use_pca)

    n_inner_splits = max(2, min(INNER_CV_FOLDS, int(y_train.sum())))
    models = build_models(n_inner_splits, n_jobs)

    scores = {}
    with tqdm_joblib(f"  fold {fold_idx + 1}/{N_FOLDS}"):
        for name, model in models.items():
            model.fit(X_train, y_train)
            scores[name] = model.predict_proba(X_test)[:, 1]

    return {"fold": fold_idx, "test_idx": test_idx, "y_test": y_test, "scores": scores}


def main():
    args = parse_args()
    use_pca = args.use_pca == "yes"

    df = pd.read_parquet(args.review_dataset, columns=["review_url", "is_toxic"])
    y = df["is_toxic"].astype(int).to_numpy()
    X = np.load(args.embeddings, mmap_mode="r")
    info(f"[{args.language}] Loaded {len(y):,} row(s), {y.sum():,} toxic ({y.mean():.4%}), use_pca={use_pca}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_indices = list(skf.split(np.zeros(len(y)), y))

    all_scores = {name: np.empty(len(y)) for name in ["logreg", "histgb"]}

    # --- Checkpoint: each fold's out-of-fold scores are saved as it finishes, so an
    # interrupted run (timeout, OOM, ...) resumes instead of restarting from zero. ---
    import json
    checkpoint_dir = args.output.parent / f".{args.output.stem}_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.json"
    current_config = {"language": args.language, "n_folds": N_FOLDS, "use_pca": use_pca, "n_total": int(len(y))}
    if config_path.exists():
        saved_config = json.loads(config_path.read_text())
        if saved_config != current_config:
            raise SystemExit(
                f"Checkpoint at {checkpoint_dir} was produced with different arguments "
                f"(saved -> current): {saved_config} -> {current_config}. Resuming would mix "
                f"incompatible results - remove the checkpoint directory to start fresh."
            )
    else:
        config_path.write_text(json.dumps(current_config, indent=2))

    completed_folds = set()
    for fold_idx in range(N_FOLDS):
        fold_ckpt = checkpoint_dir / f"fold_{fold_idx}.npz"
        if fold_ckpt.exists():
            saved = np.load(fold_ckpt)
            for name in all_scores:
                all_scores[name][saved["test_idx"]] = saved[name]
            completed_folds.add(fold_idx)
    if completed_folds:
        info(f"[{args.language}] [checkpoint] Resuming: {len(completed_folds)} fold(s) already done {sorted(completed_folds)}")

    bar = tqdm(total=N_FOLDS, desc=f"[{args.language}]", unit="fold")
    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        if fold_idx in completed_folds:
            bar.update(1)
            continue
        result = Parallel(n_jobs=1)(  # one fold at a time (memory-bounded for very large en) - parallelism happens INSIDE each fold's inner-CV search instead, via args.n_jobs
            [delayed(run_fold)(fold_idx, train_idx, test_idx, X, y, use_pca, args.n_jobs)]
        )[0]
        for name in all_scores:
            all_scores[name][result["test_idx"]] = result["scores"][name]
        np.savez(
            checkpoint_dir / f"fold_{fold_idx}.npz",
            test_idx=result["test_idx"],
            logreg=result["scores"]["logreg"],
            histgb=result["scores"]["histgb"],
        )
        info(f"[{args.language}] [checkpoint] Saved fold {fold_idx + 1}/{N_FOLDS}")
        bar.update(1)
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    bar.close()

    oof = pd.DataFrame({"review_url": df["review_url"], "y_true": y})
    summary = {"language": args.language, "n_folds": N_FOLDS, "use_pca": use_pca, "n_total": int(len(y)), "n_positive": int(y.sum())}
    for name in all_scores:
        oof[f"y_score_{name}"] = all_scores[name]
        auc = average_precision_score(y, all_scores[name])
        summary[f"auc_pr_{name}"] = round(float(auc), 6)
        info(f"[{args.language}] [{name}] AUC-PR (out-of-fold) = {auc:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.output, index=False)
    info(f"Saved out-of-fold predictions to: {args.output}")
    save_summary(summary, args.output.with_suffix(".summary.json"))

    import shutil
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    info(f"[checkpoint] Cleaned up: {checkpoint_dir}")


if __name__ == "__main__":
    main()
