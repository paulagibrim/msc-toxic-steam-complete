#!/usr/bin/env python3
"""Merge the per-fold prediction files produced by

    02_train_mil_distilbert.py --fold N

into a single out-of-fold predictions parquet + summary, in the EXACT same format
as a sequential (all-folds) run. AUC-PR is computed over the concatenated OOF, so
the result is identical to running every fold in one process - the fold split uses
a fixed seed, this only parallelises the work.

Usage:
    python merge_folds.py \
        --fold-glob '/path/steam-data/step06-output/phase3_distilbert/phase3_en_oof.fold*.parquet' \
        --output    '/path/steam-data/step06-output/phase3_distilbert/phase3_en_oof.parquet' \
        --population en --expected-folds 5
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold-glob", required=True, help="Glob matching the per-fold parquet files (QUOTE it in the shell)")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the merged OOF predictions parquet")
    parser.add_argument("--population", default="", help="Population label for the summary (pt/en/union)")
    parser.add_argument("--expected-folds", type=int, default=None, help="If set, refuse to merge unless exactly this many fold files are found")
    args = parser.parse_args()

    files = sorted(glob.glob(args.fold_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.fold_glob}")
    if args.expected_folds is not None and len(files) != args.expected_folds:
        raise SystemExit(
            f"Expected {args.expected_folds} fold file(s) but found {len(files)} - not all folds finished?\n"
            + "\n".join(f"  {f}" for f in files)
        )

    print(f"Merging {len(files)} fold file(s):")
    for f in files:
        print(f"  {f}")

    oof = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    oof = oof.sort_values(["fold", "user_url"]).reset_index(drop=True)

    n_dup = int(oof["user_url"].duplicated().sum())
    if n_dup:
        raise SystemExit(f"{n_dup} user_url(s) appear in more than one fold - the fold files overlap, refusing to merge.")

    n_folds = oof["fold"].nunique()
    auc_pr = average_precision_score(oof["y_true"], oof["y_score"])
    # ROC-AUC alongside AUC-PR: at these prevalences AUC-PR alone is not
    # comparable against the frozen-encoder tables, which report ROC-AUC as
    # the prevalence-independent headline. Both are computed over the
    # concatenated OOF, never averaged across folds.
    roc_auc = roc_auc_score(oof["y_true"], oof["y_score"])
    prevalence = float(oof["y_true"].mean())
    print(f"\n[{args.population}] out-of-fold ({n_folds}-fold): ROC-AUC {roc_auc:.4f}  |  "
          f"AUC-PR {auc_pr:.5f} (prevalence {prevalence:.6f}, lift {auc_pr / prevalence:.2f}x)  |  "
          f"{len(oof):,} users, {int(oof['y_true'].sum()):,} positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.output, index=False)
    print(f"Saved out-of-fold predictions to: {args.output}")

    summary = {
        "population": args.population,
        "n_users": int(len(oof)),
        "n_positive": int(oof["y_true"].sum()),
        "n_folds": int(n_folds),
        "auc_pr": round(float(auc_pr), 6),
        "roc_auc": round(float(roc_auc), 6),
        "prevalence": round(prevalence, 8),
        "lift": round(float(auc_pr) / prevalence, 4),
        "merged_from": [str(f) for f in files],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
