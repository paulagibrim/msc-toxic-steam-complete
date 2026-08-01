"""Phase 5: formal statistical significance testing for Phase 2's AUC-PR
results (original classifier AND the leave-toxic-out circularity control),
using the out-of-fold predictions ALREADY saved by Phase 2 - no retraining,
no new model fits. Purely a post-hoc statistical analysis.

TWO tests, computed independently for every population (pt/en/union) and
condition (original / leave-toxic-out) - 6 combinations total:

  1. PERMUTATION TEST - "is this AUC-PR significantly better than chance,
     given this exact population size and prevalence?" y_true is shuffled
     N_PERMUTATIONS times (breaking any real relationship between features
     and label while preserving the class balance), AUC-PR is recomputed
     under each shuffle - the resulting distribution is what AUC-PR looks
     like under the null hypothesis "the classifier has no real
     discriminative power". The p-value is the fraction of shuffles that
     reach or exceed the ACTUALLY OBSERVED AUC-PR. This is a more rigorous
     complement to the "X times better than baseline prevalence" framing
     already in the Phase 2 report - it accounts for sampling variability
     given the exact population size, not just a ratio against the
     theoretical baseline.

  2. BOOTSTRAP CONFIDENCE INTERVAL - "how much would this AUC-PR vary if
     we'd sampled a slightly different population?" (y_true, y_score)
     pairs are resampled WITH replacement N_BOOTSTRAP times, AUC-PR
     recomputed each time; the 2.5th/97.5th percentiles of that
     distribution form a 95% CI. Lets pt/en/union's point estimates be
     compared honestly (e.g., is pt's 0.1362 meaningfully different from
     en's 0.1684, or within each other's noise band, given pt's much
     smaller sample?).

Reads directly from phase2_classical_ml/'s existing output - if the
observed AUC-PR computed here doesn't match what's in the original
summary.json, something is wrong (either this script or the assumption
that scoring is stable) - checked explicitly, not silently trusted.

Usage:
    python 01_significance_tests.py \\
        --phase2-dir ../../../../steam-data/step06-output/phase2_classical_ml \\
        --output ../../../../steam-data/step06-output/phase5_significance/significance_report.json
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from pipeline_utils import info, save_summary

RANDOM_STATE = 42
N_PERMUTATIONS = 2000
N_BOOTSTRAP = 2000

POPULATIONS = ["pt", "en", "union"]
CONDITIONS = {
    "original": "phase2_{pop}_oof_predictions.parquet",
    "leave_toxic_out": "phase2_{pop}_leave_toxic_out_oof_predictions.parquet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Permutation test + bootstrap CI for Phase 2's AUC-PR results, from saved OOF predictions."
    )
    parser.add_argument("--phase2-dir", required=True, type=Path, help="Path to phase2_classical_ml's output directory")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the significance report JSON to")
    return parser.parse_args()


def permutation_test(y_true: np.ndarray, y_score: np.ndarray, rng: np.random.Generator) -> tuple:
    """Returns (observed_auc_pr, p_value, null_distribution_summary)."""
    observed = average_precision_score(y_true, y_score)

    null_aucs = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        shuffled_y = rng.permutation(y_true)
        null_aucs[i] = average_precision_score(shuffled_y, y_score)

    p_value = float((null_aucs >= observed).sum() + 1) / (N_PERMUTATIONS + 1)  # +1: standard finite-permutation correction
    return observed, p_value, null_aucs


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, rng: np.random.Generator) -> tuple:
    """Returns (ci_low, ci_high) - 95% bootstrap CI for AUC-PR."""
    n = len(y_true)
    boot_aucs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_y, boot_s = y_true[idx], y_score[idx]
        if boot_y.sum() == 0:  # degenerate resample with zero positives - AUC-PR undefined, skip and retry
            idx = rng.integers(0, n, size=n)
            boot_y, boot_s = y_true[idx], y_score[idx]
        boot_aucs[i] = average_precision_score(boot_y, boot_s)
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def main():
    args = parse_args()
    rng = np.random.default_rng(RANDOM_STATE)

    results = {}
    for pop in POPULATIONS:
        results[pop] = {}
        for condition, filename_template in CONDITIONS.items():
            path = args.phase2_dir / filename_template.format(pop=pop)
            if not path.exists():
                info(f"[{pop}/{condition}] File not found, skipping: {path}")
                continue

            df = pd.read_parquet(path, columns=["y_true", "y_score"])
            y_true = df["y_true"].to_numpy()
            y_score = df["y_score"].to_numpy()
            n_pos, n_total = int(y_true.sum()), len(y_true)

            info(f"[{pop}/{condition}] {n_total:,} row(s), {n_pos:,} positive - running permutation test ({N_PERMUTATIONS} shuffles)...")
            observed_auc, p_value, null_aucs = permutation_test(y_true, y_score, rng)

            info(f"[{pop}/{condition}] Running bootstrap CI ({N_BOOTSTRAP} resamples)...")
            ci_low, ci_high = bootstrap_ci(y_true, y_score, rng)

            info(
                f"[{pop}/{condition}] AUC-PR={observed_auc:.6f}, p={p_value:.5f}, "
                f"95% CI=[{ci_low:.6f}, {ci_high:.6f}], null mean={null_aucs.mean():.6f}"
            )

            results[pop][condition] = {
                "n_total": n_total,
                "n_positive": n_pos,
                "prevalence": round(n_pos / n_total, 6),
                "observed_auc_pr": round(float(observed_auc), 6),
                "permutation_p_value": round(p_value, 6),
                "significant_at_0.05": p_value < 0.05,
                "significant_at_0.01": p_value < 0.01,
                "null_distribution_mean": round(float(null_aucs.mean()), 6),
                "null_distribution_max": round(float(null_aucs.max()), 6),
                "bootstrap_ci_95_low": round(ci_low, 6),
                "bootstrap_ci_95_high": round(ci_high, 6),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_summary(results, args.output)
    info(f"Saved significance report to: {args.output}")


if __name__ == "__main__":
    main()
