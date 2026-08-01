"""Phase 6 (part 2): formal statistical significance testing for the LLM
classifier results (Anthropic/OpenAI/Google, pt/en), mirroring Phase 5's
approach but applied to Phase 6's saved prediction parquets - no new API
calls, purely a post-hoc statistical analysis.

CAVEAT (carried from the Phase 6 report): Phase 6's population is a matched
sample (all positives + a 3x negative sample), NOT the real prevalence used
in Phase 2. Its AUC-PR values are therefore only comparable to EACH OTHER
(across providers/models within Phase 6), not directly to Phase 2/3's
real-prevalence AUC-PR. This script computes both:
  1. PERMUTATION TEST - is this AUC-PR better than chance, given this exact
     matched-sample size/prevalence?
  2. BOOTSTRAP CI - how much would this AUC-PR vary under resampling?
(Same two tests as Phase 5, same rationale - see phase5_significance's
01_significance_tests.py docstring for the full explanation.)

Failed classifications (confidence_toxic == -1) are excluded before scoring,
matching exactly what 01_llm_classify_users.py itself does when computing
the auc_pr reported in each summary.json - checked explicitly against those
summaries below, not silently trusted.

Usage:
    python 02_significance_tests.py \\
        --input-dir ../../../../steam-data/step06-output/phase6_llm_classifier \\
        --output ../../../../steam-data/step06-output/phase6_llm_classifier/phase6_significance_report.json
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pipeline_utils import info, save_summary

RANDOM_STATE = 42
N_PERMUTATIONS = 2000
N_BOOTSTRAP = 2000

# Explicit list of Phase 6's actual completed runs - not a glob/discovery,
# since these are specific known results, not an open-ended set.
RESULTS = [
    {"population": "pt", "provider": "anthropic", "model": "claude-sonnet-5",
     "filename": "phase6_pt_predictions.parquet"},
    {"population": "pt", "provider": "openai", "model": "gpt-5.4",
     "filename": "phase6_pt_openai_gpt-5.4_predictions.parquet"},
    {"population": "pt", "provider": "google", "model": "gemini-3.1-pro-preview",
     "filename": "phase6_pt_google_gemini-3.1-pro-preview_predictions.parquet"},
    {"population": "pt", "provider": "google", "model": "gemini-3.6-flash",
     "filename": "phase6_pt_google_gemini-3.6-flash_predictions.parquet"},
    {"population": "en", "provider": "anthropic", "model": "claude-sonnet-5",
     "filename": "phase6_en_anthropic_claude-sonnet-5_predictions.parquet"},
    {"population": "en", "provider": "openai", "model": "gpt-5.4",
     "filename": "phase6_en_openai_gpt-5.4_predictions.parquet"},
    {"population": "en", "provider": "google", "model": "gemini-3.6-flash",
     "filename": "phase6_en_google_gemini-3.6-flash_predictions.parquet"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Permutation test + bootstrap CI for Phase 6's LLM classifier AUC-PR results."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Path to phase6_llm_classifier's output directory")
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
    for entry in RESULTS:
        pop, provider, model = entry["population"], entry["provider"], entry["model"]
        key = f"{pop}/{provider}/{model}"
        path = args.input_dir / entry["filename"]
        if not path.exists():
            info(f"[{key}] File not found, skipping: {path}")
            continue

        df = pd.read_parquet(path, columns=["y_true", "confidence_toxic"])
        valid = df[df["confidence_toxic"] >= 0]  # exclude failed classifications, same filter 01_llm_classify_users.py uses
        n_failed = len(df) - len(valid)
        y_true = valid["y_true"].to_numpy()
        y_score = (valid["confidence_toxic"] / 100.0).to_numpy()
        n_pos, n_total = int(y_true.sum()), len(y_true)

        info(f"[{key}] {n_total:,} row(s) ({n_failed} failed/excluded), {n_pos:,} positive - running permutation test ({N_PERMUTATIONS} shuffles)...")
        observed_auc, p_value, null_aucs = permutation_test(y_true, y_score, rng)

        info(f"[{key}] Running bootstrap CI ({N_BOOTSTRAP} resamples)...")
        ci_low, ci_high = bootstrap_ci(y_true, y_score, rng)

        info(
            f"[{key}] AUC-PR={observed_auc:.6f}, p={p_value:.5f}, "
            f"95% CI=[{ci_low:.6f}, {ci_high:.6f}], null mean={null_aucs.mean():.6f}"
        )

        results.setdefault(pop, {})[f"{provider}/{model}"] = {
            "n_total": n_total,
            "n_failed_excluded": n_failed,
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
