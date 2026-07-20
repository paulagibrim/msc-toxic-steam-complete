"""TF-IDF term-weight comparison across the same three mutually-exclusive
Venn regions used elsewhere in this project (visualizations/
sentiment_by_toxicity_venn_set.py, run_toxicity_venn.py): PERSPECTIVE ONLY
(perspective_score >= 0.7, detoxify_score < 0.9), DETOXIFY ONLY
(detoxify_score >= 0.9, perspective_score < 0.7), and BOTH (both
thresholds met) - same union rule and thresholds as every other toxicity
decision in this project.

DATA SOURCE: step04-output, not step02 like the original toxic-vs-non-toxic
comparison (tfidf_analysis.py) - step04's files already carry
perspective_score, detoxify_score AND review_text together per review, so
no join against step02 is needed. Unlike the original comparison, only
FLAGGED reviews are read here (rows outside all three sets - i.e. flagged
by neither model - are irrelevant to a Venn-region comparison and are
dropped up front, not scored).

REUSES tfidf_analysis.py's text cleaning, stopword list, and vectorizer
fitting unchanged - only the group definition (3 Venn regions instead of
toxic/non-toxic) and the comparison arithmetic differ.

THREE-WAY COMPARISON ARITHMETIC (generalizes tfidf_analysis.py's
2-class diferenca/log_ratio, which only had one "other" to compare
against): for each Venn region, its diferenca/log_ratio compares that
region's mean TF-IDF against the mean of the OTHER TWO regions combined -
"how much does this term stand out in this region, versus the other two
regions' reviews" - not against any single other region, which would need
three separate pairwise tables.

Usage: see run_tfidf_venn_set.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import tfidf_analysis as ta
from pipeline_utils import info, list_parquet_files

PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9
VENN_SETS = ["perspective_only", "detoxify_only", "both"]


def load_flagged_reviews_for_language(step04_dir: Path, lang: str) -> pd.DataFrame:
    """Reads step04's files, restricted to `lang` (both review_lang and
    perspective_declared_language must agree, same defense-in-depth check
    as tfidf_analysis.py's load_reviews_for_language), valid scores, and
    at least one of the two models flagging the review - reviews flagged
    by neither model don't belong to any of the three Venn regions.

    Filters each file down to its flagged rows BEFORE concatenating (not
    load-everything-then-filter) - keeps peak memory proportional to one
    file's worth of rows plus the accumulated (much smaller) flagged
    subset, and gives a per-file progress bar instead of one opaque wait."""
    files = list_parquet_files(step04_dir / f"review_lang={lang}")

    partials = []
    rows_loaded = 0
    bar = tqdm(files, desc=f"[{lang}]", unit="file")
    for f in bar:
        df = pd.read_parquet(f)
        rows_loaded += len(df)

        df = df[(df["review_lang"] == lang) & (df["perspective_declared_language"] == lang)]

        perspective_valid = df["perspective_score"].between(0, 1)
        detoxify_valid = df["detoxify_score"].between(0, 1)
        df = df[perspective_valid & detoxify_valid]

        is_p = df["perspective_score"] >= PERSPECTIVE_THRESHOLD
        is_d = df["detoxify_score"] >= DETOXIFY_THRESHOLD

        df = df.copy()
        df["venn_set"] = pd.NA
        df.loc[is_p & ~is_d, "venn_set"] = "perspective_only"
        df.loc[~is_p & is_d, "venn_set"] = "detoxify_only"
        df.loc[is_p & is_d, "venn_set"] = "both"
        df = df[df["venn_set"].notna()]

        if not df.empty:
            partials.append(df)
        bar.set_postfix(flagged=sum(len(p) for p in partials))

    info(f"[{lang}] Concatenating {len(partials)} partial(s)...")
    result = pd.concat(partials, ignore_index=True)
    info(f"[{lang}] Loaded {rows_loaded:,} row(s), {len(result):,} flagged (valid score, Perspective and/or Detoxify)")
    return result


def compute_set_means(df: pd.DataFrame, vectorizer, chunk_size: int = 500_000) -> dict:
    """Mean TF-IDF weight per term, separately for each of the three Venn
    regions - same chunked-accumulation pattern as tfidf_analysis.py's
    compute_group_means (bounds peak memory instead of transforming the
    whole corpus into one sparse matrix at once)."""
    n_features = len(vectorizer.get_feature_names_out())
    sums = {s: np.zeros(n_features, dtype=np.float64) for s in VENN_SETS}
    counts = {s: 0 for s in VENN_SETS}

    chunk_starts = list(range(0, len(df), chunk_size))
    for start in tqdm(chunk_starts, desc="[tfidf]", unit="chunk"):
        chunk = df.iloc[start:start + chunk_size]
        texts = chunk["review_text_clean"].fillna("").astype(str)
        X = vectorizer.transform(texts)

        for venn_set in VENN_SETS:
            mask = (chunk["venn_set"] == venn_set).to_numpy()
            if mask.any():
                sums[venn_set] += np.asarray(X[mask].sum(axis=0)).ravel()
                counts[venn_set] += int(mask.sum())

    means = {s: (sums[s] / counts[s] if counts[s] > 0 else sums[s]) for s in VENN_SETS}
    return {"means": means, "counts": counts}


def build_lexicon_table(vectorizer, result: dict) -> pd.DataFrame:
    """One row per term, with each Venn region's mean TF-IDF plus, for
    each region, its diferenca/log_ratio against the OTHER TWO regions
    combined (see module docstring for why - not a single pairwise
    comparison, since there isn't one "other" class here)."""
    terms = np.array(vectorizer.get_feature_names_out())
    means = result["means"]
    table = pd.DataFrame({"termo": terms})
    for venn_set in VENN_SETS:
        table[f"tfidf_{venn_set}"] = means[venn_set]

    for venn_set in VENN_SETS:
        others = [s for s in VENN_SETS if s != venn_set]
        other_mean = np.mean([means[s] for s in others], axis=0)
        table[f"diferenca_{venn_set}"] = table[f"tfidf_{venn_set}"] - other_mean
        table[f"log_ratio_{venn_set}"] = np.log2((table[f"tfidf_{venn_set}"] + 1e-9) / (other_mean + 1e-9))

    return table


def export_lexicon_table(table: pd.DataFrame, output_dir: Path, lang: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tfidf_venn_lexicon_{lang}.csv"
    table.to_csv(output_path, index=False)
    info(f"Exported Venn-set TF-IDF lexicon table ({lang}) to: {output_path}")
    return output_path
