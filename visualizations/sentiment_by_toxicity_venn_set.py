"""Sentiment profile of reviews, grouped by which toxicity-detection set
they fall into - the same five sets run_toxicity_venn.py's Venn diagram is
built from: PERSPECTIVE (total), DETOXIFY (total), PERSPECTIVE ONLY,
DETOXIFY ONLY, and BOTH (perspective_score >= 0.7 AND detoxify_score >=
0.9 - same union rule/thresholds as everywhere else in this project).

Reads step04-output directly (review_lang=pt/en partitions), which already
carries perspective_score, detoxify_score AND sentiment_score together per
review - no join needed. Rows with an invalid/sentinel score in EITHER
model (Detoxify's -1.0 "failed to score") are excluded from every set, not
just the ones that touch that model directly - same defense-in-depth rule
used throughout this project (e.g. detoxify_scoring.py, text_cleaning.py).

WHY THE CHART SHOWS ONLY THREE GROUPS, NOT FIVE: PERSPECTIVE (total) and
DETOXIFY (total) are not disjoint from BOTH - they're the two full circles
of the Venn, each already containing the overlap. Plotting all five
side-by-side would silently double-count the overlap's reviews into two
bars. The chart shows the three MUTUALLY EXCLUSIVE Venn regions instead
(PERSPECTIVE ONLY / DETOXIFY ONLY / BOTH, using chart_style's established
SERIES_1/SERIES_2/BOTH_COLOR triple - the same three colours
run_toxicity_venn.py already uses, so a reader can carry the mapping from
one figure to the other) - the two totals are still reported, in the
summary statistics table only, where a reader expects an aggregate rather
than a plotted region.

Usage:
    python sentiment_by_toxicity_venn_set.py \\
        --step04-dir ../../steam-data/step04-output \\
        --output-dir ../../steam-data/visualizations/sentiment_by_toxicity_set
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import chart_style as cs
from pipeline_utils import info


def save_summary(summary: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    info(f"Saved run summary to: {output_path}")
    return output_path

PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9
LANGUAGES = ["pt", "en"]
LANG_LABEL = {"pt": "Portuguese", "en": "English", "total": "Portuguese + English"}

EXCLUSIVE_SETS = ["perspective_only", "detoxify_only", "both"]
SET_COLOR = {"perspective_only": cs.SERIES_1, "detoxify_only": cs.SERIES_2, "both": cs.BOTH_COLOR}
SET_LABEL = {"perspective_only": "PERSPECTIVE ONLY", "detoxify_only": "DETOXIFY ONLY", "both": "BOTH MODELS"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sentiment profile of reviews per Perspective/Detoxify toxicity-detection set."
    )
    parser.add_argument("--step04-dir", required=True, type=Path, help="Path to step04's output directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write charts/report to")
    return parser.parse_args()


def load_sentiment_by_set(step04_dir: Path) -> pd.DataFrame:
    """Single sweep over step04's files: for each valid-scored review,
    records its language and sentiment_score, tagged with which of the
    three mutually-exclusive Venn regions it falls in. Returns a long
    DataFrame: review_lang, venn_set, sentiment_score."""
    files = sorted(step04_dir.rglob("*.parquet"))
    info(f"[sentiment] Sweeping across {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step04_dir} (searched recursively).")

    columns = ["review_lang", "perspective_score", "detoxify_score", "sentiment_score"]
    partials = []
    n_reviews = 0

    bar = tqdm(files, desc="[sentiment]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=columns)

        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]
        if df.empty:
            bar.set_postfix(reviews=n_reviews)
            continue

        is_p = df["perspective_score"] >= PERSPECTIVE_THRESHOLD
        is_d = df["detoxify_score"] >= DETOXIFY_THRESHOLD

        venn_set = pd.Series(pd.NA, index=df.index, dtype="object")
        venn_set[is_p & ~is_d] = "perspective_only"
        venn_set[~is_p & is_d] = "detoxify_only"
        venn_set[is_p & is_d] = "both"

        kept = pd.DataFrame({
            "review_lang": df["review_lang"],
            "venn_set": venn_set,
            "sentiment_score": df["sentiment_score"],
        })
        kept = kept[kept["venn_set"].notna()]
        if kept.empty:
            bar.set_postfix(reviews=n_reviews)
            continue

        partials.append(kept)
        n_reviews += len(kept)
        bar.set_postfix(reviews=n_reviews)

    info("[sentiment] Concatenating partials...")
    result = pd.concat(partials, ignore_index=True)
    info(f"[sentiment] {len(result):,} flagged review(s) (Perspective and/or Detoxify) with a valid sentiment score")
    return result


def _stats_for(values: pd.Series) -> dict:
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 4) if len(values) else None,
        "median": round(float(values.median()), 4) if len(values) else None,
        "std": round(float(values.std()), 4) if len(values) else None,
        "p25": round(float(values.quantile(0.25)), 4) if len(values) else None,
        "p75": round(float(values.quantile(0.75)), 4) if len(values) else None,
    }


def compute_summary(data: pd.DataFrame, lang_scope: str) -> dict:
    """Five-set statistics table for one language scope ('pt', 'en', or
    'total' for both combined) - the two totals (perspective/detoxify) are
    reported here even though the chart only plots the three exclusive
    regions (see module docstring for why)."""
    subset = data if lang_scope == "total" else data[data["review_lang"] == lang_scope]

    only_p = subset.loc[subset["venn_set"] == "perspective_only", "sentiment_score"]
    only_d = subset.loc[subset["venn_set"] == "detoxify_only", "sentiment_score"]
    both = subset.loc[subset["venn_set"] == "both", "sentiment_score"]
    perspective_total = pd.concat([only_p, both])
    detoxify_total = pd.concat([only_d, both])

    return {
        "perspective_total": _stats_for(perspective_total),
        "detoxify_total": _stats_for(detoxify_total),
        "perspective_only": _stats_for(only_p),
        "detoxify_only": _stats_for(only_d),
        "both": _stats_for(both),
    }


def _style_axes(ax, ylabel: str, font_prop) -> None:
    ax.set_facecolor(cs.SURFACE)
    ax.set_ylabel(ylabel, fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop)
    ax.tick_params(axis="x", labelsize=10, colors=cs.INK)
    ax.tick_params(axis="y", labelsize=9, colors=cs.INK_MUTED)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(cs.SPINE)
    ax.grid(axis="y", color=cs.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_sentiment_by_set(data: pd.DataFrame, output_path: Path, kind: str = "box") -> tuple:
    """Grouped box/violin plot: for each language scope (pt, en, total),
    the sentiment distribution of the three mutually-exclusive Venn
    regions, side by side."""
    font_prop = cs.load_font()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=cs.SURFACE)

    scopes = ["pt", "en", "total"]
    group_centers = []
    for i, scope in enumerate(scopes):
        subset = data if scope == "total" else data[data["review_lang"] == scope]
        center = i * 5
        group_centers.append(center)

        for j, venn_set in enumerate(EXCLUSIVE_SETS):
            values = subset.loc[subset["venn_set"] == venn_set, "sentiment_score"].to_numpy()
            offset = (j - 1) * 1.1
            color = SET_COLOR[venn_set]
            if len(values) < 2:
                continue
            if kind == "box":
                bp = ax.boxplot(
                    [values], positions=[center + offset], widths=1.0, patch_artist=True,
                    showfliers=False, medianprops={"color": cs.INK, "linewidth": 1.5}, zorder=3,
                )
                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(0.85)
                bp["boxes"][0].set_edgecolor(cs.INK)
                for element in ("whiskers", "caps"):
                    for line in bp[element]:
                        line.set_color(cs.INK_MUTED)
            else:
                parts = ax.violinplot([values], positions=[center + offset], widths=1.0, showmedians=True, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(cs.INK)
                    body.set_alpha(0.85)
                    body.set_zorder(3)
                parts["cmedians"].set_color(cs.INK)
                parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([LANG_LABEL[s].upper() for s in scopes])
    ax.set_ylim(1.0, 5.0)
    _style_axes(ax, "SENTIMENT SCORE (1-5)", font_prop)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SET_COLOR[s], edgecolor=cs.INK, alpha=0.85) for s in EXCLUSIVE_SETS]
    legend = ax.legend(handles, [SET_LABEL[s] for s in EXCLUSIVE_SETS], loc="lower left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    suffix = "box" if kind == "box" else "violin"
    png_path, pdf_path = cs.save_figure(fig, output_path.parent / f"{output_path.name}_{suffix}")
    plt.close(fig)
    return png_path, pdf_path


def plot_sentiment_cdf(data: pd.DataFrame, output_path: Path) -> tuple:
    """Cumulative distribution of sentiment_score, one subplot per language
    scope (pt/en/total) - not one axis with 9 lines - so the three-way
    contrast that matters (Perspective only / Detoxify only / Both) stays
    visually dominant within each scope, same layout convention as
    sentiment_validation_diagnostics.py's plot_sentiment_cdf."""
    font_prop = cs.load_font()
    scopes = ["pt", "en", "total"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=cs.SURFACE, sharey=True)

    for ax, scope in zip(axes, scopes):
        subset = data if scope == "total" else data[data["review_lang"] == scope]

        for venn_set in EXCLUSIVE_SETS:
            values = np.sort(subset.loc[subset["venn_set"] == venn_set, "sentiment_score"].to_numpy())
            if len(values) == 0:
                continue
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(
                values, y, color=SET_COLOR[venn_set], linewidth=2,
                label=f"{SET_LABEL[venn_set]} (n={len(values):,})", zorder=3,
            )

        ax.set_xlim(1.0, 5.0)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{int(v * 100)}%")
        ax.set_xlabel("SENTIMENT SCORE", fontsize=10, color=cs.INK_SECONDARY, fontproperties=font_prop)
        ax.set_title(LANG_LABEL[scope].upper(), fontsize=10, color=cs.INK_SECONDARY, fontproperties=font_prop, pad=10)
        _style_axes(ax, "CUMULATIVE SHARE OF SET" if scope == "pt" else "", font_prop)

        legend = ax.legend(loc="upper left", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)
            text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path.parent / f"{output_path.name}_cdf")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    data = load_sentiment_by_set(args.step04_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_sentiment_by_set(data, args.output_dir / "sentiment_by_toxicity_set", kind="box")
    plot_sentiment_by_set(data, args.output_dir / "sentiment_by_toxicity_set", kind="violin")
    plot_sentiment_cdf(data, args.output_dir / "sentiment_by_toxicity_set")

    summary = {scope: compute_summary(data, scope) for scope in ["pt", "en", "total"]}
    for scope, stats in summary.items():
        info(f"=== {LANG_LABEL[scope]} ===")
        for set_name, s in stats.items():
            info(f"  [{set_name}] n={s['n']:,} mean={s['mean']} median={s['median']} std={s['std']}")

    save_summary(summary, args.output_dir / "sentiment_by_toxicity_set.summary.json")


if __name__ == "__main__":
    main()
