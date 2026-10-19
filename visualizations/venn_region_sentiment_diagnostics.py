"""Charts for step06_user_toxicity/phase1_feature_engineering/diagnostics/
05_sentiment_facets_report.py's part (ii): sentiment_score distribution
per toxicity Venn region (perspective_only/detoxify_only/both/neither),
side by side per language (pt/en) - shows visually what facet each
detector captures (that script's JSON only has summary stats; this reads
step04 directly for the raw per-review values a box/violin plot needs).

SAME FILTERS as 05_sentiment_facets_report.py: agreement-matched rows only
(review_lang == perspective_declared_language), valid perspective/detoxify
scores, valid (non-null) sentiment_score.

SUBSAMPLING: "neither" (non-toxic) has ~27M rows - a violin's KDE (and even
a boxplot's rendering) gains nothing from plotting all of them over a
capped random sample, so every region is capped at MAX_SAMPLE_PER_GROUP
(default 50,000, fixed random_state for reproducibility) if it's larger;
smaller regions (perspective_only/detoxify_only/both) are used in full
whenever they're already below the cap. Only affects what's DRAWN - the
underlying counts (n=...) shown in each group's label are always the TRUE
full-population count from before subsampling, not the sample size.

Color mapping (four Venn regions, matching this project's Venn-diagram
convention in run_toxicity_venn.py): perspective_only=SERIES_1 (blue),
detoxify_only=SERIES_2 (amber), both=BOTH_COLOR (the deck's near-black,
already established as "both models agree"), neither=UND_COLOR (grey -
the large, non-highlighted background group, not a fourth "real"
category in the same sense as the other three).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import chart_style as cs
from pipeline_utils import info

LANGUAGES = ["pt", "en"]
PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9
VENN_REGIONS = ["perspective_only", "detoxify_only", "both", "neither"]
REGION_LABEL = {
    "perspective_only": "Perspective only",
    "detoxify_only": "Detoxify only",
    "both": "Both",
    "neither": "Neither",
}
REGION_COLOR = {
    "perspective_only": cs.SERIES_1,
    "detoxify_only": cs.SERIES_2,
    "both": cs.BOTH_COLOR,
    "neither": cs.UND_COLOR,
}
MAX_SAMPLE_PER_GROUP = 50_000
RANDOM_STATE = 42


def _venn_region(perspective_flag: pd.Series, detoxify_flag: pd.Series) -> pd.Series:
    region = pd.Series("neither", index=perspective_flag.index, dtype="object")
    region[perspective_flag & ~detoxify_flag] = "perspective_only"
    region[~perspective_flag & detoxify_flag] = "detoxify_only"
    region[perspective_flag & detoxify_flag] = "both"
    return region


def load_sampled_sentiment_by_region(step04_dir: Path) -> dict:
    """Single sweep over step04's files. Returns {(lang, region): {"values": np.ndarray (subsampled),
    "n_total": int}} - values capped at MAX_SAMPLE_PER_GROUP via reservoir-style
    concat-then-sample per file batch (simplest correct approach here: collect
    all matching values first since even the full "neither" columns fit
    comfortably as a single float32 column across the whole corpus, then
    subsample once at the end - same shuffle-free "collect partials, single
    final combine" pattern as this project's other diagnostics)."""
    files = sorted(step04_dir.rglob("*.parquet"))
    info(f"[sweep] Reading {len(files)} file(s)...")
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {step04_dir} (searched recursively).")

    columns = ["review_lang", "perspective_declared_language", "perspective_score", "detoxify_score", "sentiment_score"]
    partials = {(lang, region): [] for lang in LANGUAGES for region in VENN_REGIONS}

    bar = tqdm(files, desc="[sweep]", unit="file")
    for f in bar:
        df = pd.read_parquet(f, columns=columns)
        df = df[df["review_lang"] == df["perspective_declared_language"]]
        p_valid = df["perspective_score"].between(0, 1)
        d_valid = df["detoxify_score"].between(0, 1)
        df = df[p_valid & d_valid]
        df = df[df["sentiment_score"].notna()]
        df = df[df["review_lang"].isin(LANGUAGES)]
        if df.empty:
            continue

        p_flag = df["perspective_score"] >= PERSPECTIVE_THRESHOLD
        d_flag = df["detoxify_score"] >= DETOXIFY_THRESHOLD
        region = _venn_region(p_flag, d_flag)

        for lang in LANGUAGES:
            lang_mask = df["review_lang"] == lang
            for reg in VENN_REGIONS:
                mask = lang_mask & (region == reg)
                if mask.any():
                    partials[(lang, reg)].append(df.loc[mask, "sentiment_score"].to_numpy(dtype="float32"))
        bar.set_postfix(rows=len(df))

    info("[sweep] Concatenating + subsampling per group...")
    rng = np.random.default_rng(RANDOM_STATE)
    result = {}
    for key, chunks in partials.items():
        if not chunks:
            result[key] = {"values": np.array([], dtype="float32"), "n_total": 0}
            continue
        values = np.concatenate(chunks)
        n_total = len(values)
        if n_total > MAX_SAMPLE_PER_GROUP:
            values = rng.choice(values, size=MAX_SAMPLE_PER_GROUP, replace=False)
        result[key] = {"values": values, "n_total": n_total}
        info(f"[sweep] [{key[0]}] [{key[1]}] n_total={n_total:,}, sampled={len(values):,}")

    return result


def _style_axes(ax, ylabel: str, font_prop) -> None:
    ax.set_facecolor(cs.SURFACE)
    ax.set_ylabel(ylabel, fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop)
    ax.tick_params(axis="x", labelsize=9, colors=cs.INK)
    ax.tick_params(axis="y", labelsize=9, colors=cs.INK_MUTED)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(cs.SPINE)
    ax.grid(axis="y", color=cs.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_venn_region_boxplot(data: dict, output_path: Path) -> tuple:
    font_prop = cs.load_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor=cs.SURFACE, sharey=True)

    for ax, lang in zip(axes, LANGUAGES):
        box_data = [data[(lang, region)]["values"] for region in VENN_REGIONS]
        colors = [REGION_COLOR[region] for region in VENN_REGIONS]

        bp = ax.boxplot(
            box_data, positions=range(len(VENN_REGIONS)), widths=0.6, patch_artist=True,
            showfliers=False, medianprops={"color": cs.SURFACE, "linewidth": 1.5}, zorder=3,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
            patch.set_edgecolor(cs.INK)
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color(cs.INK_MUTED)

        labels = [
            f"{REGION_LABEL[region]}\n(n={data[(lang, region)]['n_total']:,})" for region in VENN_REGIONS
        ]
        ax.set_xticks(range(len(VENN_REGIONS)))
        ax.set_xticklabels(labels)
        ax.set_ylim(1.0, 5.0)
        ax.set_title(lang.upper(), fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop, pad=10)
        _style_axes(ax, "SENTIMENT SCORE (1-5)" if lang == "pt" else "", font_prop)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_venn_region_violin(data: dict, output_path: Path) -> tuple:
    font_prop = cs.load_font()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), facecolor=cs.SURFACE, sharey=True)

    for ax, lang in zip(axes, LANGUAGES):
        for i, region in enumerate(VENN_REGIONS):
            values = data[(lang, region)]["values"]
            if len(values) < 2:
                continue
            parts = ax.violinplot([values], positions=[i], widths=0.7, showmedians=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(REGION_COLOR[region])
                body.set_edgecolor(cs.INK)
                body.set_alpha(0.9)
                body.set_zorder(3)
            parts["cmedians"].set_color(cs.INK)
            parts["cmedians"].set_linewidth(1.5)

        labels = [
            f"{REGION_LABEL[region]}\n(n={data[(lang, region)]['n_total']:,})" for region in VENN_REGIONS
        ]
        ax.set_xticks(range(len(VENN_REGIONS)))
        ax.set_xticklabels(labels)
        ax.set_ylim(1.0, 5.0)
        ax.set_title(lang.upper(), fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop, pad=10)
        _style_axes(ax, "SENTIMENT SCORE (1-5)" if lang == "pt" else "", font_prop)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path
