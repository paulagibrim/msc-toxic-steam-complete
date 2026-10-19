"""Charts for step06's sentiment-based validation of toxic_user_labels.parquet
(see step06_user_toxicity/phase1_feature_engineering/diagnostics/03_validate_
labels_with_sentiment.py for the underlying comparison and why it's valid to
compare toxic vs non-toxic users' sentiment using ALL their reviews - this
is the PRIMARY validation, not something the leave-toxic-out control
corrects).

Reads:
  - phase1_feature_engineering/pipeline/toxic_user_labels.parquet (eligible_*,
    is_toxic_* per population)
  - phase1_feature_engineering/pipeline/user_sentiment_table.parquet
    (avg_sentiment_pt/en/union, nlptown's 1.0-5.0 expected-star-rating)

THREE POPULATIONS (pt/en/union), each with TWO groups (toxic, non-toxic) -
same eligibility definition as everywhere else in this project (users with
eligible_<pop>=True). SERIES_1 (blue) is non-toxic (the baseline majority),
SERIES_2 (amber) is toxic (the flagged group) - matches this project's
established two-way categorical pairing.

Three chart types, all comparing the same two groups per population:
  - plot_sentiment_boxplot: median/IQR summary, side-by-side per population.
  - plot_sentiment_violin: full distribution shape (boxplot alone hides
    bimodality/skew that matters here, since sentiment_score's plausible
    range is bounded 1.0-5.0 and often piles up near the ends).
  - plot_sentiment_cdf: cumulative distribution, one subplot per population
    (rather than 6 lines on one axis) so the toxic/non-toxic comparison
    within each population stays the visually dominant contrast.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chart_style as cs
from pipeline_utils import info

UNION_KEY = "pt+en"
POPULATIONS = ["pt", "en", UNION_KEY]
POP_LABEL = {"pt": "Portuguese", "en": "English", UNION_KEY: "Portuguese + English (union)"}
NONTOXIC_COLOR = cs.SERIES_1
TOXIC_COLOR = cs.SERIES_2


def load_data(labels_path: Path, sentiment_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    info(f"Loaded toxic user labels: {len(labels):,} user(s) from {labels_path}")
    sentiment = pd.read_parquet(sentiment_path)
    info(f"Loaded sentiment table: {len(sentiment):,} user(s) from {sentiment_path}")
    table = labels.merge(sentiment, on="user_url", how="left")
    return table


def _population_groups(table: pd.DataFrame, population: str) -> tuple:
    """Returns (nontoxic_sentiment, toxic_sentiment) Series for one
    population - eligible users only, dropping anyone with no sentiment
    score (never scored, or no valid agreement-matched review)."""
    sentiment_label = "union" if population == UNION_KEY else population
    label_suffix = population

    eligible = table[f"eligible_{label_suffix}"].fillna(False)
    subset = table.loc[eligible]
    is_toxic = subset[f"is_toxic_{label_suffix}"].fillna(False)

    sentiment_col = f"avg_sentiment_{sentiment_label}"
    has_sentiment = subset[sentiment_col].notna()
    subset = subset.loc[has_sentiment]
    is_toxic = is_toxic.loc[has_sentiment]

    return subset.loc[~is_toxic, sentiment_col], subset.loc[is_toxic, sentiment_col]


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


def plot_sentiment_boxplot(table: pd.DataFrame, output_path: Path) -> tuple:
    """Grouped boxplot: for each population, a non-toxic box (blue) next to
    a toxic box (amber) - median/IQR summary of avg_sentiment per user."""
    font_prop = cs.load_font()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=cs.SURFACE)

    positions = []
    box_data = []
    colors = []
    counts = []
    for i, pop in enumerate(POPULATIONS):
        nontoxic, toxic = _population_groups(table, pop)
        center = i * 3
        positions.extend([center - 0.4, center + 0.4])
        box_data.extend([nontoxic.to_numpy(), toxic.to_numpy()])
        colors.extend([NONTOXIC_COLOR, TOXIC_COLOR])
        counts.extend([len(nontoxic), len(toxic)])

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.7, patch_artist=True,
        showfliers=False, medianprops={"color": cs.INK, "linewidth": 1.5}, zorder=3,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(cs.INK)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(cs.INK_MUTED)

    group_centers = [i * 3 for i in range(len(POPULATIONS))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([POP_LABEL[pop] for pop in POPULATIONS])

    ax.set_ylim(1.0, 5.0)
    _style_axes(ax, "AVERAGE SENTIMENT PER USER (1-5)", font_prop)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=NONTOXIC_COLOR, edgecolor=cs.INK, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, facecolor=TOXIC_COLOR, edgecolor=cs.INK, alpha=0.85),
    ]
    legend = ax.legend(handles, ["NON-TOXIC", "TOXIC"], loc="lower left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_sentiment_violin(table: pd.DataFrame, output_path: Path) -> tuple:
    """Grouped violin: full distribution shape per population/group - a
    boxplot alone hides bimodality/piling near sentiment_score's 1.0/5.0
    bounds, which matters here since reviews often land near the extremes."""
    font_prop = cs.load_font()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=cs.SURFACE)

    group_centers = []
    for i, pop in enumerate(POPULATIONS):
        nontoxic, toxic = _population_groups(table, pop)
        center = i * 3
        group_centers.append(center)

        for offset, data, color in [(-0.4, nontoxic, NONTOXIC_COLOR), (0.4, toxic, TOXIC_COLOR)]:
            if len(data) < 2:
                continue
            parts = ax.violinplot(
                [data.to_numpy()], positions=[center + offset], widths=0.7,
                showmedians=True, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(cs.INK)
                body.set_alpha(0.85)
                body.set_zorder(3)
            parts["cmedians"].set_color(cs.INK)
            parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([POP_LABEL[pop] for pop in POPULATIONS])

    ax.set_ylim(1.0, 5.0)
    _style_axes(ax, "AVERAGE SENTIMENT PER USER (1-5)", font_prop)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=NONTOXIC_COLOR, edgecolor=cs.INK, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, facecolor=TOXIC_COLOR, edgecolor=cs.INK, alpha=0.85),
    ]
    legend = ax.legend(handles, ["NON-TOXIC", "TOXIC"], loc="lower left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_sentiment_cdf(table: pd.DataFrame, output_path: Path) -> tuple:
    """Cumulative distribution of avg_sentiment, one subplot per population
    (not 6 lines on one axis) so the toxic/non-toxic contrast within each
    population - the comparison that matters - stays visually dominant."""
    font_prop = cs.load_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=cs.SURFACE, sharey=True)

    for ax, pop in zip(axes, POPULATIONS):
        nontoxic, toxic = _population_groups(table, pop)

        for data, color, label in [(nontoxic, NONTOXIC_COLOR, "NON-TOXIC"), (toxic, TOXIC_COLOR, "TOXIC")]:
            values = np.sort(data.to_numpy())
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, color=color, linewidth=2, label=f"{label} (n={len(values):,})", zorder=3)

        ax.set_xlim(1.0, 5.0)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{int(v * 100)}%")
        ax.set_xlabel(
            "AVG SENTIMENT PER USER", fontsize=10, color=cs.INK_SECONDARY, fontproperties=font_prop,
        )
        ax.set_title(POP_LABEL[pop].upper(), fontsize=10, color=cs.INK_SECONDARY, fontproperties=font_prop, pad=10)
        _style_axes(ax, "CUMULATIVE SHARE OF GROUP" if pop == "pt" else "", font_prop)

        legend = ax.legend(loc="upper left", frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)
            text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path
