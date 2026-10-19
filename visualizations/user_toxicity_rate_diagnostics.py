"""Diagnostic charts for deciding step06's T_rate threshold (the minimum
share of a user's in-language reviews that must be toxic to label them a
toxic user), from the real distribution rather than a guess.

Reads step06_user_toxicity/build_user_rate_table.py's output: one row per
user, with n_total (every language), and n_<lang>/n_<lang>_toxic
(agreement-matched review/toxic-review counts per language - step02's
population, the same one every other analysis in this project uses).

THREE POPULATIONS, each analyzed on its own terms - not the full 10M+ user
roster, which would mostly show "this user has 0 reviews in this language",
a population-membership fact, not a coverage data point worth plotting:
  - pt: users with n_pt > 0 (at least one agreement-matched pt review).
  - en: users with n_en > 0.
  - pt+en (union): users with n_pt + n_en > 0.
For each population, two things are computed only within that population:
  - coverage: what share of the user's TOTAL activity (n_total, every
    language they wrote in at all, including ones never scored here) this
    language/union represents - e.g. 2 pt + 4 en + 5 es reviews gives pt
    coverage 2/11, en coverage 4/11, union coverage 6/11.
  - toxic rate: n_<lang>_toxic / n_<lang> (or the summed equivalent for the
    union) - restricted to that same language's own reviews, never mixed
    with the other language's counts.

Meant to run LOCALLY (not on the remote GPU box) - same reason as
run_language_distribution.py: needs the "Heading Now" font.

Three charts:
  - plot_coverage_cdf: cumulative distribution of coverage, computed only
    within each population (as defined above) - pt, en, and their union.
  - plot_rate_distribution: within each population, the distribution of
    toxic rate, restricted to users with rate > 0 (at least one toxic
    review) - the zero-rate majority is reported as a headline number
    instead of dominating the histogram.
  - plot_prevalence_vs_threshold: for a grid of candidate T_rate values,
    what fraction of each population would be labeled toxic at that rate.

pt/en are drawn with chart_style's fixed categorical pair (SERIES_1/
SERIES_2); the union series uses BOTH_COLOR (chart_style's "combination of
both" token, already established for exactly this kind of "both together"
series in the Venn diagrams) rather than introducing a third ad hoc color.
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
POP_COLOR = {"pt": cs.SERIES_1, "en": cs.SERIES_2, UNION_KEY: cs.BOTH_COLOR}
POP_LABEL = {"pt": "Portuguese", "en": "English", UNION_KEY: "Portuguese + English (union)"}


def load_user_rate_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    info(f"Loaded user rate table: {len(df)} user(s) from {path}")
    return df


def _n_and_toxic(table: pd.DataFrame, population: str) -> tuple:
    """Returns (n, n_toxic) Series for one population - pt/en read straight
    off their columns, the union summed from both."""
    if population == UNION_KEY:
        return table["n_pt"] + table["n_en"], table["n_pt_toxic"] + table["n_en_toxic"]
    return table[f"n_{population}"], table[f"n_{population}_toxic"]


def population_mask(table: pd.DataFrame, population: str) -> pd.Series:
    """Users who belong to this population: at least one agreement-matched
    review in the language (or in either language, for the union) - NOT
    "coverage >= some threshold". A user with zero pt reviews has no pt
    analysis at all; they're excluded from the pt population entirely,
    rather than plotted as a 0%-coverage data point."""
    n, _ = _n_and_toxic(table, population)
    return n > 0


def coverage(table: pd.DataFrame, population: str) -> pd.Series:
    """Within `population` only: share of the user's TOTAL activity
    (n_total, every language) this population's reviews represent."""
    mask = population_mask(table, population)
    n, _ = _n_and_toxic(table, population)
    return (n[mask] / table.loc[mask, "n_total"])


def toxic_rate(table: pd.DataFrame, population: str) -> pd.Series:
    """Within `population` only: fraction of its own reviews that are
    toxic. Never mixes a population's numerator with another's
    denominator (e.g. pt's rate is always n_pt_toxic / n_pt)."""
    mask = population_mask(table, population)
    n, n_toxic = _n_and_toxic(table, population)
    return (n_toxic[mask] / n[mask])


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


def plot_coverage_cdf(table: pd.DataFrame, output_path: Path, candidate_thresholds: list = None) -> tuple:
    """CDF of coverage, computed only within each population (see module
    docstring) - x is the coverage fraction (0-1), y is the cumulative
    share of THAT POPULATION at or below it. Vertical guide lines at
    `candidate_thresholds` (default 0.80/0.90/0.95)."""
    candidate_thresholds = candidate_thresholds or [0.80, 0.90, 0.95]
    font_prop = cs.load_font()

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=cs.SURFACE)

    for pop in POPULATIONS:
        cov = coverage(table, pop).sort_values().to_numpy()
        y = np.arange(1, len(cov) + 1) / len(cov)
        ax.plot(
            cov, y, color=POP_COLOR[pop], linewidth=2,
            label=f"{POP_LABEL[pop].upper()} (n={len(cov):,})", zorder=3,
        )

    for t in candidate_thresholds:
        ax.axvline(t, color=cs.INK_MUTED, linewidth=0.8, linestyle="--", zorder=2)
        ax.text(
            t, 1.02, f"{int(t * 100)}%", ha="center", va="bottom",
            fontsize=8, color=cs.INK_MUTED, fontproperties=font_prop,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v * 100)}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(v * 100)}%")
    ax.set_xlabel(
        "SHARE OF USER'S TOTAL ACTIVITY IN THIS POPULATION",
        fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop,
    )
    _style_axes(ax, "CUMULATIVE SHARE OF POPULATION", font_prop)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_rate_distribution(table: pd.DataFrame, output_path: Path, bins: int = 30) -> tuple:
    """Histogram of toxic rate within each population - restricted to
    users with rate > 0 (at least one toxic review), since the zero-rate
    majority (matching the corpus's low overall toxicity) would otherwise
    fill the chart with a single dominant bar and hide the shape of
    interest. The zero-rate share is reported as a text annotation instead
    (it's the number that matters there, not a bar)."""
    font_prop = cs.load_font()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=cs.SURFACE)

    max_rate = 0.0
    for pop in POPULATIONS:
        rate = toxic_rate(table, pop)
        n_pop = len(rate)
        n_nonzero = int((rate > 0).sum())
        pct_nonzero = 100 * n_nonzero / n_pop if n_pop else 0.0
        info(
            f"[{pop}] {n_pop} user(s) in population - "
            f"{n_nonzero} ({pct_nonzero:.2f}%) have at least one toxic review"
        )

        nonzero = rate[rate > 0]
        if len(nonzero):
            max_rate = max(max_rate, nonzero.max())
        ax.hist(
            nonzero, bins=bins, range=(0, 1), color=POP_COLOR[pop], alpha=0.6,
            label=f"{POP_LABEL[pop].upper()} ({pct_nonzero:.1f}% of population have any toxic review)",
            zorder=3, edgecolor=cs.SURFACE, linewidth=0.5,
        )

    ax.set_xlim(0, min(1.0, max_rate * 1.1) if max_rate else 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v * 100)}%")
    ax.set_xlabel(
        "TOXIC-REVIEW RATE (AMONG USERS WITH >=1 TOXIC REVIEW)",
        fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop,
    )
    _style_axes(ax, "USER COUNT", font_prop)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_prevalence_vs_threshold(table: pd.DataFrame, output_path: Path, thresholds: list = None) -> tuple:
    """For each candidate T_rate, the resulting toxic-user prevalence (% of
    that population whose toxic rate clears it) - pt, en, and their union.
    Lets a threshold be picked by reading off a target prevalence (e.g.
    this project's reference paper: 4.0% for its count label, 12.8% for
    its rate label) or a visible elbow in the curve."""
    thresholds = thresholds if thresholds is not None else [t / 1000 for t in range(0, 501, 5)]
    font_prop = cs.load_font()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=cs.SURFACE)

    for pop in POPULATIONS:
        rate = toxic_rate(table, pop)
        n_pop = len(rate)
        prevalence = [100 * (rate >= t).sum() / n_pop if n_pop else 0.0 for t in thresholds]
        ax.plot(
            [t * 100 for t in thresholds], prevalence,
            color=POP_COLOR[pop], linewidth=2, label=POP_LABEL[pop].upper(), zorder=3,
        )

    ax.set_xlabel("CANDIDATE RATE THRESHOLD (T_RATE)", fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    _style_axes(ax, "RESULTING TOXIC-USER PREVALENCE", font_prop)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path
