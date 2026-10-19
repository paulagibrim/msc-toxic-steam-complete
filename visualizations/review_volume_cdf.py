"""CDF of review volume (n_pt / n_en) per user, within each population's
own coverage-filtered eligibility (n_<lang> > 0 AND coverage >= 90% - the
same T_coverage decided for step06's toxic-user labels). Built to visually
inspect how much of each population sits at very low review counts (n=1,2,3...),
since that discreteness is what makes both the T_coverage and T_rate/alpha
thresholds behave in sharp, non-smooth steps rather than continuous curves.

Reads step06_user_toxicity/build_user_rate_table.py's output directly (not
the toxic-user labels) - this is about raw review-count volume, independent
of any toxicity labeling threshold.

Usage:
    python review_volume_cdf.py \\
        --table ../../steam-data/step06-output/user_rate_table.parquet \\
        --output ../../steam-data/step06-output/charts/review_volume_cdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chart_style as cs
from pipeline_utils import info

T_COVERAGE = 0.90
LANGUAGES = ["pt", "en"]
LANG_COLOR = {"pt": cs.SERIES_1, "en": cs.SERIES_2}
LANG_LABEL = {"pt": "Portuguese", "en": "English"}


def parse_args():
    parser = argparse.ArgumentParser(description="CDF of per-user review volume (n_pt/n_en), within each language's eligible population.")
    parser.add_argument("--table", required=True, type=Path, help="Path to build_user_rate_table.py's output parquet")
    parser.add_argument("--output", required=True, type=Path, help="Path (no extension) to write the chart to")
    parser.add_argument("--max-n", type=int, default=20, help="X-axis cutoff (default 20) - most of the mass is at very low n")
    return parser.parse_args()


def eligible_n(table: pd.DataFrame, lang: str) -> pd.Series:
    n = table[f"n_{lang}"]
    coverage = n / table["n_total"]
    mask = (n > 0) & (coverage >= T_COVERAGE)
    return n[mask]


def main():
    args = parse_args()
    table = pd.read_parquet(args.table)
    info(f"Loaded user rate table: {len(table)} user(s) from {args.table}")

    font_prop = cs.load_font()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=cs.SURFACE)

    for lang in LANGUAGES:
        n = eligible_n(table, lang).sort_values().to_numpy()
        y = np.arange(1, len(n) + 1) / len(n)
        ax.plot(n, y, color=LANG_COLOR[lang], linewidth=2, label=f"{LANG_LABEL[lang].upper()} (n={len(n):,})", zorder=3)

        pct_at_1 = 100 * (n == 1).mean()
        pct_at_2 = 100 * (n <= 2).mean()
        info(f"[{lang}] {pct_at_1:.1f}% of eligible users have exactly 1 review; {pct_at_2:.1f}% have <= 2")

    ax.set_xlim(1, args.max_n)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(v * 100)}%")
    ax.set_xlabel("REVIEWS PER USER (IN-LANGUAGE, ELIGIBLE POPULATION)", fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop)
    ax.set_ylabel("CUMULATIVE SHARE OF POPULATION", fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop)
    ax.set_facecolor(cs.SURFACE)
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

    legend = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, args.output)
    plt.close(fig)
    info(f"Saved chart to {png_path} / {pdf_path}")


if __name__ == "__main__":
    main()
