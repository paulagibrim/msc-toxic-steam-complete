"""Publications per year (bars) with cumulative share (line) for the
literature review's selected papers - reads `year` straight out of the
Selected-papers.bib entries, since that field is all the bib file itself
carries. The other four charts the review wants (platform, game, data-type
counts, and a sample-size boxplot) need per-paper coding that isn't present
in a .bib (no abstracts, no methodology fields) - see the run script's
docstring.

Usage:
    python literature_review_year_distribution.py \\
        --bib Selected-papers.bib \\
        --output ../step0x_literature_review/charts/publications_per_year
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chart_style as cs
from pipeline_utils import info


def parse_args():
    parser = argparse.ArgumentParser(description="Publications-per-year bar chart with cumulative-% line, from a .bib file.")
    parser.add_argument("--bib", required=True, type=Path, help="Path to the .bib file")
    parser.add_argument("--output", required=True, type=Path, help="Path (no extension) to write the chart to")
    parser.add_argument("--tick-size", type=float, default=20, help="Font size of the axis numbers/years (default 20)")
    parser.add_argument("--label-size", type=float, default=18, help="Font size of the axis titles (default 18)")
    parser.add_argument("--legend-size", type=float, default=16, help="Font size of the legend text (default 16)")
    parser.add_argument("--line-width", type=float, default=3.5, help="Thickness of the cumulative-%% line (default 3.5)")
    return parser.parse_args()


def years_from_bib(bib_path: Path) -> list:
    text = bib_path.read_text(encoding="utf-8")
    years = [int(y) for y in re.findall(r"year\s*=\s*\{(\d{4})\}", text)]
    return years


def main():
    args = parse_args()
    years = years_from_bib(args.bib)
    info(f"Parsed {len(years)} entries from {args.bib}")

    counts = {}
    for y in years:
        counts[y] = counts.get(y, 0) + 1
    year_range = list(range(min(counts), max(counts) + 1))
    totals = [counts.get(y, 0) for y in year_range]
    cum_pct = 100 * np.cumsum(totals) / sum(totals)

    WHITE = "#ffffff"

    font_prop = cs.load_font()
    # Wide enough that twelve year labels at --tick-size sit side by side
    # without touching; at 11in they collide from ~16pt up.
    fig, ax1 = plt.subplots(figsize=(15, 6), facecolor=WHITE)
    ax2 = ax1.twinx()

    x = np.arange(len(year_range))
    ax1.bar(x, totals, color=cs.SERIES_1, width=0.62, zorder=3, label="ARTICLES PER YEAR")
    # Markers scale with the line so the dots stay visibly larger than the
    # stroke rather than being swallowed by a thicker one.
    ax2.plot(x, cum_pct, color=cs.SERIES_2, marker="o", markersize=args.line_width * 2.2,
             linewidth=args.line_width, zorder=4, label="CUMULATIVE %")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(y) for y in year_range], fontsize=args.tick_size)
    ax1.set_ylabel("TOTAL ARTICLES", fontsize=args.label_size, color=cs.INK, fontproperties=font_prop)
    ax2.set_ylabel("CUMULATIVE PERCENTAGE (%)", fontsize=args.label_size, color=cs.INK, fontproperties=font_prop)
    ax1.set_ylim(0, max(totals) * 1.05)
    ax2.set_ylim(0, 105)

    ax1.set_facecolor(WHITE)
    ax1.tick_params(axis="x", labelsize=args.tick_size, colors=cs.INK)
    ax1.tick_params(axis="y", labelsize=args.tick_size, colors=cs.INK)
    ax2.tick_params(axis="y", labelsize=args.tick_size, colors=cs.INK)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        # cs.apply_font, not a bare set_fontproperties: the latter carries
        # font_prop's own size and silently resets whatever tick_params/
        # set_xticklabels asked for, so the size has to be (re)applied after
        # it - not before. See chart_style.apply_font's docstring.
        cs.apply_font(label, font_prop)

    for side in ("top",):
        ax1.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom", "right"):
        ax1.spines[side].set_color(cs.SPINE)
        ax2.spines[side].set_color(cs.SPINE)

    ax1.grid(axis="y", color=cs.GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)

    bars_handle, bars_label = ax1.get_legend_handles_labels()
    line_handle, line_label = ax2.get_legend_handles_labels()
    legend = ax1.legend(bars_handle + line_handle, bars_label + line_label, loc="upper left", frameon=True, fontsize=args.legend_size)
    legend.get_frame().set_facecolor(WHITE)
    legend.get_frame().set_edgecolor(cs.SPINE)
    for text in legend.get_texts():
        cs.apply_font(text, font_prop)
        text.set_color(cs.INK)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, args.output)
    plt.close(fig)
    info(f"Saved chart to {png_path} / {pdf_path}")


if __name__ == "__main__":
    main()
