"""The four non-year literature-review figures: platforms studied, games
studied, how many papers report each data type, and how large those datasets
are.

All four read literature_review_data.py - see that module for where each
number comes from and how it was checked. Nothing here parses the Notion
export at runtime; the coding is the reviewed artifact, this file only draws
it.

Every mark here is one flat fill in the project's blue (chart_style's
SERIES_1). An earlier version ran a sequential ramp across the sorted bars;
it encoded nothing the bar length and axis did not already carry, and it
left these four figures looking unlike literature_review_year_distribution
.py's chart, which has always drawn its bars in the flat SERIES_1. One
colour across all five is the chapter's convention.

Usage:
    python literature_review_charts.py --outdir path/to/charts
    python literature_review_charts.py --outdir path/to/charts --only platforms
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chart_style as cs
import literature_review_data as data
from pipeline_utils import info

# Same white surface the year chart was switched to, so the five figures in
# the chapter share one background.
WHITE = "#ffffff"

# One fill for every mark in this file - see the module docstring.
BAR_COLOR = cs.SERIES_1

# Plot-rectangle margins shared by every horizontal bar chart here - see
# horizontal_bars for why they are fixed rather than fitted per figure.
LEFT_MARGIN = 0.30
BOTTOM_MARGIN = 0.11

# Same idea for the two vertical figures (data types, dataset volumes): they
# share a canvas size, so pinning the plot rectangle too makes them the same
# figure at the same scale. Left is set by the wider of the two y-axes (the
# counts chart, which carries a y-label the boxplot does not); bottom by the
# rotated category labels both of them hang below the axis.
VERTICAL_RECT = {"left": 0.075, "right": 0.98, "top": 0.97, "bottom": 0.21}


def parse_args():
    parser = argparse.ArgumentParser(description="Literature-review figures: platforms, games, data types and dataset volumes.")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to write the charts into")
    parser.add_argument("--only", choices=["platforms", "games", "datatypes", "volumes"], help="Draw a single chart instead of all four")
    parser.add_argument("--tick-size", type=float, default=20, help="Font size of the axis numbers/categories (default 20)")
    parser.add_argument("--label-size", type=float, default=18, help="Font size of the axis titles (default 18)")
    return parser.parse_args()


def upper(labels):
    """Every label this folder draws is upper-case - axis titles, tick
    labels, legend entries, no exceptions.

    Applied here at draw time rather than by shouting in the data module, so
    literature_review_data.py stays readable as a table ("League Of Legends",
    not "LEAGUE OF LEGENDS") and the casing is a property of the figures
    instead of something every future caller has to remember to re-type.
    """
    return [label.upper() for label in labels]


def style_axes(ax, font_prop, args, grid_axis: str):
    """The furniture every figure here shares: surface, tick colour/typeface,
    a recessive grid behind the marks, and no box around the plot."""
    ax.set_facecolor(WHITE)
    ax.tick_params(axis="both", labelsize=args.tick_size, colors=cs.INK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        # apply_font, not set_fontproperties - the latter would silently
        # reset the size set by tick_params above. See chart_style.apply_font.
        cs.apply_font(label, font_prop)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(cs.SPINE)
    ax.grid(axis=grid_axis, color=cs.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def horizontal_bars(pairs, xlabel, output, font_prop, args):
    """A ranked horizontal bar chart - used for platforms and games, whose
    category names are long enough that vertical bars would need rotated
    labels."""
    names = [n for n, _ in pairs]
    counts = [c for _, c in pairs]

    # Tall enough that each bar keeps its own row at --tick-size; a fixed
    # height would crowd as categories are added.
    fig, ax = plt.subplots(figsize=(12, 0.78 * len(names) + 1.8), facecolor=WHITE)

    y = np.arange(len(names))
    # Bars descend, so the longest sits at the top where reading starts.
    ax.barh(y, counts, color=BAR_COLOR, height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(upper(names))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel.upper(), fontsize=args.label_size, color=cs.INK, fontproperties=font_prop)
    # Integer counts only - a "2.5th article" tick would be meaningless.
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlim(0, max(counts))

    style_axes(ax, font_prop, args, grid_axis="x")
    # Fixed margins instead of tight_layout, so every chart drawn by this
    # function gets the SAME plot rectangle. tight_layout sizes the left
    # margin to whichever category name happens to be longest, which made
    # platforms and games - same canvas, same eight rows - place their axis
    # at different x positions (~0.15 vs ~0.29 of the width) and their bars
    # at different scales. Side by side in the chapter they read as two
    # differently sized figures. LEFT_MARGIN is set for the longest label
    # either chart currently carries ("THE LAST OF US PART 2") plus headroom;
    # a longer one would clip rather than silently reflow the axis, which is
    # the trade this makes deliberately.
    fig.subplots_adjust(left=LEFT_MARGIN, right=0.98, top=0.97, bottom=BOTTOM_MARGIN)
    cs.save_figure(fig, output)
    plt.close(fig)


def datatype_counts(output, font_prop, args):
    """How many papers report each data type. Vertical bars, since the five
    labels are short and the comparison is of heights."""
    names = [name for name, _ in data.DATA_TYPES]
    counts = [len(rows) for _, rows in data.DATA_TYPES]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=WHITE)
    x = np.arange(len(names))
    ax.bar(x, counts, color=BAR_COLOR, width=0.72, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(upper(names), rotation=30, ha="right")
    ax.set_ylabel("TOTAL ARTICLES", fontsize=args.label_size, color=cs.INK, fontproperties=font_prop)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    style_axes(ax, font_prop, args, grid_axis="y")
    fig.subplots_adjust(**VERTICAL_RECT)
    cs.save_figure(fig, output)
    plt.close(fig)


def volume_boxplot(output, font_prop, args):
    """Dataset size per data type, on a log axis.

    The log scale is not cosmetic: these groups span comments in the tens of
    millions against interviews in the tens, so a linear axis would collapse
    every group but the first onto the baseline.
    """
    names = data.VOLUME_ORDER
    groups = [data.values(name) for name in names]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=WHITE)

    # Every line here is a hairline. The box furniture is scaffolding around
    # the fill, not a mark in its own right: drawn at the 1.2-1.6pt that
    # looks reasonable in isolation, the black outlines out-weigh the fills
    # and the five boxes read as outlined shapes rather than as a ramp.
    HAIRLINE = 0.8
    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.6,
        # Hollow circles, so an outlier reads as a single observation rather
        # than as another filled mark competing with the boxes.
        flierprops={"marker": "o", "markersize": 6, "markerfacecolor": "none",
                    "markeredgecolor": cs.INK_SECONDARY, "markeredgewidth": HAIRLINE},
        medianprops={"color": cs.INK, "linewidth": HAIRLINE},
        whiskerprops={"color": cs.INK_SECONDARY, "linewidth": HAIRLINE},
        capprops={"color": cs.INK_SECONDARY, "linewidth": HAIRLINE},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BAR_COLOR)
        patch.set_edgecolor(cs.INK_SECONDARY)
        patch.set_linewidth(HAIRLINE)

    ax.set_yscale("log")
    ax.set_xticklabels(upper(names), rotation=30, ha="right")

    style_axes(ax, font_prop, args, grid_axis="y")
    fig.subplots_adjust(**VERTICAL_RECT)
    cs.save_figure(fig, output)
    plt.close(fig)

    for name, values in zip(names, groups):
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        info(f"[{name}] n={len(values)} min={min(values):,} Q1={q1:,.0f} median={med:,.0f} Q3={q3:,.0f} max={max(values):,}")


def main():
    args = parse_args()
    font_prop = cs.load_font()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "platforms"):
        horizontal_bars(data.PLATFORMS, "TOTAL ARTICLES", args.outdir / "platforms", font_prop, args)
    if args.only in (None, "games"):
        horizontal_bars(data.GAMES, "TOTAL ARTICLES", args.outdir / "games", font_prop, args)
    if args.only in (None, "datatypes"):
        datatype_counts(args.outdir / "data_types", font_prop, args)
    if args.only in (None, "volumes"):
        volume_boxplot(args.outdir / "dataset_volumes", font_prop, args)


if __name__ == "__main__":
    main()
