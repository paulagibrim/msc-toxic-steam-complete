"""Plots, per language, how the two toxicity models divide up the reviews
either of them flags - as a Venn diagram, one per entry in step02's
toxicity_report.json.

The question is agreement on the *label*, not on the score: a review is
toxic to Perspective at >= 0.7 and to Detoxify at >= 0.9, and the diagram
shows how much of the flagged set rests on only one model's say-so. That
is what a Venn is for - two sets, their overlap, and the parts that do not
overlap - and the areas are drawn in proportion, so the shape carries the
finding before any number is read.

Shares its font, palette and surface with this folder's other charts via
chart_style, rather than repeating them here (see that module's docstring).
Perspective takes categorical slot 1 and Detoxify slot 2, in the palette's
fixed order - not by preference, and not per-language: the same model is
the same colour in every diagram, so the three files can be read side by
side.

Usage:
    python run_toxicity_venn.py \\
        --input ../../steam-data/step02-output/toxicity_report.json \\
        --output-dir ../../steam-data/visualizations/venn
"""
import argparse
import json
from pathlib import Path

import chart_style as cs
from pipeline_utils import info

PERSPECTIVE_LABEL = "PERSPECTIVE ONLY (≥ 0.7)"
DETOXIFY_LABEL = "DETOXIFY ONLY (≥ 0.9)"
BOTH_LABEL = "BOTH MODELS"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plots a Venn diagram of Perspective/Detoxify label agreement per language."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to step02's toxicity_report.json",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write the diagrams to",
    )
    return parser.parse_args()


def plot_venn(subset: dict, output_dir: Path, font_prop) -> tuple:
    """Draws one language's diagram. Returns (png_path, pdf_path)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    from matplotlib_venn import venn2

    label = subset["label"]
    only_perspective = subset["perspective_only"]
    only_detoxify = subset["detoxify_only"]
    both = subset["both"]
    total = subset["total_selected"]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=cs.SURFACE)
    ax.set_facecolor(cs.SURFACE)

    venn = venn2(
        subsets=(only_perspective, only_detoxify, both),
        # Identity goes in a legend instead: venn2 anchors these labels to
        # its circles, and the circles here differ enormously in size (one
        # model flags 5x what the other does), which drove the labels into
        # the fills. A legend also satisfies the dataviz rule that two or
        # more series always carry one.
        set_labels=(None, None),
        set_colors=(cs.SERIES_1, cs.SERIES_2),
        # Opaque: alpha would let matplotlib_venn mix the lens additively
        # and come out lighter than both circles - see chart_style's
        # BOTH_COLOR for why the lens is painted explicitly instead.
        alpha=1.0,
        ax=ax,
    )

    region_colors = {"10": cs.SERIES_1, "01": cs.SERIES_2, "11": cs.BOTH_COLOR}
    for region_id, color in region_colors.items():
        patch = venn.get_patch_by_id(region_id)
        if patch is None:
            continue  # venn2 omits a region's patch when the region is empty
        patch.set_color(color)
        # A surface-coloured ring, per the dataviz mark spec: it separates
        # the fills where they meet, so the lens edge stays legible instead
        # of the three regions reading as one silhouette.
        patch.set_edgecolor(cs.SURFACE)
        patch.set_linewidth(2)

    for region_id in ("10", "01", "11"):
        text = venn.get_label_by_id(region_id)
        if text is None:
            continue  # venn2 omits a region's label when the region is empty
        count = {"10": only_perspective, "01": only_detoxify, "11": both}[region_id]
        text.set_text(f"{count:,}\n{count / total * 100:.1f}%" if total else f"{count:,}")
        text.set_fontproperties(font_prop)
        text.set_fontsize(15)
        # Ink chosen per region from its own fill, not fixed: the three
        # fills do not share a lightness, so one colour cannot be legible on
        # all of them. These labels are also the "relief" the palette's
        # contrast WARN requires - SERIES_2 sits under 3:1 against the
        # surface, so the fill alone cannot carry identity.
        text.set_color(cs.text_on(region_colors[region_id]))

    # No title and no legend: these are dissertation figures. The caption
    # names the figure, and the legend is the same three swatches in all
    # three diagrams, so it is written once by plot_legend and placed once
    # in the document instead of being reprinted around every copy.
    #
    # The total still has to appear, because every region is a part of it
    # and the percentages are shares of it - a reader cannot recover it from
    # the diagram, and without it the percentages have no denominator. It
    # goes in the corner as "N = ...", the convention a thesis reader
    # already knows, small and muted so it stays an annotation.
    #
    # Anchored to where the circles actually are, which is neither the axes
    # box (venn2 fixes the aspect ratio, so the drawing fills only part of
    # it) nor the axes limits (venn2 pads those by ~0.1 beyond the circles).
    # Both put the label in a corner of empty space that a tight crop then
    # preserves, leaving it stranded away from the diagram it annotates.
    circles = Bbox.union([
        patch.get_path().get_extents()
        for patch in (venn.get_patch_by_id(r) for r in region_colors)
        if patch is not None
    ])
    ax.text(
        circles.x0, circles.y1 + 0.03, f"N = {total:,}",
        ha="left", va="bottom",
        fontproperties=font_prop, fontsize=11, color=cs.INK_MUTED,
    )

    png_path, pdf_path = cs.save_figure(fig, output_dir / f"venn_toxicity_{label}", tight=True)
    plt.close(fig)
    return png_path, pdf_path


def plot_legend(output_dir: Path, font_prop) -> tuple:
    """Writes the shared legend as its own figure - the three diagrams use
    the same swatches, so the document places this once rather than
    carrying three copies of it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(9, 0.5), facecolor=cs.SURFACE)
    handles = [
        Patch(facecolor=cs.SERIES_1, label=PERSPECTIVE_LABEL),
        Patch(facecolor=cs.SERIES_2, label=DETOXIFY_LABEL),
        Patch(facecolor=cs.BOTH_COLOR, label=BOTH_LABEL),
    ]
    legend = fig.legend(
        handles=handles, loc="center", ncol=3, frameon=False,
        handlelength=1.2, handleheight=1.2, columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_fontsize(10)
        text.set_color(cs.INK_SECONDARY)

    png_path, pdf_path = cs.save_figure(fig, output_dir / "venn_toxicity_legend", tight=True)
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    with open(args.input) as f:
        report = json.load(f)

    subsets = report.get("toxicity_sets", [])
    if not subsets:
        raise SystemExit(f"No 'toxicity_sets' in {args.input} - is this step02's toxicity_report.json?")

    font_prop = cs.load_font()
    plot_legend(args.output_dir, font_prop)
    for subset in subsets:
        if not subset.get("total_selected"):
            info(f"[{subset['label']}] no reviews flagged - skipping")
            continue
        plot_venn(subset, args.output_dir, font_prop)


if __name__ == "__main__":
    main()
