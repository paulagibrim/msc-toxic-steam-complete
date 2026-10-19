"""Counts reviews per `review_lang` across step01's Stage 2 output and
plots the top N languages by volume.

Ported from dissertacao-steam/data_refactor/2-toxicity/language_distribution.py,
adapted to count langdetect's OWN classification (`review_lang`) rather
than Steam's declared `language` field the original counted from the raw
scrape - per explicit request, this chart is specifically about what
langdetect found, not what Steam's (untrustworthy - see
langdetect_revalidation.py's module docstring) field says.

Reads from step01's reviews_cleaned.parquet (Stage 2's output). review_lang
is a plain column there, not a directory partition (see clean_reviews.py's
module docstring), so every file already holds every language mixed
together - no per-language folder to iterate.

Stage 2 (run_detect_language.py) already computes these exact per-language
counts once, as `language_counts` in its own reviews_report.json - so
load_counts_from_report reads that directly instead of re-scanning the
full ~93M-row corpus a second time. count_reviews_per_language (the full
file scan) is still here for when reviews_report.json isn't available
(e.g. counting a different/filtered slice than what Stage 2 reported on).
"""
import json
from pathlib import Path

import pandas as pd

import chart_style as cs
from pipeline_utils import info, list_parquet_files

# Font, palette and surface now live in chart_style, shared with this
# folder's other charts - see that module's docstring. This chart plots a
# single measure, so its bars take the palette's sequential hue, which is
# categorical slot 1's blue.
BAR_COLOR = cs.SERIES_1
# "und" (langdetect_revalidation.UNDETERMINED_LABEL - not imported directly
# since this module doesn't otherwise depend on step01; too short/
# low-signal text to classify confidently, see
# identify_language_langdetect's min_alpha_length) is a real review_lang
# value, not a missing one - kept in the ranking (per explicit request),
# just colored separately from genuine languages since it's a
# data-quality bucket, not a language.
UND_LABEL = "und"
UND_COLOR = cs.UND_COLOR

# Only rows with NO review_lang value at all get this - see
# count_reviews_per_language's docstring. Genuinely missing, so still
# dropped from the ranking (unlike "und").
MISSING_LANGUAGE_LABEL = "(missing)"

# langdetect's own code set (ISO 639-1, plus its "zh-cn"/"zh-tw" Chinese
# variants) -> full English name, for chart labels. Covers every code
# langdetect can emit (its full supported-language list), not just the 56
# seen in one report, so a future run with a different top-N/language mix
# doesn't need this dict revisited.
LANGUAGE_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "und": "Undetermined",
}


def _language_label(code: str) -> str:
    """Full English name for a review_lang code, per LANGUAGE_NAMES - falls
    back to the raw code (uppercased, matching this chart's previous
    behaviour) for anything not in the map, so an unrecognised code still
    renders instead of raising."""
    name = LANGUAGE_NAMES.get(str(code).lower())
    if name is None:
        info(f"No full-name mapping for review_lang code {code!r} - using the raw code as the label.")
        return str(code).upper()
    return name


def load_counts_from_report(report_path: Path) -> pd.Series:
    """Reads Stage 2's own reviews_report.json and returns its
    `language_counts` dict as a Series - the exact same shape
    count_reviews_per_language would produce, without re-scanning the
    corpus (Stage 2 already computed this once via
    `df["review_lang"].value_counts()` while it had the data in memory)."""
    with open(report_path) as f:
        report = json.load(f)
    counts = pd.Series(report["language_counts"], dtype="int64")
    info(f"Loaded {len(counts)} language count(s) from {report_path} ({int(counts.sum())} review(s) total)")
    return counts.sort_values(ascending=False)


def count_reviews_per_language(reviews_cleaned_dir: Path) -> pd.Series:
    """Counts rows per `review_lang` value across every file in step01's
    Stage 2 output. Reads only that one column (fast, low memory even
    though the files are large).

    Rows with no review_lang value at all are counted under
    MISSING_LANGUAGE_LABEL rather than silently dropped by
    value_counts()'s default dropna=True - shouldn't happen in practice
    (detect_review_language always assigns something, even "und"), but
    kept for the same defensive reason as the original module."""
    files = list_parquet_files(reviews_cleaned_dir)
    counts = pd.Series(dtype="int64")

    for i, f in enumerate(files, start=1):
        s = pd.read_parquet(f, columns=["review_lang"])["review_lang"].value_counts(dropna=False)
        s.index = s.index.fillna(MISSING_LANGUAGE_LABEL)
        counts = counts.add(s, fill_value=0)
        if i % 20 == 0 or i == len(files):
            info(f"[{i}/{len(files)}] counted so far")

    return counts.astype("int64").sort_values(ascending=False)


def save_language_counts(counts: pd.Series, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts.rename_axis("review_lang").reset_index(name="review_count").to_csv(output_path, index=False)
    info(f"Saved language counts to: {output_path}")
    return output_path


def plot_top_languages(counts: pd.Series, output_path: Path, n: int = 15) -> tuple:
    """Plots languages by review volume as vertical bars (tallest on the
    left, descending to the right), saved as both PNG and PDF. Returns
    (png_path, pdf_path). Excludes MISSING_LANGUAGE_LABEL (no review_lang
    value at all - a data artifact, not a language). "und" (langdetect's
    own "undetermined" result) IS kept in the ranking, per explicit
    request, just colored gray (UND_COLOR) instead of BAR_COLOR to set it
    apart from genuine languages.

    n=15 (the default, per explicit request) - pass a different number or
    None (every language found) to override."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_prop = cs.load_font()
    number_font_prop = font_prop

    ranked = counts.drop(index=MISSING_LANGUAGE_LABEL, errors="ignore")
    ranked = ranked.sort_values(ascending=False)  # tallest first, left to right
    top = ranked if n is None else ranked.head(n)

    # Widen the figure with the number of bars instead of a fixed size, so
    # every language code gets breathing room along the x-axis.
    fig_width = max(8.0, len(top) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6), facecolor=cs.SURFACE)
    ax.set_facecolor(cs.SURFACE)

    positions = range(len(top))
    bar_colors = [UND_COLOR if str(label) == UND_LABEL else BAR_COLOR for label in top.index]
    bars = ax.bar(positions, top.values, color=bar_colors, width=0.6, zorder=3)

    # Value labels rotated 90 degrees above each bar - with this many bars
    # packed into the figure, a horizontal label is wider than a single bar
    # slot and would collide with its neighbors.
    max_value = top.values.max()
    for bar, value in zip(bars, top.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.01,
            f"{int(value):,}",
            va="bottom", ha="center", rotation=90,
            fontsize=9, color=cs.INK, fontproperties=number_font_prop,
        )

    ax.set_ylabel("REVIEWS", fontsize=11, color=cs.INK_SECONDARY, fontproperties=font_prop)
    # set_xticks before set_xticklabels: avoids matplotlib's "FixedFormatter
    # should only be used together with FixedLocator" warning.
    ax.set_xticks(list(positions))
    # Full names (e.g. "Portuguese") run noticeably wider than the old
    # 2-letter codes ("PT") - angled labels avoid neighbours colliding at
    # 15 bars, where horizontal text no longer fits each bar's slot.
    ax.set_xticklabels([_language_label(label).upper() for label in top.index], rotation=40, ha="right")

    # Y-axis ticks in millions ("5M", "36.8M", ...) - per-bar labels above
    # keep the exact comma-formatted count, this is just the axis scale.
    def _millions_formatter(value, _):
        millions = value / 1_000_000
        text = f"{millions:.1f}".rstrip("0").rstrip(".")
        return f"{text}M"

    ax.yaxis.set_major_formatter(_millions_formatter)
    ax.tick_params(axis="x", labelsize=11, colors=cs.INK)
    ax.tick_params(axis="y", labelsize=9, colors=cs.INK_MUTED)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(number_font_prop)

    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(cs.SPINE)
    ax.grid(axis="y", color=cs.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max_value * 1.35)

    fig.tight_layout()
    png_path, pdf_path = cs.save_figure(fig, output_path)
    plt.close(fig)
    return png_path, pdf_path
