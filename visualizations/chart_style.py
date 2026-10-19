"""The one place this folder's charts get their font, palette and surface
from.

Every value here was already in language_distribution.py; the point of the
module is that there is now exactly one copy. The font lookup in particular
was duplicated into run_toxicity_venn.py, which is how "the same font as
the other chart" degrades into "the same font until someone edits one of
them" - a shared import keeps the charts consistent by construction rather
than by remembering.

Palette values come from the dataviz skill's reference palette
(`references/palette.md`), not from taste: SURFACE is its light chart
surface, SERIES_1/SERIES_2 are categorical slots 1 and 2 in their fixed
order, and the ink/grid greys are its text and recessive-furniture tokens.
The pair validates clean on the light surface - worst adjacent CVD
separation ΔE 73.6 (deutan), far above the 12 floor - with one caveat that
constrains callers: SERIES_2 sits at 2.74:1 against SURFACE, under the 3:1
bar, so any mark filled with it must carry a visible direct label rather
than relying on the fill alone to be seen. Both charts here do.
"""
from pathlib import Path

from pipeline_utils import info

# Requested font - see dissertacao-steam's language_distribution.py for the
# history of why this specific file (not the earlier trial font, which
# watermarked digit glyphs).
FONT_CANDIDATES = [
    Path.home() / "Library/Fonts/Heading Now.ttf",
    Path.home() / "Downloads/OnlineWebFonts_COM_07eba947aace0b22cead9a83b4368d04/Heading Now/Heading Now.ttf",
]

SURFACE = "#fcfcfb"          # dataviz skill: light chart surface

# Taken from the "Toxic Steam" deck (Canva), so a figure lifted into the
# talk carries the same identity as the slide around it: SERIES_1 is that
# deck's accent blue exactly, and it validates clean on its own (inside the
# lightness band, above the chroma floor, and over 3:1 against SURFACE).
#
# SERIES_2 is NOT the deck's yellow (#f5f44f), and cannot be. That yellow is
# the deck's *background*, and a fill cannot be the colour of the surface it
# sits on: it measures L 0.94 against a 0.43-0.77 band, contrast 1.14
# against SURFACE, and exactly 1.0 against the deck's own yellow - it would
# be invisible. #eda100 is the same yellow brought down into the legible
# band (the dataviz palette's amber), which keeps the deck's blue+yellow
# reading while surviving as a mark. The pair separates far above the CVD
# floor: worst adjacent dE 134.9 (protan), 78.5 (tritan), floor 12.
SERIES_1 = "#2647bb"         # the deck's accent blue
SERIES_2 = "#eda100"         # amber - the deck's yellow, made fillable

# The overlap of SERIES_1 and SERIES_2, i.e. a Venn's lens.
#
# It must be set explicitly: left to composite, matplotlib_venn mixes the
# two fills *additively* and the lens comes out LIGHTER than both circles,
# which makes the most important region the most recessive one and reads as
# a separate lighter category. (The version of run_toxicity_venn.py this
# replaced hard-coded its own dark override for the same reason.)
#
# It is also not their blend. SERIES_1 and SERIES_2 are near-complementary,
# so mixing them lands on a muddy brown (#8a745e) that reads as a third
# category rather than as "both". The deck's own near-black is used instead:
# it belongs to the same identity, it cannot be mistaken for either parent,
# and being the darkest region matches what it means - the reviews both
# models agree on.
#
# It sits below the validator's lightness band (L 0.214 vs a 0.43 floor),
# accepted deliberately: that band's scope is categorical palettes, where
# every hue must hold its own identity at a glance among peers. A lens is
# not a slot - it is defined by the two circles it lies between, and it
# always carries a direct label. What does matter for it, separation, is
# far above the floor: dE 91.6 (protan) from SERIES_2, floor 12.
BOTH_COLOR = "#1a1918"

# "und" is a data-quality bucket, not a language, so it is deliberately not
# a categorical slot - grey keeps it from reading as another series.
UND_COLOR = "#9c9a93"

INK = "#0b0b0b"              # primary text
INK_SECONDARY = "#52514e"    # axis titles
INK_MUTED = "#898781"        # tick labels
GRID = "#e1e0d9"
SPINE = "#c3c2b7"


def _relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a #rrggbb colour."""
    channels = []
    for i in (1, 3, 5):
        c = int(hex_color[i:i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_on(fill: str) -> str:
    """The ink to write on `fill` - whichever of SURFACE/INK contrasts with
    it more.

    Computed rather than chosen per chart, because a fixed choice only
    survives while every fill happens to be dark. It stopped being true the
    moment the palette gained SERIES_2's amber: white on it measures 2.11:1
    where black measures 9.09:1, so a hard-coded white label would have gone
    quietly unreadable on the largest region of every Venn.
    """
    contrast_with = lambda ink: (
        (max(_relative_luminance(fill), _relative_luminance(ink)) + 0.05)
        / (min(_relative_luminance(fill), _relative_luminance(ink)) + 0.05)
    )
    return SURFACE if contrast_with(SURFACE) > contrast_with(INK) else INK


def load_font():
    """Returns a FontProperties for the requested font, falling back to
    DejaVu Sans (and saying so) when it isn't installed - a missing font
    should change how a chart looks, never stop it rendering."""
    import matplotlib.font_manager as fm

    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            fm.fontManager.addfont(str(candidate))
            info(f"Using font: {candidate}")
            return fm.FontProperties(fname=str(candidate))

    info(f"Font not found in any of {[str(p) for p in FONT_CANDIDATES]} - using DejaVu Sans instead.")
    return fm.FontProperties(family="DejaVu Sans")


def save_figure(fig, output_path: Path, tight: bool = False) -> tuple:
    """Writes both a raster (PNG, quick to preview/embed) and a vector
    (PDF, scales cleanly for print) from one figure, preserving the surface
    colour - matplotlib defaults to a white background on save regardless
    of the figure's own facecolor, which would put every chart on a
    different surface than the one it was designed against. Returns
    (png_path, pdf_path).

    `tight` crops the saved file to what was actually drawn, discarding the
    figure's margins. Use it for a figure that will be placed in a document,
    where the surrounding whitespace is the document's job and the figure's
    own padding only fights it. Left off by default: a chart with axes wants
    its margins, and cropping to content makes the saved size depend on the
    data, so two runs can come out different sizes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    extra = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    fig.savefig(png_path, dpi=200, facecolor=fig.get_facecolor(), **extra)
    # dpi is irrelevant for a vector format
    fig.savefig(pdf_path, facecolor=fig.get_facecolor(), **extra)
    info(f"Saved chart to: {png_path} and {pdf_path}")
    return png_path, pdf_path
