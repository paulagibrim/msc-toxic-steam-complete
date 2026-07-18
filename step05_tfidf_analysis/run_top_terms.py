"""Reads a TF-IDF lexicon (tfidf_analysis.py's tfidf_lexicon_<lang>.csv) and
writes out its top-N terms for the toxic and the neutral class.

Both classes are already scored in that CSV - every term carries a
tfidf_toxico (its mean TF-IDF over toxic reviews) and a tfidf_neutro (over
neutral ones) side by side - so the toxic and neutral top lists are just
two different sorts of the same file, nothing is recomputed.

RANKING CRITERION (--by): defaults to the class's own TF-IDF score, which
is what "top terms by TF-IDF" means literally. Be aware of what that
column is: it is how *heavily weighted* a term is within the class, not how
*distinctive* it is of the class. A word common to both classes (pt "nao",
"bom") scores high in toxic simply by being frequent, even though it says
little about toxicity. The lexicon's own default sort is `diferenca`
(tfidf_toxico - tfidf_neutro), which surfaces terms a class uses far more
than the other; `log_ratio` goes further toward disproportion. All three
are offered so the two views can be compared - the class-score ranking for
"what dominates this class", the difference/ratio ones for "what sets it
apart".

Usage (one language):
    python run_top_terms.py \\
        --input ../../steam-data/step05-output-pt/tfidf_lexicon_pt.csv \\
        --output ../../steam-data/step05-output-pt/top_terms_pt.csv

Usage (both, in one call):
    python run_top_terms.py \\
        --input ../../steam-data/step05-output-pt/tfidf_lexicon_pt.csv \\
        --input ../../steam-data/step05-output-en/tfidf_lexicon_en.csv \\
        --output ../../steam-data/step05-output/top_terms.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import info

# Each class and the lexicon column holding its per-term TF-IDF score.
CLASS_SCORE = {"toxic": "tfidf_toxico", "neutral": "tfidf_neutro"}

# What --by can sort on. The class score ranks by weight within the class;
# diferenca/log_ratio rank by how much the classes diverge on a term (both
# are signed toward toxic, so for the neutral list they are negated).
SORT_COLUMNS = {"tfidf", "diferenca", "log_ratio"}


def _lang_of(path: Path) -> str:
    """Best-effort language tag from the filename (tfidf_lexicon_<lang>.csv),
    falling back to the stem so output is still labelled if the name differs."""
    stem = path.stem
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def top_terms(lexicon: pd.DataFrame, toxic_class: bool, n: int, by: str) -> pd.DataFrame:
    """Top n rows for one class. `by='tfidf'` sorts on that class's own
    score; `diferenca`/`log_ratio` sort on class divergence, flipped for the
    neutral class so 'top' always means 'most characteristic of this class'."""
    score_col = CLASS_SCORE["toxic" if toxic_class else "neutral"]
    if by == "tfidf":
        ranked = lexicon.sort_values(score_col, ascending=False)
    else:
        # diferenca/log_ratio point toward toxic; the neutral class wants the
        # most-negative end, so it is sorted the opposite way.
        ranked = lexicon.sort_values(by, ascending=not toxic_class)

    out = ranked.head(n).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(1, "class", "toxic" if toxic_class else "neutral")
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Writes the top-N toxic and neutral terms from a TF-IDF lexicon CSV."
    )
    parser.add_argument(
        "--input", required=True, type=Path, action="append", dest="inputs",
        help="Path to a tfidf_lexicon_<lang>.csv (repeat for more than one language)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the combined top-terms CSV to")
    parser.add_argument("--top-n", type=int, default=15, help="Terms per class per language (default: 15)")
    parser.add_argument(
        "--by", choices=sorted(SORT_COLUMNS), default="tfidf",
        help="Ranking criterion: 'tfidf' (score within the class, default), "
             "'diferenca' or 'log_ratio' (how much the class stands out). See module docstring.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    frames = []
    for path in args.inputs:
        lang = _lang_of(path)
        lexicon = pd.read_csv(path)
        for toxic_class in (True, False):
            block = top_terms(lexicon, toxic_class, args.top_n, args.by)
            block.insert(0, "language", lang)
            frames.append(block)
            top_term = block.iloc[0]["termo"] if len(block) else "(none)"
            info(f"[{lang}] {'toxic' if toxic_class else 'neutral'}: {len(block)} term(s) by {args.by}, top = {top_term!r}")

    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    info(f"Wrote {len(result)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
