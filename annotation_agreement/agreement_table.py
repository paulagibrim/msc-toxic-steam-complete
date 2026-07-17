"""Rebuilds the inter-annotator agreement table: for each toxicity bin of
each model, how much three human annotators agreed on whether the reviews
in that bin are toxic.

Reconstructed from `dissertacao-steam/data_refactor/2-toxicity/
03_compute_agreement_table.ipynb`, which no longer exists - the notebook
and its whole repository are gone from disk, and only a partial excerpt
survives in old session transcripts. This module reproduces the published
table cell-for-cell (verified against it, 40/40) and fixes the three
things that made the original fragile. Each is documented where it is
handled below.

WHAT THE TABLE ACTUALLY MEASURES:
Not Perspective vs. Detoxify. The annotators are three people (Paula,
Lorena, Marcus) labelling reviews toxic/non-toxic; the model only decides
which *bin* a review falls in. So a row reads: "of the reviews this model
scored in [0.7, 0.8), how much did the humans agree with each other, and
what did they conclude?" - i.e. whether the model's confidence band
corresponds to anything humans recognise as toxic.

THE STATISTIC IS RANDOLPH'S KAPPA, NOT FLEISS'.
The original called `fleiss_kappa(table, method='rand')` from a function
named `calcular_fleiss_kappa`; `method='rand'` selects Randolph's
free-marginal kappa, not Fleiss' fixed-marginal one. They are different
statistics (Randolph assumes chance agreement is 1/k for k categories,
rather than deriving it from the observed marginals) and Randolph
generally reports higher values. The name is corrected here; the
statistic is deliberately kept, so the numbers stay comparable to what
was already published.

WITH 20 ITEMS AND 3 ANNOTATORS, KAPPA IS QUANTISED.
For 2 categories, Randolph's kappa reduces to (u - 5) / 15, where u is the
number of unanimously-labelled items. Only six values are attainable
between 0.6667 and 1.0, in steps of 0.0667. Small differences between
bins are therefore not meaningful - a bin cannot land "just below"
another.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from pipeline_utils import info

# The two candidate stratification keys. Each spreadsheet is a sample
# drawn evenly across one model's bins; the other model's bin is recorded
# too but is incidental (and unevenly filled).
BIN_COLUMNS = ["perspective_bin", "detoxify_bin"]

# Annotator column names differ per language - the pt sheets use full
# names, the en sheets initials. Same three people either way.
ANNOTATOR_SETS = [["paula", "marcus", "lorena"], ["P", "L", "M"]]

TOXIC, NON_TOXIC, TIE = "toxic", "non-toxic", "tie"


def detect_annotators(df: pd.DataFrame) -> list:
    for candidate in ANNOTATOR_SETS:
        if all(c in df.columns for c in candidate):
            return candidate
    raise SystemExit(
        f"No known annotator columns in {list(df.columns)} - expected one of {ANNOTATOR_SETS}"
    )


def bin_lower_bound(label) -> float:
    """Sorts/identifies a bin by the number in its label, never by matching
    the label string.

    The original notebook hard-coded a list of exact label strings, which
    silently broke: the pt sheets write the last bin as
    'Extremamente Alto [0.9, 1.0]' and the en sheets as
    '... [0.9, 1.0)'. Whichever spelling the list did not contain produced
    an empty selection reported as "insufficient data" - losing the most
    toxic bin, the one the whole table exists to characterise. Parsing the
    lower bound makes both spellings, and any relabelling, equivalent.
    """
    match = re.search(r"\[\s*([0-9.]+)\s*,", str(label))
    if not match:
        raise ValueError(f"Cannot parse a bin lower bound from label: {label!r}")
    return float(match.group(1))


def detect_stratification(df: pd.DataFrame) -> str:
    """Returns which model's bin column this sheet was sampled evenly
    across, by looking at the data rather than at the filename.

    The filenames cannot be trusted for this: in pt, `bins_pt_1` is the
    perspective-stratified sheet, but in en it is `bins_en_2` - the
    numbering is inverted between languages. Reading the suffix as if it
    meant the same thing in both would silently swap the two halves of the
    table. The stratification key is the column whose bins all hold the
    same number of rows.
    """
    for column in BIN_COLUMNS:
        counts = df[column].value_counts()
        if len(counts) > 1 and counts.nunique() == 1:
            return column
    raise SystemExit(
        f"Neither {BIN_COLUMNS[0]} nor {BIN_COLUMNS[1]} is evenly sampled - "
        f"cannot tell which model this sheet is stratified by"
    )


def _resolve_missing(annotations: pd.DataFrame, assume_missing_agrees: bool):
    """Handles annotator cells left blank. Returns (annotations, n_missing).

    `bins_en_1.xlsx` is missing four of annotator P's labels. The published
    table can only be reproduced by assuming P agreed with the other two on
    all four (in one of them both said 'toxic', so filling blanks with
    'non-toxic' does not reproduce it) - which is evidence the annotations
    existed when the table was computed and were lost from the spreadsheet
    afterwards, not that P abstained.

    That assumption is reconstruction, not data, so it is off by default:
    normally the incomplete rows are dropped and the count is reported, and
    `assume_missing_agrees` opts in to reproducing the published figures.
    Note the assumption cannot lower agreement - it can only manufacture
    unanimity - so a table built with it on is biased upward by
    construction.
    """
    missing = annotations.isna().any(axis=1)
    n_missing = int(missing.sum())
    if not n_missing:
        return annotations, 0

    if not assume_missing_agrees:
        return annotations[~missing], n_missing

    filled = annotations.copy()
    for i in filled.index[missing]:
        present = filled.loc[i].dropna()
        if present.empty:
            continue
        filled.loc[i] = filled.loc[i].fillna(present.mode()[0])
    return filled, n_missing


def _majority(annotations: pd.DataFrame) -> str:
    """The bin's verdict: each review is decided by its own 2-of-3 vote,
    then the bin follows the majority of those per-review verdicts.

    Deliberately not the mean of every annotation - that collapses the
    per-review structure and disagrees with the published table (in the
    en [0.7, 0.8) perspective bin, 11 of 20 reviews are voted toxic while
    the mean over all 60 annotations is 0.4833, which would report the bin
    as non-toxic).

    An even bin size makes an exact tie reachable, and one bin does tie:
    pt perspective [0.4, 0.5) splits 10/10. A tie is reported as such, not
    broken - comparing against a threshold silently resolves it to
    whichever side the comparison happens to favour, which would hide the
    one bin where the annotators are most divided behind a verdict that
    looks like every other bin's.
    """
    per_review_toxic = annotations.mean(axis=1) > 0.5
    toxic, non_toxic = int(per_review_toxic.sum()), int((~per_review_toxic).sum())
    if toxic == non_toxic:
        return TIE
    return TOXIC if toxic > non_toxic else NON_TOXIC


def bin_agreement(annotations: pd.DataFrame, assume_missing_agrees: bool) -> dict:
    annotations, n_missing = _resolve_missing(annotations, assume_missing_agrees)
    n_reviews = len(annotations)
    result = {
        "n_reviews": n_reviews,
        "n_rows_missing_an_annotation": n_missing,
    }
    if n_reviews == 0:
        result["error"] = "no reviews in this bin"
        return result

    table, _ = aggregate_raters(annotations.values)
    # A single-category bin makes statsmodels evaluate 0/0 and warn on its
    # way to returning NaN. That case is expected and handled just below,
    # so the warning is suppressed here rather than left to imply something
    # went wrong - narrowly, around this one call.
    with np.errstate(invalid="ignore"):
        kappa = fleiss_kappa(table, method="rand")

    # A bin where every annotator chose the same single category has no
    # variance, so kappa is 0/0. The original silently substituted 1.0; it
    # is kept (the published 1.0000 cells are all this case) but flagged,
    # because "undefined" and "perfect agreement" are different claims and
    # the table cannot distinguish them on its own.
    kappa_undefined = bool(np.isnan(kappa)) and table.shape[1] == 1
    if kappa_undefined:
        kappa = 1.0

    result.update({
        "kappa": round(float(kappa), 4),
        "kappa_undefined_forced_to_one": kappa_undefined,
        "n_unanimous_reviews": int((annotations.nunique(axis=1) == 1).sum()),
        "majority": _majority(annotations),
    })
    return result


def agreement_table(path: Path, assume_missing_agrees: bool = False) -> dict:
    """Builds one spreadsheet's half of the table (one model's bins)."""
    df = pd.read_excel(path)
    annotators = detect_annotators(df)
    bin_column = detect_stratification(df)
    model = bin_column.replace("_bin", "")
    info(f"{path.name}: stratified by {bin_column} -> '{model}' column; annotators {annotators}")

    bins = []
    for label in sorted(df[bin_column].dropna().unique(), key=bin_lower_bound):
        rows = df[df[bin_column] == label]
        entry = {"bin": str(label), "bin_lower_bound": bin_lower_bound(label)}
        entry.update(bin_agreement(rows[annotators], assume_missing_agrees))
        bins.append(entry)

    return {
        "model": model,
        "source_file": path.name,
        "annotators": annotators,
        "reviews_total": len(df),
        "bins": bins,
    }
