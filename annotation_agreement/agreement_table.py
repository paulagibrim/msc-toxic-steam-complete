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

BOTH KAPPAS ARE REPORTED, AND NEITHER IS CALLED JUST "KAPPA".
The original called `fleiss_kappa(table, method='rand')` from a function
named `calcular_fleiss_kappa`. `method='rand'` selects Randolph's
free-marginal kappa, not Fleiss' fixed-marginal one - so despite every
name around it, the published table has always been Randolph. The two are
different statistics and differ by up to 0.36 on this data (en's [0.9,
1.0) bin: Randolph 0.80, Fleiss 0.44), which a single unnamed "Kappa"
column cannot reveal. Hence `kappa_randolph` and `kappa_fleiss`, never
`kappa`.

Randolph is the appropriate primary figure here, and the reason is the
design rather than the number it produces. The two differ only in what
they treat as chance agreement: Fleiss derives it from the observed
marginals, Randolph fixes it at 1/k for k categories. Fleiss' assumption
is that a skewed marginal reveals a rater's prior leaning, which should
be discounted - true when raters work to a quota. These annotators had
none; they judged each review on its own, so the marginal is an outcome,
not a constraint. Worse, each bin's prevalence is imposed by this study's
own stratification (a bin of >= 0.9 model scores really is almost all
toxic), so Fleiss' correction penalises the sampling design and reads it
as annotator bias: it scores en's [0.9, 1.0) bin at 0.44 despite 17 of 20
reviews being unanimous, the highest agreement in the table. Fleiss is
kept alongside for comparability - it is always <= Randolph (Warrens
2010), so reporting only Randolph would be reporting only the upper
bound.

References: Fleiss (1971), Psychological Bulletin 76(5); Randolph (2005),
"Free-Marginal Multirater Kappa: An Alternative to Fleiss' Fixed-Marginal
Multirater Kappa"; Warrens (2010), Advances in Data Analysis and
Classification 4(4).

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


def _resolve_missing(annotations: pd.DataFrame, fill_missing_with_majority: bool):
    """Handles annotator cells left blank. Returns (annotations, n_missing).

    `bins_en_1.xlsx` is missing four of annotator P's labels. Filling each
    with the majority of the annotators who did label that review
    reproduces the published table exactly; filling them all with
    'non-toxic' does not (in one, both other annotators said 'toxic'). That
    is evidence P's labels existed when the table was computed and were
    lost from the spreadsheet afterwards, not that P abstained.

    Filling is reconstruction, not data, so it is off by default: normally
    the incomplete rows are dropped and the count is reported, and
    `fill_missing_with_majority` opts in to reproducing the published
    figures. Filling can only ever *raise* agreement - a filled vote always
    joins the side that is already ahead, never the one behind - so a table
    built with it on is biased upward by construction.

    A majority needs one to exist. With three annotators, a blank leaves
    two, and two annotators only have a majority when they agree - so this
    rule and "assume the missing annotator agreed with the others" are the
    same rule here, and become undefined at the same point. Reviews where
    the remaining annotators are split (or where none are left) are dropped
    rather than guessed: taking the mode would pick whichever label sorts
    first, inventing a vote and breaking the tie silently. Neither case
    occurs in the current spreadsheets - all four blanks sit beside two
    annotators who agree, and all four are kept and filled - so this only
    guards future annotation rounds, where a fourth annotator would let a
    majority exist without unanimity.
    """
    missing = annotations.isna().any(axis=1)
    n_missing = int(missing.sum())
    if not n_missing:
        return annotations, 0

    if not fill_missing_with_majority:
        return annotations[~missing], n_missing

    filled = annotations.copy()
    no_majority = []
    for i in filled.index[missing]:
        votes = filled.loc[i].dropna().value_counts()
        if votes.empty or (len(votes) > 1 and votes.iloc[0] == votes.iloc[1]):
            no_majority.append(i)
            continue
        filled.loc[i] = filled.loc[i].fillna(votes.index[0])

    if no_majority:
        info(
            f"  {len(no_majority)} review(s) dropped: an annotation is missing and the "
            f"remaining annotators have no majority to fill it from"
        )
        filled = filled.drop(index=no_majority)
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


def bin_agreement(annotations: pd.DataFrame, fill_missing_with_majority: bool) -> dict:
    annotations, n_missing = _resolve_missing(annotations, fill_missing_with_majority)
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
    # went wrong - narrowly, around these calls.
    with np.errstate(invalid="ignore"):
        randolph = fleiss_kappa(table, method="rand")
        fleiss = fleiss_kappa(table, method="fleiss")

    # A bin where every annotator chose the same single category has no
    # variance, so kappa is 0/0 under either definition. The original
    # silently substituted 1.0; it is kept (the published 1.0000 cells are
    # all this case) but flagged, because "undefined" and "perfect
    # agreement" are different claims and a bare number cannot distinguish
    # them.
    kappa_undefined = bool(np.isnan(randolph)) and table.shape[1] == 1
    if kappa_undefined:
        randolph = fleiss = 1.0

    result.update({
        # Both are reported, and neither is called just "kappa". A bare
        # "Kappa" is what let the published table be labelled Fleiss while
        # holding Randolph for years: the two differ by up to 0.36 here, and
        # nothing in a single unnamed column reveals which one it is.
        "kappa_randolph": round(float(randolph), 4),
        "kappa_fleiss": round(float(fleiss), 4),
        "kappa_undefined_forced_to_one": kappa_undefined,
        "n_unanimous_reviews": int((annotations.nunique(axis=1) == 1).sum()),
        "majority": _majority(annotations),
    })
    return result


def agreement_table(path: Path, fill_missing_with_majority: bool = False) -> dict:
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
        entry.update(bin_agreement(rows[annotators], fill_missing_with_majority))
        bins.append(entry)

    return {
        "model": model,
        "source_file": path.name,
        "annotators": annotators,
        "reviews_total": len(df),
        "bins": bins,
    }
