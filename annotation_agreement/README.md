# Annotation Agreement

Rebuilds the inter-annotator agreement table: for each toxicity bin of each
model, how much three human annotators agreed on whether the reviews in
that bin are toxic.

Not a pipeline stage like `step01`-`step06` - a cross-cutting tool, and the
only one that reads no pipeline output at all. Its input is the raw
annotation spreadsheets and nothing else.

## What it actually measures

**Not Perspective vs. Detoxify.** The annotators are three people (Paula,
Lorena, Marcus) labelling reviews toxic/non-toxic. The model only decides
which *bin* a review falls into. So a row reads: *"of the reviews this model
scored in [0.7, 0.8), how much did the humans agree with each other, and
what did they conclude?"* - i.e. whether the model's confidence band
corresponds to anything humans recognise as toxic.

For agreement *between the two models*, see
`step02_run_detoxify/run_score_correlation.py` instead.

## Input

`steam-data/raw/annotations/<lang>/*.xlsx` - two spreadsheets per language,
200 annotated reviews each:

| column | meaning |
|---|---|
| `review_url`, `review_text` | the review being annotated |
| `perspective_bin`, `detoxify_bin` | which 0.1-wide band each model scored it in |
| `paula`/`marcus`/`lorena` (pt), `P`/`L`/`M` (en) | 1 = toxic, 0 = non-toxic |

Each spreadsheet is sampled evenly across **one** model's bins (20 reviews
per bin); the other model's bin is recorded but unevenly filled. The two
sheets of a language are disjoint - no review appears in both.

**Which sheet belongs to which model is detected from the data, never from
the filename.** The numbering is inverted between languages: `bins_pt_1` is
perspective-stratified, but `bins_en_1` is detoxify-stratified. Reading the
suffix as if it meant the same thing in both silently swaps the two halves
of the table.

## Setup

```bash
pip install -r requirements.txt
```

`pandas`, `numpy`, `openpyxl`, `statsmodels` - no models, no GPU, runs
anywhere in seconds.

## Running it

```bash
python run_agreement_table.py \
  --input ../../steam-data/raw/annotations \
  --output ../../steam-data/annotation-output/agreement_table.json
```

- `--lang` - language subfolder to process (repeatable). Defaults to every
  one found.
- `--fill-missing-with-majority` - see below.

## Reading the output

Per language, per model, per bin: `kappa`, `majority`, `n_reviews`,
`n_unanimous_reviews`, plus the two flags below. The conventions used are
recorded at the top of the JSON itself, so the file explains itself without
this README.

### The statistic is Randolph's kappa, not Fleiss'

The original notebook called `fleiss_kappa(table, method='rand')` from a
function named `calcular_fleiss_kappa`. `method='rand'` selects **Randolph's
free-marginal kappa**, not Fleiss' fixed-marginal one - different
statistics (Randolph fixes chance agreement at 1/k for k categories instead
of deriving it from the observed marginals), and Randolph generally reports
higher values. The statistic is deliberately kept so the numbers stay
comparable to what was already published; only the name is corrected. **Cite
it as Randolph's.**

### Kappa is quantised, so small gaps mean nothing

With 20 reviews, 3 annotators and 2 categories, Randolph's kappa reduces to
`(u - 5) / 15`, where `u` is the number of unanimously-labelled reviews.
Only six values are attainable between 0.6667 and 1.0, in steps of 0.0667.
A bin cannot land "just below" another.

### `kappa_undefined_forced_to_one`

When all three annotators use a single category, kappa is 0/0 - undefined,
not perfect. The original silently substituted 1.0; that is kept, but
flagged per bin, because "undefined" and "perfect agreement" are different
claims and a bare 1.0000 cannot distinguish them. Only the `[0, 0.1)` bins
are affected; the other 1.0000 cells are genuinely measured.

### `--fill-missing-with-majority`

`bins_en_1.xlsx` is missing four of annotator P's labels. Filling each with
the majority vote of the two annotators who did label that review
reproduces the published table exactly - and in one of the four both said
*toxic*, so filling blanks with *non-toxic* does **not** reproduce it. That
is evidence P's labels existed when the table was computed and were lost
from the spreadsheet afterwards, not that P abstained.

It is reconstruction rather than data, so it is opt-in:

- **default** - reviews with a blank annotation are dropped from their bin,
  and the count is reported per bin (`n_rows_missing_an_annotation`).
- **`--fill-missing-with-majority`** - reproduces the published table
  exactly (verified, 80/80 cells across both languages). A filled vote
  always joins the side that is already ahead, never the one behind, so it
  can only ever *raise* agreement: a table built with it on is biased upward
  by construction.

Running both ways shows what the fill is worth: it moves three en detoxify
cells (0.7895 → 0.8000, 0.9298 → 0.9333, 0.7895 → 0.8000) and nothing else.

All four blanks are kept and filled - `0`, `0`, `1`, `0`, matching the two
annotators beside each. None are dropped.

A majority needs one to exist. With three annotators a blank leaves two, and
two annotators only have a majority when they agree - so this rule and
"assume the missing annotator agreed with the others" are the same rule
here, and become undefined at the same point. Reviews where the remaining
annotators are split (or where none are left) are dropped rather than
guessed. Neither case occurs in the current spreadsheets; the guard is for
future rounds, where a fourth annotator would let a majority exist without
unanimity.

### `tie`

`majority` is decided per review by a 2-of-3 vote, then per bin by the
majority of those verdicts - not by averaging every annotation, which
collapses the per-review structure and disagrees with the published table.

An even bin size makes an exact tie reachable, and one bin does tie: pt
perspective `[0.4, 0.5)` splits 10/10. It is reported as `tie`, not broken -
a threshold comparison would silently resolve it to whichever side the
comparison favours, hiding the one bin where the annotators are most divided
behind a verdict that looks like every other bin's.

## Provenance

Reconstructed from `dissertacao-steam/data_refactor/2-toxicity/
03_compute_agreement_table.ipynb`, which no longer exists - the notebook and
its whole repository are gone from disk, and only a partial excerpt survives
in old session transcripts. This folder reproduces the published table
cell-for-cell and fixes three things that made the original fragile: bins
matched by hard-coded label strings (the pt and en sheets spell the last bin
differently, so one language silently lost its most toxic bin), model
identity taken from the filename, and blank annotations handled by accident
rather than by decision.
