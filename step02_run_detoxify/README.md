# Step 02 — Run Detoxify

Scores step01's cleaned pt/en reviews with Detoxify, keeping only the
`toxicity` output (renamed `detoxify_score`) - the other six sub-scores
Detoxify also computes (`severe_toxicity`, `obscene`, `identity_attack`,
`insult`, `threat`, `sexual_explicit`) are discarded.

This is a **HEAVY** step - Detoxify is a transformer model, meant to run on
a GPU machine (the 2x A100 machine, not the 48-core CPU one step01 used).

## What it does, in order

1. Reads one file at a time from step01's output
   (`reviews_cleaned.parquet/*.parquet`) - `review_lang` is a plain column
   there, not a subfolder, so every file can hold a mix of languages and
   gets read once per `--lang` value processed (e.g. twice total for the
   pt+en default).
2. Filters to rows where BOTH `review_lang == lang` AND
   `perspective_declared_language == lang` (step01's agreement mask - see
   `agreement_mask.py`). Rows that fail either check are dropped.
3. Strips known boilerplate phrases (Steam's own early-access/refund
   notices) from a **copy** of the text used only to feed the model -
   `review_text` in the saved output is the original, untouched.
4. Runs Detoxify (`multilingual` model) in batches, keeping only `toxicity`.
5. Writes every column that was already in the file (`review_text`,
   `game_id`, `user_url`, `perspective_score`,
   `perspective_declared_language`, `review_lang`, etc.) **plus** the new
   `detoxify_score` column - nothing from step01 is dropped.

Unlike `step01`'s masks (`toxicity_mask.py`/`language_revalidation.py`/
`agreement_mask.py`), which only export slim companion tables, this step
keeps the full row: Detoxify already needs `review_text` in memory to run
inference, so keeping the rest of the row costs nothing extra, and almost
every downstream analysis needs `review_text`/`game_id` (and eventually
user data) right alongside `detoxify_score` anyway - merging them back
together later would be a recurring cost for no benefit.

## Setup (on the GPU machine)

1. Copy this whole folder (`step02_run_detoxify/`) over - it's
   self-contained (its own `pipeline_utils.py`, doesn't depend on
   `step01_cleaning_and_language_detection/` being present).
2. Copy step01's output over too - specifically
   `steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet/`.
   Keep `steam-data/` next to `msc-toxic-steam-complete/` (same parent
   folder, wherever that ends up on this machine) - the commands below use
   `../../steam-data/...`, relative to this folder, so nothing needs
   editing per machine as long as that layout holds.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Only `pandas`, `pyarrow`, `torch`, `detoxify` - much lighter than
   step01's requirements (no `dask`, no `langdetect` needed here).

## Running it

```bash
python run_detoxify.py \
  --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \
  --output-dir ../../steam-data/step02-output
```

- `--input` is step01's `reviews_cleaned.parquet` directory (`review_lang`
  is a plain column there, not a subfolder) - not the raw reviews.
- Scores both `pt` and `en` by default; pass `--lang` (repeatable) to
  restrict to just one, e.g. `--lang pt`.
- Device auto-detects `cuda` > `mps` > `cpu`. Pass `--device cuda:0` (or
  `cuda:1`) to pin a specific GPU on the 2-GPU machine, or `--device cpu`/
  `--device mps` to force a fallback.
- `--batch-size` (default 32) and `--max-chars` (default 1200, review text
  is truncated to this before scoring) are also overridable.

## Output

Unlike step01's input (flat, review_lang as a column), this step's own
output IS still partitioned into `review_lang=<lang>/` folders - one
output file per input file, same filename, under
`<output-dir>/review_lang=<lang>/`. This step's data doesn't get
reclassified the way language detection does, so the folder-partitioning
tradeoffs that caused problems in step01 don't apply here.

```
step02-output/
  review_lang=pt/
    part.0.parquet
    part.1.parquet
    ...
  review_lang=en/
    part.0.parquet
    ...
```

Each file has every column from step01's output plus `detoxify_score`:

| column | source |
|---|---|
| `review_url`, `review_text`, `game_id`, `user_url`, `review_date`, `hours_played`, `is_recommended` | step01 (from the raw scrape, cleaned) |
| `perspective_score` | step01 (from the raw scrape, alongside Perspective's own toxicity call) |
| `perspective_declared_language` | step01 (the raw `language` field, kept for reference - no longer drives partitioning) |
| `review_lang`, `detection_confidence` | step01 (langdetect's own guess) |
| `detoxify_score` | **this step** - Detoxify's `toxicity` output |

Only rows where `review_lang` and `perspective_declared_language` agree are
present - see step01's `agreement_mask.py` for why.

## Resuming an interrupted run

Each output file is written only after that one input file finishes
scoring. If the run stops partway through, just re-run the same command -
`score_file` skips any file whose output already exists, so already-scored
files aren't reprocessed.

## Checking score validity (light)

Reports and saves how many rows have a valid `perspective_score`/
`detoxify_score` (both inside `[0, 1]`) vs. an invalid/sentinel value -
Detoxify writes `-1.0` when a batch fails to score (see
`detoxify_scoring.py`), and this is the main thing that shows up here.
Same report-only pattern as step01's `agreement_mask.py`: the check itself
is a cheap `.between(0, 1)` on columns already present, so only the small
aggregate counts are saved, not a filtered copy of the data.

```bash
python run_validity_mask.py \
  --input ../../steam-data/step02-output \
  --output ../../steam-data/step02-output/validity_report.json
```

`--input` is this step's own output directory (the scored data
`run_detoxify.py` already produced). Writes `validity_report.json` -
`rows_total`, `rows_valid`, `rows_invalid_either`,
`rows_invalid_perspective`, `rows_invalid_detoxify`, `valid_pct` per
language. To apply the mask itself in your own analysis code, call
`validity_mask.apply_validity_mask(df)` directly.

## Correlating the two models (light)

How far Perspective and Detoxify agree, per language. Both scores already
live on the same row of this step's output, so it is a read-and-correlate -
no join against step01 needed.

```bash
python run_score_correlation.py \
  --input ../../steam-data/step02-output \
  --output ../../steam-data/step02-output/score_correlation_report.json
```

Only the languages this step scored can appear (pt/en - Detoxify never ran
on the other 54 step01 detected), and the population is narrower than "all
pt/en reviews": it is step01's agreement-mask subset, the same rows
`run_detoxify.py` scored. The `n` per language matches
`agreement_report.json`'s `rows_agree` exactly, which is a useful
cross-check that the two reports describe the same population.

Rows with a score outside [0, 1] are excluded via
`validity_mask.apply_validity_mask` - a sentinel dragged into a correlation
acts as a high-leverage outlier, not merely as noise. None occur in
practice: zero invalid rows across all 27.4M.

**Both Pearson and Spearman are reported, and the pair matters more than
either alone.** Measured, Spearman comes out *below* Pearson (en: 0.716 vs
0.774; pt: 0.601 vs 0.760) - the reverse of the usual expectation for two
differently-calibrated models. Both scores pile up near zero (mean
perspective_score 0.126 for en, 0.083 for pt) and the two models order that
dense low-toxicity mass close to arbitrarily; Spearman weights it like
everything else, while Pearson is carried by the toxic tail where they do
move together. Reading Pearson alone would overstate how much the models
agree on any single review - and pt is where they diverge most.

For agreement between *humans* rather than models, see
`annotation_agreement/`.
