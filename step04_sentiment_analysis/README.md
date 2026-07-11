# Step 04 — Sentiment Analysis

Scores step02's pt/en reviews with
[`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) -
a 5-star multilingual classifier trained on product reviews, the closest
available domain match to Steam reviews, and multilingual enough to cover
pt and en with a single model.

This is a **HEAVY** step - the sentiment model is a transformer, meant to
run on a GPU machine (the same one step02/step03 used), not a laptop.

## What it does, in order

1. Reads one file at a time from step02's output
   (`step02-output/review_lang=<lang>/*.parquet`).
2. Strips known boilerplate phrases (Steam's own early-access/refund
   notices) from a **copy** of the text used only to feed the model -
   `review_text` in the saved output is the original, untouched.
3. Runs the sentiment model in batches, and instead of keeping just the
   argmax star label (a coarse 1-5 integer), computes a **continuous**
   `sentiment_score`: the expected value over the model's 5-class softmax
   distribution (`sum(p_i * i)` for `i=1..5`). Two reviews that both land
   on "3 stars" but with different underlying probabilities aren't
   collapsed to the same value - this keeps `sentiment_score` on the same
   continuous footing as `perspective_score`/`detoxify_score` for
   correlation analysis, per explicit request for a fine-grained intensity
   signal rather than a discrete class.
4. Writes every column that was already in the file (`review_text`,
   `game_id`, `user_url`, `perspective_score`, `detoxify_score`,
   `perspective_declared_language`, `review_lang`, etc.) **plus** the new
   `sentiment_score` column - nothing from step02 is dropped.

No language-agreement re-filtering here - step02's output is already
agreement-filtered (`perspective_declared_language == review_lang`), and
this step reads directly from it.

A batch that fails to score gets `sentiment_score = -1.0` - outside the
valid `[1, 5]` range, so it can't be mistaken for a real (low) score, same
convention as `detoxify_score`'s `-1.0` sentinel for its `[0, 1]` range.
Exclude these rows (`sentiment_score < 1` or `> 5`) before any analysis
that averages or thresholds on this column.

## Setup (on the GPU machine)

1. Copy this whole folder (`step04_sentiment_analysis/`) over - it's
   self-contained (its own `pipeline_utils.py`, doesn't depend on any
   other step's folder being present).
2. Copy step02's output over too - specifically
   `steam-data/step02-output/review_lang=en/` and `review_lang=pt/`. Keep
   `steam-data/` next to `msc-toxic-steam-complete/` (same parent folder) -
   the command below uses `../../steam-data/...`, relative to this folder,
   so nothing needs editing per machine as long as that layout holds.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Only `pandas`, `pyarrow`, `torch`, `transformers` - the sentiment model
   loads directly via `transformers`, no extra wrapper package needed
   (unlike step02's `detoxify` package).

## Running it

```bash
python run_sentiment.py \
  --input ../../steam-data/step02-output \
  --output-dir ../../steam-data/step04-output
```

- `--input` is step02's output directory (the one containing the
  `review_lang=*` subfolders) - not step01's output.
- Scores both `pt` and `en` by default; pass `--lang` (repeatable) to
  restrict to just one, e.g. `--lang pt`.
- Device auto-detects `cuda` > `mps` > `cpu`. Pass `--device cuda:0` (or
  `cuda:1`) to pin a specific GPU on the 2-GPU machine, or `--device cpu`/
  `--device mps` to force a fallback.
- Resumable per file - if a run is interrupted, re-running the same command
  skips any `review_lang=<lang>/<file>` that already has a scored output.

## Output

One output file per input file, same filename, under
`<output-dir>/review_lang=<lang>/`:

```
step04-output/
  review_lang=pt/
    part.0.parquet
    part.1.parquet
    ...
  review_lang=en/
    part.0.parquet
    ...
```

Each file has every column from step02's output plus `sentiment_score`:

| column | source |
|---|---|
| `review_url`, `review_text`, `game_id`, `user_url`, `review_date`, `hours_played`, `is_recommended` | step01 (from the raw scrape, cleaned) |
| `perspective_score` | step01 (from the raw scrape, alongside Perspective's own toxicity call) |
| `perspective_declared_language` | step01 (the raw `language` field, kept for reference) |
| `review_lang`, `detection_confidence` | step01 (langdetect's own guess) |
| `detoxify_score` | step02 - Detoxify's `toxicity` output |
| `sentiment_score` | **this step** - expected star rating (1.0-5.0, continuous) |
