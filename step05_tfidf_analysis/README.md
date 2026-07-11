# Step 05 — TF-IDF Toxic vs. Non-Toxic Term Analysis

Labels toxicity and compares toxic vs. non-toxic TF-IDF term weights across
step02's pt/en reviews - which terms are disproportionately associated with
toxic reviews vs. non-toxic ones. Ported from
`dissertacao-steam/data_refactor/2-toxicity/tfidf_analysis.py` (previously
driven by a notebook), adapted to this project's column names
(`detoxify_score` instead of `toxicity`) and to a plain CLI instead of a
notebook.

This is a light-medium step (CPU-only, no model inference) - can run
anywhere, though it does load a full language's worth of reviews into
memory at once (both toxic and non-toxic, unlike step03/step04's inputs).

## What it does, in order

1. Reads and concatenates every file from step02's
   `review_lang=<lang>/*.parquet` - **every** review, toxic and non-toxic
   alike (the whole point of this analysis is comparing the two groups).
2. Labels `is_toxic` using the same union rule and thresholds as everywhere
   else in this project (`perspective_score >= 0.7` OR `detoxify_score >=
   0.9`). Rows with an invalid score (either score outside `[0, 1]` -
   Detoxify's `-1.0` "failed to score" sentinel) are **dropped**, not
   labeled non-toxic - their true toxicity is unknown.
3. Cleans review text for TF-IDF specifically (lowercase, strip URLs/
   accents/non-alpha characters, collapse whitespace) - a heavier
   normalization than the boilerplate-only stripping used before Detoxify/
   sentiment scoring, since this feeds a bag-of-words vectorizer, not a
   transformer model.
4. Fits a `TfidfVectorizer` (unigrams, `min_df=100`, `max_df=0.8`,
   `max_features=100000`) using nltk's per-language stopword list plus a
   few domain-specific extras (`game`, `steam`, etc. - see
   `tfidf_analysis.STOPWORD_EXTRAS`).
5. Computes the mean TF-IDF weight per term, separately for toxic and
   non-toxic rows, processed in 500K-row chunks to bound peak memory rather
   than materializing the whole corpus as one sparse matrix.
6. Builds a lexicon table (`termo`, `tfidf_toxico`, `tfidf_neutro`,
   `diferenca`, `proporcao_toxico_neutro`, `log_ratio`), sorted by
   `diferenca` descending - the terms most disproportionately toxic first.

## Setup

1. Copy this whole folder (`step05_tfidf_analysis/`) over - it's
   self-contained (its own `pipeline_utils.py`, doesn't depend on any
   other step's folder being present).
2. Copy step02's output over too - specifically
   `steam-data/step02-output/review_lang=en/` and `review_lang=pt/`. Keep
   `steam-data/` next to `msc-toxic-steam-complete/` (same parent folder) -
   the command below uses `../../steam-data/...`, relative to this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   nltk's stopword corpus downloads automatically on first run
   (`nltk.download("stopwords", quiet=True)`).

## Running it

```bash
python run_tfidf.py \
  --input ../../steam-data/step02-output \
  --output-dir ../../steam-data/step05-output
```

- `--input` is step02's output directory (the one containing the
  `review_lang=*` subfolders) - not step01's output.
- Processes both `pt` and `en` by default; pass `--lang` (repeatable) to
  restrict to just one, e.g. `--lang pt`.

## Output

Per language, in `--output-dir`:

- `reviews_cleaned_labeled_<lang>.parquet` - slim intermediate (`review_text`,
  `game_id`, `review_text_clean`, `is_toxic`), so the lexicon step can be
  re-run without redoing loading/labeling/cleaning.
- `tfidf_lexicon_<lang>.csv` - the final term-weight comparison table.
- `tfidf_report_<lang>.json` - run summary (`rows_loaded`,
  `rows_excluded_invalid`, `rows_toxic`, `rows_non_toxic`,
  `vocabulary_size`, `count_toxic`, `count_neutral`, `lexicon_path`).
