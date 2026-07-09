# Step 01 — Cleaning and Language Detection

Cleans the raw games/users/reviews files, deduplicates reviews, assigns
each review's language purely from `langdetect`, and (optionally) checks
how often that agrees with the language originally bundled alongside the
Perspective API scrape.

Two of these five steps are heavy (review cleaning+dedup+language-detect,
standalone language validation) and are meant to run on a many-core
machine, not a laptop - they're marked **HEAVY** below. Games/users
cleaning and the agreement check are light and can run anywhere.

## Setup (on the powerful machine)

1. Copy this whole folder (`step01_cleaning_and_language_detection/`) and
   `requirements.txt` (one level up, at the project root) over.
2. Copy `steam-data/` over too (or mount/sync it) - specifically the
   `raw/games`, `raw/users`, `raw/reviews` subfolders.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. `cd` into this folder before running anything below - every script
   assumes its sibling `.py` files (`clean_reviews.py`, `pipeline_utils.py`,
   etc.) are importable from the current directory.

All commands below use `../../steam-data/...` - the data folder is always
relative to the code: from this folder
(`msc-toxic-steam-complete/step01_cleaning_and_language_detection/`), go up
twice to leave `msc-toxic-steam-complete/` and into its sibling
`steam-data/`. This works unchanged on any machine as long as
`msc-toxic-steam-complete/` and `steam-data/` are copied/mounted next to
each other (same parent folder) - no path needs editing per machine.

## 1. Clean games (light)

```bash
python run_clean_games.py \
  --input ../../steam-data/raw/games/todos_jogos.json \
  --output ../../steam-data/step01-output/games/games.parquet
```

Writes `games.parquet`, `null_summary_games.csv`, `sample_games.csv`,
`games_report.json` into `step01-output/games/`.

## 2. Clean users (light-medium, ~25M rows)

```bash
python run_clean_users.py \
  --input ../../steam-data/raw/users \
  --output ../../steam-data/step01-output/users/all_users.parquet
```

Writes `all_users.parquet`, `null_summary_users.csv`, `sample_users.csv`,
`users_report.json` into `step01-output/users/`.

## 3. Clean + deduplicate + language-detect reviews — HEAVY

```bash
python run_clean_reviews.py \
  --input ../../steam-data/raw/reviews \
  --output-dir ../../steam-data/step01-output/reviews_by_lang \
  --n-workers 24 --threads-per-worker 2 --memory-limit 16GB --blocksize 256MB \
  --local-directory ../../steam-data/step01-output/reviews_by_lang/dask-worker-space
```

No `--lang` flag anymore - the `language` field bundled alongside the
Perspective API scrape is ignored entirely (kept only as
`perspective_declared_language`, for reference) and every review's
`review_lang` is assigned purely from `langdetect`, computed via
`map_partitions` (Dask's own worker pool parallelizes it, no separate
process pool). Output has one `review_lang=<code>` folder per language
`langdetect` actually finds - however many that turns out to be, including
`review_lang=und` for reviews too short/low-signal to classify confidently
(see `clean_reviews.detect_review_language`'s docstring).

Deduplicating by `review_url` is a Dask *shuffle* across ~97M raw rows -
this is what crashed a 24GB machine before (see `clean_reviews.py`'s module
docstring). `--n-workers`/`--threads-per-worker`/`--memory-limit` configure
a `dask.distributed.Client` with real per-worker memory limits so Dask
spills to disk under pressure instead of exhausting RAM - don't be shy
about going high on a machine with RAM to spare (leave some headroom for
the OS/scheduler, don't allocate 100% of RAM to workers). Two other things
tuned in for the same reason:

`--blocksize 256MB` caps how much data gets bundled into a single raw-file
read task. Without it, Dask's own optimizer decides how many files to fuse
into one read, and picked a fusion large enough to exceed `--memory-limit`
outright on this project's data - workers died reading the raw files,
before the shuffle or langdetect even started.

There's also an optional `--n-partitions` to repartition right before
language detection, in case the raw ~25 files leave too few partitions for
`langdetect` (the most parallelizable, CPU-heavy step) to spread across all
workers - **it's off by default and shouldn't usually be needed**: the
dedup shuffle already redistributes the data into a much larger, balanced
partition count on its own (observed: 106+ partitions out of ~25 files in).
An explicit `.repartition()` turned out to be its own shuffle in this Dask
version too - not a free/local split - and crashed workers the same way an
undersized `--memory-limit` does for the dedup shuffle. Only reach for it
if the post-dedup partition count (logged every run) is genuinely too low
for your worker count, and expect it to need the same memory headroom as
the dedup shuffle.

Also worth knowing: `langdetect` is pure Python and holds the GIL, so for
that specific step, true parallelism comes from the *number of worker
processes* (`--n-workers`), not threads - `--threads-per-worker` mostly
helps the pandas/pyarrow-based steps, which release the GIL during their
C-level work. That's why the recommended config above favors more workers
with moderate memory each, over fewer workers with more memory each.

The script also calls `.persist()` right after language detection, so the
dedup shuffle and the per-row `langdetect` pass only run once - without it,
every later `len()`/`.compute()` call (row counts, language counts, the
final `to_parquet()`) would silently redo both from scratch each time.

Writes `reviews_cleaned.parquet/` (partitioned by `review_lang`),
`sample_reviews.csv`, `reviews_report.json` (includes `languages_detected`
and a `language_counts` breakdown) into `step01-output/reviews_by_lang/`.

## 4. Standalone language validation (PT and EN) — HEAVY, optional

Step 3 already assigns `review_lang` from `langdetect` directly, so this
step isn't required for the cleaned output anymore. It's a separate,
narrower cross-check: for a *given* declared language (the one bundled
alongside the Perspective API scrape), how much of what it labeled that way
does `langdetect` agree with - useful if you want that specific before/after
comparison on record. Reads straight from the **raw** review files (not
step 3's output). Run once per language:

```bash
python run_langdetect_revalidation.py \
  --input ../../steam-data/raw/reviews \
  --lang pt \
  --output-dir ../../steam-data/step01-output/language_detection \
  --n-jobs 48

python run_langdetect_revalidation.py \
  --input ../../steam-data/raw/reviews \
  --lang en \
  --output-dir ../../steam-data/step01-output/language_detection \
  --n-jobs 48
```

`--n-jobs` defaults to every core on the machine (`os.cpu_count()`) if
omitted - pass it explicitly to leave headroom, e.g. if running alongside
step 3 at the same time.

Writes `language_validation_{lang}.parquet` and
`language_mismatch_breakdown_{lang}.csv` into `step01-output/language_detection/`.

## 5. Check langdetect/Perspective agreement (light)

For pt and en, reports how many reviews in step 3's `review_lang=<lang>`
partitions also have the Perspective-scrape-declared language
(`perspective_declared_language`) agreeing - i.e. both sources say the same
thing. This is a report only, not a filter that gets saved: the check is a
plain `==` on a column already in the data (no model inference), cheap
enough to apply on demand wherever it's needed instead of writing out a
second copy of a multi-million-row partition (see `agreement_mask.py`'s
module docstring). To actually use the filter in your own analysis code,
call `agreement_mask.apply_agreement_mask(df, lang)` directly.

```bash
python run_agreement_mask.py \
  --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \
  --lang pt --lang en
```

`--input` is step 3's `reviews_cleaned.parquet` output directory (not the
raw reviews).

## Bringing results back

Steps 1 and 2 already ran locally, so only steps 3 and 4's outputs need to
be copied back - they're far smaller than the raw input (slim per-review
tables, not the full text corpus duplicated). Step 5 doesn't produce any
files (just a printed report), so there's nothing to copy back for it.
