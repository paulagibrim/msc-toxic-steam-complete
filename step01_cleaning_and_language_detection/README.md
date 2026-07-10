# Step 01 — Cleaning and Language Detection

Cleans the raw games/users/reviews files, deduplicates reviews, assigns
each review's language purely from `langdetect`, and (optionally) checks
how often that agrees with the language originally bundled alongside the
Perspective API scrape.

Three of these six steps are heavy (review cleaning+dedup, language
detection, standalone language validation) and are meant to run on a
many-core machine, not a laptop - they're marked **HEAVY** below.
Games/users cleaning and the agreement check are light and can run
anywhere.

**Review cleaning is two separate scripts/steps (3 and 4), not one.**
Deduplication is a Dask *shuffle* (every worker exchanges data with every
other worker) and wants a cluster shaped like "few workers, lots of memory
each." Language detection is pure-Python, CPU-bound, per-row work with no
shuffle at all, and wants the opposite shape: "many worker processes,
modest memory each" (`langdetect` holds the GIL, so real parallelism there
comes from process count, not threads). Running both under one Dask
Client - which is what an earlier version of this step did - meant no
single worker/memory configuration served both well, and was the source of
repeated instability (worker OOM-restarts, then, once that was tuned away,
inter-worker connection timeouts under CPU load). Splitting them also means
the expensive dedup shuffle is checkpointed to disk and only ever runs
once - if language detection needs retuning or crashes, dedup never needs
to be redone.

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

## 3. Clean + deduplicate reviews — HEAVY

```bash
python run_clean_reviews_dedup.py \
  --input ../../steam-data/raw/reviews \
  --output ../../steam-data/step01-output/reviews_deduped.parquet \
  --n-workers 8 --threads-per-worker 4 --memory-limit 40GB --blocksize 256MB \
  --local-directory ../../steam-data/step01-output/dask-worker-space-dedup
```

Deduplicating by `review_url` is a Dask *shuffle* across ~97M raw rows -
this is what crashed a 24GB machine before (see `clean_reviews.py`'s module
docstring). `--n-workers`/`--threads-per-worker`/`--memory-limit` configure
a `dask.distributed.Client` with real per-worker memory limits so Dask
spills to disk under pressure instead of exhausting RAM. **Favor few
workers with lots of memory each for this step** - a shuffle means every
worker exchanges data with every other worker, so fewer workers means less
all-to-all connection overhead as well as more headroom to buffer the
transfer; 8 workers x 40GB (320GB total) is what actually worked reliably
on the 48-core/430GB machine this was built for, after higher-worker-count
configurations repeatedly failed (worker OOM-restarts, then inter-worker
connection timeouts under load, depending on what got tuned).

`--blocksize 256MB` caps how much data gets bundled into a single raw-file
read task. Without it, Dask's own optimizer decides how many files to fuse
into one read, and picked a fusion large enough to exceed `--memory-limit`
outright on this project's data - workers died reading the raw files,
before the shuffle even started.

The script calls `.persist()` right after deduplication and prints a live
progress bar (`distributed.progress`) until it's done, then writes the
result as a checkpoint (`reviews_deduped.parquet`) - not partitioned by
language yet, that's step 4. Also writes `dedup_report.json` (row counts at
each stage, final partition count) next to the checkpoint.

**This is the step that only needs to run once.** If step 4 (language
detection) needs retuning or crashes, you don't need to redo this - just
re-run step 4 against the same checkpoint.

### 3b. If the shuffle keeps failing: shuffle-free dedup

In practice, the shuffle above proved unstable even after extensive
tuning (memory_limit raised from 4GB up through 90GB, blocksize lowered
to 64MB, threads_per_worker dropped to 1, comm timeouts raised to 120s,
fewer/bigger workers) - a single worker exceeding its own `--memory-limit`
during shuffle buffering gets killed and restarted by the nanny, which
poisons the whole shuffle's state (`P2PConsistencyError`) and forces a
full restart. This sometimes recovers, sometimes cascades into total
failure - observed as late as 99% complete.

`run_clean_reviews_dedup_noshuffle.py` produces the identical output
(same checkpoint format, same columns) without ever invoking Dask's
shuffle: every partition hashes its own rows' `review_url` into one of
200 buckets and writes each group to its own subfolder (no cross-worker
communication - can't be poisoned by another worker dying), then every
bucket is deduplicated independently with plain single-process pandas
(a `ProcessPoolExecutor`, no Dask cluster at all for this half) - safe
because every row sharing a `review_url` is guaranteed to land in the
same bucket.

```bash
python run_clean_reviews_dedup_noshuffle.py \
  --input ../../steam-data/raw/reviews \
  --output ../../steam-data/step01-output/reviews_deduped.parquet \
  --n-workers 16 --threads-per-worker 1 --memory-limit 16GB --blocksize 64MB \
  --local-directory ../../steam-data/step01-output/dask-worker-space-scatter
```

The scatter phase (writing to buckets) has no shuffle, so it can use many
workers with modest memory each, same shape as step 4's language
detection. The dedup-buckets phase runs after Dask's client closes, using
every CPU core by default (`--n-jobs-dedup` to override). Writes the same
`reviews_deduped.parquet/` checkpoint and `dedup_report.json` as the
shuffle-based version - `run_detect_language.py` (step 4) doesn't need to
know or care which one produced it.

## 4. Detect language — HEAVY

```bash
python run_detect_language.py \
  --input ../../steam-data/step01-output/reviews_deduped.parquet \
  --output-dir ../../steam-data/step01-output/reviews_by_lang \
  --n-workers 32 --threads-per-worker 1 --memory-limit 8GB \
  --local-directory ../../steam-data/step01-output/dask-worker-space-langdetect
```

Reads step 3's checkpoint (not the raw reviews) and assigns every review's
`review_lang` purely from `langdetect`, computed via `map_partitions`
(Dask's own worker pool parallelizes it, no separate process pool) - the
`language` field bundled alongside the Perspective API scrape is ignored
entirely for this (kept only as `perspective_declared_language`, for
reference). Output has one `review_lang=<code>` folder per language
`langdetect` actually finds - however many that turns out to be, including
`review_lang=und` for reviews too short/low-signal to classify confidently
(see `clean_reviews.detect_review_language`'s docstring).

**Favor many workers with modest memory each for this step** - the
opposite of step 3's advice, and deliberately so: `langdetect` is pure
Python and holds the GIL, so real parallelism comes from the *number of
worker processes*, not threads (`--threads-per-worker 1` is usually right
here). There's also no shuffle in this step at all (reading the checkpoint
is a plain partitioned read, and `export_reviews`'s partitioned write
splits each worker's own rows into the right subfolder locally, no
cross-worker data movement) - so it tolerates a high worker count far
better than step 3's dedup does.

The script also calls `.persist()` right after language detection, so the
langdetect pass only runs once - without it, every later `len()`/
`.compute()` call (row counts, language counts, the final `to_parquet()`)
would silently redo it from scratch each time.

Review text is truncated to 2000 characters (`langdetect_revalidation.
MAX_CHARS`) before detection - langdetect doesn't need a whole review to
guess its language, and without a cap, a handful of unusually long reviews
(some run to thousands of characters, especially ones padded with ASCII
art) can make a few tasks take far longer than the rest, stalling the
progress bar on a couple of stragglers with no error to explain why.

Writes `reviews_cleaned.parquet/` (partitioned by `review_lang`),
`sample_reviews.csv`, `reviews_report.json` (includes `languages_detected`
and a `language_counts` breakdown) into `step01-output/reviews_by_lang/`.

## 5. Standalone language validation (PT and EN) — HEAVY, optional

Step 4 already assigns `review_lang` from `langdetect` directly, so this
step isn't required for the cleaned output anymore. It's a separate,
narrower cross-check: for a *given* declared language (the one bundled
alongside the Perspective API scrape), how much of what it labeled that way
does `langdetect` agree with - useful if you want that specific before/after
comparison on record. Reads straight from the **raw** review files (not
step 3/4's output). Run once per language:

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
step 3/4 at the same time.

Writes `language_validation_{lang}.parquet` and
`language_mismatch_breakdown_{lang}.csv` into `step01-output/language_detection/`.

## 6. Check langdetect/Perspective agreement (light)

For pt and en (default - pass `--lang` to override), reports and **saves**
how many reviews in step 4's `review_lang=<lang>` partitions also have the
Perspective-scrape-declared language (`perspective_declared_language`)
agreeing - i.e. both sources say the same thing. The underlying *data* is
never duplicated - the check is a plain `==` on a column already present
(no model inference), cheap enough to apply on demand wherever it's needed
instead of writing out a second copy of a multi-million-row partition (see
`agreement_mask.py`'s module docstring). The *aggregate counts* themselves
(a handful of numbers per language) are small enough to persist as a
report. To apply the filter itself in your own analysis code, call
`agreement_mask.apply_agreement_mask(df, lang)` directly.

```bash
python run_agreement_mask.py \
  --input ../../steam-data/step01-output/reviews_by_lang/reviews_cleaned.parquet \
  --output ../../steam-data/step01-output/agreement_report.json
```

`--input` is step 4's `reviews_cleaned.parquet` output directory (not the
raw reviews). Writes `agreement_report.json` - `rows_total`, `rows_agree`,
`rows_disagree`, `agree_pct` per language.

## Bringing results back

Steps 1 and 2 already ran locally, so only steps 3, 4, and 5's outputs need
to be copied back - they're far smaller than the raw input (slim
per-review tables, not the full text corpus duplicated; step 3's
`reviews_deduped.parquet` checkpoint is the one exception, roughly the same
size as the deduplicated corpus, so only worth bringing back if you want it
for something else - step 4 already consumed it into the final, smaller
`reviews_cleaned.parquet`). Step 6 doesn't produce any files (just a
printed report), so there's nothing to copy back for it.
