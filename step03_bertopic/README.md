# Step 03 — BERTopic Topic Modeling

Discovers topics inside the **toxic** reviews from step02_run_detoxify, one
BERTopic model per language (pt, en). Ported from
`dissertacao-steam/bertopic_pipeline/` (7 stages: clean → embed → search →
stability → train → infer → export), adapted to this project's column names
(`detoxify_score` instead of `toxicity`) and to keep all data outside the
code repo, in the sibling `steam-data/` folder.

Two config files drive the two language runs - `config_en.yaml` and
`config_pt.yaml` - each pointing at its own `step02-output/review_lang=<lang>`
input and writing to its own `step03-output/<lang>/` output tree. **Never
edit `src/` for a configuration change** - every path, threshold, and
hyperparameter lives in the config YAML.

Every `run/*.py` script accepts a repeatable `--lang` flag and **defaults to
running both `en` and `pt`, one after another, in a single invocation** -
pass `--lang en` (or `--lang pt`) to restrict to just one. Config files are
resolved by naming convention (`--lang en` → `config_en.yaml`), so there's
nothing else to pass.

Stages 2, 3, 4, 5, and 6 are **HEAVY** - they run the embedding model and/or
BERTopic itself, and are meant for a GPU machine (the same one step02 used),
not a laptop. Stage 1 and 7 are light/medium and can run anywhere.

Stage 1 also re-checks the language-agreement mask
(`perspective_declared_language == lang_code`) before labelling, even though
step02 already guarantees every row in its output agrees - a cheap, explicit
double-check kept in case this stage is ever pointed at un-filtered input.

## Setup (on the GPU machine)

1. Copy this whole folder (`step03_bertopic/`) over.
2. Copy step02's output over too - specifically
   `steam-data/step02-output/review_lang=en/` and `review_lang=pt/`. Keep
   `steam-data/` next to `msc-toxic-steam-complete/` (same parent folder) -
   every path in the configs is `../../steam-data/...`, relative to this
   folder, so nothing needs editing per machine as long as that layout holds.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. `cd` into this folder before running anything below - every `run/*.py`
   script assumes `src/` is importable from the current directory.

```bash
python run/01_clean.py              # runs en, then pt
python run/01_clean.py --lang en    # just en
```

## 1. Text Cleaning (light-medium)

```bash
python run/01_clean.py
```

Reads step02's `review_lang=<lang>/*.parquet`, re-checks the language
agreement mask, drops rows with an invalid score (Detoxify's `-1.0` "failed
to score" sentinel - excluded, not treated as non-toxic, per this project's
established `toxicity_mask.py` convention), labels `is_toxic` (Perspective
`>= 0.7` OR Detoxify `>= 0.9` - same union rule and thresholds as
`toxicity_mask.py`), and cleans the review text for BERTopic (lowercase,
strip URLs/boilerplate/accents/non-alpha, collapse whitespace). One output
file per input file, `--resume` skips files already cleaned.

Writes to `paths.cleaned_data_dir` (`step03-output/<lang>/cleaned/`).

## 2. Embedding Generation — HEAVY

```bash
python run/02_embed.py
```

Loads every toxic, non-empty cleaned review, encodes it with a multilingual
SentenceTransformer, reduces to 50 dimensions via PCA, and saves the raw
embeddings, PCA embeddings, fitted PCA model, and a `toxic_index.parquet`
(row → `review_url`/`game_id`/text mapping) - so every downstream stage
reuses these instead of re-encoding. Auto-detects `cuda` > `cpu`.

**Only re-run this if you change `embedding.model_name` or
`embedding.pca_components`** - changing either invalidates everything
downstream.

## 3. Hyperparameter Search (Optuna) — HEAVY

```bash
python run/03_search.py
```

Searches UMAP/HDBSCAN hyperparameters over a 100K-document sample, minimizing
`outlier_rate - coherence_weight * coherence`. Persisted to a SQLite study
(`--resume` continues an interrupted search instead of losing completed
trials). Writes `best_params.json`, consumed by every stage after this one.

## 4. Stability Analysis — HEAVY

```bash
python run/04_stability.py
```

Trains BERTopic (with Stage 3's best params) at increasing sample sizes
(100K → 400K) and checks whether the topic structure converges early via
c-TF-IDF cosine similarity between consecutive sizes. Writes
`stability_report.json` with a recommended training size for Stage 5 - if
nothing stabilizes, recommends the full toxic dataset.

## 5. Final Training — HEAVY

```bash
python run/05_train.py
python run/05_train.py --lang en --sample-size 200000
```

Trains the final BERTopic model on Stage 4's recommended size (`--sample-size
auto`, the default), a fixed count, or the full dataset (`--sample-size
all`). Saves the model in safetensors format to `models_dir/final_model/`.
Note: `--sample-size` applies to every language in the run, so pass `--lang`
alongside it if you want a different sample size per language.

## 6. Batch Inference — HEAVY

```bash
python run/06_infer.py
```

Classifies every toxic document (not just the training sample) with the
fixed trained model - `.transform()`, not `.fit_transform()`, so the learned
topic structure doesn't change. Processes one cleaned file at a time,
writing one batch parquet per input file to `results_dir/batches/`.
`--resume` skips batches already classified.

## 7. Export (light)

```bash
python run/07_export.py
```

Merges Stage 6's batches, recomputes per-topic counts from the full
classified dataset (Stage 5's counts only reflect the training sample), and
writes the final artifacts to `results_dir/`:

- `classified_toxic.parquet` - one row per toxic document (`review_url`,
  `game_id`, `review_text_clean`, `topic`)
- `topic_info_real_counts.csv` - topic summary, full-dataset counts
- `topic_info.csv` - topic summary, training-sample counts (as BERTopic
  reports them)

## Bringing results back

Only `results_dir/` (small) and `models_dir/final_model/` (the trained model,
needed if you want to run inference again later) are worth copying back -
`cleaned_data_dir` and `embeddings_dir` are large, derived, and cheap to
regenerate by re-running Stages 1-2 if ever needed again.

## MLflow

Every heavy stage logs to `paths.mlruns_dir` (`step03-output/<lang>/mlruns/`).
Inspect with:

```bash
mlflow ui --backend-store-uri ../../steam-data/step03-output/en/mlruns
```
