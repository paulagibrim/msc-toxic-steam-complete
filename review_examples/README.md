# Review Examples

Generates example-message files for manual inspection - sampled reviews
matching a language + toxicity + (optionally) a text term / game tag, with
whatever perspective/detoxify/sentiment/topic data is available joined in.

Not a pipeline stage like `step01`-`step05` - a cross-cutting tool that
reads *outputs* from several of them (games from step01, scores from
step02, sentiment from step04, topics from step03), so it doesn't fit
under any single step's number. Self-contained, doesn't depend on any step
folder being present - only their output files.

## What it pulls from where

| column | source | always present? |
|---|---|---|
| `game_id`, `review_url`, `review_text`, `review_lang`, `perspective_score`, `detoxify_score` | step02's output | yes |
| `game_name` | step01's `games.parquet` (joined by `game_id`) | yes |
| `sentiment_score` | step04's output (joined by `review_url`) | only if `--step04-dir` is passed and that language has been scored |
| `topic` | step03's `classified_toxic.parquet` (joined by `review_url`) | only if `--step03-results` is passed, that language's Stage 7 has run, **and** the review is toxic (BERTopic only ever scores the toxic subset) |

Toxicity labeling uses the same union rule and thresholds as everywhere
else in this project (`perspective_score >= 0.7` OR `detoxify_score >=
0.9`), with rows carrying an invalid/sentinel score excluded before
filtering (not labeled non-toxic).

`sentiment_score`/`topic` are only joined for the **sampled** rows, not
the whole corpus - avoids loading a second multi-million-row dataset just
to label a handful of examples.

## Setup

```bash
pip install -r requirements.txt
```

Just `pandas` and `pyarrow` - no models, no GPU, runs anywhere.

## Running it

```bash
python run_show_examples.py \
  --lang pt --toxic --n 50 \
  --games ../../steam-data/step01-output/games/games.parquet \
  --step02-dir ../../steam-data/step02-output \
  --output ../../steam-data/examples/toxic_pt.csv
```

With the optional filters and the other two data sources:

```bash
python run_show_examples.py \
  --lang pt --toxic --n 50 \
  --games ../../steam-data/step01-output/games/games.parquet \
  --step02-dir ../../steam-data/step02-output \
  --step04-dir ../../steam-data/step04-output \
  --step03-results ../../steam-data/step03-output/pt/results/classified_toxic.parquet \
  --contains "trash" --game-tag "Action" --seed 42 \
  --output ../../steam-data/examples/toxic_pt_trash_action.csv
```

- `--lang` - language code (e.g. `pt`, `en`).
- `--toxic` / `--non-toxic` - required, mutually exclusive.
- `--n` - how many examples to sample (returns fewer if not enough match).
- `--games` / `--step02-dir` - required.
- `--step04-dir` / `--step03-results` - optional; omit if that step hasn't
  run yet for this language - the corresponding column comes back empty
  rather than erroring.
- `--contains` - optional substring the review text must contain
  (case-insensitive, plain substring match, not regex).
- `--game-tag` - optional; game must have this tag in its `popular_tags`
  (case-insensitive).
- `--seed` - optional random seed, for a reproducible sample.

## Using it from code directly

```python
from show_review_examples import get_review_examples

examples = get_review_examples(
    lang="pt", toxic=True, n=50,
    games_path="../../steam-data/step01-output/games/games.parquet",
    step02_dir="../../steam-data/step02-output",
    step04_dir="../../steam-data/step04-output",       # optional
    step03_results_path="../../steam-data/step03-output/pt/results/classified_toxic.parquet",  # optional
    contains="trash", game_tag="Action", seed=42,
)
```

Returns a `pandas.DataFrame`, same columns as the CSV output.
