"""Generates example-message files for manual inspection - sampled reviews
matching a language + toxicity + (optionally) a text term / game tag, with
whatever score/topic data is available joined in.

Pulls from up to four sources, cross-referenced by review_url/game_id:
  - step02's output (review_text, perspective_score, detoxify_score,
    review_url, game_id, perspective_declared_language) - the base table
    this filters and samples from.
  - step01's games.parquet (game title, popular_tags) - joined by game_id,
    for the game_name column and the optional game_tag filter.
  - step04's output (sentiment_score), if that language's folder is
    provided/exists - joined by review_url, only for the sampled rows (not
    the whole corpus, to avoid loading a second multi-million-row dataset
    just to label a handful of examples).
  - step03's classified_toxic.parquet (topic), if that language's Stage 7
    export is provided/exists - joined by review_url, only for the sampled
    rows. Only toxic reviews ever have a topic (BERTopic trains on the
    toxic subset only - see step03's README), so this is always empty for
    non-toxic examples.

`review_lang` (langdetect's own guess) is never a real column in any of
these files - step01 wrote it as Hive-style partitioning
(partition_on=["review_lang"]), so it only exists as the `review_lang=<lang>`
folder name, never as leaf-file data. Reading from that folder guarantees
review_lang == lang by construction (there's no data value left to check).
`perspective_declared_language` IS a real column, though - step02's own
detoxify_scoring.py already filters to perspective_declared_language ==
lang before scoring, so every row here should already agree, but that's
re-verified explicitly in load_scored_reviews below (not just trusted)
before anything else runs, same as step03's text_cleaning.py does.

Toxicity uses the same union rule and thresholds as everywhere else in
this project (perspective_score >= 0.7 OR detoxify_score >= 0.9, rows with
an invalid/sentinel score excluded rather than labeled non-toxic).
"""
import re
from pathlib import Path

import pandas as pd

from pipeline_utils import info, list_parquet_files

PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9

# Same boilerplate patterns stripped before scoring in step02/step03/step04/
# step05 - review_text itself is never altered anywhere in this project, so
# this is applied to a separate review_text_clean column here, purely so
# examples show what the models actually saw, not a change to review_text.
BOILERPLATE_PATTERNS = [
    r"an[aá]lise de acesso antecipado",
    r"produto recebido de gra[cç]a",
    r"produto reembolsado",
]

OUTPUT_COLUMNS = [
    "game_id", "game_name", "review_url", "review_text", "review_text_clean", "review_lang",
    "perspective_score", "detoxify_score", "sentiment_score", "topic",
]


def clean_review_text(text):
    """Strips known boilerplate phrases from review text (case-insensitive).
    Non-string input (e.g. NaN) passes through unchanged. Same logic as
    detoxify_scoring.py/sentiment_scoring.py's clean_review_text."""
    if not isinstance(text, str):
        return text
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_games(games_path: Path) -> pd.DataFrame:
    return pd.read_parquet(games_path, columns=["game_id", "title", "popular_tags"])


def load_scored_reviews(step02_dir: Path, lang: str) -> pd.DataFrame:
    """Base table: every step02-scored review for one language.

    review_lang == lang is guaranteed by construction (see module
    docstring - it's the folder we're reading from, not a real column).
    perspective_declared_language IS a real column, so it's explicitly
    re-checked here (== lang) rather than trusted from step02 alone -
    same double-check step03's text_cleaning.py does. Any row that
    disagrees is dropped and logged (unexpected in the normal flow, since
    step02 should have already filtered these)."""
    partition_dir = step02_dir / f"review_lang={lang}"
    files = list_parquet_files(partition_dir)
    columns = [
        "review_url", "review_text", "game_id",
        "perspective_score", "detoxify_score", "perspective_declared_language",
    ]
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    df = pd.concat(frames, ignore_index=True)

    rows_before_agreement = len(df)
    df = df[df["perspective_declared_language"] == lang].copy()
    n_excluded_disagreement = rows_before_agreement - len(df)
    if n_excluded_disagreement:
        info(
            f"[{lang}] Excluded {n_excluded_disagreement} row(s) where "
            f"perspective_declared_language != '{lang}' (unexpected - step02 "
            f"should have already filtered these)"
        )

    df["review_lang"] = lang
    return df.drop(columns=["perspective_declared_language"])


def _game_has_tag(tags, game_tag: str) -> bool:
    if tags is None:
        return False
    game_tag = game_tag.lower()
    if isinstance(tags, str):
        return game_tag in tags.lower()
    try:
        return any(game_tag in str(t).lower() for t in tags)
    except TypeError:
        return False  # not a string and not iterable (e.g. NaN)


def filter_reviews(
    df: pd.DataFrame,
    games: pd.DataFrame,
    toxic: bool,
    contains: str = None,
    game_tag: str = None,
) -> pd.DataFrame:
    """Applies toxicity labeling (same union rule/thresholds/invalid-score
    exclusion as elsewhere in this project), then the optional text/tag
    filters. Returns the filtered rows, NOT yet sampled."""
    perspective_valid = df["perspective_score"].between(0, 1)
    detoxify_valid = df["detoxify_score"].between(0, 1)
    df = df[perspective_valid & detoxify_valid].copy()

    is_toxic = (df["perspective_score"] >= PERSPECTIVE_THRESHOLD) | (df["detoxify_score"] >= DETOXIFY_THRESHOLD)
    df = df[is_toxic] if toxic else df[~is_toxic]

    if contains:
        df = df[df["review_text"].str.contains(contains, case=False, na=False, regex=False)]

    if game_tag:
        tagged_game_ids = set(
            games.loc[games["popular_tags"].apply(lambda t: _game_has_tag(t, game_tag)), "game_id"]
        )
        df = df[df["game_id"].isin(tagged_game_ids)]

    return df


def sample_reviews(df: pd.DataFrame, n: int, seed: int = None) -> pd.DataFrame:
    if len(df) <= n:
        info(f"Only {len(df)} matching review(s) available, requested {n} - returning all of them")
        return df.copy()
    return df.sample(n=n, random_state=seed)


def attach_game_names(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        games[["game_id", "title"]].rename(columns={"title": "game_name"}), on="game_id", how="left"
    )


def attach_sentiment(df: pd.DataFrame, step04_dir: Path, lang: str) -> pd.DataFrame:
    """Left-joins sentiment_score for the sampled rows only, if step04's
    output exists for this language. No-op (adds an empty column) if not -
    per this function's contract, this data may not be available yet."""
    partition_dir = (step04_dir / f"review_lang={lang}") if step04_dir else None
    if not partition_dir or not partition_dir.exists():
        info(f"No step04 output found for [{lang}] - sentiment_score will be empty")
        df["sentiment_score"] = pd.NA
        return df

    files = list_parquet_files(partition_dir)
    frames = [pd.read_parquet(f, columns=["review_url", "sentiment_score"]) for f in files]
    sentiment = pd.concat(frames, ignore_index=True)
    return df.merge(sentiment, on="review_url", how="left")


def attach_topics(df: pd.DataFrame, step03_results_path: Path) -> pd.DataFrame:
    """Left-joins BERTopic's `topic` for the sampled rows only, if step03's
    Stage 7 export is provided/exists for this language. Only ever
    populated for toxic reviews (BERTopic trains on the toxic subset only)."""
    if not step03_results_path or not Path(step03_results_path).exists():
        info(f"No step03 classified_toxic.parquet found at {step03_results_path} - topic will be empty")
        df["topic"] = pd.NA
        return df

    topics = pd.read_parquet(step03_results_path, columns=["review_url", "topic"])
    return df.merge(topics, on="review_url", how="left")


def get_review_examples(
    lang: str,
    toxic: bool,
    n: int,
    games_path: Path,
    step02_dir: Path,
    step04_dir: Path = None,
    step03_results_path: Path = None,
    contains: str = None,
    game_tag: str = None,
    seed: int = None,
) -> pd.DataFrame:
    """Main entry point - see module docstring for what's pulled from where.

    Args:
        lang: language code, e.g. "pt" or "en".
        toxic: True samples toxic reviews, False samples non-toxic ones.
        n: number of examples to sample (returns fewer if not enough match).
        games_path: path to step01's games.parquet.
        step02_dir: path to step02's output directory.
        step04_dir: path to step04's output directory - optional, omit if
            that step hasn't run yet for this language.
        step03_results_path: path to step03's classified_toxic.parquet for
            this language - optional, omit if Stage 7 hasn't run yet.
        contains: optional substring the review text must contain
            (case-insensitive).
        game_tag: optional popular_tag the game must have (case-insensitive).
        seed: optional random seed for reproducible sampling.
    """
    games = load_games(games_path)
    reviews = load_scored_reviews(step02_dir, lang)

    filtered = filter_reviews(reviews, games, toxic=toxic, contains=contains, game_tag=game_tag)
    info(f"[{lang}] {len(filtered)} review(s) match (toxic={toxic}, contains={contains!r}, game_tag={game_tag!r})")

    sample = sample_reviews(filtered, n=n, seed=seed)
    sample["review_text_clean"] = sample["review_text"].apply(clean_review_text)
    sample = attach_game_names(sample, games)
    sample = attach_sentiment(sample, step04_dir, lang)
    sample = attach_topics(sample, step03_results_path)

    available = [c for c in OUTPUT_COLUMNS if c in sample.columns]
    return sample[available].reset_index(drop=True)
