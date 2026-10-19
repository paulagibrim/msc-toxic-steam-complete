"""Generates example-message files for manual inspection - sampled reviews
matching a language + toxicity + (optionally) a text term / game tag, with
whatever score/topic data is available joined in.

Pulls from up to four sources, cross-referenced by review_url/game_id:
  - step02's output (review_text, perspective_score, detoxify_score,
    review_url, game_id, perspective_declared_language, plus every other
    column step01 originally scraped - review_date, hours_played,
    is_recommended, user_url, detection_confidence) - the base table this
    filters and samples from.
  - step01's games.parquet (game title, popular_tags) - joined by game_id,
    for the game_name column and the optional game_tag filter.
  - step04's output (sentiment_score), if it's provided/exists - joined by
    review_url, only for the sampled rows (not the whole corpus, to avoid
    loading a second multi-million-row dataset just to label a handful of
    examples).
  - step03's classified_toxic.parquet (topic), if that language's Stage 7
    export is provided/exists - joined by review_url, only for the sampled
    rows. Only toxic reviews ever have a topic (BERTopic trains on the
    toxic subset only - see step03's README), so this is always empty for
    non-toxic examples.

step01/step02/step04's output is flat (review_lang is a plain column, not
a directory partition - see step02_run_detoxify/detoxify_scoring.py's
module docstring): every language is scored together in the same file, so
`review_lang == lang AND perspective_declared_language == lang` is what
actually selects this language's rows, not just a defensive double-check -
applied explicitly in load_scored_reviews below, same as step03's
text_cleaning.py and step05's tfidf_analysis.py.

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
    "game_id", "game_name", "review_url", "user_url", "review_date", "is_recommended",
    "hours_played", "review_text", "review_text_clean", "review_lang", "detection_confidence",
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


def _resolve_lang_source(base_dir: Path, lang: str) -> Path:
    """step02/step04's output has been observed in two different layouts
    across this project's lifetime: subfolders (base_dir/review_lang=<lang>/
    *.parquet) and flat (every language together in base_dir directly, with
    review_lang as a column). Rather than hardcode one, check which shape
    is actually present and use that - avoids silently breaking again if
    the layout changes."""
    subfolder = base_dir / f"review_lang={lang}"
    return subfolder if subfolder.is_dir() else base_dir


def load_scored_reviews(step02_dir: Path, lang: str, chunk_filter_fn=None) -> pd.DataFrame:
    """Base table: every step02-scored review for one language. Handles
    both the subfolder layout (read the review_lang=<lang>/ subfolder
    directly - review_lang isn't a real column there) and the flat layout
    (every language together, filtered by the review_lang column) - see
    _resolve_lang_source."""
    source = _resolve_lang_source(step02_dir, lang)
    is_subfolder = source != step02_dir

    files = list_parquet_files(source)
    columns = [
        "review_url", "review_text", "game_id", "perspective_score", "detoxify_score",
        "user_url", "review_date", "is_recommended", "hours_played", "detection_confidence",
    ]
    if not is_subfolder:
        columns += ["review_lang"]
    columns += ["perspective_declared_language"]

    frames = []
    total_excluded = 0
    for f in files:
        df = pd.read_parquet(f, columns=columns)
        if is_subfolder:
            df["review_lang"] = lang

        rows_before_mask = len(df)
        df = df[
            (df["review_lang"] == lang) & (df["perspective_declared_language"] == lang)
        ].copy()
        total_excluded += (rows_before_mask - len(df))
        
        df = df.drop(columns=["perspective_declared_language"])
        
        if chunk_filter_fn:
            df = chunk_filter_fn(df)
            
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[c for c in columns if c != "perspective_declared_language"])

    if total_excluded:
        info(
            f"[{lang}] Excluded {total_excluded} row(s) not matching "
            f"review_lang == perspective_declared_language == '{lang}'"
        )

    return df


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
    toxic: bool = None,
    venn_set: str = None,
    contains: str = None,
    game_tag: str = None,
) -> pd.DataFrame:
    """Applies toxicity labeling (same union rule/thresholds/invalid-score
    exclusion as elsewhere in this project), then the optional text/tag
    filters. Returns the filtered rows, NOT yet sampled.

    Exactly one of `toxic` (binary toxic/non-toxic) or `venn_set`
    ('perspective_only' | 'detoxify_only' | 'both' - the same three
    mutually-exclusive Venn regions used in run_toxicity_venn.py and
    step05's tfidf_venn_set_analysis.py) must be given - venn_set exists
    specifically to pull qualitative examples for a TF-IDF finding (e.g.
    "detoxify_only reviews look neutral by term frequency - are they
    actually toxic in context, or genuinely mislabeled?") that term-level
    statistics alone can't answer."""
    if (toxic is None) == (venn_set is None):
        raise ValueError("Exactly one of `toxic` or `venn_set` must be given")

    perspective_valid = df["perspective_score"].between(0, 1)
    detoxify_valid = df["detoxify_score"].between(0, 1)
    df = df[perspective_valid & detoxify_valid].copy()

    if venn_set is not None:
        is_p = df["perspective_score"] >= PERSPECTIVE_THRESHOLD
        is_d = df["detoxify_score"] >= DETOXIFY_THRESHOLD
        region_mask = {
            "perspective_only": is_p & ~is_d,
            "detoxify_only": ~is_p & is_d,
            "both": is_p & is_d,
        }
        if venn_set not in region_mask:
            raise ValueError(f"venn_set must be one of {list(region_mask)}, got {venn_set!r}")
        df = df[region_mask[venn_set]]
    else:
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
    output exists. No-op (adds an empty column) if not - per this
    function's contract, this data may not be available yet.

    Handles both the subfolder and flat layouts - see _resolve_lang_source."""
    if not step04_dir or not step04_dir.exists():
        info(f"No step04 output found for [{lang}] - sentiment_score will be empty")
        df["sentiment_score"] = pd.NA
        return df

    source = _resolve_lang_source(step04_dir, lang)
    is_subfolder = source != step04_dir

    files = list_parquet_files(source)
    if not files:
        info(f"No step04 files found for [{lang}] at {source} - sentiment_score will be empty")
        df["sentiment_score"] = pd.NA
        return df

    columns = ["review_url", "sentiment_score"] + ([] if is_subfolder else ["review_lang"])
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    sentiment = pd.concat(frames, ignore_index=True)
    if not is_subfolder:
        sentiment = sentiment[sentiment["review_lang"] == lang].drop(columns=["review_lang"])
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
    n: int,
    games_path: Path,
    step02_dir: Path,
    toxic: bool = None,
    venn_set: str = None,
    step04_dir: Path = None,
    step03_results_path: Path = None,
    contains: str = None,
    game_tag: str = None,
    seed: int = None,
    light_mode: bool = False,
) -> pd.DataFrame:
    """Main entry point - see module docstring for what's pulled from where.

    Args:
        lang: language code, e.g. "pt" or "en".
        n: number of examples to sample (returns fewer if not enough match).
        games_path: path to step01's games.parquet.
        step02_dir: path to step02's output directory.
        toxic: True samples toxic reviews, False samples non-toxic ones -
            exactly one of `toxic`/`venn_set` must be given (see
            filter_reviews).
        venn_set: 'perspective_only' | 'detoxify_only' | 'both' - samples
            from that Venn region instead of the binary toxic/non-toxic
            split (see filter_reviews).
        step04_dir: path to step04's output directory - optional, omit if
            that step hasn't run yet for this language.
        step03_results_path: path to step03's classified_toxic.parquet for
            this language - optional, omit if Stage 7 hasn't run yet.
        contains: optional substring the review text must contain
            (case-insensitive).
        game_tag: optional popular_tag the game must have (case-insensitive).
        seed: optional random seed for reproducible sampling.
        light_mode: if True, filters files one by one to drastically reduce RAM usage.
    """
    games = load_games(games_path)

    if light_mode:
        def filter_chunk(chunk_df):
            return filter_reviews(chunk_df, games, toxic=toxic, venn_set=venn_set, contains=contains, game_tag=game_tag)
        filtered = load_scored_reviews(step02_dir, lang, chunk_filter_fn=filter_chunk)
    else:
        reviews = load_scored_reviews(step02_dir, lang)
        filtered = filter_reviews(reviews, games, toxic=toxic, venn_set=venn_set, contains=contains, game_tag=game_tag)
        
    selector = f"venn_set={venn_set}" if venn_set is not None else f"toxic={toxic}"
    info(f"[{lang}] {len(filtered)} review(s) match ({selector}, contains={contains!r}, game_tag={game_tag!r})")

    sample = sample_reviews(filtered, n=n, seed=seed)
    sample["review_text_clean"] = sample["review_text"].apply(clean_review_text)
    sample = attach_game_names(sample, games)
    sample = attach_sentiment(sample, step04_dir, lang)
    sample = attach_topics(sample, step03_results_path)

    available = [c for c in OUTPUT_COLUMNS if c in sample.columns]
    return sample[available].reset_index(drop=True)
