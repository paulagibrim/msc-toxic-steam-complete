"""Cleaning steps for the raw games dataset (steam-data/raw/games/todos_jogos.json).

Ported from dissertacao-steam/data_refactor/0-cleaning/clean_games.py -
same logic, this project just doesn't load paths from a config.yaml (see
run_clean_games.py for the CLI wrapper).
"""
from pathlib import Path

import pandas as pd

from pipeline_utils import info, replace_sentinels, safe_astype, warn_if_not_materialized

RENAME_MAP = {
    "titulo": "title",
    "gameId": "game_id",
    "dataLancamento": "release_date",
    "dataLancamentoCedo": "early_access_date",
    "genres": "genres",
    "desenvolvedor": "developer",
    "destribuidor": "publisher",
    "serie": "series",
    "marcadoresPopulares": "popular_tags",
    "gameRating": "game_rating",
    "reviewsRecentes": "recent_reviews_rating",
    "qntdReviewsRecentes": "recent_reviews_count",
    "porcentagemRecente": "recent_reviews_percentage",
    "reviews": "all_time_reviews_rating",
    "qntdReviews": "all_time_reviews_count",
    "porcentagem": "all_time_reviews_percentage",
    "preco": "price",
    "requisitos": "requirements",
    "requisitosMinimos": "minimum_requirements",
    "metascore": "metascore",
    "texto_ia": "ai_disclosure",
    "texto_mature": "mature_content_disclosure",
    "qntdCuradores": "curators_count",
    "dlc_id": "dlc_id",
    "grafico": "has_review_bombing",
    "idiomas_interface": "languages_interface",
    "idiomas_dublagem": "languages_audio",
    "idiomas_legendas": "languages_subtitles",
}

# Dropped for high null share and low relevance to analysis.
COLUMNS_TO_DROP = ["series", "dlc_id"]

CATEGORICAL_COLUMNS = ["recent_reviews_rating", "all_time_reviews_rating", "game_rating"]


def load_raw_games(games_raw_file: Path) -> pd.DataFrame:
    warn_if_not_materialized(games_raw_file)
    return pd.read_json(games_raw_file)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP)


def drop_low_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if existing:
        info(f"Dropping low-value columns: {existing}")
    return df.drop(columns=existing)


def replace_missing_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    return replace_sentinels(df)


def fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["release_date", "early_access_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def fix_price(df: pd.DataFrame) -> pd.DataFrame:
    df["price"] = df["price"].replace("Free", 0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def fix_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def drop_missing_titles(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["title", "game_id"])
    info(f"Dropped {before - len(df)} row(s) with missing title or game_id")
    return df


def drop_duplicate_games(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["game_id"])
    info(f"Dropped {before - len(df)} duplicate row(s) by game_id")
    return df


def export_games(df: pd.DataFrame, processed_games_file: Path) -> None:
    processed_games_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_games_file, index=False)
    info(f"Exported cleaned games dataframe to: {processed_games_file}")
