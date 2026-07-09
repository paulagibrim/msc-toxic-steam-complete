"""Cleaning steps for the raw users dataset (steam-data/raw/users/*.parquet).

Ported from dissertacao-steam/data_refactor/0-cleaning/clean_users.py - same
logic, this project just doesn't load paths from a config.yaml (see
run_clean_users.py for the CLI wrapper).
"""
from pathlib import Path

import pandas as pd

from pipeline_utils import info, read_parquet_dir, replace_sentinels, safe_astype

RENAME_MAP = {
    "id": "steam_id",
    "Nome": "steam_url",
    "Cidade": "city",
    "Estado": "state/region",
    "Pais": "country",
    "Descricao Perfil": "profile_description",
    "Nivel": "profile_level",
    "Ban": "has_ban",
    "Dias Ultimo Ban": "days_since_last_ban",
    "Motivo Ban": "ban_reason",
    "Premios": "awards",
    "Insignias": "insignias",
    "Total de Jogos": "library_size",
    "Capturas de Tela": "screenshots",
    "Itens da Oficina": "workshop_items",
    "Quantidade de Guias": "guides",
    "Quantidade de Artes": "arts",
    "Quantidade de Grupos": "groups",
    "Quantidade de Amigos": "friends_count",
    "idiomas_comentarios": "reviews_langs",
}

# Dropped: the raw file arrived with Perspective-API toxicity calculations
# (and counts derived from them) already computed by the data source, but
# using a methodology that wasn't reliable (poorly calculated) - so these
# columns are removed here instead of trusted as-is; toxicity is
# (re)computed properly elsewhere in the pipeline.
COLUMNS_TO_DROP = [
    "comentarios_nao_toxicos",
    "comentarios_toxicos",
    "Quantidade de Comentarios",
    "Quantidade de Comentarios no Perfil",
    "toxicidade_media",
    "toxicidade_desvio_padrao",
    "num_comentarios_total",
    "num_comentarios_recomendados",
    "num_comentarios_nao_recomendados",
]

TYPE_MAPPING = {
    "steam_id": "string",
    "steam_url": "string",
    "city": "string",
    "state/region": "string",
    "country": "string",
    "profile_description": "string",
    "profile_level": "Int64",
    "has_ban": "boolean",
    "days_since_last_ban": "Int64",
    "ban_reason": "string",
    "awards": "Int64",
    "insignias": "Int64",
    "library_size": "Int64",
    "screenshots": "Int64",
    "workshop_items": "Int64",
    "guides": "Int64",
    "arts": "Int64",
    "groups": "Int64",
    "friends_count": "Int64",
}


def load_raw_users(users_raw_dir: Path) -> pd.DataFrame:
    return read_parquet_dir(users_raw_dir, engine="pandas")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP)


def drop_untrusted_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if existing:
        info(f"Dropping untrusted/unreliable columns: {existing}")
    return df.drop(columns=existing)


def replace_missing_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    return replace_sentinels(df)


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    return safe_astype(df, TYPE_MAPPING)


def drop_duplicate_users(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["steam_id"])
    info(f"Dropped {before - len(df)} duplicate row(s) by steam_id")
    return df


def export_users(df: pd.DataFrame, processed_users_file: Path) -> None:
    processed_users_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_users_file, index=False)
    info(f"Exported cleaned users dataframe to: {processed_users_file}")
