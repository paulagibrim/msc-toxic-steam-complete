"""Labels toxicity and compares toxic vs. non-toxic TF-IDF term weights
across step02's pt/en reviews.

Ported from dissertacao-steam/data_refactor/2-toxicity/tfidf_analysis.py
(previously driven by a notebook, 05_run_tfidf.ipynb - this project's
convention is a plain run_*.py CLI instead), with column names updated to
match this project's step02 output (`detoxify_score` instead of `toxicity`).

Toxicity labeling uses the same union rule and thresholds as every other
toxicity decision in this project (perspective_score >= 0.7 OR
detoxify_score >= 0.9 - see step01's toxicity discussion / step03's
text_cleaning.py) and the same invalid-score exclusion (rows where either
score falls outside [0, 1] - Detoxify's -1.0 "failed to score" sentinel -
are dropped rather than labeled non-toxic, since their true toxicity is
unknown).

Unlike step03 (BERTopic) and the toxic-only slice it trains on, this reads
EVERY review in step02's output, toxic and non-toxic alike - the whole
point of this analysis is comparing term weights BETWEEN the two groups.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline_utils import info, list_parquet_files

PERSPECTIVE_THRESHOLD = 0.7
DETOXIFY_THRESHOLD = 0.9

# Extra stopwords beyond nltk's per-language list, matching the original
# notebook's picks (generic words like "game"/"jogo" dominate every review
# regardless of toxicity, so they add noise rather than signal here).
STOPWORD_EXTRAS = {
    "en": ["game", "steam", "play", "playing", "games"],
    "pt": ["jogo", "game", "steam", "jogar", "pra", "jogos", "games"],
}


def load_reviews_for_language(step02_dir: Path, lang: str) -> pd.DataFrame:
    """Reads and concatenates every review_lang=<lang> file from step02's
    output - both toxic and non-toxic rows, unlike step03/step04's inputs."""
    partition_dir = step02_dir / f"review_lang={lang}"
    files = list_parquet_files(partition_dir)
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def label_toxicity(df: pd.DataFrame) -> pd.DataFrame:
    """Excludes rows with an invalid score (either perspective_score or
    detoxify_score outside [0, 1] - Detoxify's -1.0 "failed to score"
    sentinel) rather than labeling them non-toxic, since their true
    toxicity is unknown. Returns a copy of `df` restricted to valid rows,
    with an `is_toxic` column added (perspective_score >= 0.7 OR
    detoxify_score >= 0.9)."""
    perspective_valid = df["perspective_score"].between(0, 1)
    detoxify_valid = df["detoxify_score"].between(0, 1)
    valid = perspective_valid & detoxify_valid
    n_excluded_invalid = int((~valid).sum())

    valid_df = df[valid].copy()
    valid_df["is_toxic"] = (
        (valid_df["perspective_score"] >= PERSPECTIVE_THRESHOLD)
        | (valid_df["detoxify_score"] >= DETOXIFY_THRESHOLD)
    )
    if n_excluded_invalid:
        info(f"Dropped {n_excluded_invalid} row(s) with an invalid score before labeling")
    return valid_df


def _clean_one(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_tfidf(series: pd.Series) -> pd.Series:
    """Lowercases, strips URLs, strips accents, keeps only letters and
    spaces, collapses whitespace - a heavier normalization meant only for
    TF-IDF term extraction, separate from the boilerplate-only stripping
    used before Detoxify/sentiment scoring."""
    return series.apply(_clean_one)


def export_cleaned_labeled(df: pd.DataFrame, output_dir: Path, lang: str) -> Path:
    """Exports the slim columns needed for the lexicon computation, in case
    it needs to be re-run without redoing loading/labeling/cleaning:
    review_text, game_id, review_text_clean, is_toxic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reviews_cleaned_labeled_{lang}.parquet"
    df[["review_text", "game_id", "review_text_clean", "is_toxic"]].to_parquet(output_path, index=False)
    info(f"Exported cleaned+labeled dataset ({lang}) to: {output_path}")
    return output_path


def build_stop_words(nltk_language: str, extra: list) -> list:
    """Combines nltk's per-language stopword list with STOPWORD_EXTRAS -
    downloads the nltk stopwords corpus on first use if not already present."""
    import nltk
    from nltk.corpus import stopwords as _sw

    nltk.download("stopwords", quiet=True)
    base = set(_sw.words(nltk_language))
    base.update(extra)
    return sorted(base)


def fit_vocabulary(texts: pd.Series, stopwords: list, min_df=100, max_df=0.8, max_features=100000):
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        dtype=np.float32,
    )
    vectorizer.fit(texts)
    return vectorizer


def compute_group_means(df: pd.DataFrame, vectorizer: TfidfVectorizer, chunk_size: int = 500_000) -> dict:
    """Mean TF-IDF weight per term, separately for toxic and non-toxic rows.
    Processes in chunks and accumulates sums (rather than transforming the
    whole corpus into one sparse matrix at once) to bound peak memory."""
    n_features = len(vectorizer.get_feature_names_out())
    sum_toxic = np.zeros(n_features, dtype=np.float64)
    sum_neutral = np.zeros(n_features, dtype=np.float64)
    count_toxic = 0
    count_neutral = 0

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        texts = chunk["review_text_clean"].fillna("").astype(str)
        X = vectorizer.transform(texts)
        mask_toxic = chunk["is_toxic"].fillna(False).to_numpy(dtype=bool)

        if mask_toxic.any():
            sum_toxic += np.asarray(X[mask_toxic].sum(axis=0)).ravel()
            count_toxic += int(mask_toxic.sum())
        if (~mask_toxic).any():
            sum_neutral += np.asarray(X[~mask_toxic].sum(axis=0)).ravel()
            count_neutral += int((~mask_toxic).sum())

    mean_toxic = sum_toxic / count_toxic if count_toxic > 0 else sum_toxic
    mean_neutral = sum_neutral / count_neutral if count_neutral > 0 else sum_neutral
    return {
        "mean_toxic": mean_toxic,
        "mean_neutral": mean_neutral,
        "count_toxic": count_toxic,
        "count_neutral": count_neutral,
    }


def build_lexicon_table(vectorizer: TfidfVectorizer, means: dict) -> pd.DataFrame:
    terms = np.array(vectorizer.get_feature_names_out())
    table = pd.DataFrame({
        "termo": terms,
        "tfidf_toxico": means["mean_toxic"],
        "tfidf_neutro": means["mean_neutral"],
    })
    table["diferenca"] = table["tfidf_toxico"] - table["tfidf_neutro"]
    table["proporcao_toxico_neutro"] = table["tfidf_toxico"] / (table["tfidf_neutro"] + 1e-9)
    table["log_ratio"] = np.log2((table["tfidf_toxico"] + 1e-9) / (table["tfidf_neutro"] + 1e-9))
    return table.sort_values("diferenca", ascending=False).reset_index(drop=True)


def export_lexicon_table(table: pd.DataFrame, output_dir: Path, lang: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tfidf_lexicon_{lang}.csv"
    table.to_csv(output_path, index=False)
    info(f"Exported TF-IDF lexicon table ({lang}) to: {output_path}")
    return output_path
