import pandas as pd
from pipeline_utils import info, save_summary, list_parquet_files
from pathlib import Path

def apply_toxicity_mask(df: pd.DataFrame, perspective_threshold: float = 0.7, detoxify_threshold: float = 0.9) -> pd.Series:
    """True where EITHER perspective_score >= perspective_threshold OR
    detoxify_score >= detoxify_threshold."""
    return (df["perspective_score"] >= perspective_threshold) | (df["detoxify_score"] >= detoxify_threshold)

def summarize_toxicity(df: pd.DataFrame, label: str, perspective_threshold: float = 0.7, detoxify_threshold: float = 0.9) -> dict:
    persp_mask = df["perspective_score"] >= perspective_threshold
    detox_mask = df["detoxify_score"] >= detoxify_threshold
    
    only_persp = int((persp_mask & ~detox_mask).sum())
    only_detox = int((~persp_mask & detox_mask).sum())
    both = int((persp_mask & detox_mask).sum())
    total_selected = only_persp + only_detox + both
    rows_total = len(df)
    
    return {
        "label": label,
        "rows_total": rows_total,
        "perspective_only": only_persp,
        "detoxify_only": only_detox,
        "both": both,
        "total_selected": total_selected,
    }
