"""
settings.py — Configuration loader and validator.

Loads a config YAML (config_en.yaml or config_pt.yaml), resolves all paths
relative to this pipeline's own folder (step03_bertopic/), and exposes a
single Settings object consumed by every other module.

Ported from dissertacao-steam/bertopic_pipeline/src/settings.py. Paths
changed to resolve relative to step03_bertopic/ itself (not a project
root one level up) - matches step01/step02's "../../steam-data/..."
convention, since this project keeps all data in the sibling steam-data/
folder rather than inside the code repo.

Design rationale:
- A single source of truth (the config YAML) prevents parameter drift
  between pipeline stages.
- Paths are resolved to absolute Path objects here so callers never deal
  with relative-path ambiguity.
- The Settings object is immutable after construction; all pipeline stages
  receive the same instance.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

# Newer MLflow versions refuse a plain filesystem tracking URI (paths.mlruns_dir,
# a bare directory) by default, raising unless this is set - see
# https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store. This
# pipeline only needs basic params/metrics/artifact logging, so the file
# store (not a SQLite/DB backend) is fine; set once here since every stage
# imports settings before touching mlflow.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# settings.py lives in step03_bertopic/src/ - step03_bertopic/ itself is
# where every relative path in the config YAML is resolved from.
_PIPELINE_ROOT = Path(__file__).parent.parent.resolve()


def _resolve(root: Path, rel: str) -> Path:
    """Return an absolute path by joining root with a relative string."""
    p = Path(rel)
    return p if p.is_absolute() else root / p


class Settings:
    """Typed, validated view of a config YAML."""

    def __init__(self, cfg: dict) -> None:
        r = _PIPELINE_ROOT

        # ── Reproducibility ────────────────────────────────────────────────────
        self.seed: int = int(cfg["seed"])

        # ── Paths ──────────────────────────────────────────────────────────────
        p = cfg["paths"]
        self.detoxify_data_dir: Path  = _resolve(r, p["detoxify_data_dir"])
        self.cleaned_data_dir: Path   = _resolve(r, p["cleaned_data_dir"])
        self.embeddings_dir: Path     = _resolve(r, p["embeddings_dir"])
        self.models_dir: Path         = _resolve(r, p["models_dir"])
        self.stability_dir: Path      = _resolve(r, p["stability_dir"])
        self.results_dir: Path        = _resolve(r, p["results_dir"])
        self.mlruns_dir: Path         = _resolve(r, p["mlruns_dir"])
        self.optuna_db: Path          = _resolve(r, p["optuna_db"])
        self.best_params_path: Path   = _resolve(r, p["best_params"])
        self.pca_model_path: Path     = _resolve(r, p["pca_model"])

        # Derived paths used across stages
        self.raw_embeddings_path: Path = self.embeddings_dir / "embeddings_raw.npy"
        self.pca_embeddings_path: Path = self.embeddings_dir / "embeddings_pca.npy"
        self.toxic_index_path: Path    = self.embeddings_dir / "toxic_index.parquet"
        self.final_model_path: Path    = self.models_dir / "final_model"
        self.final_results_path: Path  = self.results_dir / "classified_toxic.parquet"
        self.topic_info_path: Path     = self.results_dir / "topic_info.csv"

        # Optuna SQLite URI used by optuna.create_study(storage=...)
        self.optuna_storage: str = f"sqlite:///{self.optuna_db}"

        # ── Data ───────────────────────────────────────────────────────────────
        d = cfg["data"]
        self.perspective_threshold: float  = float(d["toxicity_threshold"]["perspective_score"])
        self.detoxify_threshold: float     = float(d["toxicity_threshold"]["detoxify_score"])
        self.text_column: str              = d["text_column"]
        self.language: str                 = d.get("language", "english")
        self.lang_code: str                = d["lang_code"]

        # ── Embedding ──────────────────────────────────────────────────────────
        e = cfg["embedding"]
        self.embedding_model_name: str   = e["model_name"]
        self.embedding_batch_size: int   = int(e["batch_size"])
        self.normalize_embeddings: bool  = bool(e["normalize_embeddings"])
        self.pca_components: Optional[int] = (
            int(e["pca_components"]) if e.get("pca_components") else None
        )

        # ── Optuna ─────────────────────────────────────────────────────────────
        o = cfg["optuna"]
        self.optuna_n_trials: int         = int(o["n_trials"])
        self.optuna_n_jobs: int           = int(o["n_jobs"])
        self.optuna_sample_fraction: float = float(o["optuna_sample_fraction"])
        self.coherence_metric: str        = o["coherence_metric"]
        self.coherence_top_n: int         = int(o["top_n_words"])
        self.coherence_weight: float      = float(o["coherence_weight"])
        self.optuna_study_name: str       = o["study_name"]
        self.umap_search_space: dict      = o["search_space"]["umap"]
        self.hdbscan_search_space: dict   = o["search_space"]["hdbscan"]

        # ── Stability ──────────────────────────────────────────────────────────
        s = cfg["stability"]
        self.sample_fractions: List[float]   = [float(x) for x in s["sample_fractions"]]
        self.stability_threshold: float      = float(s["stability_threshold"])
        self.top_n_topics_comparison: int    = int(s["top_n_topics_for_comparison"])

        # ── Training fallback defaults ─────────────────────────────────────────
        t = cfg["training"]
        self.default_umap_params: dict    = dict(t["umap"])
        self.default_hdbscan_params: dict = dict(t["hdbscan"])

        # ── Stop words ─────────────────────────────────────────────────────────
        self.extra_stop_words: List[str] = list(cfg.get("stop_words", []))

    # ── Convenience methods ────────────────────────────────────────────────────

    def create_directories(self) -> None:
        """Create all output directories that do not yet exist."""
        dirs = [
            self.cleaned_data_dir,
            self.embeddings_dir,
            self.models_dir,
            self.stability_dir,
            self.results_dir,
            self.mlruns_dir,
            self.optuna_db.parent,
            self.best_params_path.parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("All output directories confirmed.")

    def load_best_params(self) -> dict:
        """Load the best hyperparameters written by Stage 3 (Optuna)."""
        if not self.best_params_path.exists():
            logger.warning(
                "best_params.json not found at %s. "
                "Falling back to default training parameters.",
                self.best_params_path,
            )
            return {
                **{f"umap_{k}": v for k, v in self.default_umap_params.items()},
                **{f"hdbscan_{k}": v for k, v in self.default_hdbscan_params.items()},
            }
        with open(self.best_params_path) as f:
            params = json.load(f)
        logger.info("Loaded best params from %s: %s", self.best_params_path, params)
        return params

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        """Load Settings from a YAML file (config_en.yaml or config_pt.yaml)."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        logger.info("Configuration loaded from %s", config_path)
        return cls(cfg)

    @classmethod
    def from_lang(cls, lang: str) -> "Settings":
        """Load Settings by language code, via the config_<lang>.yaml naming
        convention (e.g. "en" -> step03_bertopic/config_en.yaml). Lets every
        run/*.py script accept a --lang flag instead of a raw --config path."""
        config_path = _PIPELINE_ROOT / f"config_{lang}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config found for lang='{lang}' (expected {config_path})."
            )
        return cls.from_yaml(config_path)
