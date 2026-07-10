"""
utils.py — Shared utilities: seed control, hardware detection, logging, timing.

Keeping these helpers centralised prevents the seed-setting boilerplate from
being duplicated (and potentially missed) in every pipeline stage.

Ported unchanged from dissertacao-steam/bertopic_pipeline/src/utils.py - no
project-specific paths or column names in here.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Pin every source of randomness to make runs reproducible.

    Note: UMAP must also be called with random_state=seed and n_jobs=1.
    These kwargs are set in each module that instantiates UMAP.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic mode trades some performance for bit-exact reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug("Global seed set to %d.", seed)


# ── Hardware detection ─────────────────────────────────────────────────────────

def detect_hardware() -> dict:
    """Return a snapshot of available compute resources.

    Used by pipeline stages to choose device, batch size, and worker count
    at runtime rather than hard-coding values that may not fit the target server.
    """
    info: dict = {}

    # CPU
    info["cpu_count"] = os.cpu_count() or 1

    # RAM
    mem = psutil.virtual_memory()
    info["ram_total_gb"]     = round(mem.total     / (1024 ** 3), 1)
    info["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)

    # GPU
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["gpu_count"]     = torch.cuda.device_count()
        info["gpu_name"]      = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
    else:
        info["gpu_available"] = False
        info["gpu_count"]     = 0
        info["gpu_name"]      = None
        info["vram_gb"]       = 0.0

    return info


def get_embedding_device(hw: dict) -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if hw["gpu_available"] else "cpu"


def get_pandarallel_workers(hw: dict) -> int:
    """Return the number of pandarallel workers.

    Reserves 8 cores for the OS and other concurrent processes on shared
    servers.  On a workstation with few cores the minimum returned is 1.
    """
    return max(1, hw["cpu_count"] - 8)


def log_hardware(hw: dict) -> None:
    """Write a hardware summary to the logger at INFO level."""
    logger.info(
        "Hardware — CPU cores: %d | RAM: %.1f GB (%.1f GB free) | "
        "GPU: %s (%.1f GB VRAM)",
        hw["cpu_count"],
        hw["ram_total_gb"],
        hw["ram_available_gb"],
        hw["gpu_name"] or "none",
        hw["vram_gb"],
    )


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> None:
    """Configure root logger to write to console and optionally to a file.

    Call once at the top of each run/ entrypoint.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
    handlers: list = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


# ── Timing ────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str):
    """Context manager that logs the elapsed time of the enclosed block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.1f s (%.1f min).", label, elapsed, elapsed / 60)


# ── Stop words ────────────────────────────────────────────────────────────────

def build_stop_words(language: str, extra: List[str]) -> List[str]:
    """Return a combined stop word list for use in text cleaning and BERTopic.

    Merges the NLTK stop word list for the given language with any additional
    domain-specific terms from the config YAML → stop_words.

    This function is the single place where stop words are assembled so that
    both the text cleaner (Stage 1) and BERTopic's CountVectorizer (Stages 3–6)
    always receive an identical, consistent list.

    Args:
        language: any language name accepted by nltk.corpus.stopwords,
                  e.g. "english", "portuguese", "spanish".
        extra:    domain-specific terms from the config YAML → stop_words.

    Returns:
        Deduplicated list of stop word strings.
    """
    from nltk.corpus import stopwords as _sw

    base = set(_sw.words(language))
    base.update(extra)
    return sorted(base)
