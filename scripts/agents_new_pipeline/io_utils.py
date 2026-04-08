"""Input/output helpers and cache-aware wrappers for AGENTS_NEW."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import scripts.build_user_course_features_AGENT as legacy_pipeline

from .config import FIG_DIR, OUT_DIR


def ensure_output_dirs() -> None:
    """Create AGENTS_NEW output folders if they do not yet exist."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a UTF-8 CSV file from disk."""

    return pd.read_csv(path, encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Save a DataFrame to UTF-8 CSV and return the saved path."""

    ensure_output_dirs()
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def artifact_exists(path: Path) -> bool:
    """Check whether an artifact already exists on disk."""

    return path.exists() and path.is_file()


@contextmanager
def legacy_output_context():
    """Temporarily redirect the legacy monolithic pipeline into AGENTS_NEW."""

    ensure_output_dirs()
    previous_out = legacy_pipeline.OUT_DIR
    previous_fig = legacy_pipeline.FIG_DIR
    legacy_pipeline.OUT_DIR = OUT_DIR
    legacy_pipeline.FIG_DIR = FIG_DIR
    legacy_pipeline.ensure_output_dirs()
    try:
        yield legacy_pipeline
    finally:
        legacy_pipeline.OUT_DIR = previous_out
        legacy_pipeline.FIG_DIR = previous_fig
