"""Data models used by the AGENTS_NEW modular pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class BlockResult:
    """Container returned by every notebook-facing pipeline function."""

    name: str
    data: pd.DataFrame
    summary: pd.DataFrame
    feature_summary: pd.DataFrame
    validation: pd.DataFrame | None = None
    extra_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    plots: list[Path] = field(default_factory=list)
    from_cache: bool = False
    notes: str = ""
