"""Plot helpers for notebook-visible diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIG_DIR
from .io_utils import ensure_output_dirs


def _show_if_interactive() -> None:
    """Show a figure only when matplotlib runs in an interactive backend."""

    if "agg" not in plt.get_backend().lower():
        plt.show()


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    filename: str,
    rotate: int = 45,
) -> Path:
    """Create, save, and display a bar chart."""

    ensure_output_dirs()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x].astype(str), df[y], color="#4472c4")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150)
    _show_if_interactive()
    plt.close(fig)
    return path


def plot_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    filename: str,
    bins: int = 40,
) -> Path:
    """Create, save, and display a histogram."""

    ensure_output_dirs()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=bins, color="#4472c4", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150)
    _show_if_interactive()
    plt.close(fig)
    return path


def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    """Create, save, and display a line chart."""

    ensure_output_dirs()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x], df[y], color="#4472c4", linewidth=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150)
    _show_if_interactive()
    plt.close(fig)
    return path
