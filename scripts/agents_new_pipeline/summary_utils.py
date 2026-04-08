"""Summary creation and notebook display helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import BlockResult

try:
    from IPython.display import Markdown, display
except ImportError:  # pragma: no cover
    Markdown = None
    display = None


def _safe_display(obj) -> None:
    """Display an object in notebooks and fall back to print in scripts."""

    if display is not None:
        display(obj)
    else:  # pragma: no cover
        print(obj)


def build_table_summary(
    df: pd.DataFrame,
    block_name: str,
    key_column: str | None,
    from_cache: bool,
    notes: str = "",
) -> pd.DataFrame:
    """Create a compact summary table for a block output."""

    metrics = [
        {"metric": "rows", "value": int(len(df))},
        {"metric": "columns", "value": int(df.shape[1])},
        {"metric": "from_cache", "value": int(from_cache)},
    ]
    if key_column and key_column in df.columns:
        metrics.extend(
            [
                {"metric": "unique_key_rows", "value": int(df[key_column].nunique(dropna=True))},
                {"metric": "duplicate_key_rows", "value": int(df.duplicated([key_column]).sum())},
                {"metric": "missing_key_rows", "value": int(df[key_column].isna().sum())},
            ]
        )
    if notes:
        metrics.append({"metric": "notes", "value": notes})
    summary = pd.DataFrame(metrics)
    summary.insert(0, "block_name", block_name)
    return summary


def build_feature_summary(
    df: pd.DataFrame,
    key_column: str | None,
    top_n: int = 12,
) -> pd.DataFrame:
    """Create a compact per-feature overview for user-facing review."""

    feature_rows = []
    for column in df.columns:
        if column == key_column:
            continue
        value_counts = df[column].value_counts(dropna=False, normalize=True)
        top_share = float(value_counts.iloc[0]) if not value_counts.empty else 0.0
        feature_rows.append(
            {
                "feature_name": column,
                "dtype": str(df[column].dtype),
                "missing_share": round(float(df[column].isna().mean()), 4),
                "nunique": int(df[column].nunique(dropna=True)),
                "top_value_share": round(top_share, 4),
            }
        )
    feature_summary = pd.DataFrame(feature_rows)
    return feature_summary.sort_values(["missing_share", "top_value_share"], ascending=[False, False]).head(top_n)


def display_block_result(result: BlockResult, preview_rows: int = 5) -> None:
    """Render summary tables, validation tables, and a small preview."""

    title = f"### {result.name}"
    subtitle = "Loaded from cache." if result.from_cache else "Built from raw inputs and saved to AGENTS_NEW."
    if Markdown is not None:
        _safe_display(Markdown(f"{title}\n\n{subtitle}\n\n{result.notes}"))
    else:  # pragma: no cover
        print(title)
        print(subtitle)
        print(result.notes)

    _safe_display(result.summary)
    _safe_display(result.feature_summary)

    if result.validation is not None and not result.validation.empty:
        _safe_display(result.validation)

    for label, table in result.extra_tables.items():
        if Markdown is not None:
            _safe_display(Markdown(f"**{label}**"))
        else:  # pragma: no cover
            print(label)
        _safe_display(table)

    _safe_display(result.data.head(preview_rows))


def attach_summary_and_features(
    name: str,
    data: pd.DataFrame,
    key_column: str | None,
    from_cache: bool,
    validation: pd.DataFrame | None = None,
    extra_tables: dict[str, pd.DataFrame] | None = None,
    artifact_paths: dict[str, Path] | None = None,
    plots: list[Path] | None = None,
    notes: str = "",
) -> BlockResult:
    """Build a complete BlockResult object."""

    return BlockResult(
        name=name,
        data=data,
        summary=build_table_summary(data, name, key_column, from_cache, notes=notes),
        feature_summary=build_feature_summary(data, key_column),
        validation=validation,
        extra_tables=extra_tables or {},
        artifact_paths=artifact_paths or {},
        plots=plots or [],
        from_cache=from_cache,
        notes=notes,
    )
