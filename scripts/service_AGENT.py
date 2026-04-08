import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config_AGENT import (
    BOOL_COLS_MAP,
    CATEGORY_COLS_MAP,
    CORE_ENTITY_KEY,
    DATA_RAW_DIR,
    DATE_COLS,
    DROP_COLS_MAP,
    EXPORT_DATE_FORMAT,
    FILES,
    FIGURES_DIR,
    FLOAT_COLS_MAP,
    ID_COLS_MAP,
    RAW_USECOLS_MAP,
    SUMMARIES_DIR,
    TABLES_DIR,
)


# =============================================================================
# Filesystem utilities
# =============================================================================


def ensure_output_directories() -> None:
    """Create output folders if they do not exist."""
    for path in [TABLES_DIR, SUMMARIES_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_table_path(table_name: str) -> Path:
    """Return the raw CSV path for a configured table."""
    return DATA_RAW_DIR / FILES[table_name]


# =============================================================================
# Column cleaning and type parsing
# =============================================================================


def clean_column_name(column_name: str) -> str:
    """Convert a raw CSV column name into a stable ASCII snake_case name."""
    ascii_name = (
        unicodedata.normalize("NFKD", str(column_name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    ascii_name = ascii_name.strip().lower()
    ascii_name = re.sub(r"[^a-z0-9]+", "_", ascii_name)
    ascii_name = re.sub(r"_+", "_", ascii_name).strip("_")
    return ascii_name


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed columns and normalize the remaining column names."""
    renamed = {
        column: clean_column_name(column)
        for column in df.columns
        if not str(column).lower().startswith("unnamed")
    }
    df = df.loc[:, list(renamed)].copy()
    df = df.rename(columns=renamed)
    return df


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse mixed datetime formats used across the raw exports."""
    raw = series.astype("string").str.strip()
    raw = raw.replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA})
    try:
        return pd.to_datetime(raw, errors="coerce", format="mixed", dayfirst=True)
    except TypeError:
        return pd.to_datetime(raw, errors="coerce", dayfirst=True)


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    """Convert a noisy boolean-like column into pandas boolean dtype."""
    bool_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    raw = series.astype("string").str.strip().str.lower()
    return raw.map(bool_map).astype("boolean")


def _normalize_int_like_series(series: pd.Series) -> pd.Series:
    """Strip thousands separators and cast a numeric ID-like column."""
    raw = series.astype("string").str.replace(",", "", regex=False).str.strip()
    raw = raw.replace({"": pd.NA, "nan": pd.NA})
    return pd.to_numeric(raw, errors="coerce").astype("Int64")


def _normalize_float_like_series(series: pd.Series) -> pd.Series:
    """Strip thousands separators and cast a float-like column."""
    raw = series.astype("string").str.replace(",", "", regex=False).str.strip()
    raw = raw.replace({"": pd.NA, "nan": pd.NA})
    return pd.to_numeric(raw, errors="coerce").astype("Float64")


def _apply_table_typing(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply configured date, bool, category, and float conversions."""
    for column in DATE_COLS.get(table_name, []):
        if column in df.columns:
            df[column] = _parse_datetime_series(df[column])

    for column in BOOL_COLS_MAP.get(table_name, []):
        if column in df.columns:
            df[column] = _normalize_bool_series(df[column])

    for column in FLOAT_COLS_MAP.get(table_name, []):
        if column in df.columns:
            df[column] = _normalize_float_like_series(df[column])

    for column in CATEGORY_COLS_MAP.get(table_name, []):
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().astype("category")

    return df


# =============================================================================
# Data loading and table audit
# =============================================================================




def read_raw_table(table_name: str) -> pd.DataFrame:
    """Load a configured raw CSV table, normalize column names, and apply configured typing."""
    path = resolve_table_path(table_name)
    usecols = RAW_USECOLS_MAP.get(table_name)

    df = pd.read_csv(
        path,
        usecols=usecols,
        encoding="utf-8-sig",
        low_memory=False,
    )

    # Для stats__module_* сохраняем исходные русские названия столбцов
    if not table_name.startswith("stats__module_"):
        df = standardize_columns(df)

    df = _apply_table_typing(table_name, df)

    return df


def load_all_tables(table_names: Iterable[str] | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load all configured working tables and return a simple audit overview."""
    ensure_output_directories()
    selected_names = list(table_names or FILES.keys())
    dfs: dict[str, pd.DataFrame] = {}

    for table_name in selected_names:
        dfs[table_name] = read_raw_table(table_name)

    overview = build_table_overview(dfs)
    return dfs, overview


def infer_table_grain(df: pd.DataFrame) -> str:
    """Heuristically infer the likely grain of a table."""
    cols = set(df.columns)

    if "users_course_id" in cols:
        return "users_course_id-level / user-course"
    if {"user_id", "course_id"}.issubset(cols):
        return "user_id + course_id-level"
    if "user_id" in cols:
        return "user_id-level or user-event-level"
    if "course_id" in cols:
        return "course_id-level or course-structure-level"
    if "lesson_id" in cols:
        return "lesson_id-level"
    if "task_id" in cols:
        return "task_id-level or task-event-level"
    if "training_id" in cols:
        return "training_id-level or training-event-level"
    if "id" in cols:
        return "entity table with primary id"
    return "grain not inferred"


def infer_table_role(table_name: str) -> str:
    """Return a human-readable role of the table in the pipeline."""
    roles = {
        "users_courses": "base enrollment / user-course registry",
        "users": "user profile / demographics",
        "lessons": "course structure / lesson catalog",
        "lesson_tasks": "lesson-task linkage",
        "trainings": "training catalog",
        "user_lessons": "user progress by lesson",
        "user_trainings": "user progress by training",
        "user_answers": "answer event log",
        "wk_users_courses_actions": "course activity event log",
        "wk_media_view_sessions": "media consumption event log",
        "user_access_histories": "access history / administrative timeline",
        "user_award_badges": "user achievements",
        "award_badges": "badge dictionary",
        "groups": "webinars / live groups",
        "homeworks": "homework containers",
        "homework_items": "homework-item linkage",
        "stats__module_1": "module summary / progress snapshot",
        "stats__module_2": "module summary / progress snapshot",
        "stats__module_3": "module summary / progress snapshot",
        "stats__module_4": "module summary / progress snapshot",
    }
    return roles.get(table_name, "role not specified")


def build_table_overview(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create an EDA-friendly overview of loaded tables."""
    rows = []

    for table_name, df in dfs.items():
        key_like_cols = [c for c in df.columns if c.endswith("_id") or c == "id"]
        date_cols = [c for c in df.columns if c in DATE_COLS.get(table_name, [])]
        missing_share = df.isna().mean()

        rows.append(
            {
                "table_name": table_name,
                "role": infer_table_role(table_name),
                "grain_guess": infer_table_grain(df),
                "rows": len(df),
                "cols": df.shape[1],
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                "duplicate_full_rows": int(df.duplicated().sum()),
                "all_null_columns": int(df.isna().all().sum()),
                "mean_missing_share": round(float(missing_share.mean()), 4),
                "max_missing_share": round(float(missing_share.max()), 4),
                "date_columns": ", ".join(date_cols),
                "key_like_columns": ", ".join(key_like_cols[:10]),
                "all_columns": ", ".join(df.columns),
            }
        )

    overview = pd.DataFrame(rows).sort_values(
        ["rows", "cols"], ascending=[False, False]
    ).reset_index(drop=True)

    return overview

def describe_loaded_table(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a detailed per-column overview for one loaded table."""
    summary = pd.DataFrame({
        "column_name": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_share": df.isna().mean().round(4).values,
        "nunique": df.nunique(dropna=True).values,
    })

    summary["is_key_like"] = summary["column_name"].apply(
        lambda c: c.endswith("_id") or c == "id"
    )
    summary["is_datetime_expected"] = summary["column_name"].isin(DATE_COLS.get(table_name, []))

    return summary.sort_values(
        ["is_key_like", "missing_share", "column_name"],
        ascending=[False, False, True]
    ).reset_index(drop=True)


def normalize_id_columns(dfs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Normalize configured ID-like columns across all tables."""
    summary_rows = []
    cleaned_dfs: dict[str, pd.DataFrame] = {}

    for table_name, df in dfs.items():
        table_df = df.copy()
        for column in ID_COLS_MAP.get(table_name, []):
            if column in table_df.columns:
                before_missing = table_df[column].isna().sum()
                table_df[column] = _normalize_int_like_series(table_df[column])
                after_missing = table_df[column].isna().sum()
                summary_rows.append(
                    {
                        "table_name": table_name,
                        "column_name": column,
                        "missing_before": int(before_missing),
                        "missing_after": int(after_missing),
                        "dtype_after": str(table_df[column].dtype),
                    }
                )
        cleaned_dfs[table_name] = table_df

    summary_df = pd.DataFrame(summary_rows)
    return cleaned_dfs, summary_df


def drop_service_columns(dfs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Drop explicitly configured service columns after inspection."""
    summary_rows = []
    cleaned_dfs: dict[str, pd.DataFrame] = {}

    for table_name, df in dfs.items():
        drop_cols = [column for column in DROP_COLS_MAP.get(table_name, []) if column in df.columns]
        cleaned_dfs[table_name] = df.drop(columns=drop_cols).copy()
        summary_rows.append(
            {
                "table_name": table_name,
                "dropped_columns": ", ".join(drop_cols),
                "dropped_count": len(drop_cols),
            }
        )

    return cleaned_dfs, pd.DataFrame(summary_rows)


def infer_reference_timestamp(dfs: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Infer a stable reference timestamp from the latest observed dates."""
    max_values: list[pd.Timestamp] = []

    for table_name, date_cols in DATE_COLS.items():
        if table_name not in dfs:
            continue
        for column in date_cols:
            if column in dfs[table_name].columns:
                col_max = dfs[table_name][column].max()
                if pd.notna(col_max):
                    max_values.append(pd.Timestamp(col_max))

    if not max_values:
        return pd.Timestamp.utcnow().tz_localize(None)

    return max(max_values).tz_localize(None) if getattr(max(max_values), "tzinfo", None) else max(max_values)


# =============================================================================
# Validation helpers
# =============================================================================


def validate_key_uniqueness(df: pd.DataFrame, key_cols: list[str]) -> dict[str, Any]:
    """Return a compact uniqueness diagnostic for a candidate key."""
    non_null = df.dropna(subset=key_cols)
    duplicate_rows = int(non_null.duplicated(key_cols).sum())
    unique_keys = int(non_null[key_cols].drop_duplicates().shape[0])
    return {
        "key_cols": key_cols,
        "rows": int(len(df)),
        "non_null_rows": int(len(non_null)),
        "unique_key_rows": unique_keys,
        "duplicate_rows": duplicate_rows,
        "is_unique": duplicate_rows == 0,
    }


def build_key_diagnostics(df: pd.DataFrame, candidate_keys: list[list[str]]) -> pd.DataFrame:
    """Evaluate several candidate keys for the same dataframe."""
    diagnostics = [validate_key_uniqueness(df, key_cols) for key_cols in candidate_keys]
    return pd.DataFrame(diagnostics)

def build_direct_link_diagnostics(
    dfs: dict[str, pd.DataFrame],
    table_specs: list[dict],
) -> pd.DataFrame:
    """Build diagnostics for tables that already contain users_course_id."""
    rows = []

    for spec in table_specs:
        table_name = spec["table_name"]
        user_col = spec.get("user_col")
        df = dfs[table_name]

        row = {
            "table_name": table_name,
            "rows": len(df),
            "distinct_users_course_id": df["users_course_id"].nunique(dropna=True),
            "missing_users_course_id": int(df["users_course_id"].isna().sum()),
            "missing_users_course_id_share": round(df["users_course_id"].isna().mean(), 6),
        }

        if user_col and user_col in df.columns:
            row["users_per_users_course_gt_1"] = int(
                df.dropna(subset=["users_course_id", user_col])
                  .groupby("users_course_id")[user_col]
                  .nunique()
                  .gt(1)
                  .sum()
            )
        else:
            row["users_per_users_course_gt_1"] = pd.NA

        rows.append(row)

    return pd.DataFrame(rows)


def build_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a column-level missingness table sorted by missing share."""
    summary = pd.DataFrame(
        {
            "column_name": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_share": (df.isna().mean().round(6)).values,
            "dtype": df.dtypes.astype(str).values,
        }
    )
    return summary.sort_values(["missing_share", "column_name"], ascending=[False, True]).reset_index(drop=True)


def validate_left_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str | list[str],
    right_name: str,
) -> dict[str, Any]:
    """Compute metric-based diagnostics for a left join without mutating inputs."""
    join_cols = [on] if isinstance(on, str) else list(on)
    marker_col = f"__{right_name}_matched"

    right_augmented = right_df.copy()
    right_augmented[marker_col] = 1

    merged = left_df.merge(
        right_augmented,
        on=join_cols,
        how="left",
        validate="m:1",
        indicator=True,
    )

    matched_rows = int((merged["_merge"] == "both").sum())
    unmatched_rows = int((merged["_merge"] == "left_only").sum())
    added_cols = [
        column
        for column in right_df.columns
        if column not in join_cols and column not in left_df.columns
    ]
    added_missing_share = float(merged[added_cols].isna().mean().mean()) if added_cols else 0.0

    diagnostics = {
        "right_name": right_name,
        "join_key": ", ".join(join_cols),
        "left_rows_before": int(len(left_df)),
        "left_rows_after": int(len(merged)),
        "right_rows": int(len(right_df)),
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "coverage_ratio": round(matched_rows / len(left_df), 6) if len(left_df) else 0.0,
        "left_key_unique_before": validate_key_uniqueness(left_df, [CORE_ENTITY_KEY]).get("is_unique", False)
        if CORE_ENTITY_KEY in left_df.columns
        else None,
        "left_key_unique_after": validate_key_uniqueness(merged, [CORE_ENTITY_KEY]).get("is_unique", False)
        if CORE_ENTITY_KEY in merged.columns
        else None,
        "right_key_unique": validate_key_uniqueness(right_df, join_cols).get("is_unique", False),
        "row_multiplier_delta": int(len(merged) - len(left_df)),
        "added_columns": ", ".join(added_cols),
        "added_columns_mean_missing_share": round(added_missing_share, 6),
    }
    return diagnostics


def build_route_coverage(
    source_df: pd.DataFrame,
    matched_mask: pd.Series,
    route_name: str,
) -> dict[str, Any]:
    """Return compact route-resolution diagnostics for an intermediate merge path."""
    rows_total = int(len(source_df))
    rows_matched = int(matched_mask.fillna(False).sum())
    rows_unmatched = rows_total - rows_matched

    return {
        "route_name": route_name,
        "rows_total": rows_total,
        "rows_matched": rows_matched,
        "rows_unmatched": rows_unmatched,
        "coverage_ratio": round(rows_matched / rows_total, 6) if rows_total else 0.0,
    }

# =============================================================================
# Export helpers
# =============================================================================


def save_dataframe(df: pd.DataFrame, file_name: str, folder: Path = TABLES_DIR, index: bool = False) -> Path:
    """Save a dataframe as UTF-8 CSV inside the AGENT workspace."""
    ensure_output_directories()
    path = folder / f"{file_name}.csv"
    df.to_csv(path, index=index, encoding="utf-8", date_format=EXPORT_DATE_FORMAT)
    return path


def save_summary(summary: Any, file_name: str, folder: Path = SUMMARIES_DIR) -> Path:
    """Save a summary object as CSV or JSON depending on its type."""
    ensure_output_directories()

    if isinstance(summary, pd.DataFrame):
        path = folder / f"{file_name}.csv"
        summary.to_csv(path, index=False, encoding="utf-8", date_format=EXPORT_DATE_FORMAT)
        return path

    path = folder / f"{file_name}.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
