"""Validation helpers for keys, coverage, and bridge quality."""

from __future__ import annotations

import pandas as pd


def validate_unique_key(df: pd.DataFrame, key_column: str, block_name: str) -> pd.DataFrame:
    """Validate that a block remains unique on its intended key."""

    return pd.DataFrame(
        [
            {
                "block_name": block_name,
                "key_column": key_column,
                "rows": int(len(df)),
                "unique_keys": int(df[key_column].nunique(dropna=True)),
                "duplicate_key_rows": int(df.duplicated([key_column]).sum()),
                "missing_key_rows": int(df[key_column].isna().sum()),
            }
        ]
    )


def validate_block_coverage(
    base_df: pd.DataFrame,
    block_df: pd.DataFrame,
    key_column: str,
    block_name: str,
) -> pd.DataFrame:
    """Validate coverage of a block against the user-course base table."""

    matched = base_df[key_column].isin(set(block_df[key_column].dropna()))
    return pd.DataFrame(
        [
            {
                "block_name": block_name,
                "rows_in_base": int(len(base_df)),
                "rows_in_block": int(len(block_df)),
                "matched_rows": int(matched.sum()),
                "unmatched_rows": int((~matched).sum()),
                "coverage_share": round(float(matched.mean()), 4),
            }
        ]
    )
