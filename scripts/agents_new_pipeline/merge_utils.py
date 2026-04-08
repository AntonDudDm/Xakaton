"""Merge helpers for explicit block-by-block assembly."""

from __future__ import annotations

import pandas as pd

import scripts.build_user_course_features_AGENT as legacy_pipeline


def merge_feature_block(
    base_df: pd.DataFrame,
    block_df: pd.DataFrame,
    block_name: str,
    join_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-merge a block into the base table and return merge diagnostics."""

    new_columns = [column for column in block_df.columns if column != join_key]
    merged_df = base_df.merge(block_df, on=join_key, how="left")
    validation = pd.DataFrame(
        [
            legacy_pipeline.build_merge_validation(
                base=base_df,
                merged=merged_df,
                block_name=block_name,
                new_columns=new_columns,
            )
        ]
    )
    return merged_df, validation
