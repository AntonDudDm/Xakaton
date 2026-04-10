from __future__ import annotations

from typing import Any

import pandas as pd

from .config_AGENT import CORE_ENTITY_KEY
from .service_AGENT import validate_left_merge


# =============================================================================
# Merge diagnostics
# =============================================================================


def merge_feature_block(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: str | list[str],
    block_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform a validated left merge and return diagnostics."""
    diagnostics = validate_left_merge(left_df=left_df, right_df=right_df, on=on, right_name=block_name)
    merged = left_df.merge(right_df, on=on, how="left", validate="m:1")
    return merged, diagnostics


# =============================================================================
# Master table assembly
# =============================================================================


def assemble_master_user_course_table(
    users_courses_base: pd.DataFrame,
    user_features: pd.DataFrame,
    course_features: pd.DataFrame,
    user_lessons_features: pd.DataFrame,
    user_training_features: pd.DataFrame,
    user_answer_features: pd.DataFrame,
    course_action_features: pd.DataFrame,
    media_features: pd.DataFrame,
    access_history_features: pd.DataFrame,
    stats_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the final master table with metric-based merge diagnostics."""
    merged = users_courses_base.copy()
    merge_rows: list[dict[str, Any]] = []

    merge_steps = [
        ("user_features", user_features, "user_id"),
        ("course_features", course_features, "course_id"),
        ("user_lessons_features", user_lessons_features, CORE_ENTITY_KEY),
        ("user_training_features", user_training_features, CORE_ENTITY_KEY),
        ("user_answer_features", user_answer_features, CORE_ENTITY_KEY),
        ("course_action_features", course_action_features, CORE_ENTITY_KEY),
        ("media_features", media_features, CORE_ENTITY_KEY),
        ("access_history_features", access_history_features, CORE_ENTITY_KEY),
    ]

    if stats_features is not None:
        merge_steps.append(("stats_module_features", stats_features, CORE_ENTITY_KEY))

    for block_name, block_df, join_key in merge_steps:
        merged, diagnostics = merge_feature_block(
            left_df=merged,
            right_df=block_df,
            on=join_key,
            block_name=block_name,
        )
        merge_rows.append(diagnostics)

    merge_report_df = pd.DataFrame(merge_rows)
    return merged, merge_report_df
