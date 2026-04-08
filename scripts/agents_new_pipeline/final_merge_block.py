"""Final merge block for AGENTS_NEW."""

from __future__ import annotations

from .access_history_block import build_access_history_block
from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .course_actions_block import build_course_actions_block
from .course_block import build_course_block
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .media_sessions_block import build_media_sessions_block
from .plot_utils import plot_histogram
from .summary_utils import attach_summary_and_features, display_block_result
from .user_answers_block import build_user_answers_block
from .user_block import build_user_block
from .user_lessons_block import build_user_lessons_block
from .user_trainings_block import build_user_trainings_block
from .validation_utils import validate_unique_key


def build_final_feature_table_block(
    base_result=None,
    course_result=None,
    user_result=None,
    user_lessons_result=None,
    user_trainings_result=None,
    user_answers_result=None,
    course_actions_result=None,
    media_result=None,
    access_result=None,
    show_output: bool = True,
):
    """Build or load the final user-course feature table."""

    ensure_output_dirs()
    table_path = OUT_DIR / "final_user_course_features_AGENT.csv"
    merge_validation_path = OUT_DIR / "merge_validation_AGENT.csv"
    missingness_path = OUT_DIR / "final_feature_missingness_AGENT.csv"
    block_summary_path = OUT_DIR / "feature_block_summary_AGENT.csv"

    from_cache = all(
        artifact_exists(path)
        for path in [table_path, merge_validation_path, missingness_path, block_summary_path]
    )
    if from_cache:
        final_df = read_csv(table_path)
        merge_validation = read_csv(merge_validation_path)
        missingness = read_csv(missingness_path)
        block_summary = read_csv(block_summary_path)
    else:
        if base_result is None:
            base_result = build_base_entity_block(show_output=False)
        if course_result is None:
            course_result = build_course_block(base_result=base_result, show_output=False)
        if user_result is None:
            user_result = build_user_block(show_output=False)
        if user_lessons_result is None:
            user_lessons_result = build_user_lessons_block(base_result=base_result, show_output=False)
        if user_trainings_result is None:
            user_trainings_result = build_user_trainings_block(base_result=base_result, show_output=False)
        if user_answers_result is None:
            user_answers_result = build_user_answers_block(base_result=base_result, show_output=False)
        if course_actions_result is None:
            course_actions_result = build_course_actions_block(base_result=base_result, show_output=False)
        if media_result is None:
            media_result = build_media_sessions_block(base_result=base_result, show_output=False)
        if access_result is None:
            access_result = build_access_history_block(base_result=base_result, show_output=False)

        with legacy_output_context() as legacy:
            final_df, merge_validation, missingness, block_summary = legacy.assemble_final_feature_table(
                base=base_result.data,
                course_features=course_result.data,
                user_features=user_result.data,
                user_lessons_agg=user_lessons_result.data,
                user_trainings_agg=user_trainings_result.data,
                user_answers_agg=user_answers_result.data,
                course_actions_agg=course_actions_result.data,
                media_sessions_agg=media_result.data,
                access_history_agg=access_result.data,
            )

    key_validation = validate_unique_key(final_df, PRIMARY_KEYS["final_features"], "final_features")
    plots = [
        plot_histogram(
            final_df["days_from_course_start_to_first_observed_activity"],
            title="Delay to first observed activity",
            xlabel="days_from_course_start_to_first_observed_activity",
            filename="final_first_activity_delay_AGENTS_NEW.png",
        ),
        plot_histogram(
            final_df["days_from_last_observed_activity_to_access_end"],
            title="Gap from last activity to access end",
            xlabel="days_from_last_observed_activity_to_access_end",
            filename="final_last_activity_gap_AGENTS_NEW.png",
        ),
    ]
    result = attach_summary_and_features(
        name="Final feature table block",
        data=final_df,
        key_column=PRIMARY_KEYS["final_features"],
        from_cache=from_cache,
        validation=key_validation,
        extra_tables={
            "Merge validation": merge_validation,
            "Top feature missingness": missingness.head(12),
            "Feature block summary": block_summary,
        },
        artifact_paths={
            "table": table_path,
            "merge_validation": merge_validation_path,
            "missingness": missingness_path,
            "block_summary": block_summary_path,
        },
        plots=plots,
        notes="Final table keeps one row per users_course_id and stays target-free for future attachment.",
    )
    if show_output:
        display_block_result(result)
    return result
