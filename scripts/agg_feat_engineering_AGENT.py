from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config_AGENT import CORE_ENTITY_KEY
from .service_AGENT import validate_key_uniqueness


# =============================================================================
# Shared helpers
# =============================================================================


def _bool_to_int(series: pd.Series) -> pd.Series:
    """Convert nullable booleans into stable integer flags."""
    return series.fillna(False).astype("int8")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a safe numeric ratio with missing values when the denominator is 0."""
    denominator = denominator.replace({0: np.nan})
    return numerator.astype("Float64") / denominator.astype("Float64")


def _timedelta_in_days(series: pd.Series) -> pd.Series:
    """Convert a timedelta series into float days."""
    return series.dt.total_seconds() / 86400.0


def _row_summary(
    block_name: str,
    source_tables: list[str],
    df: pd.DataFrame,
    key_cols: list[str],
    feature_list: list[str],
    coverage_notes: str,
    important_warnings: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized summary for a feature block."""
    uniqueness = validate_key_uniqueness(df, key_cols)
    summary = {
        "block_name": block_name,
        "source_tables": source_tables,
        "target_level": ", ".join(key_cols),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "key_cols": key_cols,
        "is_key_unique": uniqueness["is_unique"],
        "feature_count": int(len(feature_list)),
        "new_feature_count": int(len(feature_list)),
        "coverage_notes": coverage_notes,
        "important_warnings": important_warnings,
    }
    if extra:
        summary.update(extra)
    return summary


# =============================================================================
# Core entity block
# =============================================================================


def build_users_courses_base(
    users_courses: pd.DataFrame,
    users: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build the main user-course entity table at users_course_id grain."""
    pupil_ids = set(users.loc[users["type"] == "User::Pupil", "id"].dropna().tolist())

    base = users_courses.copy()
    base = base.rename(columns={"id": CORE_ENTITY_KEY})
    base = base.loc[base["user_id"].isin(pupil_ids)].copy()

    base["uc_is_active"] = (base["state"] == "active").astype("int8")
    base["uc_is_inactive"] = (base["state"] == "inactive").astype("int8")
    base["uc_has_official_start"] = base["wk_officially_started_at"].notna().astype("int8")
    base["uc_has_completion_record"] = base["wk_course_completed_at"].notna().astype("int8")
    base["uc_has_access_end"] = base["access_finished_at"].notna().astype("int8")
    base["uc_points_ratio"] = _safe_ratio(base["wk_points"], base["wk_max_points"])
    base["uc_points_positive_flag"] = base["wk_points"].fillna(0).gt(0).astype("int8")
    base["uc_full_points_flag"] = (
        base["wk_points"].notna()
        & base["wk_max_points"].notna()
        & base["wk_points"].eq(base["wk_max_points"])
    ).astype("int8")
    base["uc_tasks_per_viewable_lesson"] = _safe_ratio(
        base["wk_max_task_count"],
        base["wk_max_viewable_lessons"],
    )
    base["uc_enrollment_age_days"] = _timedelta_in_days(reference_timestamp - base["created_at"])
    base["uc_access_window_days"] = _timedelta_in_days(base["access_finished_at"] - base["created_at"])
    base["uc_official_start_delay_days"] = _timedelta_in_days(
        base["wk_officially_started_at"] - base["created_at"]
    )
    base["uc_completion_delay_days"] = _timedelta_in_days(
        base["wk_course_completed_at"] - base["created_at"]
    )

    feature_list = [
        "uc_is_active",
        "uc_is_inactive",
        "uc_has_official_start",
        "uc_has_completion_record",
        "uc_has_access_end",
        "uc_points_ratio",
        "uc_points_positive_flag",
        "uc_full_points_flag",
        "uc_tasks_per_viewable_lesson",
        "uc_enrollment_age_days",
        "uc_access_window_days",
        "uc_official_start_delay_days",
        "uc_completion_delay_days",
    ]

    summary = _row_summary(
        block_name="users_courses_base",
        source_tables=["users_courses", "users"],
        df=base,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes=(
            "Base table is filtered to User::Pupil accounts to keep the entity aligned "
            "with the student analytics use case."
        ),
        important_warnings=(
            "Administrative completion and state fields are kept for EDA, but they should "
            "be reviewed later against the final target horizon to avoid leakage."
        ),
        extra={
            "excluded_non_pupil_rows": int(len(users_courses) - len(base)),
            "distinct_users": int(base["user_id"].nunique(dropna=True)),
            "distinct_courses": int(base["course_id"].nunique(dropna=True)),
        },
    )
    return base, feature_list, summary


# =============================================================================
# User-level aggregation
# =============================================================================


def build_users_base_features(
    users: pd.DataFrame,
    user_award_badges: pd.DataFrame,
    award_badges: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build user-level metadata and profile stability features."""
    user_df = users.loc[users["type"] == "User::Pupil"].copy()
    user_df = user_df.rename(columns={"id": "user_id"})

    user_df["user_is_subscribed_flag"] = _bool_to_int(user_df["subscribed"])
    user_df["user_account_age_days"] = _timedelta_in_days(reference_timestamp - user_df["created_at"])
    user_df["user_profile_age_days"] = _timedelta_in_days(user_df["updated_at"] - user_df["created_at"])
    user_df["user_grade_change_delay_days"] = _timedelta_in_days(
        user_df["grade_changed_at"] - user_df["created_at"]
    )
    user_df["user_has_grade_flag"] = user_df["grade_id"].notna().astype("int8")
    user_df["user_has_timezone_flag"] = user_df["timezone"].notna().astype("int8")
    user_df["user_has_region_flag"] = user_df["d_wk_region_id"].notna().astype("int8")
    user_df["user_has_municipal_flag"] = user_df["d_wk_municipal_id"].notna().astype("int8")
    user_df["user_has_school_flag"] = user_df["d_wk_school_id"].notna().astype("int8")
    user_df["user_has_gender_flag"] = user_df["wk_gender"].notna().astype("int8")

    badge_agg = (
        user_award_badges.merge(
            award_badges.rename(columns={"id": "award_badge_id"}),
            on="award_badge_id",
            how="left",
        )
        .assign(
            award_badge_is_special=lambda frame: _bool_to_int(frame["special"]),
        )
        .groupby("user_id", as_index=False)
        .agg(
            user_badges_total_count=("award_badge_id", "count"),
            user_badges_unique_count=("award_badge_id", "nunique"),
            user_special_badges_count=("award_badge_is_special", "sum"),
            user_first_badge_at=("created_at", "min"),
            user_last_badge_at=("created_at", "max"),
        )
    )
    badge_agg["user_badge_span_days"] = _timedelta_in_days(
        badge_agg["user_last_badge_at"] - badge_agg["user_first_badge_at"]
    )

    result = user_df.merge(badge_agg, on="user_id", how="left", validate="1:1")

    feature_list = [
        "sign_in_count",
        "grade_id",
        "timezone",
        "wk_gender",
        "d_wk_region_id",
        "d_wk_municipal_id",
        "d_wk_school_id",
        "user_is_subscribed_flag",
        "user_account_age_days",
        "user_profile_age_days",
        "user_grade_change_delay_days",
        "user_has_grade_flag",
        "user_has_timezone_flag",
        "user_has_region_flag",
        "user_has_municipal_flag",
        "user_has_school_flag",
        "user_has_gender_flag",
        "user_badges_total_count",
        "user_badges_unique_count",
        "user_special_badges_count",
        "user_badge_span_days",
    ]

    keep_cols = ["user_id"] + feature_list + ["user_first_badge_at", "user_last_badge_at"]
    result = result[keep_cols]

    summary = _row_summary(
        block_name="users_base_features",
        source_tables=["users", "user_award_badges", "award_badges"],
        df=result,
        key_cols=["user_id"],
        feature_list=feature_list,
        coverage_notes="User-level features are aggregated to one row per user_id before joining to the user-course base table.",
        important_warnings="Badge-based signals are platform-wide and should be interpreted as enrichment rather than course-specific behavior.",
    )
    return result, feature_list, summary


# =============================================================================
# Course-level aggregation
# =============================================================================


def build_course_structure_features(
    lessons: pd.DataFrame,
    lesson_tasks: pd.DataFrame,
    trainings: pd.DataFrame,
    groups: pd.DataFrame,
    homeworks: pd.DataFrame,
    homework_items: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build stable course-level structure features."""
    lesson_agg = (
        lessons.groupby("course_id", as_index=False)
        .agg(
            course_lessons_count=("id", "count"),
            course_lessons_with_tasks_count=("task_expected", lambda s: int(s.fillna(False).sum())),
            course_lessons_with_conspect_count=("conspect_expected", lambda s: int(s.fillna(False).sum())),
            course_lessons_survival_count=(
                "wk_survival_training_expected",
                lambda s: int(s.fillna(False).sum()),
            ),
            course_lessons_scratch_count=(
                "wk_scratch_playground_enabled",
                lambda s: int(s.fillna(False).sum()),
            ),
            course_lessons_attendance_count=(
                "wk_attendance_tracking_enabled",
                lambda s: int(s.fillna(False).sum()),
            ),
            course_lesson_number_max=("lesson_number", "max"),
            course_lessons_max_points_sum=("wk_max_points", "sum"),
            course_lessons_task_count_sum=("wk_task_count", "sum"),
            course_video_duration_sum=("wk_video_duration", "sum"),
            course_video_duration_mean=("wk_video_duration", "mean"),
        )
    )
    lesson_agg["course_lessons_with_tasks_share"] = _safe_ratio(
        lesson_agg["course_lessons_with_tasks_count"],
        lesson_agg["course_lessons_count"],
    )
    lesson_agg["course_lessons_with_conspect_share"] = _safe_ratio(
        lesson_agg["course_lessons_with_conspect_count"],
        lesson_agg["course_lessons_count"],
    )
    lesson_agg["course_lessons_survival_share"] = _safe_ratio(
        lesson_agg["course_lessons_survival_count"],
        lesson_agg["course_lessons_count"],
    )
    lesson_agg["course_lessons_scratch_share"] = _safe_ratio(
        lesson_agg["course_lessons_scratch_count"],
        lesson_agg["course_lessons_count"],
    )
    lesson_agg["course_lessons_attendance_share"] = _safe_ratio(
        lesson_agg["course_lessons_attendance_count"],
        lesson_agg["course_lessons_count"],
    )

    task_enriched = lesson_tasks.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
    )
    task_agg = (
        task_enriched.groupby("course_id", as_index=False)
        .agg(
            course_task_link_count=("id_x", "count"),
            course_unique_task_count=("task_id", "nunique"),
            course_required_task_count=("task_required", lambda s: int(s.fillna(False).sum())),
        )
    )
    task_agg["course_required_task_share"] = _safe_ratio(
        task_agg["course_required_task_count"],
        task_agg["course_task_link_count"],
    )

    training_enriched = trainings.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
    )
    training_agg = (
        training_enriched.groupby("course_id", as_index=False)
        .agg(
            course_trainings_count=("id_x", "count"),
            course_training_task_templates_sum=("task_templates_count", "sum"),
            course_training_difficulty_mean=("difficulty", "mean"),
        )
    )

    group_enriched = groups.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
    )
    group_agg = (
        group_enriched.groupby("course_id", as_index=False)
        .agg(
            course_groups_count=("id_x", "count"),
            course_groups_with_video_count=("video_available", lambda s: int(s.fillna(False).sum())),
            course_groups_duration_sum=("duration", "sum"),
            course_groups_actual_duration_sum=("wk_duration_actual", "sum"),
        )
    )
    group_agg["course_groups_with_video_share"] = _safe_ratio(
        group_agg["course_groups_with_video_count"],
        group_agg["course_groups_count"],
    )

    course_homeworks = homeworks.loc[homeworks["resource_type"] == "Lesson"].copy()
    course_homeworks = course_homeworks.merge(
        lessons[["id", "course_id"]],
        left_on="resource_id",
        right_on="id",
        how="left",
        validate="m:1",
    )
    homework_agg = (
        course_homeworks.groupby("course_id", as_index=False)
        .agg(
            course_homeworks_count=("id_x", "count"),
            course_unique_homework_types=("homework_type", "nunique"),
        )
    )

    homework_items_enriched = homework_items.merge(
        course_homeworks[["id_x", "course_id"]].rename(columns={"id_x": "homework_id"}),
        on="homework_id",
        how="left",
        validate="m:1",
    )
    homework_items_agg = (
        homework_items_enriched.groupby("course_id", as_index=False)
        .agg(
            course_homework_item_count=("id", "count"),
            course_homework_task_item_count=(
                "resource_type",
                lambda s: int((s == "Task").sum()),
            ),
        )
    )

    result = lesson_agg.copy()
    for block in [task_agg, training_agg, group_agg, homework_agg, homework_items_agg]:
        result = result.merge(block, on="course_id", how="left", validate="1:1")

    feature_list = [column for column in result.columns if column != "course_id"]

    summary = _row_summary(
        block_name="course_structure_features",
        source_tables=["lessons", "lesson_tasks", "trainings", "groups", "homeworks", "homework_items"],
        df=result,
        key_cols=["course_id"],
        feature_list=feature_list,
        coverage_notes="Course structure is aggregated before any merge to the user-course base. Lessons are used as the stable course skeleton.",
        important_warnings="Homework-derived structure only covers homework rows that resolve to lesson resources.",
    )
    return result, feature_list, summary


# =============================================================================
# User-lesson block
# =============================================================================


def build_user_lesson_features(
    user_lessons: pd.DataFrame,
    lessons: pd.DataFrame,
    course_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Aggregate lesson progress signals to the user-course level."""
    enriched = user_lessons.merge(
        lessons[["id", "course_id", "lesson_number"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
    )
    enriched["ul_solved_tasks_resolved"] = (
        enriched["wk_solved_task_count"]
        .fillna(enriched["solved_tasks_count"])
        .fillna(0)
        .astype("Float64")
    )
    enriched["ul_task_count_gap"] = (
        enriched["solved_tasks_count"].fillna(0) - enriched["wk_solved_task_count"].fillna(0)
    )

    result = (
        enriched.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(
            ul_lesson_rows=("lesson_id", "count"),
            ul_lessons_touched_count=("lesson_id", "nunique"),
            ul_lessons_solved_count=("solved", lambda s: int(s.fillna(False).sum())),
            ul_video_visited_count=("video_visited", lambda s: int(s.fillna(False).sum())),
            ul_video_viewed_count=("video_viewed", lambda s: int(s.fillna(False).sum())),
            ul_translation_visited_count=("translation_visited", lambda s: int(s.fillna(False).sum())),
            ul_points_sum=("wk_points", "sum"),
            ul_points_mean=("wk_points", "mean"),
            ul_solved_tasks_sum=("ul_solved_tasks_resolved", "sum"),
            ul_furthest_lesson_number=("lesson_number", "max"),
            ul_task_count_gap_sum=("ul_task_count_gap", "sum"),
        )
    )

    ratios = course_features[
        [
            "course_id",
            "course_lessons_count",
            "course_lesson_number_max",
            "course_lessons_max_points_sum",
            "course_lessons_task_count_sum",
        ]
    ]
    entity_map = (
        enriched[[CORE_ENTITY_KEY, "course_id"]]
        .dropna(subset=[CORE_ENTITY_KEY, "course_id"])
        .drop_duplicates()
    )
    result = result.merge(entity_map, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result = result.merge(ratios, on="course_id", how="left", validate="m:1")

    result["ul_lessons_touched_ratio"] = _safe_ratio(
        result["ul_lessons_touched_count"],
        result["course_lessons_count"],
    )
    result["ul_furthest_lesson_ratio"] = _safe_ratio(
        result["ul_furthest_lesson_number"],
        result["course_lesson_number_max"],
    )
    result["ul_points_ratio_vs_course"] = _safe_ratio(
        result["ul_points_sum"],
        result["course_lessons_max_points_sum"],
    )
    result["ul_solved_tasks_ratio_vs_course"] = _safe_ratio(
        result["ul_solved_tasks_sum"],
        result["course_lessons_task_count_sum"],
    )

    feature_list = [column for column in result.columns if column not in [CORE_ENTITY_KEY, "course_id"]]
    summary = _row_summary(
        block_name="user_lesson_features",
        source_tables=["user_lessons", "lessons", "course_structure_features"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes="Lesson progress is already keyed by users_course_id, so the aggregation is direct and low risk.",
        important_warnings="The solved task count keeps the resolved wk_solved_task_count fallback but also retains the aggregate gap signal for auditability.",
    )
    return result.drop(columns=["course_id"]), feature_list, summary


# =============================================================================
# User-training block
# =============================================================================


def build_user_training_features(
    user_trainings: pd.DataFrame,
    trainings: pd.DataFrame,
    lessons: pd.DataFrame,
    users_courses_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Resolve and aggregate training activity to the user-course level."""
    training_map = trainings.merge(
        lessons[["id", "course_id", "lesson_number"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
    )

    resolved = user_trainings.merge(
        training_map.rename(columns={"id_x": "training_id"}),
        on="training_id",
        how="left",
        validate="m:1",
    )
    resolved = resolved.merge(
        users_courses_base[[CORE_ENTITY_KEY, "user_id", "course_id", "created_at"]],
        on=["user_id", "course_id"],
        how="left",
        validate="m:1",
    )

    resolved["training_first_timestamp"] = resolved[["started_at", "finished_at", "mark_saved_at"]].min(axis=1)
    resolved["training_last_timestamp"] = resolved[["started_at", "finished_at", "mark_saved_at"]].max(axis=1)
    resolved["training_checked_flag"] = (resolved["state"] == "checked").astype("int8")
    resolved["training_started_flag"] = resolved["started_at"].notna().astype("int8")
    resolved["training_finished_flag"] = resolved["finished_at"].notna().astype("int8")
    resolved["training_high_mark_flag"] = resolved["mark"].fillna(-1).ge(4).astype("int8")

    matched = resolved.loc[resolved[CORE_ENTITY_KEY].notna()].copy()
    result = (
        matched.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(
            training_records_count=("training_id", "count"),
            training_unique_count=("training_id", "nunique"),
            training_unique_lessons_count=("lesson_id", "nunique"),
            training_started_count=("training_started_flag", "sum"),
            training_checked_count=("training_checked_flag", "sum"),
            training_finished_count=("training_finished_flag", "sum"),
            training_attempts_sum=("attempts", "sum"),
            training_attempts_mean=("attempts", "mean"),
            training_attempts_max=("attempts", "max"),
            training_submitted_answers_sum=("submitted_answers_count", "sum"),
            training_solved_tasks_sum=("solved_tasks_count", "sum"),
            training_earned_points_sum=("earned_points", "sum"),
            training_mark_mean=("mark", "mean"),
            training_mark_max=("mark", "max"),
            training_high_mark_count=("training_high_mark_flag", "sum"),
            training_first_activity_at=("training_first_timestamp", "min"),
            training_last_activity_at=("training_last_timestamp", "max"),
            training_max_lesson_number=("lesson_number", "max"),
        )
    )
    result["training_activity_span_days"] = _timedelta_in_days(
        result["training_last_activity_at"] - result["training_first_activity_at"]
    )

    feature_list = [column for column in result.columns if column != CORE_ENTITY_KEY]
    summary = _row_summary(
        block_name="user_training_features",
        source_tables=["user_trainings", "trainings", "lessons", "users_courses_base"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes=(
            f"Training rows are resolved through training -> lesson -> course -> user-course. "
            f"Matched coverage: {matched.shape[0] / max(len(resolved), 1):.4%}."
        ),
        important_warnings="A small unmatched tail remains because some training rows do not resolve to lessons or enrolled user-course pairs.",
        extra={
            "route_rows_total": int(len(resolved)),
            "route_rows_matched": int(len(matched)),
            "route_rows_unmatched": int(len(resolved) - len(matched)),
        },
    )
    return result, feature_list, summary


# =============================================================================
# User-answer block
# =============================================================================


def build_user_answer_features(
    user_answers: pd.DataFrame,
    lessons: pd.DataFrame,
    trainings: pd.DataFrame,
    homeworks: pd.DataFrame,
    users_courses_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Resolve answer events to user-course and aggregate answer behavior."""
    lesson_map = lessons[["id", "course_id"]].rename(columns={"id": "resource_id"})

    training_map = trainings.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
    )[["id_x", "course_id"]].rename(columns={"id_x": "resource_id"})

    homework_map = (
        homeworks.loc[homeworks["resource_type"] == "Lesson", ["id", "resource_id"]]
        .rename(columns={"id": "homework_id", "resource_id": "lesson_id"})
        .merge(
            lessons[["id", "course_id"]],
            left_on="lesson_id",
            right_on="id",
            how="left",
            validate="m:1",
        )[["homework_id", "course_id"]]
        .rename(columns={"homework_id": "resource_id"})
    )

    lesson_answers = user_answers.loc[user_answers["resource_type"] == "Lesson"].merge(
        lesson_map,
        on="resource_id",
        how="left",
        validate="m:1",
    )
    training_answers = user_answers.loc[user_answers["resource_type"] == "Training"].merge(
        training_map,
        on="resource_id",
        how="left",
        validate="m:1",
    )
    homework_answers = user_answers.loc[user_answers["resource_type"] == "Homework"].merge(
        homework_map,
        on="resource_id",
        how="left",
        validate="m:1",
    )

    resolved = pd.concat([lesson_answers, training_answers, homework_answers], ignore_index=True)
    resolved = resolved.merge(
        users_courses_base[[CORE_ENTITY_KEY, "user_id", "course_id", "created_at"]],
        on=["user_id", "course_id"],
        how="left",
        validate="m:1",
    )

    resolved["answer_solved_flag"] = _bool_to_int(resolved["solved"])
    resolved["answer_skipped_flag"] = _bool_to_int(resolved["skipped"])
    resolved["answer_partial_flag"] = _bool_to_int(resolved["wk_partial_answer"])
    resolved["answer_unsolved_flag"] = (
        resolved["answer_solved_flag"].eq(0) & resolved["answer_skipped_flag"].eq(0)
    ).astype("int8")
    resolved["answer_async_pending_flag"] = (
        resolved["async_check_status"].astype("string").str.lower() == "pending"
    ).astype("int8")
    resolved["answer_async_failed_flag"] = (
        resolved["async_check_status"].astype("string").str.lower() == "failed"
    ).astype("int8")
    resolved["answer_day"] = resolved["submitted_at"].dt.normalize()

    matched = resolved.loc[resolved[CORE_ENTITY_KEY].notna()].copy()
    matched["answer_resource_key"] = (
        matched["resource_type"].astype("string") + "_" + matched["resource_id"].astype("string")
    )

    result = (
        matched.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(
            answer_total_count=("task_id", "count"),
            answer_task_unique_count=("task_id", "nunique"),
            answer_resource_unique_count=("answer_resource_key", "nunique"),
            answer_solved_count=("answer_solved_flag", "sum"),
            answer_unsolved_count=("answer_unsolved_flag", "sum"),
            answer_skipped_count=("answer_skipped_flag", "sum"),
            answer_partial_count=("answer_partial_flag", "sum"),
            answer_attempts_sum=("attempts", "sum"),
            answer_attempts_mean=("attempts", "mean"),
            answer_attempts_max=("attempts", "max"),
            answer_points_sum=("points", "sum"),
            answer_points_mean=("points", "mean"),
            answer_points_max=("points", "max"),
            answer_first_at=("submitted_at", "min"),
            answer_last_at=("submitted_at", "max"),
            answer_active_days=("answer_day", "nunique"),
            answer_async_pending_count=("answer_async_pending_flag", "sum"),
            answer_async_failed_count=("answer_async_failed_flag", "sum"),
        )
    )

    type_counts = pd.crosstab(matched[CORE_ENTITY_KEY], matched["resource_type"])
    type_counts = type_counts.rename(
        columns={
            "Lesson": "answer_lesson_count",
            "Training": "answer_training_count",
            "Homework": "answer_homework_count",
        }
    ).reset_index()
    result = result.merge(type_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result["answer_activity_span_days"] = _timedelta_in_days(
        result["answer_last_at"] - result["answer_first_at"]
    )

    feature_list = [column for column in result.columns if column != CORE_ENTITY_KEY]
    summary = _row_summary(
        block_name="user_answer_features",
        source_tables=["user_answers", "lessons", "trainings", "homeworks", "users_courses_base"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes=(
            f"Answer rows are resolved by resource type before mapping to user-course. "
            f"Matched coverage: {matched.shape[0] / max(len(resolved), 1):.4%}."
        ),
        important_warnings=(
            "Answer-to-course linkage intentionally avoids task_id as a primary bridge because task_id is not globally unique across courses."
        ),
        extra={
            "route_rows_total": int(len(resolved)),
            "route_rows_matched": int(len(matched)),
            "route_rows_unmatched": int(len(resolved) - len(matched)),
        },
    )
    return result, feature_list, summary


# =============================================================================
# Course action block
# =============================================================================


def build_course_action_features(
    wk_users_courses_actions: pd.DataFrame,
    users_courses_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Aggregate course action logs to user-course level."""
    entity_dates = users_courses_base[[CORE_ENTITY_KEY, "created_at", "access_finished_at"]].rename(
        columns={"created_at": "course_created_at"}
    )
    action_df = wk_users_courses_actions.merge(
        entity_dates,
        on=CORE_ENTITY_KEY,
        how="left",
        validate="m:1",
    )
    action_df["action_day"] = action_df["created_at"].dt.normalize()
    action_df["days_from_enrollment"] = _timedelta_in_days(
        action_df["created_at"] - action_df["course_created_at"]
    )
    action_df["days_to_access_end"] = _timedelta_in_days(
        action_df["access_finished_at"] - action_df["created_at"]
    )
    action_df["action_first_14d_flag"] = action_df["days_from_enrollment"].between(
        0,
        14,
        inclusive="both",
    ).astype("int8")
    action_df["action_first_30d_flag"] = action_df["days_from_enrollment"].between(
        0,
        30,
        inclusive="both",
    ).astype("int8")
    action_df["action_last_14d_flag"] = action_df["days_to_access_end"].between(
        0,
        14,
        inclusive="both",
    ).astype("int8")

    result = (
        action_df.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(
            action_total_count=("action", "count"),
            action_unique_types_count=("action", "nunique"),
            action_unique_lessons_count=("lesson_id", "nunique"),
            action_active_days=("action_day", "nunique"),
            action_first_at=("created_at", "min"),
            action_last_at=("created_at", "max"),
            action_first_14d_count=("action_first_14d_flag", "sum"),
            action_first_30d_count=("action_first_30d_flag", "sum"),
            action_last_14d_count=("action_last_14d_flag", "sum"),
        )
    )

    action_counts = pd.crosstab(action_df[CORE_ENTITY_KEY], action_df["action"])
    action_counts = action_counts.add_prefix("action_type_count_").reset_index()
    result = result.merge(action_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result["action_span_days"] = _timedelta_in_days(result["action_last_at"] - result["action_first_at"])
    result["action_per_active_day"] = _safe_ratio(result["action_total_count"], result["action_active_days"])

    feature_list = [column for column in result.columns if column != CORE_ENTITY_KEY]
    summary = _row_summary(
        block_name="course_action_features",
        source_tables=["wk_users_courses_actions", "users_courses_base"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes="Course actions already carry users_course_id, so the block can be aggregated directly at the target grain.",
        important_warnings="Window counts use enrollment and access dates from the current user-course record and should be revisited if a future rolling cutoff is introduced.",
    )
    return result, feature_list, summary


# =============================================================================
# Media block
# =============================================================================


def build_media_features(
    wk_media_view_sessions: pd.DataFrame,
    groups: pd.DataFrame,
    lessons: pd.DataFrame,
    users_courses_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Resolve media sessions to user-course and aggregate media behavior."""
    lesson_media_map = lessons[["id", "course_id"]].rename(columns={"id": "resource_id"})
    group_media_map = groups.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
    )[["id_x", "course_id"]].rename(columns={"id_x": "resource_id"})

    lesson_sessions = wk_media_view_sessions.loc[wk_media_view_sessions["resource_type"] == "Lesson"].merge(
        lesson_media_map,
        on="resource_id",
        how="left",
        validate="m:1",
    )
    group_sessions = wk_media_view_sessions.loc[wk_media_view_sessions["resource_type"] == "Group"].merge(
        group_media_map,
        on="resource_id",
        how="left",
        validate="m:1",
    )

    resolved = pd.concat([lesson_sessions, group_sessions], ignore_index=True)
    resolved = resolved.merge(
        users_courses_base[[CORE_ENTITY_KEY, "user_id", "course_id"]],
        left_on=["viewer_id", "course_id"],
        right_on=["user_id", "course_id"],
        how="left",
        validate="m:1",
    )
    resolved["media_view_fraction"] = _safe_ratio(
        resolved["viewed_segments_count"],
        resolved["segments_total"],
    )
    resolved["media_fully_watched_flag"] = resolved["media_view_fraction"].fillna(0).ge(0.95).astype("int8")
    resolved["media_day"] = resolved["started_at"].dt.normalize()

    matched = resolved.loc[resolved[CORE_ENTITY_KEY].notna()].copy()
    matched["media_resource_key"] = (
        matched["resource_type"].astype("string") + "_" + matched["resource_id"].astype("string")
    )

    result = (
        matched.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(
            media_session_count=("resource_id", "count"),
            media_resource_unique_count=("media_resource_key", "nunique"),
            media_total_segments=("segments_total", "sum"),
            media_total_viewed_segments=("viewed_segments_count", "sum"),
            media_view_fraction_mean=("media_view_fraction", "mean"),
            media_view_fraction_max=("media_view_fraction", "max"),
            media_fully_watched_count=("media_fully_watched_flag", "sum"),
            media_first_at=("started_at", "min"),
            media_last_at=("started_at", "max"),
            media_active_days=("media_day", "nunique"),
        )
    )

    kind_counts = pd.crosstab(matched[CORE_ENTITY_KEY], matched["kind"]).add_prefix("media_kind_count_").reset_index()
    type_counts = (
        pd.crosstab(matched[CORE_ENTITY_KEY], matched["resource_type"])
        .rename(columns={"Lesson": "media_lesson_session_count", "Group": "media_group_session_count"})
        .reset_index()
    )
    result = result.merge(kind_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result = result.merge(type_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result["media_span_days"] = _timedelta_in_days(result["media_last_at"] - result["media_first_at"])

    feature_list = [column for column in result.columns if column != CORE_ENTITY_KEY]
    summary = _row_summary(
        block_name="media_features",
        source_tables=["wk_media_view_sessions", "groups", "lessons", "users_courses_base"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes=(
            f"Media rows are resolved through Lesson/Group resource bridges. "
            f"Matched coverage: {matched.shape[0] / max(len(resolved), 1):.4%}."
        ),
        important_warnings="Media linkage is highly reliable in this dataset, but a tiny unmatched tail remains and should be kept in diagnostics.",
        extra={
            "route_rows_total": int(len(resolved)),
            "route_rows_matched": int(len(matched)),
            "route_rows_unmatched": int(len(resolved) - len(matched)),
        },
    )
    return result, feature_list, summary


# =============================================================================
# Access history block
# =============================================================================


def build_access_history_features(
    user_access_histories: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Aggregate access history records directly at user-course grain."""
    access_df = user_access_histories.copy()
    access_df["access_period_days"] = _timedelta_in_days(
        access_df["access_expired_at"] - access_df["access_started_at"]
    )
    access_df["access_revoke_flag"] = (
        access_df["activator_class"].astype("string").str.contains("Revoke", case=False, na=False)
    ).astype("int8")
    access_df["access_extension_flag"] = (
        access_df["activator_class"].astype("string").str.contains(
            "ChangeAccessDuration|MonthPremium",
            case=False,
            na=False,
        )
    ).astype("int8")
    access_df["access_premium_flag"] = (
        access_df["activator_class"].astype("string").str.contains("Premium", case=False, na=False)
    ).astype("int8")

    result = (
        access_df.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(
            access_period_count=("access_started_at", "count"),
            access_total_duration_days=("access_period_days", "sum"),
            access_first_start_at=("access_started_at", "min"),
            access_last_end_at=("access_expired_at", "max"),
            access_revoke_count=("access_revoke_flag", "sum"),
            access_extension_count=("access_extension_flag", "sum"),
            access_premium_count=("access_premium_flag", "sum"),
        )
    )
    result["access_multiple_periods_flag"] = result["access_period_count"].gt(1).astype("int8")
    result["access_span_days"] = _timedelta_in_days(
        result["access_last_end_at"] - result["access_first_start_at"]
    )

    feature_list = [column for column in result.columns if column != CORE_ENTITY_KEY]
    summary = _row_summary(
        block_name="access_history_features",
        source_tables=["user_access_histories"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes="Access history records already use users_course_id and therefore aggregate directly to the target grain.",
        important_warnings="Access period features are administrative and may need horizon control in downstream modeling.",
    )
    return result, feature_list, summary


# =============================================================================
# Second-stage time features
# =============================================================================


def build_time_window_features(
    final_master_df: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Create second-stage time-aware features on the assembled master table."""
    result = final_master_df.copy()
    observed_end = result["access_finished_at"].fillna(reference_timestamp)

    result["time_observed_window_days"] = _timedelta_in_days(observed_end - result["created_at"])
    result["time_to_first_action_days"] = _timedelta_in_days(result["action_first_at"] - result["created_at"])
    result["time_to_first_answer_days"] = _timedelta_in_days(result["answer_first_at"] - result["created_at"])
    result["time_to_first_training_days"] = _timedelta_in_days(
        result["training_first_activity_at"] - result["created_at"]
    )
    result["time_to_first_media_days"] = _timedelta_in_days(result["media_first_at"] - result["created_at"])

    first_cols = ["action_first_at", "answer_first_at", "training_first_activity_at", "media_first_at"]
    last_cols = ["action_last_at", "answer_last_at", "training_last_activity_at", "media_last_at"]
    result["time_first_any_activity_at"] = result[first_cols].min(axis=1)
    result["time_last_any_activity_at"] = result[last_cols].max(axis=1)
    result["time_to_first_any_activity_days"] = _timedelta_in_days(
        result["time_first_any_activity_at"] - result["created_at"]
    )
    result["time_inactivity_gap_days"] = _timedelta_in_days(observed_end - result["time_last_any_activity_at"])

    result["time_action_recency_to_access_end_days"] = _timedelta_in_days(observed_end - result["action_last_at"])
    result["time_answer_recency_to_access_end_days"] = _timedelta_in_days(observed_end - result["answer_last_at"])
    result["time_training_recency_to_access_end_days"] = _timedelta_in_days(
        observed_end - result["training_last_activity_at"]
    )
    result["time_media_recency_to_access_end_days"] = _timedelta_in_days(observed_end - result["media_last_at"])

    result["time_action_intensity_per_observed_day"] = _safe_ratio(
        result["action_total_count"],
        result["time_observed_window_days"],
    )
    result["time_answer_intensity_per_observed_day"] = _safe_ratio(
        result["answer_total_count"],
        result["time_observed_window_days"],
    )
    result["time_training_intensity_per_observed_day"] = _safe_ratio(
        result["training_records_count"],
        result["time_observed_window_days"],
    )
    result["time_media_intensity_per_observed_day"] = _safe_ratio(
        result["media_session_count"],
        result["time_observed_window_days"],
    )
    result["time_points_per_observed_day"] = _safe_ratio(
        result["wk_points"],
        result["time_observed_window_days"],
    )

    result["time_action_first_14d_share"] = _safe_ratio(
        result["action_first_14d_count"],
        result["action_total_count"],
    )
    result["time_action_first_30d_share"] = _safe_ratio(
        result["action_first_30d_count"],
        result["action_total_count"],
    )
    result["time_action_last_14d_share"] = _safe_ratio(
        result["action_last_14d_count"],
        result["action_total_count"],
    )
    result["time_action_late_to_early_ratio"] = _safe_ratio(
        result["action_last_14d_count"],
        result["action_first_14d_count"],
    )

    channel_flags = result[["action_total_count", "answer_total_count", "training_records_count", "media_session_count"]]
    result["time_active_channels_count"] = channel_flags.fillna(0).gt(0).sum(axis=1).astype("int8")

    feature_list = [
        "time_observed_window_days",
        "time_to_first_action_days",
        "time_to_first_answer_days",
        "time_to_first_training_days",
        "time_to_first_media_days",
        "time_first_any_activity_at",
        "time_last_any_activity_at",
        "time_to_first_any_activity_days",
        "time_inactivity_gap_days",
        "time_action_recency_to_access_end_days",
        "time_answer_recency_to_access_end_days",
        "time_training_recency_to_access_end_days",
        "time_media_recency_to_access_end_days",
        "time_action_intensity_per_observed_day",
        "time_answer_intensity_per_observed_day",
        "time_training_intensity_per_observed_day",
        "time_media_intensity_per_observed_day",
        "time_points_per_observed_day",
        "time_action_first_14d_share",
        "time_action_first_30d_share",
        "time_action_last_14d_share",
        "time_action_late_to_early_ratio",
        "time_active_channels_count",
    ]

    summary = _row_summary(
        block_name="time_window_features",
        source_tables=["final_master_user_course_features"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes="Time-aware features are computed only after all blocks are aligned to the same user-course grain.",
        important_warnings="Observed-window features use access end as the default horizon and should be revisited if the modeling cutoff changes.",
    )
    return result, feature_list, summary
