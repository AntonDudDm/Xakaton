from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config_AGENT import CORE_ENTITY_KEY
from .service_AGENT import build_route_coverage, validate_key_uniqueness


# =============================================================================
# Shared helpers
# =============================================================================


def _bool_to_int(series: pd.Series) -> pd.Series:
    """Convert nullable booleans into stable integer flags."""
    return series.fillna(False).astype("int8")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a safe numeric ratio with missing values when the denominator is 0."""
    numerator = pd.Series(numerator, copy=False)
    denominator = pd.Series(denominator, copy=False)

    denominator = denominator.replace({0: np.nan})
    return numerator.astype("Float64") / denominator.astype("Float64")


def _timedelta_in_days(series: pd.Series) -> pd.Series:
    """Convert a timedelta series into float days."""
    return series.dt.total_seconds() / 86400.0


def _normalize_text_state(series: pd.Series) -> pd.Series:
    """Normalize free-text categorical states for simple rule-based flags."""
    return series.astype("string").str.strip().str.lower()


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

    # -------------------------------------------------------------------------
    # Existing base flags and ratios
    # -------------------------------------------------------------------------
    base["uc_active_flag"] = (base["state"] == "active").astype("int8")
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

    # -------------------------------------------------------------------------
    # New additive features: keep old logic intact, only enrich the base block
    # -------------------------------------------------------------------------

    # Cleaner semantic aliases for the most important lifecycle events.
    base["uc_started_flag"] = base["uc_has_official_start"]
    base["uc_completed_flag"] = base["uc_has_completion_record"]
    base["uc_has_access_end_flag"] = base["uc_has_access_end"]

    # Existence / structural availability flags.
    base["uc_has_points_flag"] = base["wk_points"].notna().astype("int8")
    base["uc_has_max_points_flag"] = base["wk_max_points"].notna().astype("int8")
    base["uc_has_viewable_lessons_flag"] = base["wk_max_viewable_lessons"].fillna(0).gt(0).astype("int8")
    base["uc_has_tasks_flag"] = base["wk_max_task_count"].fillna(0).gt(0).astype("int8")

    # Progress-presence flag: useful for separating "empty" enrollments
    # from records with any evidence of real course progress.
    base["uc_any_progress_flag"] = (
        base["wk_points"].fillna(0).gt(0)
        | base["wk_officially_started_at"].notna()
        | base["wk_course_completed_at"].notna()
    ).astype("int8")

    # Additional normalized progress features.
    base["uc_points_per_task"] = _safe_ratio(
        base["wk_points"],
        base["wk_max_task_count"],
    )
    base["uc_points_per_viewable_lesson"] = _safe_ratio(
        base["wk_points"],
        base["wk_max_viewable_lessons"],
    )
    base["uc_viewable_lessons_per_task"] = _safe_ratio(
        base["wk_max_viewable_lessons"],
        base["wk_max_task_count"],
    )

    # Simple diagnostic consistency flags.
    base["uc_completed_without_points_flag"] = (
        base["wk_course_completed_at"].notna()
        & base["wk_points"].fillna(0).eq(0)
    ).astype("int8")
    base["uc_started_without_points_flag"] = (
        base["wk_officially_started_at"].notna()
        & base["wk_points"].fillna(0).eq(0)
    ).astype("int8")
    base["uc_has_points_without_start_flag"] = (
        base["wk_points"].fillna(0).gt(0)
        & base["wk_officially_started_at"].isna()
    ).astype("int8")

    # Stable anchor timestamp for later temporal feature engineering.
    # Prefer the official course start when available; otherwise fall back
    # to record creation time.
    base["uc_start_anchor_at"] = base["wk_officially_started_at"].fillna(base["created_at"])

    # Relative-to-anchor timing signals.
    base["uc_days_from_anchor_to_access_end"] = _timedelta_in_days(
        base["access_finished_at"] - base["uc_start_anchor_at"]
    )
    base["uc_days_from_anchor_to_completion"] = _timedelta_in_days(
        base["wk_course_completed_at"] - base["uc_start_anchor_at"]
    )

    feature_list = [
        "uc_active_flag",
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

        # New additive features
        "uc_started_flag",
        "uc_completed_flag",
        "uc_has_access_end_flag",
        "uc_has_points_flag",
        "uc_has_max_points_flag",
        "uc_has_viewable_lessons_flag",
        "uc_has_tasks_flag",
        "uc_any_progress_flag",
        "uc_points_per_task",
        "uc_points_per_viewable_lesson",
        "uc_viewable_lessons_per_task",
        "uc_completed_without_points_flag",
        "uc_started_without_points_flag",
        "uc_has_points_without_start_flag",
        "uc_start_anchor_at",
        "uc_days_from_anchor_to_access_end",
        "uc_days_from_anchor_to_completion",
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

    badge_enriched = user_award_badges.merge(
        award_badges.rename(columns={"id": "award_badge_id"}),
        on="award_badge_id",
        how="left",
        validate="m:1",
        indicator="_badge_dict_merge",
    )

    badge_agg = (
        badge_enriched
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

    # -------------------------------------------------------------------------
    # New additive badge/profile features
    # -------------------------------------------------------------------------
    badge_agg["user_has_badges_flag"] = badge_agg["user_badges_total_count"].fillna(0).gt(0).astype("int8")
    badge_agg["user_has_special_badges_flag"] = badge_agg["user_special_badges_count"].fillna(0).gt(0).astype("int8")

    result = user_df.merge(badge_agg, on="user_id", how="left", validate="1:1")

    result["user_first_badge_delay_days"] = _timedelta_in_days(
        result["user_first_badge_at"] - result["created_at"]
    )
    result["user_badges_per_account_year"] = _safe_ratio(
        result["user_badges_total_count"],
        result["user_account_age_days"] / 365.25,
    )
    result["user_unique_badges_per_account_year"] = _safe_ratio(
        result["user_badges_unique_count"],
        result["user_account_age_days"] / 365.25,
    )

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

        # New additive features
        "user_has_badges_flag",
        "user_has_special_badges_flag",
        "user_first_badge_delay_days",
        "user_badges_per_account_year",
        "user_unique_badges_per_account_year",
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
        extra={
            "badge_dictionary_route_coverage": build_route_coverage(
                badge_enriched,
                badge_enriched["_badge_dict_merge"] == "both",
                "user_award_badges -> award_badges",
            ),
        },
    )
    return result, feature_list, summary

# =============================================================================
# User-level aggregation
# =============================================================================


def build_users_base_features(
    users: pd.DataFrame,
    user_award_badges: pd.DataFrame,
    award_badges: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build stable user-level profile and badge features."""

    # -------------------------------------------------------------------------
    # 1. Base pupil subset
    # -------------------------------------------------------------------------
    user_df = users.loc[users["type"] == "User::Pupil"].copy()
    user_df = user_df.rename(columns={"id": "user_id"})

    # -------------------------------------------------------------------------
    # 2. Core profile features
    # Здесь собраны базовые пользовательские признаки, которые легко
    # интерпретируются и стабильно мержатся в master-table по user_id.
    # -------------------------------------------------------------------------
    user_df["user_is_subscribed_flag"] = _bool_to_int(user_df["subscribed"])
    user_df["user_account_age_days"] = _timedelta_in_days(
        reference_timestamp - user_df["created_at"]
    )
    user_df["user_profile_age_days"] = _timedelta_in_days(user_df["updated_at"] - user_df["created_at"])

    user_df["user_has_grade_flag"] = user_df["grade_id"].notna().astype("int8")
    user_df["user_has_timezone_flag"] = user_df["timezone"].notna().astype("int8")
    user_df["user_has_region_flag"] = user_df["d_wk_region_id"].notna().astype("int8")
    user_df["user_has_municipal_flag"] = user_df["d_wk_municipal_id"].notna().astype("int8")
    user_df["user_has_school_flag"] = user_df["d_wk_school_id"].notna().astype("int8")
    user_df["user_has_gender_flag"] = user_df["wk_gender"].notna().astype("int8")

    # -------------------------------------------------------------------------
    # 3. Badge route resolution
    # Здесь можно менять логику сопоставления пользовательских наград
    # со справочником наград.
    # -------------------------------------------------------------------------
    badge_enriched = user_award_badges.merge(
        award_badges.rename(columns={"id": "award_badge_id"}),
        on="award_badge_id",
        how="left",
        validate="m:1",
        indicator="_badge_dict_merge",
    )

    # -------------------------------------------------------------------------
    # 4. Badge aggregations
    # Здесь собраны основные агрегаты по наградам на уровне user_id.
    # -------------------------------------------------------------------------
    badge_agg = (
        badge_enriched
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
    badge_agg["user_has_badges_flag"] = (
        badge_agg["user_badges_total_count"].fillna(0).gt(0).astype("int8")
    )
    badge_agg["user_has_special_badges_flag"] = (
        badge_agg["user_special_badges_count"].fillna(0).gt(0).astype("int8")
    )

    # -------------------------------------------------------------------------
    # 5. Merge profile + badges
    # -------------------------------------------------------------------------
    result = user_df.merge(
        badge_agg,
        on="user_id",
        how="left",
        validate="1:1",
    )

    # -------------------------------------------------------------------------
    # 6. Derived badge tempo features
    # Здесь можно добавлять или убирать производные признаки по наградам.
    # -------------------------------------------------------------------------
    result["user_first_badge_delay_days"] = _timedelta_in_days(
        result["user_first_badge_at"] - result["created_at"]
    )
    result["user_badges_per_account_year"] = _safe_ratio(
        result["user_badges_total_count"],
        result["user_account_age_days"] / 365.25,
    )

    # -------------------------------------------------------------------------
    # 7. Feature groups
    # Здесь удобно вручную управлять составом признаков.
    # -------------------------------------------------------------------------
    profile_raw_features = [
        "sign_in_count",
        "grade_id",
        "timezone",
        "wk_gender",
        "d_wk_region_id",
        "d_wk_municipal_id",
        "d_wk_school_id",
    ]

    profile_flag_features = [
        "user_is_subscribed_flag",
        "user_account_age_days",
        "user_profile_age_days",
        "user_has_grade_flag",
        "user_has_timezone_flag",
        "user_has_region_flag",
        "user_has_municipal_flag",
        "user_has_school_flag",
        "user_has_gender_flag",
    ]

    badge_core_features = [
        "user_badges_total_count",
        "user_badges_unique_count",
        "user_special_badges_count",
        "user_badge_span_days",
        "user_has_badges_flag",
        "user_has_special_badges_flag",
        "user_first_badge_delay_days",
        "user_badges_per_account_year",
    ]

    feature_list = (
        profile_raw_features
        + profile_flag_features
        + badge_core_features
    )

    # -------------------------------------------------------------------------
    # 8. Final column selection
    # user_first_badge_at / user_last_badge_at оставляем в result для EDA,
    # но не включаем в feature_list как основные модельные признаки.
    # -------------------------------------------------------------------------
    keep_cols = ["user_id"] + feature_list + ["user_first_badge_at", "user_last_badge_at"]
    result = result[keep_cols]

    # -------------------------------------------------------------------------
    # 9. Summary
    # -------------------------------------------------------------------------
    summary = _row_summary(
        block_name="users_base_features",
        source_tables=["users", "user_award_badges", "award_badges"],
        df=result,
        key_cols=["user_id"],
        feature_list=feature_list,
        coverage_notes=(
            "User-level features are aggregated to one row per user_id "
            "before joining to the user-course base table."
        ),
        important_warnings=(
            "Badge-based signals are platform-wide and should be interpreted "
            "as enrichment rather than course-specific behavior."
        ),
        extra={
            "badge_dictionary_route_coverage": build_route_coverage(
                badge_enriched,
                badge_enriched["_badge_dict_merge"] == "both",
                "user_award_badges -> award_badges",
            ),
        },
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
    # -------------------------------------------------------------------------
    # Lesson skeleton
    # -------------------------------------------------------------------------
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

    # New additive lesson-level derived features
    lesson_agg["course_tasks_per_lesson"] = _safe_ratio(
        lesson_agg["course_lessons_task_count_sum"],
        lesson_agg["course_lessons_count"],
    )
    lesson_agg["course_points_per_lesson"] = _safe_ratio(
        lesson_agg["course_lessons_max_points_sum"],
        lesson_agg["course_lessons_count"],
    )
    lesson_agg["course_video_duration_per_lesson"] = _safe_ratio(
        lesson_agg["course_video_duration_sum"],
        lesson_agg["course_lessons_count"],
    )
    lesson_agg["course_has_video_flag"] = lesson_agg["course_video_duration_sum"].fillna(0).gt(0).astype("int8")
    lesson_agg["course_lesson_number_gap"] = (
        lesson_agg["course_lesson_number_max"].fillna(0) - lesson_agg["course_lessons_count"].fillna(0)
    ).astype("Float64")

    # -------------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------------
    task_enriched = lesson_tasks.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
        indicator="_task_course_merge",
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
    task_course_lessons_count = (
        lesson_agg.set_index("course_id")
        .reindex(task_agg["course_id"])["course_lessons_count"]
        .reset_index(drop=True)
    )

    task_agg["course_unique_tasks_per_lesson"] = _safe_ratio(
        task_agg["course_unique_task_count"],
        task_course_lessons_count,
    )
    task_agg["course_required_tasks_per_lesson"] = _safe_ratio(
        task_agg["course_required_task_count"],
        task_course_lessons_count,
    )
    task_agg["course_has_required_tasks_flag"] = task_agg["course_required_task_count"].fillna(0).gt(0).astype("int8")

    # -------------------------------------------------------------------------
    # Trainings
    # -------------------------------------------------------------------------
    training_enriched = trainings.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
        indicator="_training_course_merge",
    )
    training_agg = (
        training_enriched.groupby("course_id", as_index=False)
        .agg(
            course_trainings_count=("id_x", "count"),
            course_training_task_templates_sum=("task_templates_count", "sum"),
            course_training_difficulty_mean=("difficulty", "mean"),
            course_training_difficulty_max=("difficulty", "max"),
        )
    )
    training_course_lessons_count = (
        lesson_agg.set_index("course_id")
        .reindex(training_agg["course_id"])["course_lessons_count"]
        .reset_index(drop=True)
    )

    training_agg["course_trainings_per_lesson"] = _safe_ratio(
        training_agg["course_trainings_count"],
        training_course_lessons_count,
    )
    training_agg["course_training_templates_per_training"] = _safe_ratio(
        training_agg["course_training_task_templates_sum"],
        training_agg["course_trainings_count"],
    )
    training_agg["course_has_trainings_flag"] = training_agg["course_trainings_count"].fillna(0).gt(0).astype("int8")

    # -------------------------------------------------------------------------
    # Groups / webinars
    # -------------------------------------------------------------------------
    group_enriched = groups.merge(
        lessons[["id", "course_id"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
        indicator="_group_course_merge",
    )
    group_agg = (
        group_enriched.groupby("course_id", as_index=False)
        .agg(
            course_groups_count=("id_x", "count"),
            course_groups_with_video_count=("video_available", lambda s: int(s.fillna(False).sum())),
            course_groups_duration_sum=("duration", "sum"),
            course_groups_actual_duration_sum=("wk_duration_actual", "sum"),
            course_groups_finished_count=("state", lambda s: int((s.astype("string") == "finished").sum())),
        )
    )
    group_agg["course_groups_with_video_share"] = _safe_ratio(
        group_agg["course_groups_with_video_count"],
        group_agg["course_groups_count"],
    )
    group_agg["course_groups_finished_share"] = _safe_ratio(
        group_agg["course_groups_finished_count"],
        group_agg["course_groups_count"],
    )
    group_agg["course_groups_per_lesson"] = _safe_ratio(
        group_agg["course_groups_count"],
        lesson_agg.set_index("course_id").reindex(group_agg["course_id"])["course_lessons_count"].values,
    )
    group_agg["course_group_actual_to_planned_duration_ratio"] = _safe_ratio(
        group_agg["course_groups_actual_duration_sum"],
        group_agg["course_groups_duration_sum"],
    )
    group_agg["course_has_groups_flag"] = group_agg["course_groups_count"].fillna(0).gt(0).astype("int8")

    # -------------------------------------------------------------------------
    # Homeworks
    # -------------------------------------------------------------------------
    course_homeworks = homeworks.loc[homeworks["resource_type"] == "Lesson"].copy()
    course_homeworks = course_homeworks.merge(
        lessons[["id", "course_id"]],
        left_on="resource_id",
        right_on="id",
        how="left",
        validate="m:1",
        indicator="_homework_course_merge",
    )
    homework_agg = (
        course_homeworks.groupby("course_id", as_index=False)
        .agg(
            course_homeworks_count=("id_x", "count"),
            course_unique_homework_types=("homework_type", "nunique"),
        )
    )
    homework_course_lessons_count = (
        lesson_agg.set_index("course_id")
        .reindex(homework_agg["course_id"])["course_lessons_count"]
        .reset_index(drop=True)
    )

    homework_agg["course_homeworks_per_lesson"] = _safe_ratio(
        homework_agg["course_homeworks_count"],
        homework_course_lessons_count,
    )
    homework_agg["course_has_homeworks_flag"] = homework_agg["course_homeworks_count"].fillna(0).gt(0).astype("int8")

    homework_items_enriched = homework_items.merge(
        course_homeworks[["id_x", "course_id"]].rename(columns={"id_x": "homework_id"}),
        on="homework_id",
        how="left",
        validate="m:1",
        indicator="_homework_items_course_merge",
    )
    homework_items_agg = (
        homework_items_enriched.groupby("course_id", as_index=False)
        .agg(
            course_homework_item_count=("id", "count"),
            course_homework_task_item_count=("resource_type", lambda s: int((s == "Task").sum())),
        )
    )
    homework_course_counts = (
        homework_agg.set_index("course_id")
        .reindex(homework_items_agg["course_id"])["course_homeworks_count"]
        .reset_index(drop=True)
    )

    homework_items_agg["course_homework_items_per_homework"] = _safe_ratio(
        homework_items_agg["course_homework_item_count"],
        homework_course_counts,
    )
    homework_items_agg["course_homework_task_item_share"] = _safe_ratio(
        homework_items_agg["course_homework_task_item_count"],
        homework_items_agg["course_homework_item_count"],
    )

    # -------------------------------------------------------------------------
    # Final merge
    # -------------------------------------------------------------------------
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
        extra={
            "task_route_coverage": build_route_coverage(
                task_enriched,
                task_enriched["_task_course_merge"] == "both",
                "lesson_tasks -> lessons -> course_id",
            ),
            "training_route_coverage": build_route_coverage(
                training_enriched,
                training_enriched["_training_course_merge"] == "both",
                "trainings -> lessons -> course_id",
            ),
            "group_route_coverage": build_route_coverage(
                group_enriched,
                group_enriched["_group_course_merge"] == "both",
                "groups -> lessons -> course_id",
            ),
            "homework_route_coverage": build_route_coverage(
                course_homeworks,
                course_homeworks["_homework_course_merge"] == "both",
                "homeworks[Lesson] -> lessons -> course_id",
            ),
            "homework_items_route_coverage": build_route_coverage(
                homework_items_enriched,
                homework_items_enriched["_homework_items_course_merge"] == "both",
                "homework_items -> resolved homeworks -> course_id",
            ),
        },
    )
    return result, feature_list, summary

# def build_course_structure_features(
#     lessons: pd.DataFrame,
#     lesson_tasks: pd.DataFrame,
#     trainings: pd.DataFrame,
#     groups: pd.DataFrame,
#     homeworks: pd.DataFrame,
#     homework_items: pd.DataFrame,
# ) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
#     """Build stable course-level structure features."""
#     lesson_agg = (
#         lessons.groupby("course_id", as_index=False)
#         .agg(
#             course_lessons_count=("id", "count"),
#             course_lessons_with_tasks_count=("task_expected", lambda s: int(s.fillna(False).sum())),
#             course_lessons_with_conspect_count=("conspect_expected", lambda s: int(s.fillna(False).sum())),
#             course_lessons_survival_count=(
#                 "wk_survival_training_expected",
#                 lambda s: int(s.fillna(False).sum()),
#             ),
#             course_lessons_scratch_count=(
#                 "wk_scratch_playground_enabled",
#                 lambda s: int(s.fillna(False).sum()),
#             ),
#             course_lessons_attendance_count=(
#                 "wk_attendance_tracking_enabled",
#                 lambda s: int(s.fillna(False).sum()),
#             ),
#             course_lesson_number_max=("lesson_number", "max"),
#             course_lessons_max_points_sum=("wk_max_points", "sum"),
#             course_lessons_task_count_sum=("wk_task_count", "sum"),
#             course_video_duration_sum=("wk_video_duration", "sum"),
#             course_video_duration_mean=("wk_video_duration", "mean"),
#         )
#     )
#     lesson_agg["course_lessons_with_tasks_share"] = _safe_ratio(
#         lesson_agg["course_lessons_with_tasks_count"],
#         lesson_agg["course_lessons_count"],
#     )
#     lesson_agg["course_lessons_with_conspect_share"] = _safe_ratio(
#         lesson_agg["course_lessons_with_conspect_count"],
#         lesson_agg["course_lessons_count"],
#     )
#     lesson_agg["course_lessons_survival_share"] = _safe_ratio(
#         lesson_agg["course_lessons_survival_count"],
#         lesson_agg["course_lessons_count"],
#     )
#     lesson_agg["course_lessons_scratch_share"] = _safe_ratio(
#         lesson_agg["course_lessons_scratch_count"],
#         lesson_agg["course_lessons_count"],
#     )
#     lesson_agg["course_lessons_attendance_share"] = _safe_ratio(
#         lesson_agg["course_lessons_attendance_count"],
#         lesson_agg["course_lessons_count"],
#     )

#     task_enriched = lesson_tasks.merge(
#         lessons[["id", "course_id"]],
#         left_on="lesson_id",
#         right_on="id",
#         how="left",
#         validate="m:1",
#     )
#     task_agg = (
#         task_enriched.groupby("course_id", as_index=False)
#         .agg(
#             course_task_link_count=("id_x", "count"),
#             course_unique_task_count=("task_id", "nunique"),
#             course_required_task_count=("task_required", lambda s: int(s.fillna(False).sum())),
#         )
#     )
#     task_agg["course_required_task_share"] = _safe_ratio(
#         task_agg["course_required_task_count"],
#         task_agg["course_task_link_count"],
#     )

#     training_enriched = trainings.merge(
#         lessons[["id", "course_id"]],
#         left_on="lesson_id",
#         right_on="id",
#         how="left",
#         validate="m:1",
#     )
#     training_agg = (
#         training_enriched.groupby("course_id", as_index=False)
#         .agg(
#             course_trainings_count=("id_x", "count"),
#             course_training_task_templates_sum=("task_templates_count", "sum"),
#             course_training_difficulty_mean=("difficulty", "mean"),
#         )
#     )

#     group_enriched = groups.merge(
#         lessons[["id", "course_id"]],
#         left_on="lesson_id",
#         right_on="id",
#         how="left",
#         validate="m:1",
#     )
#     group_agg = (
#         group_enriched.groupby("course_id", as_index=False)
#         .agg(
#             course_groups_count=("id_x", "count"),
#             course_groups_with_video_count=("video_available", lambda s: int(s.fillna(False).sum())),
#             course_groups_duration_sum=("duration", "sum"),
#             course_groups_actual_duration_sum=("wk_duration_actual", "sum"),
#         )
#     )
#     group_agg["course_groups_with_video_share"] = _safe_ratio(
#         group_agg["course_groups_with_video_count"],
#         group_agg["course_groups_count"],
#     )

#     course_homeworks = homeworks.loc[homeworks["resource_type"] == "Lesson"].copy()
#     course_homeworks = course_homeworks.merge(
#         lessons[["id", "course_id"]],
#         left_on="resource_id",
#         right_on="id",
#         how="left",
#         validate="m:1",
#     )
#     homework_agg = (
#         course_homeworks.groupby("course_id", as_index=False)
#         .agg(
#             course_homeworks_count=("id_x", "count"),
#             course_unique_homework_types=("homework_type", "nunique"),
#         )
#     )

#     homework_items_enriched = homework_items.merge(
#         course_homeworks[["id_x", "course_id"]].rename(columns={"id_x": "homework_id"}),
#         on="homework_id",
#         how="left",
#         validate="m:1",
#     )
#     homework_items_agg = (
#         homework_items_enriched.groupby("course_id", as_index=False)
#         .agg(
#             course_homework_item_count=("id", "count"),
#             course_homework_task_item_count=(
#                 "resource_type",
#                 lambda s: int((s == "Task").sum()),
#             ),
#         )
#     )

#     result = lesson_agg.copy()
#     for block in [task_agg, training_agg, group_agg, homework_agg, homework_items_agg]:
#         result = result.merge(block, on="course_id", how="left", validate="1:1")

#     feature_list = [column for column in result.columns if column != "course_id"]

#     summary = _row_summary(
#         block_name="course_structure_features",
#         source_tables=["lessons", "lesson_tasks", "trainings", "groups", "homeworks", "homework_items"],
#         df=result,
#         key_cols=["course_id"],
#         feature_list=feature_list,
#         coverage_notes="Course structure is aggregated before any merge to the user-course base. Lessons are used as the stable course skeleton.",
#         important_warnings="Homework-derived structure only covers homework rows that resolve to lesson resources.",
#     )
#     return result, feature_list, summary


# =============================================================================
# User-lesson block
# =============================================================================

def build_user_lesson_features(
    user_lessons: pd.DataFrame,
    lessons: pd.DataFrame,
    course_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Aggregate user-lesson progress signals to users_course_id grain."""

    # -------------------------------------------------------------------------
    # 1. Route resolution: user_lessons -> lessons
    # Здесь подтягиваем course_id и lesson_number.
    # -------------------------------------------------------------------------
    enriched = user_lessons.merge(
        lessons[["id", "course_id", "lesson_number"]],
        left_on="lesson_id",
        right_on="id",
        how="left",
        validate="m:1",
        indicator="_lesson_route_merge",
    )

    # -------------------------------------------------------------------------
    # 2. Core resolved helper columns
    # Здесь можно менять fallback-логику для решенных задач.
    # -------------------------------------------------------------------------
    enriched["ul_solved_tasks_resolved"] = (
        enriched["wk_solved_task_count"]
        .fillna(enriched["solved_tasks_count"])
        .fillna(0)
        .astype("Float64")
    )

    enriched["ul_task_count_gap"] = (
        enriched["solved_tasks_count"].fillna(0)
        - enriched["wk_solved_task_count"].fillna(0)
    )

    # -------------------------------------------------------------------------
    # 3. Core aggregation to users_course_id
    # Здесь лежат основные lesson-level user-course фичи.
    # -------------------------------------------------------------------------
    result = (
        enriched.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(
            ul_lessons_touched_count=("lesson_id", "nunique"),
            ul_lessons_solved_count=("solved", lambda s: int(s.fillna(False).sum())),
            ul_video_visited_count=("video_visited", lambda s: int(s.fillna(False).sum())),
            ul_video_viewed_count=("video_viewed", lambda s: int(s.fillna(False).sum())),
            ul_translation_visited_count=("translation_visited", lambda s: int(s.fillna(False).sum())),
            ul_points_sum=("wk_points", "sum"),
            ul_solved_tasks_sum=("ul_solved_tasks_resolved", "sum"),
            ul_furthest_lesson_number=("lesson_number", "max"),
            ul_task_count_gap_sum=("ul_task_count_gap", "sum"),
        )
    )

    # -------------------------------------------------------------------------
    # 4. Entity map and conflict diagnostics
    # Здесь проверяем, не приводит ли enrichment к конфликту users_course_id -> course_id.
    # -------------------------------------------------------------------------
    entity_map_raw = (
        enriched[[CORE_ENTITY_KEY, "course_id"]]
        .dropna(subset=[CORE_ENTITY_KEY, "course_id"])
        .drop_duplicates()
    )

    course_conflicts = (
        entity_map_raw.groupby(CORE_ENTITY_KEY)["course_id"]
        .nunique()
        .rename("course_id_nunique_per_users_course")
        .reset_index()
    )

    entity_map = (
        entity_map_raw.groupby(CORE_ENTITY_KEY, as_index=False)
        .agg(course_id=("course_id", "first"))
    )

    # -------------------------------------------------------------------------
    # 5. Course denominators for normalized lesson progress
    # Здесь задаются только те признаки курса, которые нужны как знаменатели.
    # -------------------------------------------------------------------------
    ratio_denominators = course_features[
        [
            "course_id",
            "course_lessons_count",
            "course_lesson_number_max",
            "course_lessons_max_points_sum",
            "course_lessons_task_count_sum",
        ]
    ]

    result = result.merge(entity_map, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result = result.merge(course_conflicts, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result = result.merge(ratio_denominators, on="course_id", how="left", validate="m:1")

    # -------------------------------------------------------------------------
    # 6. Main normalized lesson features
    # Здесь удобно вручную добавлять или убирать ratio-признаки.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 7. Within-block behavioral ratios
    # Эти фичи описывают lesson-level поведение внутри уже затронутых уроков.
    # -------------------------------------------------------------------------
    result["ul_lessons_solved_ratio"] = _safe_ratio(
        result["ul_lessons_solved_count"],
        result["ul_lessons_touched_count"],
    )
    result["ul_video_visited_ratio"] = _safe_ratio(
        result["ul_video_visited_count"],
        result["ul_lessons_touched_count"],
    )
    result["ul_video_viewed_ratio"] = _safe_ratio(
        result["ul_video_viewed_count"],
        result["ul_lessons_touched_count"],
    )
    result["ul_translation_visited_ratio"] = _safe_ratio(
        result["ul_translation_visited_count"],
        result["ul_lessons_touched_count"],
    )
    result["ul_video_viewed_to_visited_ratio"] = _safe_ratio(
        result["ul_video_viewed_count"],
        result["ul_video_visited_count"],
    )
    result["ul_points_per_touched_lesson"] = _safe_ratio(
        result["ul_points_sum"],
        result["ul_lessons_touched_count"],
    )
    result["ul_solved_tasks_per_touched_lesson"] = _safe_ratio(
        result["ul_solved_tasks_sum"],
        result["ul_lessons_touched_count"],
    )

    # -------------------------------------------------------------------------
    # 8. Simple behavioral flags
    # -------------------------------------------------------------------------
    result["ul_has_any_video_visit_flag"] = (
        result["ul_video_visited_count"].fillna(0).gt(0).astype("int8")
    )
    result["ul_has_any_video_view_flag"] = (
        result["ul_video_viewed_count"].fillna(0).gt(0).astype("int8")
    )
    result["ul_has_any_translation_flag"] = (
        result["ul_translation_visited_count"].fillna(0).gt(0).astype("int8")
    )
    result["ul_has_any_solved_lesson_flag"] = (
        result["ul_lessons_solved_count"].fillna(0).gt(0).astype("int8")
    )
    result["ul_course_mapping_conflict_flag"] = (
        result["course_id_nunique_per_users_course"].fillna(0).gt(1).astype("int8")
    )

    # -------------------------------------------------------------------------
    # 9. Feature groups
    # Здесь удобно руками управлять составом финального feature_list.
    # -------------------------------------------------------------------------
    core_count_features = [
        "ul_lessons_touched_count",
        "ul_lessons_solved_count",
        "ul_video_visited_count",
        "ul_video_viewed_count",
        "ul_translation_visited_count",
        "ul_points_sum",
        "ul_solved_tasks_sum",
        "ul_furthest_lesson_number",
        "ul_task_count_gap_sum",
    ]

    ratio_features = [
        "ul_lessons_touched_ratio",
        "ul_furthest_lesson_ratio",
        "ul_points_ratio_vs_course",
        "ul_solved_tasks_ratio_vs_course",
        "ul_lessons_solved_ratio",
        "ul_video_visited_ratio",
        "ul_video_viewed_ratio",
        "ul_translation_visited_ratio",
        "ul_video_viewed_to_visited_ratio",
        "ul_points_per_touched_lesson",
        "ul_solved_tasks_per_touched_lesson",
    ]

    flag_features = [
        "ul_has_any_video_visit_flag",
        "ul_has_any_video_view_flag",
        "ul_has_any_translation_flag",
        "ul_has_any_solved_lesson_flag",
        "ul_course_mapping_conflict_flag",
    ]

    feature_list = core_count_features + ratio_features + flag_features

    # -------------------------------------------------------------------------
    # 10. Final cleanup
    # В result оставляем только итоговые признаки и ключ.
    # -------------------------------------------------------------------------
    result = result[[CORE_ENTITY_KEY] + feature_list]

    # -------------------------------------------------------------------------
    # 11. Summary
    # -------------------------------------------------------------------------
    summary = _row_summary(
        block_name="user_lesson_features",
        source_tables=["user_lessons", "lessons", "course_structure_features"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes=(
            "Lesson progress is already keyed by users_course_id, so the aggregation "
            "is direct and low risk."
        ),
        important_warnings=(
            "Solved-task counts use fallback logic between wk_solved_task_count "
            "and solved_tasks_count. Course denominators are used only for normalized ratios."
        ),
        extra={
            "lesson_route_coverage": build_route_coverage(
                enriched,
                enriched["_lesson_route_merge"] == "both",
                "user_lessons -> lessons -> course_id",
            ),
            "users_course_to_course_conflicts": int(
                course_conflicts["course_id_nunique_per_users_course"].fillna(0).gt(1).sum()
            ),
        },
    )
    return result, feature_list, summary

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
        indicator="_training_lesson_merge",
    )

    resolved = user_trainings.merge(
        training_map.rename(columns={"id_x": "training_id"}),
        on="training_id",
        how="left",
        validate="m:1",
        indicator="_user_training_route_merge",
    )
    resolved = resolved.merge(
        users_courses_base[[CORE_ENTITY_KEY, "user_id", "course_id", "created_at"]],
        on=["user_id", "course_id"],
        how="left",
        validate="m:1",
        indicator="_training_users_course_merge",
    )

    resolved["training_first_timestamp"] = resolved[["started_at", "finished_at", "mark_saved_at"]].min(axis=1)
    resolved["training_last_timestamp"] = resolved[["started_at", "finished_at", "mark_saved_at"]].max(axis=1)
    resolved["training_checked_flag"] = (resolved["state"] == "checked").astype("int8")
    resolved["training_started_flag"] = resolved["started_at"].notna().astype("int8")
    resolved["training_finished_flag"] = resolved["finished_at"].notna().astype("int8")
    resolved["training_high_mark_flag"] = resolved["mark"].fillna(-1).ge(4).astype("int8")

    # New additive flags by training type
    resolved["training_lesson_type_flag"] = (resolved["type"] == "LessonTraining").astype("int8")
    resolved["training_regular_type_flag"] = (resolved["type"] == "RegularTraining").astype("int8")
    resolved["training_olympiad_type_flag"] = (resolved["type"] == "OlympiadTraining").astype("int8")

    # New additive timing feature relative to course enrollment record
    resolved["training_first_delay_from_course_created_days"] = _timedelta_in_days(
        resolved["training_first_timestamp"] - resolved["created_at"]
    )

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

            # New additive aggregations
            training_lesson_type_count=("training_lesson_type_flag", "sum"),
            training_regular_type_count=("training_regular_type_flag", "sum"),
            training_olympiad_type_count=("training_olympiad_type_flag", "sum"),
            training_first_delay_from_course_created_days=("training_first_delay_from_course_created_days", "min"),
        )
    )

    result["training_activity_span_days"] = _timedelta_in_days(
        result["training_last_activity_at"] - result["training_first_activity_at"]
    )

    # New additive ratios
    result["training_checked_ratio"] = _safe_ratio(
        result["training_checked_count"],
        result["training_records_count"],
    )
    result["training_finished_ratio"] = _safe_ratio(
        result["training_finished_count"],
        result["training_records_count"],
    )
    result["training_high_mark_ratio"] = _safe_ratio(
        result["training_high_mark_count"],
        result["training_checked_count"],
    )
    result["training_points_per_training"] = _safe_ratio(
        result["training_earned_points_sum"],
        result["training_records_count"],
    )
    result["training_solved_tasks_per_training"] = _safe_ratio(
        result["training_solved_tasks_sum"],
        result["training_records_count"],
    )
    result["training_submitted_answers_per_training"] = _safe_ratio(
        result["training_submitted_answers_sum"],
        result["training_records_count"],
    )

    # New additive presence flags
    result["training_has_any_started_flag"] = result["training_started_count"].fillna(0).gt(0).astype("int8")
    result["training_has_any_checked_flag"] = result["training_checked_count"].fillna(0).gt(0).astype("int8")
    result["training_has_any_finished_flag"] = result["training_finished_count"].fillna(0).gt(0).astype("int8")
    result["training_has_any_high_mark_flag"] = result["training_high_mark_count"].fillna(0).gt(0).astype("int8")

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
            "training_to_lesson_route_coverage": build_route_coverage(
                training_map,
                training_map["_training_lesson_merge"] == "both",
                "trainings -> lessons -> course_id",
            ),
            "user_training_to_training_route_coverage": build_route_coverage(
                resolved,
                resolved["_user_training_route_merge"] == "both",
                "user_trainings -> trainings/lessons",
            ),
            "training_to_users_course_route_coverage": build_route_coverage(
                resolved,
                resolved["_training_users_course_merge"] == "both",
                "resolved trainings -> users_courses_base",
            ),
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
        indicator="_answer_training_lesson_merge",
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
            indicator="_answer_homework_lesson_merge",
        )[["homework_id", "course_id"]]
        .rename(columns={"homework_id": "resource_id"})
    )

    lesson_answers = user_answers.loc[user_answers["resource_type"] == "Lesson"].merge(
        lesson_map,
        on="resource_id",
        how="left",
        validate="m:1",
        indicator="_answer_lesson_merge",
    )
    training_answers = user_answers.loc[user_answers["resource_type"] == "Training"].merge(
        training_map,
        on="resource_id",
        how="left",
        validate="m:1",
        indicator="_answer_training_merge",
    )
    homework_answers = user_answers.loc[user_answers["resource_type"] == "Homework"].merge(
        homework_map,
        on="resource_id",
        how="left",
        validate="m:1",
        indicator="_answer_homework_merge",
    )

    resolved = pd.concat([lesson_answers, training_answers, homework_answers], ignore_index=True)

    base_cols = [CORE_ENTITY_KEY, "user_id", "course_id", "created_at"]
    if "uc_start_anchor_at" in users_courses_base.columns:
        base_cols.append("uc_start_anchor_at")
    if "access_finished_at" in users_courses_base.columns:
        base_cols.append("access_finished_at")

    resolved = resolved.merge(
        users_courses_base[base_cols],
        on=["user_id", "course_id"],
        how="left",
        validate="m:1",
        indicator="_answer_users_course_merge",
    )

    # -------------------------------------------------------------------------
    # Базовые флаги ответа
    # -------------------------------------------------------------------------
    resolved["answer_solved_flag"] = _bool_to_int(resolved["solved"])
    resolved["answer_skipped_flag"] = _bool_to_int(resolved["skipped"])
    resolved["answer_partial_flag"] = _bool_to_int(resolved["wk_partial_answer"])
    resolved["answer_unsolved_flag"] = (
        resolved["answer_solved_flag"].eq(0) & resolved["answer_skipped_flag"].eq(0)
    ).astype("int8")

    # Старые признаки оставляем ради совместимости
    resolved["answer_async_pending_flag"] = (
        resolved["async_check_status"].astype("string").str.lower() == "pending"
    ).astype("int8")
    resolved["answer_async_failed_flag"] = (
        resolved["async_check_status"].astype("string").str.lower() == "failed"
    ).astype("int8")

    # Корректные async-флаги по 0 / 1 / 2
    async_status_num = pd.to_numeric(resolved["async_check_status"], errors="coerce")
    resolved["answer_async_not_started_flag"] = async_status_num.eq(0).astype("int8")
    resolved["answer_async_in_progress_flag"] = async_status_num.eq(1).astype("int8")
    resolved["answer_async_completed_flag"] = async_status_num.eq(2).astype("int8")

    # -------------------------------------------------------------------------
    # Временная ось относительно личного старта курса
    # -------------------------------------------------------------------------
    resolved["answer_day"] = resolved["submitted_at"].dt.normalize()
    resolved["answer_weekday"] = resolved["submitted_at"].dt.dayofweek
    resolved["answer_weekend_flag"] = resolved["answer_weekday"].isin([5, 6]).astype("int8")
    resolved["answer_week_flag"] = resolved["answer_weekday"].isin([0, 1, 2, 3, 4]).astype("int8")

    if "uc_start_anchor_at" in resolved.columns:
        resolved["answer_start_anchor_at"] = resolved["uc_start_anchor_at"].fillna(resolved["created_at"])
    else:
        resolved["answer_start_anchor_at"] = resolved["created_at"]

    resolved["answer_days_from_start"] = _timedelta_in_days(
        resolved["submitted_at"] - resolved["answer_start_anchor_at"]
    )
    resolved["answer_week_from_start"] = (
        pd.to_numeric(resolved["answer_days_from_start"], errors="coerce") // 7
    ).astype("Float64")

    # Окна первых дней
    resolved["answer_first_7d_flag"] = resolved["answer_days_from_start"].between(0, 7, inclusive="both").astype("int8")
    resolved["answer_first_14d_flag"] = resolved["answer_days_from_start"].between(0, 14, inclusive="both").astype("int8")
    resolved["answer_first_30d_flag"] = resolved["answer_days_from_start"].between(0, 30, inclusive="both").astype("int8")

    # Фазы по 30 дней от старта
    resolved["answer_first_month30_flag"] = resolved["answer_days_from_start"].between(0, 30, inclusive="both").astype("int8")
    resolved["answer_second_month30_flag"] = resolved["answer_days_from_start"].between(31, 60, inclusive="both").astype("int8")
    resolved["answer_third_month30_flag"] = resolved["answer_days_from_start"].between(61, 90, inclusive="both").astype("int8")

    # Последние окна до конца доступа
    if "access_finished_at" in resolved.columns:
        resolved["answer_days_to_access_end"] = _timedelta_in_days(
            resolved["access_finished_at"] - resolved["submitted_at"]
        )
        resolved["answer_last_14d_flag"] = resolved["answer_days_to_access_end"].between(0, 14, inclusive="both").astype("int8")
        resolved["answer_last_30d_flag"] = resolved["answer_days_to_access_end"].between(0, 30, inclusive="both").astype("int8")
    else:
        resolved["answer_last_14d_flag"] = 0
        resolved["answer_last_30d_flag"] = 0

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

            # Корректные async-counts
            answer_async_not_started_count=("answer_async_not_started_flag", "sum"),
            answer_async_in_progress_count=("answer_async_in_progress_flag", "sum"),
            answer_async_completed_count=("answer_async_completed_flag", "sum"),

            # Окна
            answer_first_7d_count=("answer_first_7d_flag", "sum"),
            answer_first_14d_count=("answer_first_14d_flag", "sum"),
            answer_first_30d_count=("answer_first_30d_flag", "sum"),
            answer_last_14d_count=("answer_last_14d_flag", "sum"),
            answer_last_30d_count=("answer_last_30d_flag", "sum"),

            # 30-дневные фазы
            answer_first_month30_count=("answer_first_month30_flag", "sum"),
            answer_second_month30_count=("answer_second_month30_flag", "sum"),
            answer_third_month30_count=("answer_third_month30_flag", "sum"),

            # Календарная активность
            answer_weekend_count=("answer_weekend_flag", "sum"),
            answer_weekday_count=("answer_week_flag", "sum"),
            answer_active_weeks=("answer_week_from_start", "nunique"),
            answer_days_from_start_min=("answer_days_from_start", "min"),
            answer_days_from_start_max=("answer_days_from_start", "max"),
        )
    )

    # Counts по типам активностей
    type_counts = pd.crosstab(matched[CORE_ENTITY_KEY], matched["resource_type"])
    type_counts = type_counts.rename(
        columns={
            "Lesson": "answer_lesson_count",
            "Training": "answer_training_count",
            "Homework": "answer_homework_count",
        }
    ).reset_index()
    result = result.merge(type_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")

    # Counts по дням недели
    weekday_counts = pd.crosstab(matched[CORE_ENTITY_KEY], matched["answer_weekday"]).reset_index()
    weekday_counts = weekday_counts.rename(
        columns={
            0: "answer_monday_count",
            1: "answer_tuesday_count",
            2: "answer_wednesday_count",
            3: "answer_thursday_count",
            4: "answer_friday_count",
            5: "answer_saturday_count",
            6: "answer_sunday_count",
        }
    )
    result = result.merge(weekday_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")

    result["answer_activity_span_days"] = _timedelta_in_days(
        result["answer_last_at"] - result["answer_first_at"]
    )

    # -------------------------------------------------------------------------
    # Производные ratio / tempo признаки
    # -------------------------------------------------------------------------
    result["answer_solved_share"] = _safe_ratio(result["answer_solved_count"], result["answer_total_count"])
    result["answer_unsolved_share"] = _safe_ratio(result["answer_unsolved_count"], result["answer_total_count"])
    result["answer_skipped_share"] = _safe_ratio(result["answer_skipped_count"], result["answer_total_count"])
    result["answer_partial_share"] = _safe_ratio(result["answer_partial_count"], result["answer_total_count"])
    result["answer_points_per_answer"] = _safe_ratio(result["answer_points_sum"], result["answer_total_count"])
    result["answer_attempts_per_answer"] = _safe_ratio(result["answer_attempts_sum"], result["answer_total_count"])
    result["answer_points_per_active_day"] = _safe_ratio(result["answer_points_sum"], result["answer_active_days"])
    result["answer_solved_per_active_day"] = _safe_ratio(result["answer_solved_count"], result["answer_active_days"])

    result["answer_first_7d_share"] = _safe_ratio(result["answer_first_7d_count"], result["answer_total_count"])
    result["answer_first_14d_share"] = _safe_ratio(result["answer_first_14d_count"], result["answer_total_count"])
    result["answer_first_30d_share"] = _safe_ratio(result["answer_first_30d_count"], result["answer_total_count"])
    result["answer_last_14d_share"] = _safe_ratio(result["answer_last_14d_count"], result["answer_total_count"])
    result["answer_last_30d_share"] = _safe_ratio(result["answer_last_30d_count"], result["answer_total_count"])

    result["answer_first_month30_share"] = _safe_ratio(result["answer_first_month30_count"], result["answer_total_count"])
    result["answer_second_month30_share"] = _safe_ratio(result["answer_second_month30_count"], result["answer_total_count"])
    result["answer_third_month30_share"] = _safe_ratio(result["answer_third_month30_count"], result["answer_total_count"])

    result["answer_weekend_share"] = _safe_ratio(result["answer_weekend_count"], result["answer_total_count"])
    result["answer_weekday_share"] = _safe_ratio(result["answer_weekday_count"], result["answer_total_count"])

    result["answer_lesson_share"] = _safe_ratio(result.get("answer_lesson_count"), result["answer_total_count"])
    result["answer_training_share"] = _safe_ratio(result.get("answer_training_count"), result["answer_total_count"])
    result["answer_homework_share"] = _safe_ratio(result.get("answer_homework_count"), result["answer_total_count"])

    # Средняя активность по неделям / месяцам
    result["answer_observed_weeks"] = _safe_ratio(result["answer_days_from_start_max"] + 1, 7.0)
    result["answer_observed_months30"] = _safe_ratio(result["answer_days_from_start_max"] + 1, 30.0)

    result["answer_per_active_week"] = _safe_ratio(
        result["answer_total_count"],
        result["answer_active_weeks"],
    )
    result["answer_per_observed_week"] = _safe_ratio(
        result["answer_total_count"],
        result["answer_observed_weeks"],
    )
    result["answer_per_observed_month30"] = _safe_ratio(
        result["answer_total_count"],
        result["answer_observed_months30"],
    )

    # Доли по дням недели
    for day_col in [
        "answer_monday_count",
        "answer_tuesday_count",
        "answer_wednesday_count",
        "answer_thursday_count",
        "answer_friday_count",
        "answer_saturday_count",
        "answer_sunday_count",
    ]:
        if day_col in result.columns:
            result[day_col.replace("_count", "_share")] = _safe_ratio(
                result[day_col],
                result["answer_total_count"],
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
            "Answer-to-course linkage intentionally avoids task_id as a primary bridge because task_id is not globally unique across courses. "
            "Homework-based answer routing still covers only homework rows that resolve through lesson resources."
        ),
        extra={
            "lesson_answer_route_coverage": build_route_coverage(
                lesson_answers,
                lesson_answers["_answer_lesson_merge"] == "both",
                "Lesson answers -> lessons -> course_id",
            ),
            "training_answer_route_coverage": build_route_coverage(
                training_answers,
                training_answers["_answer_training_merge"] == "both",
                "Training answers -> trainings/lessons -> course_id",
            ),
            "homework_answer_route_coverage": build_route_coverage(
                homework_answers,
                homework_answers["_answer_homework_merge"] == "both",
                "Homework answers -> homeworks[Lesson] -> lessons -> course_id",
            ),
            "users_course_route_coverage": build_route_coverage(
                resolved,
                resolved["_answer_users_course_merge"] == "both",
                "Resolved answers -> users_courses_base",
            ),
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
    entity_cols = [CORE_ENTITY_KEY, "created_at", "access_finished_at"]
    if "uc_start_anchor_at" in users_courses_base.columns:
        entity_cols.append("uc_start_anchor_at")

    entity_dates = users_courses_base[entity_cols].rename(
        columns={"created_at": "course_created_at"}
    )

    action_df = wk_users_courses_actions.merge(
        entity_dates,
        on=CORE_ENTITY_KEY,
        how="left",
        validate="m:1",
        indicator="_action_entity_merge",
    )

    action_df["action_day"] = action_df["created_at"].dt.normalize()
    action_df["action_weekday"] = action_df["created_at"].dt.dayofweek
    action_df["action_weekend_flag"] = action_df["action_weekday"].isin([5, 6]).astype("int8")
    action_df["action_weekday_flag"] = action_df["action_weekday"].isin([0, 1, 2, 3, 4]).astype("int8")

    if "uc_start_anchor_at" in action_df.columns:
        action_df["action_start_anchor_at"] = action_df["uc_start_anchor_at"].fillna(action_df["course_created_at"])
    else:
        action_df["action_start_anchor_at"] = action_df["course_created_at"]

    action_df["days_from_enrollment"] = _timedelta_in_days(
        action_df["created_at"] - action_df["action_start_anchor_at"]
    )
    action_df["action_week_from_start"] = (
        pd.to_numeric(action_df["days_from_enrollment"], errors="coerce") // 7
    ).astype("Float64")

    action_df["days_to_access_end"] = _timedelta_in_days(
        action_df["access_finished_at"] - action_df["created_at"]
    )

    action_df["action_first_7d_flag"] = action_df["days_from_enrollment"].between(
        0, 7, inclusive="both"
    ).astype("int8")
    action_df["action_first_14d_flag"] = action_df["days_from_enrollment"].between(
        0, 14, inclusive="both"
    ).astype("int8")
    action_df["action_first_30d_flag"] = action_df["days_from_enrollment"].between(
        0, 30, inclusive="both"
    ).astype("int8")

    action_df["action_first_month30_flag"] = action_df["days_from_enrollment"].between(
        0, 30, inclusive="both"
    ).astype("int8")
    action_df["action_second_month30_flag"] = action_df["days_from_enrollment"].between(
        31, 60, inclusive="both"
    ).astype("int8")
    action_df["action_third_month30_flag"] = action_df["days_from_enrollment"].between(
        61, 90, inclusive="both"
    ).astype("int8")

    action_df["action_last_14d_flag"] = action_df["days_to_access_end"].between(
        0, 14, inclusive="both"
    ).astype("int8")
    action_df["action_last_30d_flag"] = action_df["days_to_access_end"].between(
        0, 30, inclusive="both"
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
            action_first_7d_count=("action_first_7d_flag", "sum"),
            action_first_14d_count=("action_first_14d_flag", "sum"),
            action_first_30d_count=("action_first_30d_flag", "sum"),
            action_last_14d_count=("action_last_14d_flag", "sum"),
            action_last_30d_count=("action_last_30d_flag", "sum"),

            # New temporal distribution features
            action_first_month30_count=("action_first_month30_flag", "sum"),
            action_second_month30_count=("action_second_month30_flag", "sum"),
            action_third_month30_count=("action_third_month30_flag", "sum"),
            action_weekend_count=("action_weekend_flag", "sum"),
            action_weekday_count=("action_weekday_flag", "sum"),
            action_active_weeks=("action_week_from_start", "nunique"),
            action_days_from_start_min=("days_from_enrollment", "min"),
            action_days_from_start_max=("days_from_enrollment", "max"),
        )
    )

    action_counts = pd.crosstab(action_df[CORE_ENTITY_KEY], action_df["action"])
    action_counts = action_counts.add_prefix("action_type_count_").reset_index()
    result = result.merge(action_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")

    weekday_counts = pd.crosstab(action_df[CORE_ENTITY_KEY], action_df["action_weekday"]).reset_index()
    weekday_counts = weekday_counts.rename(
        columns={
            0: "action_monday_count",
            1: "action_tuesday_count",
            2: "action_wednesday_count",
            3: "action_thursday_count",
            4: "action_friday_count",
            5: "action_saturday_count",
            6: "action_sunday_count",
        }
    )
    result = result.merge(weekday_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")

    result["action_span_days"] = _timedelta_in_days(result["action_last_at"] - result["action_first_at"])
    result["action_per_active_day"] = _safe_ratio(result["action_total_count"], result["action_active_days"])

    # New tempo features
    result["action_observed_weeks"] = _safe_ratio(result["action_days_from_start_max"] + 1, 7.0)
    result["action_per_active_week"] = _safe_ratio(result["action_total_count"], result["action_active_weeks"])
    result["action_per_observed_week"] = _safe_ratio(result["action_total_count"], result["action_observed_weeks"])

    # New shares
    result["action_first_7d_share"] = _safe_ratio(result["action_first_7d_count"], result["action_total_count"])
    result["action_first_14d_share"] = _safe_ratio(result["action_first_14d_count"], result["action_total_count"])
    result["action_first_30d_share"] = _safe_ratio(result["action_first_30d_count"], result["action_total_count"])
    result["action_last_14d_share"] = _safe_ratio(result["action_last_14d_count"], result["action_total_count"])
    result["action_last_30d_share"] = _safe_ratio(result["action_last_30d_count"], result["action_total_count"])

    result["action_first_month30_share"] = _safe_ratio(result["action_first_month30_count"], result["action_total_count"])
    result["action_second_month30_share"] = _safe_ratio(result["action_second_month30_count"], result["action_total_count"])
    result["action_third_month30_share"] = _safe_ratio(result["action_third_month30_count"], result["action_total_count"])

    result["action_weekend_share"] = _safe_ratio(result["action_weekend_count"], result["action_total_count"])
    result["action_weekday_share"] = _safe_ratio(result["action_weekday_count"], result["action_total_count"])

    result["action_unique_lessons_per_active_day"] = _safe_ratio(
        result["action_unique_lessons_count"], result["action_active_days"]
    )
    result["action_unique_types_per_active_day"] = _safe_ratio(
        result["action_unique_types_count"], result["action_active_days"]
    )

    # Shares by weekday
    for day_col in [
        "action_monday_count",
        "action_tuesday_count",
        "action_wednesday_count",
        "action_thursday_count",
        "action_friday_count",
        "action_saturday_count",
        "action_sunday_count",
    ]:
        if day_col in result.columns:
            result[day_col.replace("_count", "_share")] = _safe_ratio(
                result[day_col], result["action_total_count"]
            )

    # Presence flags by action type
    for action_col, out_col in [
        ("action_type_count_start_training", "action_has_start_training_flag"),
        ("action_type_count_user_answer", "action_has_user_answer_flag"),
        ("action_type_count_visit_video", "action_has_visit_video_flag"),
        ("action_type_count_visit_translation", "action_has_visit_translation_flag"),
        ("action_type_count_visit_preparation_material", "action_has_preparation_material_flag"),
        ("action_type_count_scratch_playground_visited", "action_has_scratch_flag"),
    ]:
        if action_col in result.columns:
            result[out_col] = result[action_col].fillna(0).gt(0).astype("int8")

    feature_list = [column for column in result.columns if column != CORE_ENTITY_KEY]
    summary = _row_summary(
        block_name="course_action_features",
        source_tables=["wk_users_courses_actions", "users_courses_base"],
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes="Course actions already carry users_course_id, so the block can be aggregated directly at the target grain.",
        important_warnings="Window counts use the current user-course record timing; if a future modeling cutoff is introduced, these windows should be recomputed relative to that cutoff.",
        extra={
            "action_entity_route_coverage": build_route_coverage(
                action_df,
                action_df["_action_entity_merge"] == "both",
                "wk_users_courses_actions -> users_courses_base",
            ),
        },
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
        indicator="_group_media_lesson_merge",
    )[["id_x", "course_id"]].rename(columns={"id_x": "resource_id"})

    lesson_sessions = wk_media_view_sessions.loc[
        wk_media_view_sessions["resource_type"] == "Lesson"
    ].merge(
        lesson_media_map,
        on="resource_id",
        how="left",
        validate="m:1",
        indicator="_lesson_media_merge",
    )

    group_sessions = wk_media_view_sessions.loc[
        wk_media_view_sessions["resource_type"] == "Group"
    ].merge(
        group_media_map,
        on="resource_id",
        how="left",
        validate="m:1",
        indicator="_group_media_merge",
    )

    resolved = pd.concat([lesson_sessions, group_sessions], ignore_index=True)

    base_cols = [CORE_ENTITY_KEY, "user_id", "course_id", "created_at"]
    if "uc_start_anchor_at" in users_courses_base.columns:
        base_cols.append("uc_start_anchor_at")
    if "access_finished_at" in users_courses_base.columns:
        base_cols.append("access_finished_at")

    resolved = resolved.merge(
        users_courses_base[base_cols],
        left_on=["viewer_id", "course_id"],
        right_on=["user_id", "course_id"],
        how="left",
        validate="m:1",
        indicator="_media_users_course_merge",
    )

    resolved["media_view_fraction"] = _safe_ratio(
        resolved["viewed_segments_count"],
        resolved["segments_total"],
    )
    resolved["media_fully_watched_flag"] = resolved["media_view_fraction"].fillna(0).ge(0.95).astype("int8")
    resolved["media_started_day"] = resolved["started_at"].dt.normalize()
    resolved["media_weekday"] = resolved["started_at"].dt.dayofweek
    resolved["media_weekend_flag"] = resolved["media_weekday"].isin([5, 6]).astype("int8")
    resolved["media_weekday_flag"] = resolved["media_weekday"].isin([0, 1, 2, 3, 4]).astype("int8")

    # Якорь времени
    if "uc_start_anchor_at" in resolved.columns:
        resolved["media_start_anchor_at"] = resolved["uc_start_anchor_at"].fillna(resolved["created_at"])
    else:
        resolved["media_start_anchor_at"] = resolved["created_at"]

    resolved["media_days_from_start"] = _timedelta_in_days(
        resolved["started_at"] - resolved["media_start_anchor_at"]
    )
    resolved["media_week_from_start"] = (
        pd.to_numeric(resolved["media_days_from_start"], errors="coerce") // 7
    ).astype("Float64")

    resolved["media_first_7d_flag"] = resolved["media_days_from_start"].between(0, 7, inclusive="both").astype("int8")
    resolved["media_first_14d_flag"] = resolved["media_days_from_start"].between(0, 14, inclusive="both").astype("int8")
    resolved["media_first_30d_flag"] = resolved["media_days_from_start"].between(0, 30, inclusive="both").astype("int8")

    if "access_finished_at" in resolved.columns:
        resolved["media_days_to_access_end"] = _timedelta_in_days(
            resolved["access_finished_at"] - resolved["started_at"]
        )
        resolved["media_last_14d_flag"] = resolved["media_days_to_access_end"].between(0, 14, inclusive="both").astype("int8")
        resolved["media_last_30d_flag"] = resolved["media_days_to_access_end"].between(0, 30, inclusive="both").astype("int8")
    else:
        resolved["media_last_14d_flag"] = 0
        resolved["media_last_30d_flag"] = 0

    # kind flags
    resolved["media_kind_live_flag"] = (resolved["kind"].astype("string") == "ulms_live").astype("int8")
    resolved["media_kind_vod_flag"] = (resolved["kind"].astype("string") == "ulms_vod").astype("int8")
    resolved["media_kind_kinescope_flag"] = (resolved["kind"].astype("string") == "kinescope").astype("int8")

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
            media_active_days=("media_started_day", "nunique"),

            # additive temporal features
            media_first_7d_count=("media_first_7d_flag", "sum"),
            media_first_14d_count=("media_first_14d_flag", "sum"),
            media_first_30d_count=("media_first_30d_flag", "sum"),
            media_last_14d_count=("media_last_14d_flag", "sum"),
            media_last_30d_count=("media_last_30d_flag", "sum"),
            media_active_weeks=("media_week_from_start", "nunique"),
            media_days_from_start_min=("media_days_from_start", "min"),
            media_days_from_start_max=("media_days_from_start", "max"),
            media_weekend_count=("media_weekend_flag", "sum"),
            media_weekday_count=("media_weekday_flag", "sum"),

            # additive kind counts
            media_live_count=("media_kind_live_flag", "sum"),
            media_vod_count=("media_kind_vod_flag", "sum"),
            media_kinescope_count=("media_kind_kinescope_flag", "sum"),
        )
    )

    kind_counts = (
        pd.crosstab(matched[CORE_ENTITY_KEY], matched["kind"])
        .add_prefix("media_kind_count_")
        .reset_index()
    )
    type_counts = (
        pd.crosstab(matched[CORE_ENTITY_KEY], matched["resource_type"])
        .rename(columns={"Lesson": "media_lesson_session_count", "Group": "media_group_session_count"})
        .reset_index()
    )
    result = result.merge(kind_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")
    result = result.merge(type_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")

    weekday_counts = pd.crosstab(matched[CORE_ENTITY_KEY], matched["media_weekday"]).reset_index()
    weekday_counts = weekday_counts.rename(
        columns={
            0: "media_monday_count",
            1: "media_tuesday_count",
            2: "media_wednesday_count",
            3: "media_thursday_count",
            4: "media_friday_count",
            5: "media_saturday_count",
            6: "media_sunday_count",
        }
    )
    result = result.merge(weekday_counts, on=CORE_ENTITY_KEY, how="left", validate="1:1")

    result["media_span_days"] = _timedelta_in_days(result["media_last_at"] - result["media_first_at"])

    # additive quality / tempo features
    result["media_view_fraction_weighted"] = _safe_ratio(
        result["media_total_viewed_segments"],
        result["media_total_segments"],
    )
    result["media_fully_watched_share"] = _safe_ratio(
        result["media_fully_watched_count"],
        result["media_session_count"],
    )
    result["media_segments_per_session"] = _safe_ratio(
        result["media_total_segments"],
        result["media_session_count"],
    )
    result["media_viewed_segments_per_session"] = _safe_ratio(
        result["media_total_viewed_segments"],
        result["media_session_count"],
    )
    result["media_resources_per_session"] = _safe_ratio(
        result["media_resource_unique_count"],
        result["media_session_count"],
    )
    result["media_per_active_day"] = _safe_ratio(
        result["media_session_count"],
        result["media_active_days"],
    )

    result["media_observed_weeks"] = _safe_ratio(result["media_days_from_start_max"] + 1, 7.0)
    result["media_per_active_week"] = _safe_ratio(
        result["media_session_count"],
        result["media_active_weeks"],
    )
    result["media_per_observed_week"] = _safe_ratio(
        result["media_session_count"],
        result["media_observed_weeks"],
    )

    result["media_first_7d_share"] = _safe_ratio(result["media_first_7d_count"], result["media_session_count"])
    result["media_first_14d_share"] = _safe_ratio(result["media_first_14d_count"], result["media_session_count"])
    result["media_first_30d_share"] = _safe_ratio(result["media_first_30d_count"], result["media_session_count"])
    result["media_last_14d_share"] = _safe_ratio(result["media_last_14d_count"], result["media_session_count"])
    result["media_last_30d_share"] = _safe_ratio(result["media_last_30d_count"], result["media_session_count"])

    result["media_weekend_share"] = _safe_ratio(result["media_weekend_count"], result["media_session_count"])
    result["media_weekday_share"] = _safe_ratio(result["media_weekday_count"], result["media_session_count"])

    result["media_lesson_share"] = _safe_ratio(result.get("media_lesson_session_count"), result["media_session_count"])
    result["media_group_share"] = _safe_ratio(result.get("media_group_session_count"), result["media_session_count"])
    result["media_live_share"] = _safe_ratio(result["media_live_count"], result["media_session_count"])
    result["media_vod_share"] = _safe_ratio(result["media_vod_count"], result["media_session_count"])
    result["media_kinescope_share"] = _safe_ratio(result["media_kinescope_count"], result["media_session_count"])

    for day_col in [
        "media_monday_count",
        "media_tuesday_count",
        "media_wednesday_count",
        "media_thursday_count",
        "media_friday_count",
        "media_saturday_count",
        "media_sunday_count",
    ]:
        if day_col in result.columns:
            result[day_col.replace("_count", "_share")] = _safe_ratio(
                result[day_col],
                result["media_session_count"],
            )

    # presence flags
    result["media_has_live_flag"] = result["media_live_count"].fillna(0).gt(0).astype("int8")
    result["media_has_vod_flag"] = result["media_vod_count"].fillna(0).gt(0).astype("int8")
    result["media_has_kinescope_flag"] = result["media_kinescope_count"].fillna(0).gt(0).astype("int8")
    result["media_has_fully_watched_flag"] = result["media_fully_watched_count"].fillna(0).gt(0).astype("int8")

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
            "lesson_media_route_coverage": build_route_coverage(
                lesson_sessions,
                lesson_sessions["_lesson_media_merge"] == "both",
                "Lesson media -> lessons -> course_id",
            ),
            "group_media_route_coverage": build_route_coverage(
                group_sessions,
                group_sessions["_group_media_merge"] == "both",
                "Group media -> groups/lessons -> course_id",
            ),
            "users_course_route_coverage": build_route_coverage(
                resolved,
                resolved["_media_users_course_merge"] == "both",
                "Resolved media -> users_courses_base",
            ),
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
# Stats module block
# =============================================================================


def build_stats_module_features(
    stats_tables: dict[str, pd.DataFrame],
    users_courses_base: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Resolve stats__module_* tables to users_course_id and aggregate conservatively."""
    base_keys = users_courses_base[[CORE_ENTITY_KEY, "user_id", "course_id"]].copy()
    module_prefix_map = {
        "stats__module_1": "stats_m1",
        "stats__module_2": "stats_m2",
        "stats__module_3": "stats_m3",
        "stats__module_4": "stats_m4",
    }

    route_reports: list[dict[str, Any]] = []
    aggregated_blocks: list[pd.DataFrame] = []
    feature_list: list[str] = []

    shared_numeric_cols = [
        "lessons_viewed_count",
        "lessons_viewed_80pct_count",
        "lessons_watched_count",
        "content_viewed_units",
        "final_tasks_solved_count",
        "interim_assessment_score",
    ]
    shared_text_flag_cols = [
        "viewed_80pct_lessons_or_video_flag",
        "viewed_720_video_units_and_80pct_lessons_flag",
        "attended_live_lesson_flag",
        "all_required_final_tasks_solved_flag",
        "current_control_passed_flag",
        "interim_assessment_passed_flag",
        "reflection_passed_flag",
        "final_assessment_passed_flag",
    ]
    shared_text_cols = ["track_name", "level_name", "module_status"]
    shared_date_cols = ["enrollment_date", "interim_assessment_submitted_at_msk"]

    for table_name, prefix in module_prefix_map.items():
        if table_name not in stats_tables:
            continue

        module_df = stats_tables[table_name].copy()
        resolved = module_df.merge(
            base_keys,
            on=["user_id", "course_id"],
            how="left",
            validate="m:1",
        )
        route_reports.append(
            build_route_coverage(
                source_df=resolved,
                matched_mask=resolved[CORE_ENTITY_KEY].notna(),
                route_name=f"{table_name}_to_users_course_id",
            )
        )

        matched = resolved.loc[resolved[CORE_ENTITY_KEY].notna()].copy()
        if matched.empty:
            continue

        for column in shared_text_flag_cols:
            if column in matched.columns:
                normalized = _normalize_text_state(matched[column])
                matched[f"{column}_binary"] = normalized.isin(
                    ["да", "завершил", "сдал", "пройден", "пройдена"]
                ).astype("int8")
                matched[f"{column}_negative_binary"] = normalized.isin(
                    ["нет", "не сдавал", "отчислен"]
                ).astype("int8")

        if "module_status" in matched.columns:
            status_norm = _normalize_text_state(matched["module_status"])
            matched["module_status_completed_binary"] = status_norm.eq("завершил").astype("int8")
            matched["module_status_dropout_binary"] = status_norm.eq("отчислен").astype("int8")

        agg_map: dict[str, tuple[str, str]] = {
            f"{prefix}_row_count": ("user_id", "count"),
            f"{prefix}_teacher_nunique": ("teacher_id", "nunique"),
            f"{prefix}_parallel_nunique": ("parallel_id", "nunique"),
        }

        for column in shared_numeric_cols:
            if column in matched.columns:
                agg_map[f"{prefix}_{column}"] = (column, "max")

        for column in shared_text_cols:
            if column in matched.columns:
                agg_map[f"{prefix}_{column}"] = (column, "first")

        for column in shared_date_cols:
            if column in matched.columns:
                agg_map[f"{prefix}_{column}"] = (column, "max")

        for column in shared_text_flag_cols:
            binary_col = f"{column}_binary"
            negative_binary_col = f"{column}_negative_binary"
            positive_feature_name = f"{prefix}_{column}" if column.endswith("_flag") else f"{prefix}_{column}_flag"
            if column.endswith("_flag"):
                negative_feature_name = f"{prefix}_{column[:-5]}_negative_flag"
            else:
                negative_feature_name = f"{prefix}_{column}_negative_flag"
            if binary_col in matched.columns:
                agg_map[positive_feature_name] = (binary_col, "max")
            if negative_binary_col in matched.columns:
                agg_map[negative_feature_name] = (negative_binary_col, "max")

        if "module_status_completed_binary" in matched.columns:
            agg_map[f"{prefix}_module_completed_flag"] = ("module_status_completed_binary", "max")
        if "module_status_dropout_binary" in matched.columns:
            agg_map[f"{prefix}_module_dropout_flag"] = ("module_status_dropout_binary", "max")

        aggregated = matched.groupby(CORE_ENTITY_KEY, as_index=False).agg(**agg_map)
        aggregated_blocks.append(aggregated)
        feature_list.extend([column for column in aggregated.columns if column != CORE_ENTITY_KEY])

    if aggregated_blocks:
        result = aggregated_blocks[0]
        for block in aggregated_blocks[1:]:
            result = result.merge(block, on=CORE_ENTITY_KEY, how="outer", validate="1:1")
    else:
        result = pd.DataFrame({CORE_ENTITY_KEY: pd.Series(dtype="Int64")})

    feature_list = list(dict.fromkeys(feature_list))
    route_report_df = pd.DataFrame(route_reports)
    summary = _row_summary(
        block_name="stats_module_features",
        source_tables=list(module_prefix_map.keys()),
        df=result,
        key_cols=[CORE_ENTITY_KEY],
        feature_list=feature_list,
        coverage_notes=(
            "Each stats__module_* table is first resolved through (user_id, course_id) -> users_course_id "
            "and only then aggregated to one row per users_course_id."
        ),
        important_warnings=(
            "stats__module_4 currently does not resolve to the current users_courses base and is kept only in route diagnostics unless matches appear."
        ),
        extra={
            "module_route_reports": route_report_df.to_dict(orient="records"),
        },
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
