from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "AGENT"
FIG_DIR = OUT_DIR / "figures"


RAW_TABLES = {
    "users": "users.csv",
    "users_courses": "users_courses.csv",
    "lessons": "lessons.csv",
    "lesson_tasks": "lesson_tasks.csv",
    "groups": "groups.csv",
    "trainings": "trainings.csv",
    "user_lessons": "user_lessons.csv",
    "user_trainings": "user_trainings.csv",
    "homeworks": "homeworks.csv",
    "homework_items": "homework_items.csv",
    "user_answers": "user_answers.csv",
    "wk_users_courses_actions": "wk_users_courses_actions.csv",
    "wk_media_view_sessions": "wk_media_view_sessions.csv",
    "user_access_histories": "user_access_histories.csv",
    "user_award_badges": "user_award_badges.csv",
    "award_badges": "award_badges.csv",
    "user_activity_histories": "user_activity_histories.csv",
}


ENTITY_MAP_ROWS = [
    {
        "table_name": "users",
        "represents": "Platform user profile",
        "native_grain": "one row per user",
        "candidate_key": "id",
        "target_grain": "user",
        "merge_strategy": "Merge directly on user_id after agent filtering",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "users_courses",
        "represents": "Course enrollment",
        "native_grain": "one row per user-course enrollment",
        "candidate_key": "id",
        "target_grain": "user-course",
        "merge_strategy": "Core entity; keep one row per users_course_id",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "lessons",
        "represents": "Course lesson structure",
        "native_grain": "one row per lesson",
        "candidate_key": "id",
        "target_grain": "course",
        "merge_strategy": "Aggregate to course_id before merging",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "lesson_tasks",
        "represents": "Task-to-lesson links",
        "native_grain": "one row per lesson-task link",
        "candidate_key": "(lesson_id, task_id, position)",
        "target_grain": "course",
        "merge_strategy": "Aggregate via lesson_id -> course_id",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "groups",
        "represents": "Webinar/group schedule",
        "native_grain": "one row per group",
        "candidate_key": "id",
        "target_grain": "course",
        "merge_strategy": "Aggregate via lesson_id -> course_id",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "trainings",
        "represents": "Training metadata",
        "native_grain": "one row per training template",
        "candidate_key": "id",
        "target_grain": "course",
        "merge_strategy": "Aggregate via lesson_id -> course_id",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "user_lessons",
        "represents": "User progress on lesson",
        "native_grain": "one row per user-lesson state",
        "candidate_key": "(users_course_id, lesson_id)",
        "target_grain": "user-course",
        "merge_strategy": "Aggregate on users_course_id",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "user_trainings",
        "represents": "User training attempts",
        "native_grain": "one row per user-training attempt",
        "candidate_key": "(user_id, training_id, started_at)",
        "target_grain": "user-course",
        "merge_strategy": "Resolve course via trainings -> lessons -> course_id, then map by (user_id, course_id)",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "user_answers",
        "represents": "Task answer events",
        "native_grain": "one row per submitted answer",
        "candidate_key": "event-like, no clean natural key in export",
        "target_grain": "user-course",
        "merge_strategy": "Resolve course by resource_type/resource_id before aggregation",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "wk_users_courses_actions",
        "represents": "User-course action log",
        "native_grain": "one row per action",
        "candidate_key": "event-like",
        "target_grain": "user-course",
        "merge_strategy": "Aggregate directly on users_course_id",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "wk_media_view_sessions",
        "represents": "Media sessions",
        "native_grain": "one row per media session",
        "candidate_key": "event-like",
        "target_grain": "user-course",
        "merge_strategy": "Resolve course via Lesson/Group resources, then map by (viewer_id, course_id)",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "user_access_histories",
        "represents": "Access window history",
        "native_grain": "one row per access interval event",
        "candidate_key": "(users_course_id, access_started_at, access_expired_at)",
        "target_grain": "user-course",
        "merge_strategy": "Aggregate directly on users_course_id",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "user_award_badges",
        "represents": "Assigned badge event",
        "native_grain": "one row per badge assignment",
        "candidate_key": "(user_id, award_badge_id, created_at)",
        "target_grain": "user",
        "merge_strategy": "Aggregate on user_id; use as user-level enrichment",
        "use_in_pipeline": "yes",
    },
    {
        "table_name": "award_badges",
        "represents": "Badge dictionary",
        "native_grain": "one row per badge type",
        "candidate_key": "id",
        "target_grain": "dictionary",
        "merge_strategy": "Reference only; not used as standalone feature block",
        "use_in_pipeline": "reference_only",
    },
    {
        "table_name": "user_activity_histories",
        "represents": "Alternative activity log",
        "native_grain": "one row per activity event",
        "candidate_key": "event-like",
        "target_grain": "excluded",
        "merge_strategy": "Audited but excluded from the main feature pipeline",
        "use_in_pipeline": "excluded",
    },
]


COUNT_FILL_SUFFIXES = (
    "_count",
    "_total",
    "_sum",
    "_days",
    "_flag",
    "_share",
    "_ratio",
    "_mean",
    "_max",
    "_min",
    "_median",
    "_span_days",
    "_gap_days",
    "_duration_days",
    "_duration_minutes",
    "_hours",
)


def ensure_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def raw_path(table_name: str) -> Path:
    return RAW_DIR / RAW_TABLES[table_name]


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def to_int_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).astype("Int64")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype("int8")
    normalized = series.astype("string").str.lower()
    return normalized.isin(["true", "1", "t", "yes", "y"]).astype("int8").rename(series.name)


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def safe_divide(num: pd.Series | float, den: pd.Series | float) -> pd.Series:
    num_series = pd.Series(num) if not isinstance(num, pd.Series) else num.astype(float)
    den_series = pd.Series(den) if not isinstance(den, pd.Series) else den.astype(float)
    result = num_series / den_series.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan)


def days_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 86400.0


def hours_between(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / 3600.0


def coalesce_datetime_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in columns:
        if col in df.columns:
            result = result.fillna(df[col])
    return result


def markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    return df.head(max_rows).to_markdown(index=False)


def finalize_partial_aggregate(
    partials: list[pd.DataFrame],
    key: str,
    sum_cols: list[str] | None = None,
    min_cols: list[str] | None = None,
    max_cols: list[str] | None = None,
) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame({key: pd.Series(dtype="Int64")})
    combined = pd.concat(partials, ignore_index=True)
    agg_map: dict[str, str] = {}
    for col in sum_cols or []:
        agg_map[col] = "sum"
    for col in min_cols or []:
        agg_map[col] = "min"
    for col in max_cols or []:
        agg_map[col] = "max"
    if not agg_map:
        return combined.drop_duplicates(subset=[key]).reset_index(drop=True)
    return combined.groupby(key, as_index=False).agg(agg_map)


def build_raw_table_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    missingness_rows = []

    for table_name, filename in RAW_TABLES.items():
        path = RAW_DIR / filename
        sample = pd.read_csv(path, nrows=5000, low_memory=False)
        row_count = 0
        for chunk in pd.read_csv(path, usecols=[sample.columns[0]], chunksize=300000, low_memory=False):
            row_count += len(chunk)
        rows.append(
            {
                "table_name": table_name,
                "file_name": filename,
                "row_count": row_count,
                "column_count": len(sample.columns),
                "file_size_mb": round(path.stat().st_size / (1024 ** 2), 2),
                "estimated_memory_mb": round(
                    sample.memory_usage(deep=True).sum() / max(len(sample), 1) * row_count / (1024 ** 2),
                    2,
                ),
                "id_like_columns": ", ".join([c for c in sample.columns if c == "id" or c.endswith("_id")]),
                "date_like_columns": ", ".join([c for c in sample.columns if c.endswith("_at") or c.endswith("_date")]),
                "table_role": "event" if row_count > 500000 else "entity_or_small_event",
            }
        )
        sample_missing = (
            sample.isna()
            .mean()
            .rename("missing_share")
            .reset_index()
            .rename(columns={"index": "column_name"})
        )
        sample_missing["table_name"] = table_name
        missingness_rows.append(sample_missing)

    audit = pd.DataFrame(rows).sort_values("row_count", ascending=False).reset_index(drop=True)
    missingness = (
        pd.concat(missingness_rows, ignore_index=True)
        .sort_values(["missing_share", "table_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    save_csv(audit, "raw_table_audit_AGENT.csv")
    save_csv(missingness, "raw_missingness_sample_AGENT.csv")
    return audit, missingness


def build_candidate_key_diagnostics() -> pd.DataFrame:
    specs = [
        ("users", ["id"]),
        ("users_courses", ["id"]),
        ("users_courses", ["user_id", "course_id"]),
        ("lessons", ["id"]),
        ("lesson_tasks", ["lesson_id", "task_id", "position"]),
        ("groups", ["id"]),
        ("trainings", ["id"]),
        ("user_lessons", ["users_course_id", "lesson_id"]),
        ("user_access_histories", ["users_course_id", "access_started_at", "access_expired_at"]),
        ("user_award_badges", ["user_id", "award_badge_id", "created_at"]),
    ]
    rows = []
    for table_name, key_cols in specs:
        df = pd.read_csv(raw_path(table_name), usecols=key_cols, low_memory=False)
        for col in key_cols:
            if col == "id" or col.endswith("_id"):
                df[col] = to_int_id(df[col])
            elif col.endswith("_at"):
                df[col] = to_datetime(df[col])
        rows.append(
            {
                "table_name": table_name,
                "candidate_key": ", ".join(key_cols),
                "row_count": len(df),
                "null_key_rows": int(df[key_cols].isna().any(axis=1).sum()),
                "duplicate_rows_on_key": int(df.duplicated(key_cols).sum()),
                "unique_key_rows": int(df.drop_duplicates(key_cols).shape[0]),
                "is_unique_key": int(df.duplicated(key_cols).sum() == 0),
            }
        )
    result = pd.DataFrame(rows).sort_values(["is_unique_key", "duplicate_rows_on_key", "table_name"])
    save_csv(result, "candidate_key_diagnostics_AGENT.csv")
    return result


def load_users() -> tuple[pd.DataFrame, set[int]]:
    users = pd.read_csv(
        raw_path("users"),
        usecols=[
            "id",
            "created_at",
            "type",
            "sign_in_count",
            "grade_id",
            "subscribed",
            "d_wk_school_id",
            "d_wk_municipal_id",
            "d_wk_region_id",
            "wk_gender",
        ],
        low_memory=False,
    )
    users["user_id"] = to_int_id(users["id"])
    users["user_created_at"] = to_datetime(users["created_at"])
    users["sign_in_count"] = to_numeric(users["sign_in_count"]).fillna(0)
    users["grade_id"] = to_int_id(users["grade_id"])
    users["subscribed_flag"] = to_bool(users["subscribed"])
    users["school_id"] = to_int_id(users["d_wk_school_id"])
    users["municipal_id"] = to_int_id(users["d_wk_municipal_id"])
    users["region_id"] = to_int_id(users["d_wk_region_id"])
    users["is_agent_flag"] = users["type"].astype("string").str.contains("Agent", na=False).astype("int8")
    agent_ids = set(users.loc[users["is_agent_flag"] == 1, "user_id"].dropna().astype(int))
    users = users.loc[users["is_agent_flag"] == 0].copy()
    users["has_school_id_flag"] = users["school_id"].notna().astype("int8")
    users["has_municipal_id_flag"] = users["municipal_id"].notna().astype("int8")
    users["has_region_id_flag"] = users["region_id"].notna().astype("int8")
    users["gender_known_flag"] = users["wk_gender"].notna().astype("int8")
    user_features = users[
        [
            "user_id",
            "user_created_at",
            "sign_in_count",
            "grade_id",
            "subscribed_flag",
            "school_id",
            "municipal_id",
            "region_id",
            "wk_gender",
            "has_school_id_flag",
            "has_municipal_id_flag",
            "has_region_id_flag",
            "gender_known_flag",
        ]
    ].drop_duplicates("user_id")
    return user_features, agent_ids


def load_users_courses_base(agent_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(
        raw_path("users_courses"),
        usecols=[
            "id",
            "user_id",
            "course_id",
            "state",
            "created_at",
            "access_finished_at",
            "wk_points",
            "wk_max_points",
            "wk_max_viewable_lessons",
            "wk_max_task_count",
            "wk_officially_started_at",
            "wk_course_completed_at",
        ],
        low_memory=False,
    )
    base["users_course_id"] = to_int_id(base["id"])
    base["user_id"] = to_int_id(base["user_id"])
    base["course_id"] = to_int_id(base["course_id"])
    for col in ["wk_points", "wk_max_points", "wk_max_viewable_lessons", "wk_max_task_count"]:
        base[col] = to_numeric(base[col])
    base["created_at"] = to_datetime(base["created_at"])
    base["access_finished_at"] = to_datetime(base["access_finished_at"])
    base["wk_officially_started_at"] = to_datetime(base["wk_officially_started_at"])
    base["wk_course_completed_at"] = to_datetime(base["wk_course_completed_at"])
    base = base.loc[~base["user_id"].isin(agent_ids)].copy()
    base["course_started_flag"] = base["wk_officially_started_at"].notna().astype("int8")
    base["course_completed_flag"] = base["wk_course_completed_at"].notna().astype("int8")
    base["access_is_active_flag"] = (base["state"].astype("string") == "active").astype("int8")
    base["user_course_points_ratio"] = safe_divide(base["wk_points"], base["wk_max_points"])
    base["course_anchor_at"] = coalesce_datetime_columns(base, ["wk_officially_started_at", "created_at"])
    base["days_created_to_anchor"] = days_between(base["course_anchor_at"], base["created_at"])
    base["days_anchor_to_access_end"] = days_between(base["access_finished_at"], base["course_anchor_at"])
    base["days_anchor_to_completion"] = days_between(base["wk_course_completed_at"], base["course_anchor_at"])
    base["max_points_missing_flag"] = base["wk_max_points"].isna().astype("int8")
    base["max_lessons_missing_flag"] = base["wk_max_viewable_lessons"].isna().astype("int8")
    base["max_tasks_missing_flag"] = base["wk_max_task_count"].isna().astype("int8")
    base = base.drop(columns=["id"])
    validation = pd.DataFrame(
        [
            {"metric": "users_course_id_unique", "value": int(base["users_course_id"].nunique() == len(base))},
            {
                "metric": "user_course_pair_unique",
                "value": int(base[["user_id", "course_id"]].drop_duplicates().shape[0] == len(base)),
            },
            {"metric": "rows_after_agent_filter", "value": int(len(base))},
        ]
    )
    save_csv(base.sort_values("users_course_id"), "users_courses_base_AGENT.csv")
    save_csv(validation, "users_courses_base_validation_AGENT.csv")
    return base.sort_values("users_course_id").reset_index(drop=True), validation


def load_lessons() -> pd.DataFrame:
    lessons = pd.read_csv(
        raw_path("lessons"),
        usecols=[
            "id",
            "course_id",
            "conspect_expected",
            "task_expected",
            "lesson_number",
            "wk_max_points",
            "wk_task_count",
            "wk_survival_training_expected",
            "wk_scratch_playground_enabled",
            "wk_attendance_tracking_enabled",
            "wk_video_duration",
        ],
        low_memory=False,
    )
    lessons["lesson_id"] = to_int_id(lessons["id"])
    lessons["course_id"] = to_int_id(lessons["course_id"])
    lessons["lesson_number"] = to_numeric(lessons["lesson_number"])
    lessons["wk_max_points"] = to_numeric(lessons["wk_max_points"])
    lessons["wk_task_count"] = to_numeric(lessons["wk_task_count"])
    lessons["wk_video_duration"] = to_numeric(lessons["wk_video_duration"])
    for col in [
        "conspect_expected",
        "task_expected",
        "wk_survival_training_expected",
        "wk_scratch_playground_enabled",
        "wk_attendance_tracking_enabled",
    ]:
        lessons[col] = to_bool(lessons[col])
    lessons["video_available_flag"] = lessons["wk_video_duration"].fillna(0).gt(0).astype("int8")
    lessons["lesson_number_available_flag"] = lessons["lesson_number"].notna().astype("int8")
    return lessons


def load_lesson_tasks() -> pd.DataFrame:
    lesson_tasks = pd.read_csv(
        raw_path("lesson_tasks"),
        usecols=["lesson_id", "task_id", "task_required", "position"],
        low_memory=False,
    )
    lesson_tasks["lesson_id"] = to_int_id(lesson_tasks["lesson_id"])
    lesson_tasks["task_id"] = to_int_id(lesson_tasks["task_id"])
    lesson_tasks["position"] = to_numeric(lesson_tasks["position"])
    lesson_tasks["task_required"] = to_bool(lesson_tasks["task_required"])
    return lesson_tasks


def load_groups() -> pd.DataFrame:
    groups = pd.read_csv(
        raw_path("groups"),
        usecols=[
            "id",
            "lesson_id",
            "starts_at",
            "duration",
            "state",
            "video_available",
            "pupils_notified_at",
            "wk_actual_started_at",
            "wk_actual_finished_at",
        ],
        low_memory=False,
    )
    groups["group_id"] = to_int_id(groups["id"])
    groups["lesson_id"] = to_int_id(groups["lesson_id"])
    groups["starts_at"] = to_datetime(groups["starts_at"])
    groups["duration"] = to_numeric(groups["duration"])
    groups["video_available_flag"] = to_bool(groups["video_available"])
    groups["pupils_notified_at"] = to_datetime(groups["pupils_notified_at"])
    groups["wk_actual_started_at"] = to_datetime(groups["wk_actual_started_at"])
    groups["wk_actual_finished_at"] = to_datetime(groups["wk_actual_finished_at"])
    groups["finished_group_flag"] = (groups["state"].astype("string") == "finished").astype("int8")
    groups["started_group_flag"] = groups["state"].astype("string").isin(["started", "finished"]).astype("int8")
    groups["actual_duration_minutes"] = (
        groups["wk_actual_finished_at"] - groups["wk_actual_started_at"]
    ).dt.total_seconds() / 60.0
    groups["planned_to_actual_ratio"] = safe_divide(groups["actual_duration_minutes"], groups["duration"])
    groups["notification_lead_hours"] = hours_between(groups["starts_at"], groups["pupils_notified_at"])
    return groups


def load_trainings() -> pd.DataFrame:
    trainings = pd.read_csv(
        raw_path("trainings"),
        usecols=["id", "lesson_id", "time_limit", "published_at", "difficulty", "task_templates_count"],
        low_memory=False,
    )
    trainings["training_id"] = to_int_id(trainings["id"])
    trainings["lesson_id"] = to_int_id(trainings["lesson_id"])
    trainings["time_limit"] = to_numeric(trainings["time_limit"])
    trainings["difficulty"] = to_numeric(trainings["difficulty"])
    trainings["task_templates_count"] = to_numeric(trainings["task_templates_count"])
    trainings["published_at"] = to_datetime(trainings["published_at"])
    trainings["published_flag"] = trainings["published_at"].notna().astype("int8")
    return trainings


def load_homeworks() -> pd.DataFrame:
    homeworks = pd.read_csv(
        raw_path("homeworks"),
        usecols=["id", "resource_type", "resource_id", "homework_type"],
        low_memory=False,
    )
    homeworks["homework_id"] = to_int_id(homeworks["id"])
    homeworks["resource_id"] = to_int_id(homeworks["resource_id"])
    homeworks["homework_type"] = to_numeric(homeworks["homework_type"])
    return homeworks


def build_course_features(
    base: pd.DataFrame,
    lessons: pd.DataFrame,
    lesson_tasks: pd.DataFrame,
    groups: pd.DataFrame,
    trainings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesson_features = (
        lessons.groupby("course_id", as_index=False)
        .agg(
            lessons_count=("lesson_id", "nunique"),
            tasks_total_from_lessons=("wk_task_count", "sum"),
            max_points_total_from_lessons=("wk_max_points", "sum"),
            total_video_duration=("wk_video_duration", "sum"),
            average_video_duration=("wk_video_duration", "mean"),
            median_video_duration=("wk_video_duration", "median"),
            share_of_lessons_with_tasks=("task_expected", "mean"),
            share_of_lessons_with_conspect=("conspect_expected", "mean"),
            share_of_lessons_with_survival_training=("wk_survival_training_expected", "mean"),
            share_of_lessons_with_scratch=("wk_scratch_playground_enabled", "mean"),
            share_of_lessons_with_attendance_tracking=("wk_attendance_tracking_enabled", "mean"),
            video_available_share=("video_available_flag", "mean"),
            lesson_number_available_share=("lesson_number_available_flag", "mean"),
            course_min_lesson_number=("lesson_number", "min"),
            course_max_lesson_number=("lesson_number", "max"),
        )
    )

    task_map = lesson_tasks.merge(lessons[["lesson_id", "course_id"]], on="lesson_id", how="left")
    lesson_task_features = (
        task_map.groupby("course_id", as_index=False)
        .agg(
            lesson_task_links_total=("task_id", "size"),
            unique_tasks_total=("task_id", "nunique"),
            required_tasks_total=("task_required", "sum"),
        )
    )
    lesson_task_features["required_task_share"] = safe_divide(
        lesson_task_features["required_tasks_total"],
        lesson_task_features["lesson_task_links_total"],
    )

    training_map = trainings.merge(lessons[["lesson_id", "course_id"]], on="lesson_id", how="left")
    training_features = (
        training_map.groupby("course_id", as_index=False)
        .agg(
            trainings_total=("training_id", "nunique"),
            published_trainings_count=("published_flag", "sum"),
            training_time_limit_mean=("time_limit", "mean"),
            training_time_limit_max=("time_limit", "max"),
            trainings_difficulty_mean=("difficulty", "mean"),
            training_task_templates_total=("task_templates_count", "sum"),
            training_task_templates_mean=("task_templates_count", "mean"),
            lessons_with_trainings_count=("lesson_id", "nunique"),
        )
    )

    group_map = groups.merge(lessons[["lesson_id", "course_id"]], on="lesson_id", how="left")
    group_features = (
        group_map.groupby("course_id", as_index=False)
        .agg(
            groups_count_per_course=("group_id", "nunique"),
            planned_webinar_duration_total=("duration", "sum"),
            planned_webinar_duration_mean=("duration", "mean"),
            video_available_share_groups=("video_available_flag", "mean"),
            finished_webinar_count=("finished_group_flag", "sum"),
            started_webinar_count=("started_group_flag", "sum"),
            actual_webinar_duration_mean=("actual_duration_minutes", "mean"),
            actual_webinar_duration_max=("actual_duration_minutes", "max"),
            actual_vs_planned_duration_ratio_mean=("planned_to_actual_ratio", "mean"),
            webinar_notification_lead_hours_mean=("notification_lead_hours", "mean"),
        )
    )

    course_features = lesson_features.merge(lesson_task_features, on="course_id", how="left")
    course_features = course_features.merge(training_features, on="course_id", how="left")
    course_features = course_features.merge(group_features, on="course_id", how="left")
    course_features["tasks_per_lesson_mean"] = safe_divide(
        course_features["tasks_total_from_lessons"],
        course_features["lessons_count"],
    )
    course_features["groups_per_lesson_mean"] = safe_divide(
        course_features["groups_count_per_course"],
        course_features["lessons_count"],
    )
    course_features["trainings_per_lesson_mean"] = safe_divide(
        course_features["trainings_total"],
        course_features["lessons_count"],
    )
    course_features["lessons_with_trainings_share"] = safe_divide(
        course_features["lessons_with_trainings_count"],
        course_features["lessons_count"],
    )
    course_features["published_trainings_share"] = safe_divide(
        course_features["published_trainings_count"],
        course_features["trainings_total"],
    )

    course_reference = (
        base.groupby("course_id", as_index=False)
        .agg(
            uc_wk_max_points_median=("wk_max_points", "median"),
            uc_wk_max_viewable_lessons_median=("wk_max_viewable_lessons", "median"),
            uc_wk_max_task_count_median=("wk_max_task_count", "median"),
            enrollment_rows=("users_course_id", "size"),
        )
    )
    course_validation = course_reference.merge(course_features, on="course_id", how="left")
    course_validation["max_points_delta"] = (
        course_validation["max_points_total_from_lessons"] - course_validation["uc_wk_max_points_median"]
    )
    course_validation["lesson_count_delta"] = (
        course_validation["lessons_count"] - course_validation["uc_wk_max_viewable_lessons_median"]
    )
    course_validation["task_count_delta"] = (
        course_validation["tasks_total_from_lessons"] - course_validation["uc_wk_max_task_count_median"]
    )

    save_csv(course_features.sort_values("course_id"), "course_features_AGENT.csv")
    save_csv(course_validation.sort_values("course_id"), "course_reference_validation_AGENT.csv")
    return course_features.sort_values("course_id").reset_index(drop=True), course_validation


def build_user_features(user_features: pd.DataFrame, agent_ids: set[int]) -> pd.DataFrame:
    badges = pd.read_csv(
        raw_path("user_award_badges"),
        usecols=["user_id", "award_badge_id", "created_at"],
        low_memory=False,
    )
    badges["user_id"] = to_int_id(badges["user_id"])
    badges["award_badge_id"] = to_int_id(badges["award_badge_id"])
    badges["created_at"] = to_datetime(badges["created_at"])
    badges = badges.loc[~badges["user_id"].isin(agent_ids)].copy()
    badge_agg = (
        badges.groupby("user_id", as_index=False)
        .agg(
            badge_events_total=("award_badge_id", "size"),
            unique_badges_total=("award_badge_id", "nunique"),
            first_badge_at=("created_at", "min"),
            last_badge_at=("created_at", "max"),
        )
    )
    badge_ids = sorted(badges["award_badge_id"].dropna().unique().astype(int))
    for badge_id in badge_ids:
        per_badge = (
            badges.loc[badges["award_badge_id"] == badge_id, ["user_id"]]
            .drop_duplicates()
            .assign(**{f"has_badge_{badge_id}_flag": 1})
        )
        badge_agg = badge_agg.merge(per_badge, on="user_id", how="left")
    badge_cols = [c for c in badge_agg.columns if c.endswith("_flag")]
    if badge_cols:
        badge_agg[badge_cols] = badge_agg[badge_cols].fillna(0).astype("int8")
    badge_agg["badge_activity_span_days"] = days_between(
        badge_agg["last_badge_at"],
        badge_agg["first_badge_at"],
    )

    result = user_features.merge(badge_agg, on="user_id", how="left")
    save_csv(result.sort_values("user_id"), "user_features_AGENT.csv")
    return result.sort_values("user_id").reset_index(drop=True)


def build_training_course_map(trainings: pd.DataFrame, lessons: pd.DataFrame) -> pd.DataFrame:
    return trainings.merge(
        lessons[["lesson_id", "course_id", "lesson_number"]],
        on="lesson_id",
        how="left",
    )[["training_id", "lesson_id", "course_id", "lesson_number"]]


def build_group_course_map(groups: pd.DataFrame, lessons: pd.DataFrame) -> pd.DataFrame:
    return groups.merge(
        lessons[["lesson_id", "course_id"]],
        on="lesson_id",
        how="left",
    )[["group_id", "lesson_id", "course_id"]]


def build_homework_course_map(homeworks: pd.DataFrame, lessons: pd.DataFrame) -> pd.DataFrame:
    lesson_lookup = lessons[["lesson_id", "course_id", "lesson_number"]].rename(columns={"lesson_id": "resource_id"})
    return homeworks.merge(lesson_lookup, on="resource_id", how="left")[["homework_id", "course_id", "lesson_number", "resource_type"]]


def build_merge_validation(
    base: pd.DataFrame,
    merged: pd.DataFrame,
    block_name: str,
    new_columns: list[str],
) -> dict[str, float | int | str]:
    matched_mask = merged[new_columns].notna().any(axis=1) if new_columns else pd.Series(False, index=merged.index)
    return {
        "block_name": block_name,
        "rows_before_merge": int(len(base)),
        "rows_after_merge": int(len(merged)),
        "users_course_id_duplicate_rows": int(merged.duplicated(["users_course_id"]).sum()),
        "matched_rows": int(matched_mask.sum()),
        "unmatched_rows": int((~matched_mask).sum()),
        "coverage_share": round(float(matched_mask.mean()), 4),
        "new_columns_count": len(new_columns),
        "mean_new_column_null_share": round(float(merged[new_columns].isna().mean().mean()), 4) if new_columns else 0.0,
        "max_new_column_null_share": round(float(merged[new_columns].isna().mean().max()), 4) if new_columns else 0.0,
    }


def build_user_lessons_agg(
    lessons: pd.DataFrame,
    agent_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesson_meta = lessons[["lesson_id", "course_id", "lesson_number"]]
    partials: list[pd.DataFrame] = []
    lesson_pairs_parts: list[pd.DataFrame] = []
    group_pairs_parts: list[pd.DataFrame] = []
    stats = {"source_rows": 0}

    for chunk in pd.read_csv(
        raw_path("user_lessons"),
        usecols=[
            "user_id",
            "lesson_id",
            "group_id",
            "video_visited",
            "translation_visited",
            "users_course_id",
            "solved",
            "solved_tasks_count",
            "wk_points",
            "video_viewed",
            "wk_solved_task_count",
        ],
        chunksize=400000,
        low_memory=False,
    ):
        stats["source_rows"] += len(chunk)
        chunk["user_id"] = to_int_id(chunk["user_id"])
        chunk = chunk.loc[~chunk["user_id"].isin(agent_ids)].copy()
        if chunk.empty:
            continue
        chunk["lesson_id"] = to_int_id(chunk["lesson_id"])
        chunk["group_id"] = to_int_id(chunk["group_id"])
        chunk["users_course_id"] = to_int_id(chunk["users_course_id"])
        chunk["solved_tasks_count"] = to_numeric(chunk["solved_tasks_count"])
        chunk["wk_solved_task_count"] = to_numeric(chunk["wk_solved_task_count"])
        chunk["wk_points"] = to_numeric(chunk["wk_points"])
        for col in ["video_visited", "translation_visited", "solved", "video_viewed"]:
            chunk[col] = to_bool(chunk[col])
        chunk["solved_tasks_observed"] = chunk["wk_solved_task_count"].fillna(chunk["solved_tasks_count"])
        chunk = chunk.merge(lesson_meta, on="lesson_id", how="left")
        chunk["lesson_touched_flag"] = 1
        chunk["group_present_flag"] = chunk["group_id"].notna().astype("int8")
        chunk["solved_lesson_number"] = chunk["lesson_number"].where(chunk["solved"] == 1)

        partial = (
            chunk.groupby("users_course_id", as_index=False)
            .agg(
                user_lessons_rows=("lesson_touched_flag", "sum"),
                visited_video_lessons_count=("video_visited", "sum"),
                viewed_video_completion_count=("video_viewed", "sum"),
                visited_translation_lessons_count=("translation_visited", "sum"),
                solved_lessons_count=("solved", "sum"),
                solved_tasks_total=("solved_tasks_observed", "sum"),
                points_total_over_lessons=("wk_points", "sum"),
                first_lesson_number_seen=("lesson_number", "min"),
                furthest_lesson_number_reached=("lesson_number", "max"),
                first_solved_lesson_number=("solved_lesson_number", "min"),
                unique_group_rows_count=("group_present_flag", "sum"),
            )
        )
        partials.append(partial)
        lesson_pairs_parts.append(chunk[["users_course_id", "lesson_id"]].dropna().drop_duplicates())
        group_pairs_parts.append(chunk[["users_course_id", "group_id"]].dropna().drop_duplicates())

    result = finalize_partial_aggregate(
        partials,
        "users_course_id",
        sum_cols=[
            "user_lessons_rows",
            "visited_video_lessons_count",
            "viewed_video_completion_count",
            "visited_translation_lessons_count",
            "solved_lessons_count",
            "solved_tasks_total",
            "points_total_over_lessons",
            "unique_group_rows_count",
        ],
        min_cols=["first_lesson_number_seen", "first_solved_lesson_number"],
        max_cols=["furthest_lesson_number_reached"],
    )
    result["points_mean_over_lessons"] = safe_divide(
        result["points_total_over_lessons"],
        result["user_lessons_rows"],
    )
    lesson_pairs = pd.concat(lesson_pairs_parts, ignore_index=True).drop_duplicates()
    group_pairs = pd.concat(group_pairs_parts, ignore_index=True).drop_duplicates() if group_pairs_parts else pd.DataFrame(columns=["users_course_id", "group_id"])
    unique_lessons = lesson_pairs.groupby("users_course_id", as_index=False).size().rename(columns={"size": "unique_lessons_touched"})
    unique_groups = group_pairs.groupby("users_course_id", as_index=False).size().rename(columns={"size": "unique_groups_visited"})
    result = result.merge(unique_lessons, on="users_course_id", how="left")
    result = result.merge(unique_groups, on="users_course_id", how="left")
    result["progression_span_in_lessons"] = result["furthest_lesson_number_reached"] - result["first_lesson_number_seen"]
    summary = pd.DataFrame(
        [
            {
                "block_name": "user_lessons",
                "source_rows": stats["source_rows"],
                "aggregated_rows": int(len(result)),
                "is_unique_users_course_id": int(result["users_course_id"].nunique() == len(result)),
            }
        ]
    )
    save_csv(result.sort_values("users_course_id"), "user_lessons_agg_AGENT.csv")
    save_csv(summary, "user_lessons_agg_validation_AGENT.csv")
    return result.sort_values("users_course_id").reset_index(drop=True), summary


def build_user_trainings_agg(
    base: pd.DataFrame,
    training_course_map: pd.DataFrame,
    agent_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_trainings = pd.read_csv(
        raw_path("user_trainings"),
        usecols=[
            "user_id",
            "training_id",
            "solved_tasks_count",
            "earned_points",
            "type",
            "state",
            "submitted_answers_count",
            "started_at",
            "finished_at",
            "attempts",
            "mark",
        ],
        low_memory=False,
    )
    user_trainings["user_id"] = to_int_id(user_trainings["user_id"])
    user_trainings = user_trainings.loc[~user_trainings["user_id"].isin(agent_ids)].copy()
    user_trainings["training_id"] = to_int_id(user_trainings["training_id"])
    for col in ["solved_tasks_count", "earned_points", "submitted_answers_count", "attempts", "mark"]:
        user_trainings[col] = to_numeric(user_trainings[col])
    user_trainings["started_at"] = to_datetime(user_trainings["started_at"])
    user_trainings["finished_at"] = to_datetime(user_trainings["finished_at"])
    user_trainings = user_trainings.merge(training_course_map, on="training_id", how="left")
    user_trainings = user_trainings.merge(
        base[["users_course_id", "user_id", "course_id"]],
        on=["user_id", "course_id"],
        how="left",
    )

    bridge = pd.DataFrame(
        [
            {
                "block_name": "user_trainings",
                "source_rows": int(len(user_trainings)),
                "course_resolved_rows": int(user_trainings["course_id"].notna().sum()),
                "enrollment_resolved_rows": int(user_trainings["users_course_id"].notna().sum()),
                "course_resolved_share": round(float(user_trainings["course_id"].notna().mean()), 4),
                "enrollment_resolved_share": round(float(user_trainings["users_course_id"].notna().mean()), 4),
            }
        ]
    )

    resolved = user_trainings.loc[user_trainings["users_course_id"].notna()].copy()
    resolved["trainings_started_count"] = 1
    resolved["trainings_finished_flag"] = resolved["finished_at"].notna().astype("int8")
    resolved["trainings_checked_flag"] = (resolved["state"].astype("string") == "checked").astype("int8")
    resolved["lesson_training_flag"] = (resolved["type"].astype("string") == "UserTrainings::LessonTraining").astype("int8")
    resolved["regular_training_flag"] = (resolved["type"].astype("string") == "UserTrainings::RegularTraining").astype("int8")
    resolved["olympiad_training_flag"] = (resolved["type"].astype("string") == "UserTrainings::OlympiadTraining").astype("int8")
    resolved["mark_ge_4_flag"] = resolved["mark"].ge(4).fillna(False).astype("int8")

    agg = (
        resolved.groupby("users_course_id", as_index=False)
        .agg(
            trainings_started_count=("trainings_started_count", "sum"),
            trainings_finished_count=("trainings_finished_flag", "sum"),
            trainings_checked_count=("trainings_checked_flag", "sum"),
            lesson_trainings_count=("lesson_training_flag", "sum"),
            regular_trainings_count=("regular_training_flag", "sum"),
            olympiad_trainings_count=("olympiad_training_flag", "sum"),
            training_attempts_total=("attempts", "sum"),
            training_attempts_mean=("attempts", "mean"),
            submitted_answers_total_in_trainings=("submitted_answers_count", "sum"),
            solved_tasks_total_in_trainings=("solved_tasks_count", "sum"),
            earned_points_total_in_trainings=("earned_points", "sum"),
            mark_mean=("mark", "mean"),
            mark_max=("mark", "max"),
            mark_min=("mark", "min"),
            count_mark_ge_4=("mark_ge_4_flag", "sum"),
            first_training_started_at=("started_at", "min"),
            last_training_finished_at=("finished_at", "max"),
            furthest_training_lesson_number=("lesson_number", "max"),
            first_training_lesson_number=("lesson_number", "min"),
            unique_training_lessons_count=("lesson_id", "nunique"),
            unique_training_templates_count=("training_id", "nunique"),
        )
    )
    agg["passed_training_ratio"] = safe_divide(agg["count_mark_ge_4"], agg["trainings_finished_count"])
    agg["training_activity_span_days"] = days_between(agg["last_training_finished_at"], agg["first_training_started_at"])

    save_csv(agg.sort_values("users_course_id"), "user_trainings_agg_AGENT.csv")
    save_csv(bridge, "user_trainings_bridge_validation_AGENT.csv")
    return agg.sort_values("users_course_id").reset_index(drop=True), bridge


def build_user_answers_agg(
    base: pd.DataFrame,
    lessons: pd.DataFrame,
    training_course_map: pd.DataFrame,
    homework_course_map: pd.DataFrame,
    agent_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesson_resource_map = lessons[["lesson_id", "course_id", "lesson_number"]].rename(columns={"lesson_id": "resource_id"})
    training_resource_map = training_course_map.rename(columns={"training_id": "resource_id"})
    homework_resource_map = homework_course_map.rename(columns={"homework_id": "resource_id"})

    partials: list[pd.DataFrame] = []
    task_pairs_parts: list[pd.DataFrame] = []
    answer_day_parts: list[pd.DataFrame] = []
    bridge_rows: list[dict[str, float | int | str]] = []

    for chunk in pd.read_csv(
        raw_path("user_answers"),
        usecols=[
            "user_id",
            "task_id",
            "attempts",
            "solved",
            "points",
            "max_attempts",
            "skipped",
            "resource_type",
            "resource_id",
            "submitted_at",
            "wk_partial_answer",
            "async_check_status",
        ],
        chunksize=300000,
        low_memory=False,
    ):
        chunk["user_id"] = to_int_id(chunk["user_id"])
        chunk = chunk.loc[~chunk["user_id"].isin(agent_ids)].copy()
        if chunk.empty:
            continue
        chunk["task_id"] = to_int_id(chunk["task_id"])
        chunk["resource_id"] = to_int_id(chunk["resource_id"])
        for col in ["attempts", "points", "max_attempts", "async_check_status"]:
            chunk[col] = to_numeric(chunk[col])
        chunk["submitted_at"] = to_datetime(chunk["submitted_at"])
        for col in ["solved", "skipped", "wk_partial_answer"]:
            chunk[col] = to_bool(chunk[col])

        resolved_parts: list[pd.DataFrame] = []
        for resource_type, mapping in [
            ("Lesson", lesson_resource_map),
            ("Training", training_resource_map),
            ("Homework", homework_resource_map),
        ]:
            part = chunk.loc[chunk["resource_type"] == resource_type].copy()
            if part.empty:
                continue
            part = part.merge(mapping, on="resource_id", how="left")
            part["resource_type"] = resource_type
            bridge_rows.append(
                {
                    "resource_type": resource_type,
                    "rows": int(len(part)),
                    "course_resolved_rows": int(part["course_id"].notna().sum()),
                }
            )
            resolved_parts.append(part)

        if not resolved_parts:
            continue

        resolved = pd.concat(resolved_parts, ignore_index=True)
        resolved = resolved.merge(
            base[["users_course_id", "user_id", "course_id"]],
            on=["user_id", "course_id"],
            how="left",
        )
        if resolved.empty:
            continue
        resolved["course_resolved_flag"] = resolved["course_id"].notna().astype("int8")
        resolved["enrollment_resolved_flag"] = resolved["users_course_id"].notna().astype("int8")
        resolved = resolved.loc[resolved["users_course_id"].notna()].copy()
        if resolved.empty:
            continue

        resolved["answer_event_count"] = 1
        resolved["unsolved_answer_flag"] = ((resolved["solved"] == 0) & (resolved["skipped"] == 0)).astype("int8")
        resolved["partial_answer_flag"] = resolved["wk_partial_answer"]
        resolved["async_checked_flag"] = resolved["async_check_status"].eq(2).fillna(False).astype("int8")
        resolved["lesson_answer_flag"] = (resolved["resource_type"] == "Lesson").astype("int8")
        resolved["training_answer_flag"] = (resolved["resource_type"] == "Training").astype("int8")
        resolved["homework_answer_flag"] = (resolved["resource_type"] == "Homework").astype("int8")
        resolved["first_attempt_success_flag"] = (
            (resolved["solved"] == 1) & resolved["attempts"].fillna(np.inf).le(1)
        ).astype("int8")
        resolved["attempt_pressure"] = safe_divide(resolved["attempts"], resolved["max_attempts"])
        resolved["answer_day"] = resolved["submitted_at"].dt.floor("D")

        partials.append(
            resolved.groupby("users_course_id", as_index=False)
            .agg(
                answers_count=("answer_event_count", "sum"),
                solved_answers_count=("solved", "sum"),
                unsolved_answers_count=("unsolved_answer_flag", "sum"),
                skipped_count=("skipped", "sum"),
                attempts_total=("attempts", "sum"),
                points_total=("points", "sum"),
                partial_answer_count=("partial_answer_flag", "sum"),
                async_checked_count=("async_checked_flag", "sum"),
                lesson_answer_count=("lesson_answer_flag", "sum"),
                training_answer_count=("training_answer_flag", "sum"),
                homework_answer_count=("homework_answer_flag", "sum"),
                first_attempt_success_count=("first_attempt_success_flag", "sum"),
                max_attempt_pressure=("attempt_pressure", "max"),
                first_answer_at=("submitted_at", "min"),
                last_answer_at=("submitted_at", "max"),
                first_answer_lesson_number=("lesson_number", "min"),
                furthest_answer_lesson_number=("lesson_number", "max"),
            )
        )
        task_pairs_parts.append(resolved[["users_course_id", "task_id"]].dropna().drop_duplicates())
        answer_day_parts.append(resolved[["users_course_id", "answer_day"]].dropna().drop_duplicates())

    bridge = (
        pd.DataFrame(bridge_rows)
        .groupby("resource_type", as_index=False)
        .agg(rows=("rows", "sum"), course_resolved_rows=("course_resolved_rows", "sum"))
    )
    if not bridge.empty:
        bridge["course_resolved_share"] = safe_divide(bridge["course_resolved_rows"], bridge["rows"])

    result = finalize_partial_aggregate(
        partials,
        "users_course_id",
        sum_cols=[
            "answers_count",
            "solved_answers_count",
            "unsolved_answers_count",
            "skipped_count",
            "attempts_total",
            "points_total",
            "partial_answer_count",
            "async_checked_count",
            "lesson_answer_count",
            "training_answer_count",
            "homework_answer_count",
            "first_attempt_success_count",
        ],
        min_cols=["first_answer_at", "first_answer_lesson_number"],
        max_cols=["last_answer_at", "furthest_answer_lesson_number", "max_attempt_pressure"],
    )
    result["attempts_mean"] = safe_divide(result["attempts_total"], result["answers_count"])
    result["points_mean"] = safe_divide(result["points_total"], result["answers_count"])

    task_pairs = pd.concat(task_pairs_parts, ignore_index=True).drop_duplicates() if task_pairs_parts else pd.DataFrame(columns=["users_course_id", "task_id"])
    answer_days = pd.concat(answer_day_parts, ignore_index=True).drop_duplicates() if answer_day_parts else pd.DataFrame(columns=["users_course_id", "answer_day"])
    unique_tasks = task_pairs.groupby("users_course_id", as_index=False).size().rename(columns={"size": "answered_tasks_count_unique"})
    active_days = answer_days.groupby("users_course_id", as_index=False).size().rename(columns={"size": "answer_active_days_count"})
    result = result.merge(unique_tasks, on="users_course_id", how="left")
    result = result.merge(active_days, on="users_course_id", how="left")
    result["answer_activity_span_days"] = days_between(result["last_answer_at"], result["first_answer_at"])
    result["success_ratio_over_answers"] = safe_divide(result["solved_answers_count"], result["answers_count"])
    result["average_points_per_answer"] = safe_divide(result["points_total"], result["answers_count"])
    result["proportion_of_tasks_solved_on_first_attempt"] = safe_divide(
        result["first_attempt_success_count"],
        result["answers_count"],
    )

    save_csv(result.sort_values("users_course_id"), "user_answers_agg_AGENT.csv")
    save_csv(bridge, "user_answers_bridge_validation_AGENT.csv")
    return result.sort_values("users_course_id").reset_index(drop=True), bridge


def build_course_actions_agg(
    base: pd.DataFrame,
    lessons: pd.DataFrame,
    agent_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_anchor = base[["users_course_id", "course_anchor_at", "access_finished_at"]]
    lesson_lookup = lessons[["lesson_id", "lesson_number"]]
    partials: list[pd.DataFrame] = []
    lesson_pairs_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        raw_path("wk_users_courses_actions"),
        usecols=["user_id", "users_course_id", "action", "created_at", "lesson_id"],
        chunksize=400000,
        low_memory=False,
    ):
        chunk["user_id"] = to_int_id(chunk["user_id"])
        chunk = chunk.loc[~chunk["user_id"].isin(agent_ids)].copy()
        if chunk.empty:
            continue
        chunk["users_course_id"] = to_int_id(chunk["users_course_id"])
        chunk["lesson_id"] = to_int_id(chunk["lesson_id"])
        chunk["created_at"] = to_datetime(chunk["created_at"])
        chunk = chunk.merge(base_anchor, on="users_course_id", how="left")
        chunk = chunk.merge(lesson_lookup, on="lesson_id", how="left")
        chunk["action_date"] = chunk["created_at"].dt.floor("D")
        chunk["day_from_course_start"] = (
            chunk["action_date"] - chunk["course_anchor_at"].dt.floor("D")
        ).dt.days
        chunk["action_event_count"] = 1
        chunk["start_training_count"] = (chunk["action"].astype("string") == "start_training").astype("int8")
        chunk["user_answer_action_count"] = (chunk["action"].astype("string") == "user_answer").astype("int8")
        chunk["visit_translation_count"] = (chunk["action"].astype("string") == "visit_translation").astype("int8")
        chunk["visit_video_count"] = (chunk["action"].astype("string") == "visit_video").astype("int8")
        chunk["visit_preparation_material_count"] = (
            chunk["action"].astype("string") == "visit_preparation_material"
        ).astype("int8")
        chunk["scratch_playground_visit_count"] = (
            chunk["action"].astype("string") == "scratch_playground_visited"
        ).astype("int8")

        partials.append(
            chunk.groupby("users_course_id", as_index=False)
            .agg(
                actions_total=("action_event_count", "sum"),
                start_training_count=("start_training_count", "sum"),
                user_answer_action_count=("user_answer_action_count", "sum"),
                visit_translation_count=("visit_translation_count", "sum"),
                visit_video_count=("visit_video_count", "sum"),
                visit_preparation_material_count=("visit_preparation_material_count", "sum"),
                scratch_playground_visit_count=("scratch_playground_visit_count", "sum"),
                first_action_at=("created_at", "min"),
                last_action_at=("created_at", "max"),
                first_action_lesson_number=("lesson_number", "min"),
                furthest_action_lesson_number=("lesson_number", "max"),
            )
        )
        lesson_pairs_parts.append(chunk[["users_course_id", "lesson_id"]].dropna().drop_duplicates())
        daily_parts.append(
            chunk.groupby(["users_course_id", "action_date", "day_from_course_start"], as_index=False)
            .agg(actions_on_day=("action_event_count", "sum"))
        )

    result = finalize_partial_aggregate(
        partials,
        "users_course_id",
        sum_cols=[
            "actions_total",
            "start_training_count",
            "user_answer_action_count",
            "visit_translation_count",
            "visit_video_count",
            "visit_preparation_material_count",
            "scratch_playground_visit_count",
        ],
        min_cols=["first_action_at", "first_action_lesson_number"],
        max_cols=["last_action_at", "furthest_action_lesson_number"],
    )
    lesson_pairs = pd.concat(lesson_pairs_parts, ignore_index=True).drop_duplicates() if lesson_pairs_parts else pd.DataFrame(columns=["users_course_id", "lesson_id"])
    unique_lessons = lesson_pairs.groupby("users_course_id", as_index=False).size().rename(columns={"size": "action_unique_lessons_touched"})
    result = result.merge(unique_lessons, on="users_course_id", how="left")

    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(columns=["users_course_id", "action_date", "day_from_course_start", "actions_on_day"])
    if not daily.empty:
        daily = daily.groupby(["users_course_id", "action_date", "day_from_course_start"], as_index=False)["actions_on_day"].sum()
        day_gap = daily.sort_values(["users_course_id", "action_date"]).copy()
        day_gap["gap_days"] = day_gap.groupby("users_course_id")["action_date"].diff().dt.days
        peak_day = (
            daily.sort_values(["users_course_id", "actions_on_day", "day_from_course_start"], ascending=[True, False, True])
            .drop_duplicates("users_course_id")
            .rename(columns={"day_from_course_start": "days_from_course_start_to_peak_activity", "actions_on_day": "peak_actions_in_day"})
            [["users_course_id", "days_from_course_start_to_peak_activity", "peak_actions_in_day"]]
        )
        temporal = daily.groupby("users_course_id", as_index=False).agg(
            active_days_count=("action_date", "nunique"),
            actions_first_1d=("actions_on_day", lambda s: s[daily.loc[s.index, "day_from_course_start"].between(0, 1)].sum()),
            actions_first_3d=("actions_on_day", lambda s: s[daily.loc[s.index, "day_from_course_start"].between(0, 3)].sum()),
            actions_first_7d=("actions_on_day", lambda s: s[daily.loc[s.index, "day_from_course_start"].between(0, 7)].sum()),
            actions_first_14d=("actions_on_day", lambda s: s[daily.loc[s.index, "day_from_course_start"].between(0, 14)].sum()),
            actions_first_30d=("actions_on_day", lambda s: s[daily.loc[s.index, "day_from_course_start"].between(0, 30)].sum()),
            active_days_first_7d=("day_from_course_start", lambda s: s.between(0, 7).sum()),
            active_days_first_14d=("day_from_course_start", lambda s: s.between(0, 14).sum()),
            active_days_first_30d=("day_from_course_start", lambda s: s.between(0, 30).sum()),
            active_days_after_30d=("day_from_course_start", lambda s: s.gt(30).sum()),
            last_action_day_from_start=("day_from_course_start", "max"),
        )
        gap_stats = day_gap.groupby("users_course_id", as_index=False).agg(
            mean_gap_between_active_days=("gap_days", "mean"),
            max_gap_between_active_days=("gap_days", "max"),
        )
        gap_stats["longest_inactivity_streak_days"] = gap_stats["max_gap_between_active_days"].fillna(0) - 1
        result = result.merge(temporal, on="users_course_id", how="left")
        result = result.merge(gap_stats, on="users_course_id", how="left")
        result = result.merge(peak_day, on="users_course_id", how="left")
    result["action_span_days"] = days_between(result["last_action_at"], result["first_action_at"])
    result["mean_actions_per_active_day"] = safe_divide(result["actions_total"], result["active_days_count"])
    result["unique_action_types_count"] = (
        result[
            [
                "start_training_count",
                "user_answer_action_count",
                "visit_translation_count",
                "visit_video_count",
                "visit_preparation_material_count",
                "scratch_playground_visit_count",
            ]
        ]
        .gt(0)
        .sum(axis=1)
    )
    result["preparation_material_engagement_flag"] = result["visit_preparation_material_count"].fillna(0).gt(0).astype("int8")
    result["scratch_engagement_flag"] = result["scratch_playground_visit_count"].fillna(0).gt(0).astype("int8")

    coverage = pd.DataFrame(
        [
            {
                "block_name": "course_actions",
                "source_rows": int(sum(len(part) for part in partials)),
                "aggregated_rows": int(len(result)),
                "is_unique_users_course_id": int(result["users_course_id"].nunique() == len(result)),
            }
        ]
    )
    save_csv(result.sort_values("users_course_id"), "course_actions_agg_AGENT.csv")
    save_csv(coverage, "course_actions_agg_validation_AGENT.csv")
    if not daily.empty:
        save_csv(daily.sort_values(["users_course_id", "action_date"]), "course_actions_daily_profile_AGENT.csv")
    return result.sort_values("users_course_id").reset_index(drop=True), coverage, daily


def build_media_sessions_agg(
    base: pd.DataFrame,
    lessons: pd.DataFrame,
    group_course_map: pd.DataFrame,
    agent_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesson_resource_map = lessons[["lesson_id", "course_id"]].rename(columns={"lesson_id": "resource_id"})
    group_resource_map = group_course_map[["group_id", "course_id"]].rename(columns={"group_id": "resource_id"})
    media = pd.read_csv(
        raw_path("wk_media_view_sessions"),
        usecols=["resource_type", "resource_id", "viewer_id", "segments_total", "viewed_segments_count", "started_at"],
        low_memory=False,
    )
    media["viewer_id"] = to_int_id(media["viewer_id"])
    media = media.loc[~media["viewer_id"].isin(agent_ids)].copy()
    media["resource_id"] = to_int_id(media["resource_id"])
    media["segments_total"] = to_numeric(media["segments_total"])
    media["viewed_segments_count"] = to_numeric(media["viewed_segments_count"])
    media["started_at"] = to_datetime(media["started_at"])

    lesson_part = media.loc[media["resource_type"] == "Lesson"].merge(lesson_resource_map, on="resource_id", how="left")
    group_part = media.loc[media["resource_type"] == "Group"].merge(group_resource_map, on="resource_id", how="left")
    resolved = pd.concat([lesson_part, group_part], ignore_index=True)
    bridge = (
        resolved.groupby("resource_type", as_index=False)
        .agg(rows=("resource_id", "size"), course_resolved_rows=("course_id", lambda s: s.notna().sum()))
    )
    bridge["course_resolved_share"] = safe_divide(bridge["course_resolved_rows"], bridge["rows"])

    resolved = resolved.rename(columns={"viewer_id": "user_id"})
    resolved = resolved.merge(base[["users_course_id", "user_id", "course_id"]], on=["user_id", "course_id"], how="left")
    resolved = resolved.loc[resolved["users_course_id"].notna()].copy()
    resolved["media_sessions_count"] = 1
    resolved["lesson_media_sessions_count"] = (resolved["resource_type"] == "Lesson").astype("int8")
    resolved["group_media_sessions_count"] = (resolved["resource_type"] == "Group").astype("int8")
    resolved["viewed_fraction"] = safe_divide(resolved["viewed_segments_count"], resolved["segments_total"])
    resolved["fully_watched_session_flag"] = resolved["viewed_fraction"].ge(0.95).fillna(False).astype("int8")
    resolved["media_day"] = resolved["started_at"].dt.floor("D")

    agg = (
        resolved.groupby("users_course_id", as_index=False)
        .agg(
            media_sessions_count=("media_sessions_count", "sum"),
            lesson_media_sessions_count=("lesson_media_sessions_count", "sum"),
            group_media_sessions_count=("group_media_sessions_count", "sum"),
            viewed_segments_total=("viewed_segments_count", "sum"),
            viewed_segments_mean=("viewed_segments_count", "mean"),
            viewed_fraction_mean=("viewed_fraction", "mean"),
            viewed_fraction_max=("viewed_fraction", "max"),
            fully_watched_sessions_count=("fully_watched_session_flag", "sum"),
            first_media_session_at=("started_at", "min"),
            last_media_session_at=("started_at", "max"),
        )
    )
    resource_pairs = resolved[["users_course_id", "resource_type", "resource_id"]].dropna().drop_duplicates()
    day_pairs = resolved[["users_course_id", "media_day"]].dropna().drop_duplicates()
    unique_resources = resource_pairs.groupby("users_course_id", as_index=False).size().rename(columns={"size": "unique_resources_viewed"})
    media_days = day_pairs.groupby("users_course_id", as_index=False).size().rename(columns={"size": "media_active_days_count"})
    agg = agg.merge(unique_resources, on="users_course_id", how="left")
    agg = agg.merge(media_days, on="users_course_id", how="left")
    agg["media_activity_span_days"] = days_between(agg["last_media_session_at"], agg["first_media_session_at"])

    save_csv(agg.sort_values("users_course_id"), "media_sessions_agg_AGENT.csv")
    save_csv(bridge, "media_sessions_bridge_validation_AGENT.csv")
    return agg.sort_values("users_course_id").reset_index(drop=True), bridge


def build_access_history_agg(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    access = pd.read_csv(
        raw_path("user_access_histories"),
        usecols=["users_course_id", "access_started_at", "access_expired_at", "activator_class"],
        low_memory=False,
    )
    access["users_course_id"] = to_int_id(access["users_course_id"])
    access["access_started_at"] = to_datetime(access["access_started_at"])
    access["access_expired_at"] = to_datetime(access["access_expired_at"])
    access = access.merge(base[["users_course_id"]], on="users_course_id", how="inner")
    access = access.sort_values(["users_course_id", "access_started_at", "access_expired_at"]).reset_index(drop=True)
    access["access_duration_days"] = days_between(access["access_expired_at"], access["access_started_at"]) + 1
    access["premium_access_event_count"] = access["activator_class"].astype("string").str.contains("PremiumAccess", na=False).astype("int8")
    access["revoke_access_event_count"] = access["activator_class"].astype("string").str.contains("RevokeAccess", na=False).astype("int8")
    access["change_duration_event_count"] = access["activator_class"].astype("string").str.contains("ChangeAccessDuration", na=False).astype("int8")
    access["standard_access_event_count"] = access["activator_class"].astype("string").str.contains("StandardAccess", na=False).astype("int8")
    access["month_premium_access_event_count"] = access["activator_class"].astype("string").str.contains("MonthPremium", na=False).astype("int8")
    access["prev_access_expired_at"] = access.groupby("users_course_id")["access_expired_at"].shift()
    access["gap_before_period_days"] = days_between(access["access_started_at"], access["prev_access_expired_at"])

    agg = (
        access.groupby("users_course_id", as_index=False)
        .agg(
            access_periods_count=("users_course_id", "size"),
            first_access_started_at=("access_started_at", "min"),
            last_access_expired_at=("access_expired_at", "max"),
            total_access_days=("access_duration_days", "sum"),
            current_access_window_length=("access_duration_days", "last"),
            premium_access_event_count=("premium_access_event_count", "sum"),
            revoke_access_event_count=("revoke_access_event_count", "sum"),
            change_duration_event_count=("change_duration_event_count", "sum"),
            standard_access_event_count=("standard_access_event_count", "sum"),
            month_premium_access_event_count=("month_premium_access_event_count", "sum"),
            mean_access_gap_days=("gap_before_period_days", "mean"),
            max_access_gap_days=("gap_before_period_days", "max"),
        )
    )
    agg["access_reopen_flag"] = agg["access_periods_count"].gt(1).astype("int8")

    validation = pd.DataFrame(
        [
            {
                "block_name": "access_history",
                "source_rows": int(len(access)),
                "aggregated_rows": int(len(agg)),
                "is_unique_users_course_id": int(agg["users_course_id"].nunique() == len(agg)),
            }
        ]
    )
    save_csv(agg.sort_values("users_course_id"), "access_history_agg_AGENT.csv")
    save_csv(validation, "access_history_agg_validation_AGENT.csv")
    return agg.sort_values("users_course_id").reset_index(drop=True), validation


def fill_behavior_block_defaults(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        if col.endswith(("_at", "_date")):
            continue
        if any(col.endswith(suffix) for suffix in COUNT_FILL_SUFFIXES):
            df[col] = df[col].fillna(0)
    return df


def assemble_final_feature_table(
    base: pd.DataFrame,
    course_features: pd.DataFrame,
    user_features: pd.DataFrame,
    user_lessons_agg: pd.DataFrame,
    user_trainings_agg: pd.DataFrame,
    user_answers_agg: pd.DataFrame,
    course_actions_agg: pd.DataFrame,
    media_sessions_agg: pd.DataFrame,
    access_history_agg: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merge_logs: list[dict[str, float | int | str]] = []
    final = base.copy()

    for block_name, block_df in [
        ("course_features", course_features),
        ("user_features", user_features),
        ("user_lessons", user_lessons_agg),
        ("user_trainings", user_trainings_agg),
        ("user_answers", user_answers_agg),
        ("course_actions", course_actions_agg),
        ("media_sessions", media_sessions_agg),
        ("access_history", access_history_agg),
    ]:
        key = "course_id" if block_name == "course_features" else "user_id" if block_name == "user_features" else "users_course_id"
        new_cols = [c for c in block_df.columns if c != key]
        final = final.merge(block_df, on=key, how="left")
        merge_logs.append(build_merge_validation(base, final, block_name, new_cols))

    event_block_columns = {
        "user_lessons": [c for c in user_lessons_agg.columns if c != "users_course_id"],
        "user_trainings": [c for c in user_trainings_agg.columns if c != "users_course_id"],
        "user_answers": [c for c in user_answers_agg.columns if c != "users_course_id"],
        "course_actions": [c for c in course_actions_agg.columns if c != "users_course_id"],
        "media_sessions": [c for c in media_sessions_agg.columns if c != "users_course_id"],
        "access_history": [c for c in access_history_agg.columns if c != "users_course_id"],
    }
    for cols in event_block_columns.values():
        final = fill_behavior_block_defaults(final, cols)

    final["account_age_at_enrollment_days"] = days_between(final["created_at"], final["user_created_at"])
    final["account_age_at_course_start_days"] = days_between(final["course_anchor_at"], final["user_created_at"])
    final["visited_lessons_ratio_vs_course_total"] = safe_divide(final["unique_lessons_touched"], final["lessons_count"])
    final["solved_lessons_ratio_vs_course_total"] = safe_divide(final["solved_lessons_count"], final["lessons_count"])
    final["solved_tasks_ratio_vs_course_total"] = safe_divide(final["solved_tasks_total"], final["tasks_total_from_lessons"])
    final["training_progress_ratio_over_course_structure"] = safe_divide(
        final["unique_training_lessons_count"],
        final["lessons_count"],
    )
    final["answer_density_per_lesson"] = safe_divide(final["answers_count"], final["lessons_count"])
    final["media_sessions_per_lesson"] = safe_divide(final["media_sessions_count"], final["lessons_count"])
    final["actions_per_lesson"] = safe_divide(final["actions_total"], final["lessons_count"])
    final["fraction_of_course_reached_by_lesson_number"] = safe_divide(
        final["furthest_lesson_number_reached"],
        final["course_max_lesson_number"],
    )

    final["days_from_course_start_to_first_action"] = days_between(final["first_action_at"], final["course_anchor_at"])
    final["days_from_course_start_to_first_answer"] = days_between(final["first_answer_at"], final["course_anchor_at"])
    final["days_from_course_start_to_first_training"] = days_between(final["first_training_started_at"], final["course_anchor_at"])
    final["days_from_course_start_to_first_media_session"] = days_between(final["first_media_session_at"], final["course_anchor_at"])
    final["days_from_course_start_to_last_action"] = days_between(final["last_action_at"], final["course_anchor_at"])
    final["days_from_course_start_to_last_answer"] = days_between(final["last_answer_at"], final["course_anchor_at"])
    final["days_from_course_start_to_last_training"] = days_between(final["last_training_finished_at"], final["course_anchor_at"])
    final["days_from_course_start_to_last_media_session"] = days_between(final["last_media_session_at"], final["course_anchor_at"])

    first_cols = ["first_action_at", "first_answer_at", "first_training_started_at", "first_media_session_at"]
    last_cols = ["last_action_at", "last_answer_at", "last_training_finished_at", "last_media_session_at"]
    final["first_observed_activity_at"] = pd.concat([final[col] for col in first_cols], axis=1).min(axis=1)
    final["last_observed_activity_at"] = pd.concat([final[col] for col in last_cols], axis=1).max(axis=1)
    final["days_from_course_start_to_first_observed_activity"] = days_between(
        final["first_observed_activity_at"],
        final["course_anchor_at"],
    )
    final["days_from_last_observed_activity_to_access_end"] = days_between(
        final["access_finished_at"],
        final["last_observed_activity_at"],
    )
    final["overall_observed_activity_span_days"] = days_between(
        final["last_observed_activity_at"],
        final["first_observed_activity_at"],
    )
    final["early_to_late_actions_ratio_14d"] = safe_divide(
        final["actions_first_14d"],
        (final["actions_total"] - final["actions_first_14d"]).replace({0: np.nan}),
    )
    final["frontloaded_actions_share_7d"] = safe_divide(final["actions_first_7d"], final["actions_total"])
    final["frontloaded_active_days_share_14d"] = safe_divide(final["active_days_first_14d"], final["active_days_count"])
    final["days_from_peak_activity_to_last_action"] = (
        final["last_action_day_from_start"] - final["days_from_course_start_to_peak_activity"]
    )

    leakage_or_helper_cols = [
        "course_completed_flag",
        "wk_course_completed_at",
        "days_anchor_to_completion",
    ]
    dup_cols = [col for col in final.columns if "dup" in col.lower()]
    all_missing_cols = [col for col in final.columns if final[col].isna().all()]
    final = final.drop(columns=sorted(set(leakage_or_helper_cols + dup_cols + all_missing_cols)), errors="ignore")

    constant_rows = []
    for col in final.columns:
        nunique = final[col].nunique(dropna=False)
        top_share = final[col].value_counts(dropna=False, normalize=True).iloc[0]
        constant_rows.append(
            {
                "feature_name": col,
                "nunique_including_na": int(nunique),
                "top_value_share": round(float(top_share), 4),
                "is_constant": int(nunique <= 1),
                "is_near_constant_995": int(top_share >= 0.995),
            }
        )
    constant_summary = pd.DataFrame(constant_rows).sort_values(["is_constant", "top_value_share"], ascending=[False, False])
    constant_drop_cols = constant_summary.loc[
        (constant_summary["is_constant"] == 1) & (constant_summary["feature_name"] != "users_course_id"),
        "feature_name",
    ].tolist()
    final = final.drop(columns=constant_drop_cols, errors="ignore")
    missingness = (
        final.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_share")
        .reset_index()
        .rename(columns={"index": "feature_name"})
    )
    block_summary = pd.DataFrame(
        [
            {"block_name": "base", "feature_count": len(base.columns) - 3},
            {"block_name": "course_features", "feature_count": len(course_features.columns) - 1},
            {"block_name": "user_features", "feature_count": len(user_features.columns) - 1},
            {"block_name": "user_lessons", "feature_count": len(user_lessons_agg.columns) - 1},
            {"block_name": "user_trainings", "feature_count": len(user_trainings_agg.columns) - 1},
            {"block_name": "user_answers", "feature_count": len(user_answers_agg.columns) - 1},
            {"block_name": "course_actions", "feature_count": len(course_actions_agg.columns) - 1},
            {"block_name": "media_sessions", "feature_count": len(media_sessions_agg.columns) - 1},
            {"block_name": "access_history", "feature_count": len(access_history_agg.columns) - 1},
            {"block_name": "final_total", "feature_count": len(final.columns) - 1},
        ]
    )

    save_csv(pd.DataFrame(merge_logs), "merge_validation_AGENT.csv")
    save_csv(missingness, "final_feature_missingness_AGENT.csv")
    save_csv(constant_summary, "final_constant_features_AGENT.csv")
    save_csv(pd.DataFrame({"dropped_constant_feature": constant_drop_cols}), "dropped_constant_features_AGENT.csv")
    save_csv(block_summary, "feature_block_summary_AGENT.csv")
    save_csv(final.sort_values("users_course_id"), "final_user_course_features_AGENT.csv")
    return (
        final.sort_values("users_course_id").reset_index(drop=True),
        pd.DataFrame(merge_logs),
        missingness,
        block_summary,
    )


def make_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, filename: str, rotate: int = 45) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def make_hist_plot(series: pd.Series, title: str, xlabel: str, filename: str, bins: int = 40, log1p: bool = False) -> None:
    clean = series.dropna()
    if clean.empty:
        return
    plot_values = np.log1p(clean) if log1p else clean
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(plot_values, bins=bins, color="#4472c4", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(f"log1p({xlabel})" if log1p else xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def make_line_plot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, filename: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x], df[y], color="#4472c4", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def build_visual_diagnostics(
    raw_audit: pd.DataFrame,
    course_validation: pd.DataFrame,
    training_bridge: pd.DataFrame,
    answers_bridge: pd.DataFrame,
    media_bridge: pd.DataFrame,
    merge_validation: pd.DataFrame,
    actions_daily: pd.DataFrame,
    final_features: pd.DataFrame,
    final_missingness: pd.DataFrame,
) -> None:
    make_bar_plot(raw_audit, "table_name", "row_count", "Raw table row counts", "Rows", "raw_table_rows_AGENT.png", rotate=70)
    coverage_df = pd.concat(
        [
            training_bridge.assign(metric="enrollment_resolved_share")[["block_name", "enrollment_resolved_share"]].rename(columns={"block_name": "source", "enrollment_resolved_share": "coverage_share"}),
            answers_bridge[["resource_type", "course_resolved_share"]].rename(columns={"resource_type": "source", "course_resolved_share": "coverage_share"}),
            media_bridge[["resource_type", "course_resolved_share"]].rename(columns={"resource_type": "source", "course_resolved_share": "coverage_share"}),
        ],
        ignore_index=True,
    )
    if not coverage_df.empty:
        coverage_df = coverage_df.dropna()
        make_bar_plot(coverage_df, "source", "coverage_share", "Bridge coverage diagnostics", "Coverage share", "bridge_coverage_AGENT.png", rotate=30)
    if not merge_validation.empty:
        make_bar_plot(merge_validation, "block_name", "coverage_share", "Merge coverage by feature block", "Coverage share", "merge_coverage_AGENT.png", rotate=30)
    if not actions_daily.empty:
        mean_profile = (
            actions_daily.loc[actions_daily["day_from_course_start"].between(0, 60)]
            .groupby("day_from_course_start", as_index=False)["actions_on_day"]
            .mean()
            .rename(columns={"actions_on_day": "mean_actions_on_day"})
        )
        make_line_plot(
            mean_profile,
            "day_from_course_start",
            "mean_actions_on_day",
            "Mean daily action intensity from course start",
            "Mean actions per active day",
            "action_intensity_profile_AGENT.png",
        )
    make_hist_plot(
        final_features["days_from_course_start_to_first_observed_activity"],
        "Delay to first observed activity",
        "Days",
        "first_activity_delay_AGENT.png",
    )
    make_hist_plot(
        final_features["days_from_last_observed_activity_to_access_end"],
        "Recency gap from last activity to access end",
        "Days",
        "last_activity_recency_AGENT.png",
    )
    make_hist_plot(final_features["actions_total"], "Action volume distribution", "actions_total", "actions_total_AGENT.png", log1p=True)
    make_hist_plot(
        final_features["visited_lessons_ratio_vs_course_total"],
        "Visited lessons ratio",
        "visited_lessons_ratio_vs_course_total",
        "visited_lessons_ratio_AGENT.png",
    )
    make_bar_plot(
        final_missingness.head(20).sort_values("missing_share"),
        "feature_name",
        "missing_share",
        "Top 20 missing features",
        "Missing share",
        "final_missingness_top20_AGENT.png",
        rotate=85,
    )
    if not course_validation.empty:
        cv = course_validation[["uc_wk_max_points_median", "max_points_total_from_lessons"]].dropna()
        if not cv.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(cv["uc_wk_max_points_median"], cv["max_points_total_from_lessons"], alpha=0.6)
            ax.set_title("Course structure validation: max points")
            ax.set_xlabel("users_courses median max points")
            ax.set_ylabel("lessons aggregated max points")
            fig.tight_layout()
            fig.savefig(FIG_DIR / "course_points_validation_AGENT.png", dpi=150)
            plt.close(fig)


def build_all_artifacts() -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    entity_map = pd.DataFrame(ENTITY_MAP_ROWS)
    save_csv(entity_map, "entity_map_AGENT.csv")

    raw_audit, raw_missingness = build_raw_table_audit()
    key_diagnostics = build_candidate_key_diagnostics()
    user_features_base, agent_ids = load_users()
    base, base_validation = load_users_courses_base(agent_ids)
    lessons = load_lessons()
    lesson_tasks = load_lesson_tasks()
    groups = load_groups()
    trainings = load_trainings()
    homeworks = load_homeworks()

    course_features, course_validation = build_course_features(base, lessons, lesson_tasks, groups, trainings)
    user_features = build_user_features(user_features_base, agent_ids)
    training_course_map = build_training_course_map(trainings, lessons)
    group_course_map = build_group_course_map(groups, lessons)
    homework_course_map = build_homework_course_map(homeworks, lessons)

    user_lessons_agg, user_lessons_validation = build_user_lessons_agg(lessons, agent_ids)
    user_trainings_agg, training_bridge = build_user_trainings_agg(base, training_course_map, agent_ids)
    user_answers_agg, answers_bridge = build_user_answers_agg(base, lessons, training_course_map, homework_course_map, agent_ids)
    course_actions_agg, course_actions_validation, actions_daily = build_course_actions_agg(base, lessons, agent_ids)
    media_sessions_agg, media_bridge = build_media_sessions_agg(base, lessons, group_course_map, agent_ids)
    access_history_agg, access_validation = build_access_history_agg(base)

    (
        final_features,
        merge_validation,
        final_missingness,
        feature_block_summary,
    ) = assemble_final_feature_table(
        base,
        course_features,
        user_features,
        user_lessons_agg,
        user_trainings_agg,
        user_answers_agg,
        course_actions_agg,
        media_sessions_agg,
        access_history_agg,
    )

    build_visual_diagnostics(
        raw_audit,
        course_validation,
        training_bridge,
        answers_bridge,
        media_bridge,
        merge_validation,
        actions_daily,
        final_features,
        final_missingness,
    )

    artifacts = {
        "entity_map": entity_map,
        "raw_audit": raw_audit,
        "raw_missingness": raw_missingness,
        "key_diagnostics": key_diagnostics,
        "base": base,
        "base_validation": base_validation,
        "course_features": course_features,
        "course_validation": course_validation,
        "user_features": user_features,
        "user_lessons_agg": user_lessons_agg,
        "user_lessons_validation": user_lessons_validation,
        "user_trainings_agg": user_trainings_agg,
        "training_bridge": training_bridge,
        "user_answers_agg": user_answers_agg,
        "answers_bridge": answers_bridge,
        "course_actions_agg": course_actions_agg,
        "course_actions_validation": course_actions_validation,
        "actions_daily": actions_daily,
        "media_sessions_agg": media_sessions_agg,
        "media_bridge": media_bridge,
        "access_history_agg": access_history_agg,
        "access_validation": access_validation,
        "final_features": final_features,
        "merge_validation": merge_validation,
        "final_missingness": final_missingness,
        "feature_block_summary": feature_block_summary,
    }
    return artifacts


if __name__ == "__main__":
    artifacts = build_all_artifacts()
    print("Artifacts saved to:", OUT_DIR)
    print("Final feature table shape:", artifacts["final_features"].shape)
