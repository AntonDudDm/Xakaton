from pathlib import Path


# =============================================================================
# Paths and constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_AGENT_DIR = PROJECT_ROOT / "data" / "AGENT"
TABLES_DIR = DATA_AGENT_DIR / "tables"
SUMMARIES_DIR = DATA_AGENT_DIR / "summaries"
FIGURES_DIR = DATA_AGENT_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

CORE_ENTITY_KEY = "users_course_id"
USER_KEY = "user_id"
COURSE_KEY = "course_id"

EXPORT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# Raw files
# =============================================================================

FILES = {
    "users_courses": "users_courses.csv",
    "users": "users.csv",
    "lessons": "lessons.csv",
    "lesson_tasks": "lesson_tasks.csv",
    "trainings": "trainings.csv",
    "user_lessons": "user_lessons.csv",
    "user_trainings": "user_trainings.csv",
    "user_answers": "user_answers.csv",
    "wk_users_courses_actions": "wk_users_courses_actions.csv",
    "wk_media_view_sessions": "wk_media_view_sessions.csv",
    "user_access_histories": "user_access_histories.csv",
    "user_award_badges": "user_award_badges.csv",
    "award_badges": "award_badges.csv",
    "groups": "groups.csv",
    "homeworks": "homeworks.csv",
    "homework_items": "homework_items.csv",
    "stats__module_1": "stats__module_1.csv",
    "stats__module_2": "stats__module_2.csv",
    "stats__module_3": "stats__module_3.csv",
    "stats__module_4": "stats__module_4.csv",
}


# =============================================================================
# Explicit renaming for stats__module_* tables
# =============================================================================

STATS_COLUMN_RENAME_MAP = {
    "stats__module_1": {
        "\u041a\u0440\u0443\u0436\u043e\u043a": "track_name",
        "\u0414\u0430\u0442\u0430 \u0437\u0430\u0447\u0438\u0441\u043b\u0435\u043d\u0438\u044f": "enrollment_date",
        "id \u043f\u0430\u0440\u0430\u043b\u043b\u0435\u043b\u0438": "parallel_id",
        "\u0423\u0440\u043e\u0432\u0435\u043d\u044c": "level_name",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043b \u0443\u0440\u043e\u043a\u043e\u0432": "lessons_viewed_count",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e \u043a\u043e\u043d\u0442\u0435\u043d\u0442\u0430": "content_viewed_units",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e 80% \u0443\u0440 \u0438\u043b\u0438 \u0432\u0438\u0434\u0435\u043e\u043a\u043e\u043d\u0442": "viewed_80pct_lessons_or_video_flag",
        "\u041f\u043e\u0441\u0435\u0442\u0438\u043b \u0443\u0440\u043e\u043a \u0432 \u043e\u043d\u043b\u0430\u0439\u043d\u0435": "attended_live_lesson_flag",
        "\u0420\u0435\u0448\u0435\u043d\u043e \u0418\u0417": "final_tasks_solved_count",
        "\u0420\u0435\u0448\u0435\u043d\u044b \u0432\u0441\u0435 \u043e\u0431\u044f\u0437.\u0418\u0417": "all_required_final_tasks_solved_flag",
        "\u041f\u0440\u043e\u0439\u0434\u0435\u043d \u0442\u0435\u043a.\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c": "current_control_passed_flag",
        "\u0411\u0430\u043b\u043b \u041f\u0410": "interim_assessment_score",
        "\u0421\u0434\u0430\u043b \u041f\u0410": "interim_assessment_passed_flag",
        "\u0414\u0430\u0442\u0430 \u0441\u0434\u0430\u0447\u0438 \u041f\u0410 (\u041c\u0421\u041a)": "interim_assessment_submitted_at_msk",
        "\u0421\u0442\u0430\u0442\u0443\u0441": "module_status",
    },
    "stats__module_2": {
        "\u041a\u0440\u0443\u0436\u043e\u043a": "track_name",
        "id \u043f\u0430\u0440\u0430\u043b\u043b\u0435\u043b\u0438": "parallel_id",
        "\u0423\u0440\u043e\u0432\u0435\u043d\u044c": "level_name",
        "\u041f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043b \u0443\u0440\u043e\u043a\u043e\u0432 \u043d\u0430 80%": "lessons_viewed_80pct_count",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e \u043a\u043e\u043d\u0442\u0435\u043d\u0442\u0430 (\u0435\u0434)": "content_viewed_units",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e 720\u0435\u0434 \u0432\u0438\u0434\u0435\u043e\u043a\u043e\u043d\u0442 \u0438 80% \u0443\u0440 ": "viewed_720_video_units_and_80pct_lessons_flag",
        "\u0421\u043c\u043e\u0442\u0440\u0435\u043b \u0443\u0440\u043e\u043a\u043e\u0432": "lessons_watched_count",
        "\u041f\u043e\u0441\u0435\u0442\u0438\u043b \u0443\u0440\u043e\u043a \u0432 \u043e\u043d\u043b\u0430\u0439\u043d\u0435": "attended_live_lesson_flag",
        "\u0420\u0435\u0448\u0435\u043d\u043e \u0418\u0417": "final_tasks_solved_count",
        "\u0420\u0435\u0448\u0435\u043d\u044b \u0432\u0441\u0435 \u043e\u0431\u044f\u0437.\u0418\u0417": "all_required_final_tasks_solved_flag",
        "\u041f\u0440\u043e\u0439\u0434\u0435\u043d \u0442\u0435\u043a.\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c": "current_control_passed_flag",
        "\u0411\u0430\u043b\u043b \u041f\u0410": "interim_assessment_score",
        "\u0421\u0434\u0430\u043b \u041f\u0410": "interim_assessment_passed_flag",
        "\u0414\u0430\u0442\u0430 \u0441\u0434\u0430\u0447\u0438 \u041f\u0410 (\u041c\u0421\u041a)": "interim_assessment_submitted_at_msk",
        "\u041f\u0440\u043e\u0439\u0434\u0435\u043d\u0430 \u0440\u0435\u0444\u043b\u0435\u043a\u0441\u0438\u044f": "reflection_passed_flag",
        "\u0421\u0442\u0430\u0442\u0443\u0441": "module_status",
    },
    "stats__module_3": {
        "\u041a\u0440\u0443\u0436\u043e\u043a": "track_name",
        "id \u043f\u0430\u0440\u0430\u043b\u043b\u0435\u043b\u0438": "parallel_id",
        "\u041f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043b \u0443\u0440\u043e\u043a\u043e\u0432 \u043d\u0430 80%": "lessons_viewed_80pct_count",
        "\u0421\u043c\u043e\u0442\u0440\u0435\u043b \u0443\u0440\u043e\u043a\u043e\u0432": "lessons_watched_count",
        "\u041f\u043e\u0441\u0435\u0442\u0438\u043b \u0443\u0440\u043e\u043a \u0432 \u043e\u043d\u043b\u0430\u0439\u043d\u0435": "attended_live_lesson_flag",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e \u043a\u043e\u043d\u0442\u0435\u043d\u0442\u0430 (\u0435\u0434)": "content_viewed_units",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e 720\u0435\u0434 \u0432\u0438\u0434\u0435\u043e\u043a\u043e\u043d\u0442 \u0438 80% \u0443\u0440 ": "viewed_720_video_units_and_80pct_lessons_flag",
        "\u0420\u0435\u0448\u0435\u043d\u043e \u0418\u0417": "final_tasks_solved_count",
        "\u0420\u0435\u0448\u0435\u043d\u044b \u0432\u0441\u0435 \u043e\u0431\u044f\u0437.\u0418\u0417": "all_required_final_tasks_solved_flag",
        "\u041f\u0440\u043e\u0439\u0434\u0435\u043d \u0442\u0435\u043a.\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c": "current_control_passed_flag",
        "\u0411\u0430\u043b\u043b \u041f\u0410": "interim_assessment_score",
        "\u0421\u0434\u0430\u043b \u041f\u0410": "interim_assessment_passed_flag",
        "\u0414\u0430\u0442\u0430 \u0441\u0434\u0430\u0447\u0438 \u041f\u0410 (\u041c\u0421\u041a)": "interim_assessment_submitted_at_msk",
        "\u0423\u0440\u043e\u0432\u0435\u043d\u044c": "level_name",
        "\u041f\u0440\u043e\u0439\u0434\u0435\u043d\u0430 \u0440\u0435\u0444\u043b\u0435\u043a\u0441\u0438\u044f": "reflection_passed_flag",
    },
    "stats__module_4": {
        "\u041a\u0440\u0443\u0436\u043e\u043a": "track_name",
        "id \u043f\u0430\u0440\u0430\u043b\u043b\u0435\u043b\u0438": "parallel_id",
        "\u041f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043b \u0443\u0440\u043e\u043a\u043e\u0432 \u043d\u0430 80%": "lessons_viewed_80pct_count",
        "\u0421\u043c\u043e\u0442\u0440\u0435\u043b \u0443\u0440\u043e\u043a\u043e\u0432": "lessons_watched_count",
        "\u041f\u043e\u0441\u0435\u0442\u0438\u043b \u0443\u0440\u043e\u043a \u0432 \u043e\u043d\u043b\u0430\u0439\u043d\u0435": "attended_live_lesson_flag",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e \u043a\u043e\u043d\u0442\u0435\u043d\u0442\u0430 (\u0435\u0434)": "content_viewed_units",
        "\u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u043d\u043e 720\u0435\u0434 \u0432\u0438\u0434\u0435\u043e\u043a\u043e\u043d\u0442 \u0438 80% \u0443\u0440 ": "viewed_720_video_units_and_80pct_lessons_flag",
        "\u0420\u0435\u0448\u0435\u043d\u043e \u0418\u0417": "final_tasks_solved_count",
        "\u0420\u0435\u0448\u0435\u043d\u044b \u0432\u0441\u0435 \u043e\u0431\u044f\u0437.\u0418\u0417": "all_required_final_tasks_solved_flag",
        "\u041f\u0440\u043e\u0439\u0434\u0435\u043d \u0442\u0435\u043a.\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c": "current_control_passed_flag",
        "\u0421\u0434\u0430\u043b \u041f\u0410": "interim_assessment_passed_flag",
        "\u0423\u0440\u043e\u0432\u0435\u043d\u044c": "level_name",
        "\u041f\u0440\u043e\u0439\u0434\u0435\u043d\u0430 \u0440\u0435\u0444\u043b\u0435\u043a\u0441\u0438\u044f": "reflection_passed_flag",
        "\u0421\u0434\u0430\u043b \u0418\u0410": "final_assessment_passed_flag",
    },
}


# =============================================================================
# Working column selection
# =============================================================================

RAW_USECOLS_MAP = {
    "users_courses": [
        "id",
        "user_id",
        "course_id",
        "state",
        "created_at",
        "updated_at",
        "access_finished_at",
        "wk_points",
        "wk_max_points",
        "wk_max_viewable_lessons",
        "wk_max_task_count",
        "wk_officially_started_at",
        "wk_course_completed_at",
    ],
    "users": [
        "id",
        "created_at",
        "updated_at",
        "type",
        "sign_in_count",
        "subscribed",
        "grade_id",
        "timezone",
        "grade_changed_at",
        "d_wk_school_id",
        "d_wk_municipal_id",
        "d_wk_region_id",
        "wk_gender",
    ],
    "lessons": [
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
        "wk_attendance_tracking_disabled_at",
    ],
    "lesson_tasks": [
        "id",
        "lesson_id",
        "task_id",
        "position",
        "task_required",
    ],
    "trainings": [
        "id",
        "name",
        "difficulty",
        "published_at",
        "lesson_id",
        "task_templates_count",
    ],
    "user_lessons": [
        "user_id",
        "lesson_id",
        "video_visited",
        "translation_visited",
        "users_course_id",
        "solved",
        "solved_tasks_count",
        "wk_points",
        "video_viewed",
        "wk_solved_task_count",
    ],
    "user_trainings": [
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
        "mark_saved_at",
    ],
    "user_answers": [
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
    "wk_users_courses_actions": [
        "user_id",
        "users_course_id",
        "sourceable_id",
        "action",
        "created_at",
        "updated_at",
        "lesson_id",
    ],
    "wk_media_view_sessions": [
        "resource_type",
        "resource_id",
        "viewer_id",
        "segments_total",
        "viewed_segments_count",
        "started_at",
        "kind",
    ],
    "user_access_histories": [
        "users_course_id",
        "access_started_at",
        "access_expired_at",
        "activator_class",
    ],
    "user_award_badges": [
        "award_badge_id",
        "user_id",
        "created_at",
    ],
    "award_badges": [
        "id",
        "name",
        "title",
        "level",
        "quota",
        "special",
    ],
    "groups": [
        "id",
        "lesson_id",
        "teacher_id",
        "starts_at",
        "duration",
        "state",
        "video_available",
        "wk_actual_started_at",
        "wk_actual_finished_at",
        "wk_duration_actual",
    ],
    "homeworks": [
        "id",
        "resource_type",
        "resource_id",
        "homework_type",
    ],
    "homework_items": [
        "id",
        "homework_id",
        "resource_type",
        "resource_id",
        "position",
    ],
}


# =============================================================================
# Column typing configuration
# =============================================================================

DATE_COLS = {
    "users_courses": [
        "created_at",
        "updated_at",
        "access_finished_at",
        "wk_officially_started_at",
        "wk_course_completed_at",
    ],
    "users": [
        "created_at",
        "updated_at",
        "grade_changed_at",
    ],
    "lessons": ["wk_attendance_tracking_disabled_at"],
    "trainings": ["published_at"],
    "user_trainings": ["started_at", "finished_at", "mark_saved_at"],
    "user_answers": ["submitted_at"],
    "wk_users_courses_actions": ["created_at", "updated_at"],
    "wk_media_view_sessions": ["started_at"],
    "user_access_histories": ["access_started_at", "access_expired_at"],
    "user_award_badges": ["created_at"],
    "groups": ["starts_at", "wk_actual_started_at", "wk_actual_finished_at"],
    "stats__module_1": ["enrollment_date", "interim_assessment_submitted_at_msk"],
    "stats__module_2": ["interim_assessment_submitted_at_msk"],
    "stats__module_3": ["interim_assessment_submitted_at_msk"],
    "stats__module_4": [],
}

ID_COLS_MAP = {
    "users_courses": ["id", "user_id", "course_id"],
    "users": ["id", "grade_id", "d_wk_school_id", "d_wk_municipal_id", "d_wk_region_id"],
    "lessons": ["id", "course_id"],
    "lesson_tasks": ["id", "lesson_id", "task_id", "position"],
    "trainings": ["id", "lesson_id", "difficulty", "task_templates_count"],
    "user_lessons": ["user_id", "lesson_id", "users_course_id", "solved_tasks_count", "wk_solved_task_count"],
    "user_trainings": [
        "user_id",
        "training_id",
        "solved_tasks_count",
        "earned_points",
        "submitted_answers_count",
        "attempts",
        "mark",
    ],
    "user_answers": ["user_id", "task_id", "attempts", "max_attempts", "resource_id"],
    "wk_users_courses_actions": ["user_id", "users_course_id", "sourceable_id", "lesson_id"],
    "wk_media_view_sessions": ["resource_id", "viewer_id", "segments_total", "viewed_segments_count"],
    "user_access_histories": ["users_course_id"],
    "user_award_badges": ["award_badge_id", "user_id"],
    "award_badges": ["id", "level", "quota"],
    "groups": ["id", "lesson_id", "teacher_id", "duration", "wk_duration_actual"],
    "homeworks": ["id", "resource_id"],
    "homework_items": ["id", "homework_id", "resource_id", "position"],
    "stats__module_1": ["user_id", "teacher_id", "parallel_id", "course_id"],
    "stats__module_2": ["user_id", "teacher_id", "course_id", "parallel_id"],
    "stats__module_3": ["user_id", "teacher_id", "course_id", "parallel_id"],
    "stats__module_4": ["user_id", "teacher_id", "course_id", "parallel_id"],
}

BOOL_COLS_MAP = {
    "users": ["subscribed"],
    "lessons": [
        "conspect_expected",
        "task_expected",
        "wk_survival_training_expected",
        "wk_scratch_playground_enabled",
        "wk_attendance_tracking_enabled",
    ],
    "lesson_tasks": ["task_required"],
    "user_lessons": ["video_visited", "translation_visited", "solved", "video_viewed"],
    "user_answers": ["solved", "skipped", "wk_partial_answer"],
    "award_badges": ["special"],
    "groups": ["video_available"],
}

CATEGORY_COLS_MAP = {
    "users_courses": ["state"],
    "users": ["type", "timezone", "wk_gender"],
    "user_trainings": ["type", "state"],
    "user_answers": ["resource_type", "async_check_status"],
    "wk_users_courses_actions": ["action"],
    "wk_media_view_sessions": ["resource_type", "kind"],
    "user_access_histories": ["activator_class"],
    "award_badges": ["name", "title"],
    "groups": ["state"],
    "homeworks": ["resource_type", "homework_type"],
    "homework_items": ["resource_type"],
    "stats__module_1": [
        "track_name",
        "level_name",
        "viewed_80pct_lessons_or_video_flag",
        "attended_live_lesson_flag",
        "all_required_final_tasks_solved_flag",
        "current_control_passed_flag",
        "interim_assessment_passed_flag",
        "module_status",
    ],
    "stats__module_2": [
        "track_name",
        "level_name",
        "viewed_720_video_units_and_80pct_lessons_flag",
        "attended_live_lesson_flag",
        "all_required_final_tasks_solved_flag",
        "current_control_passed_flag",
        "interim_assessment_passed_flag",
        "reflection_passed_flag",
        "module_status",
    ],
    "stats__module_3": [
        "track_name",
        "attended_live_lesson_flag",
        "viewed_720_video_units_and_80pct_lessons_flag",
        "all_required_final_tasks_solved_flag",
        "current_control_passed_flag",
        "interim_assessment_passed_flag",
        "level_name",
        "reflection_passed_flag",
    ],
    "stats__module_4": [
        "track_name",
        "attended_live_lesson_flag",
        "viewed_720_video_units_and_80pct_lessons_flag",
        "all_required_final_tasks_solved_flag",
        "current_control_passed_flag",
        "interim_assessment_passed_flag",
        "level_name",
        "reflection_passed_flag",
        "final_assessment_passed_flag",
    ],
}

FLOAT_COLS_MAP = {
    "users_courses": ["wk_points", "wk_max_points", "wk_max_viewable_lessons", "wk_max_task_count"],
    "lessons": ["lesson_number", "wk_max_points", "wk_task_count", "wk_video_duration"],
    "user_lessons": ["wk_points"],
    "user_trainings": ["mark"],
    "user_answers": ["points"],
    "stats__module_1": [
        "lessons_viewed_count",
        "content_viewed_units",
        "final_tasks_solved_count",
        "interim_assessment_score",
    ],
    "stats__module_2": [
        "lessons_viewed_80pct_count",
        "content_viewed_units",
        "lessons_watched_count",
        "final_tasks_solved_count",
        "interim_assessment_score",
    ],
    "stats__module_3": [
        "lessons_viewed_80pct_count",
        "lessons_watched_count",
        "content_viewed_units",
        "final_tasks_solved_count",
        "interim_assessment_score",
    ],
    "stats__module_4": [
        "lessons_viewed_80pct_count",
        "lessons_watched_count",
        "content_viewed_units",
        "final_tasks_solved_count",
    ],
}


# =============================================================================
# Service columns and export names
# =============================================================================

DROP_COLS_MAP = {
    "users": [],
    "user_answers": [],
}

BLOCK_EXPORT_NAMES = {
    "users_courses_base": "users_courses_base_AGENT",
    "user_features": "user_features_AGENT",
    "course_features": "course_features_AGENT",
    "user_lessons": "user_lessons_agg_AGENT",
    "user_trainings": "user_trainings_agg_AGENT",
    "user_answers": "user_answers_agg_AGENT",
    "course_actions": "course_actions_agg_AGENT",
    "media": "media_sessions_agg_AGENT",
    "access": "access_history_agg_AGENT",
    "stats": "stats_module_features_AGENT",
    "final_master": "final_user_course_features_AGENT",
}
