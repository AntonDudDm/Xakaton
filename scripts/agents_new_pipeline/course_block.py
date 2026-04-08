"""Course-level feature block for AGENTS_NEW."""

from __future__ import annotations

from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .service_utils import (
    load_groups_source_data,
    load_lesson_tasks_source_data,
    load_lessons_source_data,
    load_trainings_source_data,
)
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_unique_key


def build_course_block(base_result=None, show_output: bool = True):
    """Build or load the course-level structural feature block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "course_features_AGENT.csv"
    validation_path = OUT_DIR / "course_reference_validation_AGENT.csv"

    from_cache = artifact_exists(table_path) and artifact_exists(validation_path)
    if from_cache:
        course_df = read_csv(table_path)
        validation = read_csv(validation_path)
    else:
        if base_result is None:
            base_result = build_base_entity_block(show_output=False)
        lessons = load_lessons_source_data()
        lesson_tasks = load_lesson_tasks_source_data()
        groups = load_groups_source_data()
        trainings = load_trainings_source_data()
        with legacy_output_context() as legacy:
            course_df, validation = legacy.build_course_features(
                base=base_result.data,
                lessons=lessons,
                lesson_tasks=lesson_tasks,
                groups=groups,
                trainings=trainings,
            )

    key_validation = validate_unique_key(course_df, PRIMARY_KEYS["course_features"], "course_features")
    plots = [
        plot_histogram(
            course_df["lessons_count"],
            title="Lessons per course",
            xlabel="lessons_count",
            filename="course_lessons_count_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="Course feature block",
        data=course_df,
        key_column=PRIMARY_KEYS["course_features"],
        from_cache=from_cache,
        validation=key_validation,
        extra_tables={"Course structure reconciliation": validation.head(12)},
        artifact_paths={"table": table_path, "validation": validation_path},
        plots=plots,
        notes="Aggregates lesson, task, training, and webinar structure before enrollment-level merges.",
    )
    if show_output:
        display_block_result(result)
    return result
