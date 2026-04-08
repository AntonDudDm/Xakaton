"""User-answer aggregation block for AGENTS_NEW."""

from __future__ import annotations

from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .service_utils import (
    build_homework_course_map,
    build_training_course_map,
    load_homeworks_source_data,
    load_lessons_source_data,
    load_trainings_source_data,
    load_user_source_data,
)
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_block_coverage, validate_unique_key


def build_user_answers_block(base_result=None, show_output: bool = True):
    """Build or load the user-answer feature block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "user_answers_agg_AGENT.csv"
    bridge_path = OUT_DIR / "user_answers_bridge_validation_AGENT.csv"

    from_cache = artifact_exists(table_path) and artifact_exists(bridge_path)
    if from_cache:
        answers_df = read_csv(table_path)
        bridge_validation = read_csv(bridge_path)
    else:
        if base_result is None:
            base_result = build_base_entity_block(show_output=False)
        lessons = load_lessons_source_data()
        trainings = load_trainings_source_data()
        homeworks = load_homeworks_source_data()
        training_course_map = build_training_course_map(trainings, lessons)
        homework_course_map = build_homework_course_map(homeworks, lessons)
        _, agent_ids = load_user_source_data()
        with legacy_output_context() as legacy:
            answers_df, bridge_validation = legacy.build_user_answers_agg(
                base=base_result.data,
                lessons=lessons,
                training_course_map=training_course_map,
                homework_course_map=homework_course_map,
                agent_ids=agent_ids,
            )

    if base_result is None:
        base_result = build_base_entity_block(show_output=False)

    key_validation = validate_unique_key(answers_df, PRIMARY_KEYS["user_answers"], "user_answers")
    coverage_validation = validate_block_coverage(base_result.data, answers_df, PRIMARY_KEYS["user_answers"], "user_answers")
    plots = [
        plot_histogram(
            answers_df["answers_count"],
            title="Answer events per enrollment",
            xlabel="answers_count",
            filename="user_answers_count_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="User-answer block",
        data=answers_df,
        key_column=PRIMARY_KEYS["user_answers"],
        from_cache=from_cache,
        validation=coverage_validation,
        extra_tables={
            "Key validation": key_validation,
            "Answer bridge validation": bridge_validation,
        },
        artifact_paths={"table": table_path, "bridge_validation": bridge_path},
        plots=plots,
        notes="Maps answers through resource_type/resource_id instead of the ambiguous task_id path.",
    )
    if show_output:
        display_block_result(result)
    return result
