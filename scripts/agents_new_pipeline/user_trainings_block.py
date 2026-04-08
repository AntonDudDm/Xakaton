"""User-training aggregation block for AGENTS_NEW."""

from __future__ import annotations

from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .service_utils import build_training_course_map, load_lessons_source_data, load_trainings_source_data, load_user_source_data
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_block_coverage, validate_unique_key


def build_user_trainings_block(base_result=None, show_output: bool = True):
    """Build or load the user-training feature block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "user_trainings_agg_AGENT.csv"
    bridge_path = OUT_DIR / "user_trainings_bridge_validation_AGENT.csv"

    from_cache = artifact_exists(table_path) and artifact_exists(bridge_path)
    if from_cache:
        trainings_df = read_csv(table_path)
        bridge_validation = read_csv(bridge_path)
    else:
        if base_result is None:
            base_result = build_base_entity_block(show_output=False)
        lessons = load_lessons_source_data()
        trainings = load_trainings_source_data()
        training_course_map = build_training_course_map(trainings, lessons)
        _, agent_ids = load_user_source_data()
        with legacy_output_context() as legacy:
            trainings_df, bridge_validation = legacy.build_user_trainings_agg(
                base=base_result.data,
                training_course_map=training_course_map,
                agent_ids=agent_ids,
            )

    if base_result is None:
        base_result = build_base_entity_block(show_output=False)

    key_validation = validate_unique_key(trainings_df, PRIMARY_KEYS["user_trainings"], "user_trainings")
    coverage_validation = validate_block_coverage(base_result.data, trainings_df, PRIMARY_KEYS["user_trainings"], "user_trainings")
    plots = [
        plot_histogram(
            trainings_df["trainings_started_count"],
            title="Trainings started per enrollment",
            xlabel="trainings_started_count",
            filename="user_trainings_started_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="User-training block",
        data=trainings_df,
        key_column=PRIMARY_KEYS["user_trainings"],
        from_cache=from_cache,
        validation=coverage_validation,
        extra_tables={
            "Key validation": key_validation,
            "Training bridge validation": bridge_validation,
        },
        artifact_paths={"table": table_path, "bridge_validation": bridge_path},
        plots=plots,
        notes="Uses the explicit training -> lesson -> course path before aggregating attempts, marks, and timing features.",
    )
    if show_output:
        display_block_result(result)
    return result
