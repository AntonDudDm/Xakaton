"""User-lesson aggregation block for AGENTS_NEW."""

from __future__ import annotations

from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .service_utils import load_lessons_source_data, load_user_source_data
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_block_coverage, validate_unique_key


def build_user_lessons_block(base_result=None, show_output: bool = True):
    """Build or load the user-lesson feature block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "user_lessons_agg_AGENT.csv"
    validation_path = OUT_DIR / "user_lessons_agg_validation_AGENT.csv"

    from_cache = artifact_exists(table_path) and artifact_exists(validation_path)
    if from_cache:
        lessons_df = read_csv(table_path)
        validation = read_csv(validation_path)
    else:
        lessons_meta = load_lessons_source_data()
        _, agent_ids = load_user_source_data()
        with legacy_output_context() as legacy:
            lessons_df, validation = legacy.build_user_lessons_agg(lessons_meta, agent_ids)

    if base_result is None:
        base_result = build_base_entity_block(show_output=False)

    key_validation = validate_unique_key(lessons_df, PRIMARY_KEYS["user_lessons"], "user_lessons")
    coverage_validation = validate_block_coverage(base_result.data, lessons_df, PRIMARY_KEYS["user_lessons"], "user_lessons")
    plots = [
        plot_histogram(
            lessons_df["unique_lessons_touched"],
            title="Unique lessons touched per enrollment",
            xlabel="unique_lessons_touched",
            filename="user_lessons_unique_touched_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="User-lesson block",
        data=lessons_df,
        key_column=PRIMARY_KEYS["user_lessons"],
        from_cache=from_cache,
        validation=coverage_validation,
        extra_tables={
            "Key validation": key_validation,
            "Legacy block validation": validation,
        },
        artifact_paths={"table": table_path, "validation": validation_path},
        plots=plots,
        notes="Aggregates lesson visits, solved lessons, lesson points, and partial progression signals at enrollment level.",
    )
    if show_output:
        display_block_result(result)
    return result
