"""Media-session aggregation block for AGENTS_NEW."""

from __future__ import annotations

from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .service_utils import build_group_course_map, load_groups_source_data, load_lessons_source_data, load_user_source_data
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_block_coverage, validate_unique_key


def build_media_sessions_block(base_result=None, show_output: bool = True):
    """Build or load the media-session feature block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "media_sessions_agg_AGENT.csv"
    bridge_path = OUT_DIR / "media_sessions_bridge_validation_AGENT.csv"

    from_cache = artifact_exists(table_path) and artifact_exists(bridge_path)
    if from_cache:
        media_df = read_csv(table_path)
        bridge_validation = read_csv(bridge_path)
    else:
        if base_result is None:
            base_result = build_base_entity_block(show_output=False)
        lessons = load_lessons_source_data()
        groups = load_groups_source_data()
        group_course_map = build_group_course_map(groups, lessons)
        _, agent_ids = load_user_source_data()
        with legacy_output_context() as legacy:
            media_df, bridge_validation = legacy.build_media_sessions_agg(
                base=base_result.data,
                lessons=lessons,
                group_course_map=group_course_map,
                agent_ids=agent_ids,
            )

    if base_result is None:
        base_result = build_base_entity_block(show_output=False)

    key_validation = validate_unique_key(media_df, PRIMARY_KEYS["media_sessions"], "media_sessions")
    coverage_validation = validate_block_coverage(base_result.data, media_df, PRIMARY_KEYS["media_sessions"], "media_sessions")
    plots = [
        plot_histogram(
            media_df["media_sessions_count"],
            title="Media sessions per enrollment",
            xlabel="media_sessions_count",
            filename="media_sessions_count_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="Media-session block",
        data=media_df,
        key_column=PRIMARY_KEYS["media_sessions"],
        from_cache=from_cache,
        validation=coverage_validation,
        extra_tables={
            "Key validation": key_validation,
            "Media bridge validation": bridge_validation,
        },
        artifact_paths={"table": table_path, "bridge_validation": bridge_path},
        plots=plots,
        notes="Resolves media sessions through lesson/group resources before aggregating view-depth features.",
    )
    if show_output:
        display_block_result(result)
    return result
