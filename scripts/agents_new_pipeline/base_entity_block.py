"""Base entity block for the AGENTS_NEW modular pipeline."""

from __future__ import annotations

from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .service_utils import load_user_source_data
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_unique_key


def build_base_entity_block(show_output: bool = True):
    """Build or load the core `users_course_id` base table."""

    ensure_output_dirs()
    table_path = OUT_DIR / "users_courses_base_AGENT.csv"
    validation_path = OUT_DIR / "users_courses_base_validation_AGENT.csv"

    from_cache = artifact_exists(table_path) and artifact_exists(validation_path)
    if from_cache:
        base_df = read_csv(table_path)
        validation = read_csv(validation_path)
    else:
        _, agent_ids = load_user_source_data()
        with legacy_output_context() as legacy:
            base_df, validation = legacy.load_users_courses_base(agent_ids)

    key_validation = validate_unique_key(base_df, PRIMARY_KEYS["base_entity"], "base_entity")
    plots = [
        plot_histogram(
            base_df["user_course_points_ratio"],
            title="User-course points ratio",
            xlabel="user_course_points_ratio",
            filename="base_points_ratio_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="Base entity block",
        data=base_df,
        key_column=PRIMARY_KEYS["base_entity"],
        from_cache=from_cache,
        validation=key_validation,
        extra_tables={"Legacy validation": validation},
        artifact_paths={"table": table_path, "validation": validation_path},
        plots=plots,
        notes="Keeps one row per enrollment and preserves access/start anchors without target construction.",
    )
    if show_output:
        display_block_result(result)
    return result
