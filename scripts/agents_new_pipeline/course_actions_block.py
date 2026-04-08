"""Course-action aggregation block for AGENTS_NEW."""

from __future__ import annotations

from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram, plot_line
from .service_utils import load_lessons_source_data, load_user_source_data
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_block_coverage, validate_unique_key


def build_course_actions_block(base_result=None, show_output: bool = True):
    """Build or load the course-action feature block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "course_actions_agg_AGENT.csv"
    validation_path = OUT_DIR / "course_actions_agg_validation_AGENT.csv"
    daily_path = OUT_DIR / "course_actions_daily_profile_AGENT.csv"

    from_cache = all(artifact_exists(path) for path in [table_path, validation_path, daily_path])
    if from_cache:
        actions_df = read_csv(table_path)
        validation = read_csv(validation_path)
        daily_df = read_csv(daily_path)
    else:
        if base_result is None:
            base_result = build_base_entity_block(show_output=False)
        lessons = load_lessons_source_data()
        _, agent_ids = load_user_source_data()
        with legacy_output_context() as legacy:
            actions_df, validation, daily_df = legacy.build_course_actions_agg(
                base=base_result.data,
                lessons=lessons,
                agent_ids=agent_ids,
            )

    if base_result is None:
        base_result = build_base_entity_block(show_output=False)

    key_validation = validate_unique_key(actions_df, PRIMARY_KEYS["course_actions"], "course_actions")
    coverage_validation = validate_block_coverage(base_result.data, actions_df, PRIMARY_KEYS["course_actions"], "course_actions")
    mean_daily = (
        daily_df.loc[daily_df["day_from_course_start"].between(0, 60)]
        .groupby("day_from_course_start", as_index=False)["actions_on_day"]
        .mean()
        .rename(columns={"actions_on_day": "mean_actions_on_day"})
    )
    plots = [
        plot_histogram(
            actions_df["actions_total"],
            title="Course actions per enrollment",
            xlabel="actions_total",
            filename="course_actions_total_AGENTS_NEW.png",
        ),
        plot_line(
            mean_daily,
            x="day_from_course_start",
            y="mean_actions_on_day",
            title="Mean daily action intensity",
            ylabel="mean_actions_on_day",
            filename="course_actions_daily_AGENTS_NEW.png",
        ),
    ]
    result = attach_summary_and_features(
        name="Course-action block",
        data=actions_df,
        key_column=PRIMARY_KEYS["course_actions"],
        from_cache=from_cache,
        validation=coverage_validation,
        extra_tables={
            "Key validation": key_validation,
            "Legacy block validation": validation,
            "Daily action profile sample": daily_df.head(12),
        },
        artifact_paths={"table": table_path, "validation": validation_path, "daily_profile": daily_path},
        plots=plots,
        notes="Captures time-aware engagement, early-vs-late activity, and inactivity gaps from the action log.",
    )
    if show_output:
        display_block_result(result)
    return result
