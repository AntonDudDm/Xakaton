"""Access-history aggregation block for AGENTS_NEW."""

from __future__ import annotations

from .base_entity_block import build_base_entity_block
from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_block_coverage, validate_unique_key


def build_access_history_block(base_result=None, show_output: bool = True):
    """Build or load the access-history feature block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "access_history_agg_AGENT.csv"
    validation_path = OUT_DIR / "access_history_agg_validation_AGENT.csv"

    from_cache = artifact_exists(table_path) and artifact_exists(validation_path)
    if from_cache:
        access_df = read_csv(table_path)
        validation = read_csv(validation_path)
    else:
        if base_result is None:
            base_result = build_base_entity_block(show_output=False)
        with legacy_output_context() as legacy:
            access_df, validation = legacy.build_access_history_agg(base_result.data)

    if base_result is None:
        base_result = build_base_entity_block(show_output=False)

    key_validation = validate_unique_key(access_df, PRIMARY_KEYS["access_history"], "access_history")
    coverage_validation = validate_block_coverage(base_result.data, access_df, PRIMARY_KEYS["access_history"], "access_history")
    plots = [
        plot_histogram(
            access_df["total_access_days"],
            title="Total access days per enrollment",
            xlabel="total_access_days",
            filename="access_total_days_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="Access-history block",
        data=access_df,
        key_column=PRIMARY_KEYS["access_history"],
        from_cache=from_cache,
        validation=coverage_validation,
        extra_tables={
            "Key validation": key_validation,
            "Legacy block validation": validation,
        },
        artifact_paths={"table": table_path, "validation": validation_path},
        plots=plots,
        notes="Aggregates access windows, reopen signals, extension events, and gap statistics per enrollment.",
    )
    if show_output:
        display_block_result(result)
    return result
