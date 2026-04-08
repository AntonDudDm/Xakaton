"""User-level enrichment block for AGENTS_NEW."""

from __future__ import annotations

from .config import OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv
from .plot_utils import plot_histogram
from .service_utils import load_user_source_data
from .summary_utils import attach_summary_and_features, display_block_result
from .validation_utils import validate_unique_key


def build_user_block(show_output: bool = True):
    """Build or load the user-level enrichment block."""

    ensure_output_dirs()
    table_path = OUT_DIR / "user_features_AGENT.csv"
    from_cache = artifact_exists(table_path)

    if from_cache:
        user_df = read_csv(table_path)
    else:
        user_source_df, agent_ids = load_user_source_data()
        with legacy_output_context() as legacy:
            user_df = legacy.build_user_features(user_source_df, agent_ids)

    key_validation = validate_unique_key(user_df, PRIMARY_KEYS["user_features"], "user_features")
    plots = [
        plot_histogram(
            user_df["sign_in_count"],
            title="User sign-in count distribution",
            xlabel="sign_in_count",
            filename="user_sign_in_count_AGENTS_NEW.png",
        )
    ]
    result = attach_summary_and_features(
        name="User feature block",
        data=user_df,
        key_column=PRIMARY_KEYS["user_features"],
        from_cache=from_cache,
        validation=key_validation,
        artifact_paths={"table": table_path},
        plots=plots,
        notes="Adds user profile, account-age proxies, geography ids, and lightweight badge behavior signals.",
    )
    if show_output:
        display_block_result(result)
    return result
