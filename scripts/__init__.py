"""Support modules for the AGENT EDA pipeline."""

from .agg_feat_engineering_AGENT import (
    build_access_history_features,
    build_course_action_features,
    build_course_structure_features,
    build_media_features,
    build_time_window_features,
    build_user_answer_features,
    build_user_lesson_features,
    build_user_training_features,
    build_users_base_features,
    build_users_courses_base,
)
from .config_AGENT import (
    BLOCK_EXPORT_NAMES,
    CORE_ENTITY_KEY,
    DATA_AGENT_DIR,
    DATA_RAW_DIR,
    FIGURES_DIR,
    SUMMARIES_DIR,
    TABLES_DIR,
)
from .merge_AGENT import assemble_master_user_course_table
from .service_AGENT import (
    build_key_diagnostics,
    build_missingness_summary,
    build_table_overview,
    drop_service_columns,
    ensure_output_directories,
    infer_reference_timestamp,
    load_all_tables,
    normalize_id_columns,
    save_dataframe,
    save_summary,
    validate_key_uniqueness,
    validate_left_merge,
    describe_loaded_table,
    build_direct_link_diagnostics,
    build_route_coverage,
)
