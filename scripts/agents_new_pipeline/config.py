"""Configuration for the AGENTS_NEW modular pipeline."""

from pathlib import Path

import scripts.build_user_course_features_AGENT as legacy_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "AGENTS_NEW"
FIG_DIR = OUT_DIR / "figures"

RAW_TABLES = legacy_pipeline.RAW_TABLES
ENTITY_MAP_ROWS = legacy_pipeline.ENTITY_MAP_ROWS

PRIMARY_KEYS = {
    "audit": "table_name",
    "base_entity": "users_course_id",
    "course_features": "course_id",
    "user_features": "user_id",
    "user_lessons": "users_course_id",
    "user_trainings": "users_course_id",
    "user_answers": "users_course_id",
    "course_actions": "users_course_id",
    "media_sessions": "users_course_id",
    "access_history": "users_course_id",
    "final_features": "users_course_id",
}
