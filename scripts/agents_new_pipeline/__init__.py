"""Modular AGENTS_NEW pipeline for user-course feature engineering."""

from .audit_blocks import run_audit_block
from .base_entity_block import build_base_entity_block
from .course_block import build_course_block
from .user_block import build_user_block
from .user_lessons_block import build_user_lessons_block
from .user_trainings_block import build_user_trainings_block
from .user_answers_block import build_user_answers_block
from .course_actions_block import build_course_actions_block
from .media_sessions_block import build_media_sessions_block
from .access_history_block import build_access_history_block
from .final_merge_block import build_final_feature_table_block

__all__ = [
    "run_audit_block",
    "build_base_entity_block",
    "build_course_block",
    "build_user_block",
    "build_user_lessons_block",
    "build_user_trainings_block",
    "build_user_answers_block",
    "build_course_actions_block",
    "build_media_sessions_block",
    "build_access_history_block",
    "build_final_feature_table_block",
]
