"""Shared loading, cleaning, and dependency helpers."""

from __future__ import annotations

import scripts.build_user_course_features_AGENT as legacy_pipeline


def load_user_source_data():
    """Load cleaned user-level source data and agent ids."""

    return legacy_pipeline.load_users()


def load_lessons_source_data():
    """Load cleaned lessons source data."""

    return legacy_pipeline.load_lessons()


def load_lesson_tasks_source_data():
    """Load cleaned lesson-task links."""

    return legacy_pipeline.load_lesson_tasks()


def load_groups_source_data():
    """Load cleaned groups source data."""

    return legacy_pipeline.load_groups()


def load_trainings_source_data():
    """Load cleaned trainings source data."""

    return legacy_pipeline.load_trainings()


def load_homeworks_source_data():
    """Load cleaned homeworks source data."""

    return legacy_pipeline.load_homeworks()


def build_training_course_map(trainings, lessons):
    """Build the training-to-course bridge."""

    return legacy_pipeline.build_training_course_map(trainings, lessons)


def build_group_course_map(groups, lessons):
    """Build the group-to-course bridge."""

    return legacy_pipeline.build_group_course_map(groups, lessons)


def build_homework_course_map(homeworks, lessons):
    """Build the homework-to-course bridge."""

    return legacy_pipeline.build_homework_course_map(homeworks, lessons)
