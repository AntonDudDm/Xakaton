# Hypothesis Registry for `EDA_user_course_features_AGENT`

This file tracks the key entity, linkage, aggregation, and merge hypotheses used by the new user-course feature pipeline.

---

## H-01. `users_course_id` is the correct master-table grain

Status: CONFIRMED

Statement:
The main analytical entity should be one row per `users_course_id`, with `users_courses.id` as the stable user-course key.

Why it matters:
The final feature table must be training-ready for later target attachment, so the grain must match the intended prediction object.

Tables used:
- `users_courses`
- `user_lessons`
- `wk_users_courses_actions`
- `user_access_histories`

Validation:
- `users_courses.id` uniqueness check
- `(user_id, course_id)` uniqueness check
- direct-link consistency checks for tables that already store `users_course_id`

Result:
`users_courses.id` is unique, `(user_id, course_id)` is also unique in `users_courses`, and direct-link tables do not map the same `users_course_id` to multiple users.

Impact:
The base entity and final master table are built at `users_course_id` grain.

---

## H-02. The master table should be restricted to pupil accounts

Status: CONFIRMED

Statement:
The master table should keep only `User::Pupil` rows from `users_courses`, excluding agent/teacher accounts.

Why it matters:
The business case is about student progress, churn risk, and course completion. Non-pupil accounts would pollute behavior patterns.

Tables used:
- `users`
- `users_courses`

Validation:
- `users.type` distribution review
- enrollment-row overlap between `users_courses.user_id` and pupil user ids

Result:
`users` contains both pupils and agents. `23,629` enrollment rows belong to non-pupil accounts and are excluded from the new base table.

Impact:
The pipeline filters `users_courses` to pupil accounts before any downstream merge.

---

## H-03. Course structure should be derived from `lessons` and merged by `course_id`

Status: CONFIRMED

Statement:
Stable course-level structure should be built from `lessons` and related course metadata tables, then merged to the user-course base through `course_id`.

Why it matters:
Course structure must be aggregated once at course level before it can safely enrich the user-course table.

Tables used:
- `lessons`
- `lesson_tasks`
- `trainings`
- `groups`
- `homeworks`
- `homework_items`
- `users_courses`

Validation:
- compare the set of `course_id` values between `users_courses` and `lessons`
- check that every `course_id` present in `users_courses` exists in `lessons`

Result:
All `course_id` values used in `users_courses` are covered by `lessons`. Additional courses exist in `lessons`, but not the other way around.

Impact:
The course block uses `lessons` as the course skeleton and merges course features by `course_id`.

---

## H-04. Resolve `user_trainings` to user-course through training -> lesson -> course

Status: CONFIRMED

Statement:
Training activity should be mapped through:
`user_trainings.training_id -> trainings.id -> trainings.lesson_id -> lessons.id -> lessons.course_id -> users_courses(user_id, course_id)`.

Why it matters:
Without this bridge, training-derived features cannot be aligned to the user-course entity.

Tables used:
- `user_trainings`
- `trainings`
- `lessons`
- `users_courses`

Validation:
- training-to-lesson coverage check
- lesson-to-course coverage check
- final match rate to `(user_id, course_id)` in `users_courses`

Result:
After restricting the base to pupil accounts, the route resolves `96.0385%` of training rows to `users_course_id`. The unmatched tail is small and is retained as a documented limitation.

Impact:
The training block uses this route and aggregates only matched rows.

---

## H-05. Resolve media sessions through Lesson/Group resource bridges

Status: CONFIRMED

Statement:
Media sessions should be resolved through:
- `Lesson`: `resource_id -> lessons.id -> lessons.course_id`
- `Group`: `resource_id -> groups.id -> groups.lesson_id -> lessons.course_id`
and then mapped to `users_course_id` through `(viewer_id, course_id)`.

Why it matters:
`wk_media_view_sessions` does not store `users_course_id` directly, so linkage must be validated before aggregation.

Tables used:
- `wk_media_view_sessions`
- `groups`
- `lessons`
- `users_courses`

Validation:
- coverage check for resource-to-course resolution
- final coverage check for `(viewer_id, course_id) -> users_course_id`

Result:
The resource-to-course bridge is almost perfect, and the final student-aligned route resolves `97.7338%` of media rows to `users_course_id`. The remaining unmatched tail is mostly explained by non-pupil viewers outside the modeling base.

Impact:
The media block is included in the master table.

---

## H-06. `task_id` is not a safe universal bridge for `user_answers`

Status: FAILED

Statement:
The candidate route `user_answers.task_id -> lesson_tasks.task_id -> lessons.course_id` is not safe as a universal answer-to-course bridge.

Why it matters:
If `task_id` is reused across courses, this route would create row multiplication and ambiguous course assignment.

Tables used:
- `user_answers`
- `lesson_tasks`
- `lessons`

Validation:
- uniqueness check for `lesson_tasks.task_id`
- cardinality review of `task_id -> course_id`

Result:
`task_id` is highly non-unique across courses. Only about `3.6%` of lesson-answer rows map to a single course through `task_id`; most map to multiple courses.

Impact:
The answer block must not use `task_id` as its primary linkage route.

---

## H-07. `user_answers` can be resolved by resource-specific bridges

Status: CONFIRMED

Statement:
Answer rows should be resolved by resource-specific routes:
- `Lesson`: `resource_id -> lessons.id -> course_id`
- `Training`: `resource_id -> trainings.id -> trainings.lesson_id -> course_id`
- `Homework`: `resource_id -> homeworks.id -> lesson-backed course_id`
then matched to `users_course_id` through `(user_id, course_id)`.

Why it matters:
`user_answers` is one of the main behavioral sources, but it has no direct `users_course_id`.

Tables used:
- `user_answers`
- `lessons`
- `trainings`
- `homeworks`
- `users_courses`

Validation:
- resource-type specific coverage checks
- final match rate to `users_course_id`

Result:
The resource-specific route resolves `98.2701%` of answer rows to `users_course_id` in the final pupil-only pipeline. Homework coverage is almost perfect, while the remaining gap is concentrated in a small tail of unresolved lesson/training rows and non-pupil activity outside the modeling base.

Impact:
The answer block uses resource-based linkage and keeps the unresolved tail documented in the block summary.

---

## H-08. `wk_users_courses_actions` is the primary course-activity log for the master table

Status: CONFIRMED

Statement:
The main course-activity block should use `wk_users_courses_actions`, not raw-to-raw merges against other activity logs.

Why it matters:
This table already stores `users_course_id`, which makes it the safest source for course-level activity intensity and temporal patterns.

Tables used:
- `wk_users_courses_actions`
- `users_courses`

Validation:
- direct key availability review
- distinct `users_course_id` coverage check

Result:
`wk_users_courses_actions` directly covers `216,342` distinct `users_course_id` values and does not require ambiguous bridge logic.

Impact:
The action block is built from `wk_users_courses_actions`; `user_activity_histories` is left out of the first master-table version.

---

## H-09. User-level enrichment must be aggregated before merging to the base

Status: CONFIRMED

Statement:
Any user-level enrichment, including badges, must be aggregated to one row per `user_id` before it is merged into the user-course base.

Why it matters:
Directly merging many-to-one or many-to-many user tables into the base would break the target grain.

Tables used:
- `users`
- `user_award_badges`
- `award_badges`
- `users_courses`

Validation:
- aggregate to one row per `user_id`
- validate uniqueness of the resulting user feature block

Result:
The user feature block is unique on `user_id` and safe to merge into the user-course base.

Impact:
User enrichment is merged by `user_id` only after aggregation.

---

## H-10. Access history should be aggregated directly on `users_course_id`

Status: CONFIRMED

Statement:
`user_access_histories` should be summarized directly at `users_course_id` level and merged without any extra bridge logic.

Why it matters:
Access history is a key administrative context source, and it already matches the target grain.

Tables used:
- `user_access_histories`

Validation:
- direct-key inspection
- distinct `users_course_id` coverage review

Result:
The table already stores the correct key and covers almost the entire pupil user-course population. Final merge coverage is `99.9656%`.

Impact:
The access-history block is aggregated directly on `users_course_id`.

---

## H-11. Time-aware features should be built after master-table assembly

Status: CONFIRMED

Statement:
Delay, recency, intensity, and inactivity-gap features should be computed after all blocks are aligned to the same user-course grain.

Why it matters:
This avoids mixing incompatible granularities and keeps time-aware logic transparent.

Tables used:
- assembled user-course master table

Validation:
- verify that anchor timestamps from base, actions, answers, trainings, media, and access blocks coexist in one table

Result:
The design is cleaner and safer than building cross-block time deltas before alignment.

Impact:
The pipeline computes time-window features as a dedicated second-stage block after assembly.

---

## H-12. `stats__module_1` to `stats__module_3` can enrich the flat user-course table through `(user_id, course_id)`

Status: CONFIRMED

Statement:
The first three module-report tables can be aligned to the pupil user-course base through:
`stats__module_*.(user_id, course_id) -> users_courses.(user_id, course_id) -> users_course_id`.

Why it matters:
These tables already contain curated module-level progress and assessment signals. If the bridge is valid, they should enrich the exploratory flat table instead of staying outside the pipeline.

Tables used:
- `stats__module_1`
- `stats__module_2`
- `stats__module_3`
- `users_courses`

Validation:
- `(user_id, course_id)` key checks
- duplicate-pair checks
- match-rate checks to the pupil-only `users_courses_base`
- post-aggregation uniqueness checks on `users_course_id`

Result:
`stats__module_1` resolves `98.1294%` of rows, while `stats__module_2` and `stats__module_3` resolve `100%` of rows to `users_course_id`. Module 1 contains duplicate `(user_id, course_id)` rows, but they can be safely collapsed with conservative aggregation.

Impact:
The pipeline adds a dedicated `stats_module_features_AGENT` block with `stats_m1_*`, `stats_m2_*`, and `stats_m3_*` features merged by `users_course_id`.

---

## H-13. `stats__module_4` should remain diagnostic-only until a reliable user-course bridge is found

Status: FAILED

Statement:
`stats__module_4` should not be merged into the flat master table unless its `(user_id, course_id)` pairs can be matched reliably to the current pupil user-course base.

Why it matters:
Adding unresolved module rows would either create false joins or force unsupported assumptions about course identity.

Tables used:
- `stats__module_4`
- `users_courses`

Validation:
- direct `(user_id, course_id)` overlap check against the pupil-only `users_courses_base`
- coverage review of the candidate bridge

Result:
The current bridge coverage is `0%`: none of the observed `(user_id, course_id)` rows in `stats__module_4` map to the current pupil-only `users_courses_base`.

Impact:
`stats__module_4` is kept in the raw audit and route diagnostics, but it is not merged into the current flat user-course table.
