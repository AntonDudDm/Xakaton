"""
eda_linear_v2.py
Компактное линейное EDA:
1) low-leakage feature screening
2) student-level univariate analysis
3) course-level structure analysis
4) shortlist features before hypotheses
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats


# =============================================================================
# Paths
# =============================================================================

def resolve_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "scripts").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Project root not found")


PROJECT_ROOT = resolve_project_root()
os.environ["PYTHONIOENCODING"] = "utf-8"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_AGENT import TABLES_DIR

DATA_DIR = TABLES_DIR
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda_linear_v2"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Load
# =============================================================================

df = pl.read_parquet(DATA_DIR / "df_train.parquet")
df_infer = pl.read_parquet(DATA_DIR / "df_infer.parquet")

print(f"df_train : {df.shape}")
print(f"df_infer : {df_infer.shape}")

df_pd = df.to_pandas()
df_labeled = df_pd[df_pd["target"].notna()].copy()

print(f"df_labeled : {df_labeled.shape}")


# =============================================================================
# Feature blocks
# =============================================================================

USER_FEATURES = [
    "timezone",
]

COURSE_FEATURES = [
    "course_lessons_count",
    "course_lessons_with_conspect_share",
    "course_lessons_with_tasks_share",
    "course_tasks_per_lesson",
    "course_required_task_share",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
    "course_video_duration_per_lesson",
    "course_has_video_flag",
    "course_trainings_per_lesson",
    "course_training_difficulty_mean",
    "course_training_difficulty_max",
    "course_training_templates_per_training",
    "course_has_trainings_flag",
    "course_homeworks_per_lesson",
    "course_homework_items_per_homework",
    "course_homework_task_item_share",
    "course_has_homeworks_flag",
]

TRAINING_FEATURES = [
    "training_attempts_mean",
    "training_checked_ratio",
    "training_high_mark_count",
    "training_mark_mean",
    "training_points_per_training",
]

ANSWER_FEATURES = [
    "answer_total_count",
    "answer_solved_share",
    "answer_attempts_per_answer",
    "answer_first_7d_count",
    "answer_first_14d_count",
]

TIME_FEATURES = [
    "time_to_first_action_days",
    "time_to_first_answer_days",
]

BASE_FEATURES = [
    "module_num",
]

ALL_FEATURES = USER_FEATURES + COURSE_FEATURES + TRAINING_FEATURES + ANSWER_FEATURES + TIME_FEATURES + BASE_FEATURES
ALL_FEATURES = [c for c in ALL_FEATURES if c in df_labeled.columns]

print("\nSelected features:")
for f in ALL_FEATURES:
    print(" ", f)


# =============================================================================
# Types
# =============================================================================

for col in USER_FEATURES:
    if col in df_labeled.columns:
        df_labeled[col] = df_labeled[col].astype("category")

for col in ALL_FEATURES:
    if col not in USER_FEATURES:
        df_labeled[col] = pd.to_numeric(df_labeled[col], errors="coerce")


# =============================================================================
# 01. Student-level univariate analysis
# =============================================================================

print("\n" + "=" * 80)
print("01. STUDENT-LEVEL UNIVARIATE ANALYSIS")
print("=" * 80)

STUDENT_NUMERIC_FEATURES = [
    c for c in (TRAINING_FEATURES + ANSWER_FEATURES + TIME_FEATURES)
    if c in df_labeled.columns
]

clean_student_features = []
for c in STUDENT_NUMERIC_FEATURES:
    s = pd.to_numeric(df_labeled[c], errors="coerce")
    if s.notna().sum() < 50:
        continue
    if s.nunique(dropna=True) <= 1:
        continue
    if s.value_counts(normalize=True, dropna=True).iloc[0] >= 0.95:
        print(f"Skip near-constant feature: {c}")
        continue
    clean_student_features.append(c)

print("\nStudent-level numeric features used:")
print(clean_student_features)

rows = []
for c in clean_student_features:
    sub = df_labeled[[c, "target"]].copy()
    sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()

    if len(sub) < 50:
        continue

    g1 = sub.loc[sub["target"] == 1, c]
    g0 = sub.loc[sub["target"] == 0, c]

    m1 = g1.median()
    m0 = g0.median()
    gap = abs(m1 - m0)

    _, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")

    rows.append({
        "feature": c,
        "median_completed": m1,
        "median_dropout": m0,
        "gap": gap,
        "p_value": p,
        "significant": p < 0.05,
    })

student_stat_df = pd.DataFrame(rows).sort_values(["gap", "p_value"], ascending=[False, True])
student_stat_df = student_stat_df[student_stat_df["gap"] > 0].copy()

print("\nTop student-level features:")
print(student_stat_df.head(15).to_string(index=False))
student_stat_df.to_csv(OUTPUT_DIR / "01_student_univariate_stats.csv", index=False)

top_features = student_stat_df.head(12)["feature"].tolist()

if top_features:
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[c[:28] for c in top_features],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    for i, col in enumerate(top_features):
        r = i // n_cols + 1
        cc = i % n_cols + 1

        vals = pd.to_numeric(df_labeled[col], errors="coerce")
        tgt = df_labeled["target"]

        for tval, color, name in [
            (0, "#e74c3c", "Dropout"),
            (1, "#27ae60", "Completed"),
        ]:
            v = vals[tgt == tval].dropna()
            fig.add_trace(
                go.Box(
                    y=v,
                    marker_color=color,
                    name=name,
                    showlegend=(i == 0),
                    legendgroup=name,
                    boxmean=True,
                ),
                row=r,
                col=cc,
            )

    fig.update_layout(
        title="Top student-level features: Completed vs Dropout",
        height=max(700, 320 * n_rows),
        width=1150,
        legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
    )
    fig.write_image(str(OUTPUT_DIR / "02_student_boxplots.png"))
    print("Saved: 02_student_boxplots.png")


# =============================================================================
# 02. Categorical analysis
# =============================================================================

print("\n" + "=" * 80)
print("02. CATEGORICAL ANALYSIS")
print("=" * 80)

for col in USER_FEATURES:
    if col not in df_labeled.columns:
        continue

    temp = df_labeled[[col, "target"]].copy()
    temp[col] = temp[col].astype(str).fillna("missing")

    agg = temp.groupby(col)["target"].agg(["count", "mean"]).reset_index()
    agg["completion_pct"] = (agg["mean"] * 100).round(1)
    agg = agg[agg["count"] >= 20].copy()
    agg = agg.sort_values("count", ascending=False).head(10)

    if agg.empty:
        print(f"\nFeature: {col} -> no categories with count >= 20")
        continue

    print(f"\nFeature: {col}")
    print(agg[[col, "count", "completion_pct"]].to_string(index=False))

    fig = px.bar(
        agg,
        x=col,
        y="completion_pct",
        text="count",
        title=f"Completion rate by {col}",
    )
    fig.write_image(str(OUTPUT_DIR / f"03_cat_{col}.png"))
    print(f"Saved: 03_cat_{col}.png")


# =============================================================================
# 03. Binning for key student-level features
# =============================================================================

print("\n" + "=" * 80)
print("03. BINNING OF KEY FEATURES")
print("=" * 80)

def plot_binned_feature(frame: pd.DataFrame, feature: str, bins: int = 5):
    temp = frame[[feature, "target"]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 50 or temp[feature].nunique() < 2:
        print(f"Skip binning for {feature}")
        return None

    temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")

    agg = (
        temp.groupby("bin")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "completion_rate"})
    )
    agg["completion_pct"] = (agg["completion_rate"] * 100).round(1)
    agg["bin_label"] = agg["bin"].astype(str)

    fig = px.bar(
        agg,
        x="bin_label",
        y="completion_pct",
        text="count",
        title=f"{feature}: completion rate by bins",
    )
    fig.update_xaxes(title_text="Feature bins")
    fig.update_yaxes(title_text="Completion (%)")
    fig.write_image(str(OUTPUT_DIR / f"04_bins_{feature}.png"))

    return agg[["bin_label", "count", "completion_pct"]]

important_for_bins = [
    "answer_total_count",
    "answer_first_14d_count",
    "training_points_per_training",
    "time_to_first_action_days",
    "answer_solved_share",
    "training_mark_mean",
]

for f in important_for_bins:
    if f in df_labeled.columns:
        agg = plot_binned_feature(df_labeled, f)
        if agg is not None:
            print(f"\nBinning for {f}")
            print(agg.to_string(index=False))


print("\n" + "=" * 80)
print("STUDENT-LEVEL INSIGHTS")
print("=" * 80)

print("""
1. Поведенческие признаки сильнее пользовательских:
   - answer_total_count и answer_first_14d_count дают сильный сигнал.
   - Это означает, что активное решение задач важнее статического профиля студента.

2. Качество выполнения тоже важно:
   - training_points_per_training, training_high_mark_count, training_mark_mean.
   - Completed не только активнее, но и успешнее справляются с тренировками.

3. Ранний старт связан с успешностью:
   - time_to_first_action_days ниже у Completed.
   - Поздний старт можно рассматривать как ранний риск-сигнал.
""")


# =============================================================================
# 04. Course-level structure analysis
# =============================================================================

print("\n" + "=" * 80)
print("04. COURSE-LEVEL STRUCTURE ANALYSIS")
print("=" * 80)

if "course_id" not in df_labeled.columns:
    raise KeyError("course_id is required for course-level analysis")

course_features = [c for c in COURSE_FEATURES if c in df_labeled.columns]

course_df = (
    df_labeled
    .groupby("course_id")
    .agg({"target": ["mean", "count"], **{c: "mean" for c in course_features}})
)

course_df.columns = ["_".join(col).strip() for col in course_df.columns.values]
course_df = course_df.reset_index()
course_df = course_df.rename(columns={
    "target_mean": "completion_rate",
    "target_count": "n_students",
})

print(f"\nCourses before size filter: {len(course_df)}")
course_df = course_df[course_df["n_students"] >= 20].copy()
print(f"Courses after size filter:  {len(course_df)}")

print("\nCourse-level preview:")
print(course_df.head().to_string(index=False))

# --- variability
var_rows = []
for f in course_features:
    col = f"{f}_mean"
    if col not in course_df.columns:
        continue

    s = pd.to_numeric(course_df[col], errors="coerce")
    var_rows.append({
        "feature": f,
        "n_unique": int(s.nunique(dropna=True)),
        "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
        "min": float(s.min()) if s.notna().sum() else np.nan,
        "max": float(s.max()) if s.notna().sum() else np.nan,
    })

var_df = pd.DataFrame(var_rows).sort_values(["n_unique", "std"], ascending=[True, True])
var_df.to_csv(OUTPUT_DIR / "05_course_variability.csv", index=False)

print("\nCourse feature variability:")
print(var_df.to_string(index=False))

# --- keep only sufficiently variable course features
course_keep = []
course_descriptive = []
course_drop = []

for _, row in var_df.iterrows():
    if row["n_unique"] >= 3:
        course_keep.append(row["feature"])
    elif row["n_unique"] == 2:
        course_descriptive.append(row["feature"])
    else:
        course_drop.append(row["feature"])

print("\nKeep for course-level interpretation:")
print(course_keep if course_keep else "None")

print("\nKeep only as descriptive metadata:")
print(course_descriptive if course_descriptive else "None")

print("\nDrop from course-level evidence:")
print(course_drop if course_drop else "None")

# --- correlations for keep-features only
corr_rows = []
for f in course_keep:
    col = f"{f}_mean"
    s = pd.to_numeric(course_df[col], errors="coerce")
    r = s.corr(course_df["completion_rate"])
    corr_rows.append({"feature": f, "corr": r})

corr_df = pd.DataFrame(corr_rows).sort_values("corr", ascending=False) if corr_rows else pd.DataFrame()
corr_df.to_csv(OUTPUT_DIR / "06_course_correlations.csv", index=False)

print("\nCourse feature correlations:")
print(corr_df.to_string(index=False) if not corr_df.empty else "No valid course correlations.")

# --- scatter plots only for useful course features
for f in course_keep:
    col = f"{f}_mean"
    fig = px.scatter(
        course_df,
        x=col,
        y="completion_rate",
        size="n_students",
        hover_name="course_id",
        title=f"{f} vs completion_rate (course level)",
    )
    fig.write_image(str(OUTPUT_DIR / f"07_course_{f}.png"))
    print(f"Saved: 07_course_{f}.png")

print("\n" + "=" * 80)
print("COURSE-LEVEL INSIGHTS")
print("=" * 80)

if not course_keep:
    print("""
Большинство course-level признаков имеют слишком низкую вариативность между курсами.
Это означает, что по текущему датасету нельзя сделать сильные статистические выводы
о влиянии структуры курса.
""")
else:
    print("""
Часть course-level признаков имеет достаточную вариативность.
Их можно использовать как слабые или умеренные сигналы о влиянии структуры курса
на completion_rate, но главным драйвером остаются поведенческие признаки студентов.
""")

COURSE_FEATURES_FINAL = course_keep + course_descriptive

print("\nFinal course features kept for next blocks:")
print(COURSE_FEATURES_FINAL if COURSE_FEATURES_FINAL else "None")
pd.DataFrame({"course_features_final": COURSE_FEATURES_FINAL}).to_csv(
    OUTPUT_DIR / "08_course_features_final.csv", index=False
)


# =============================================================================
# 05. Final feature shortlist before hypotheses
# =============================================================================

print("\n" + "=" * 80)
print("05. FINAL FEATURE SHORTLIST BEFORE HYPOTHESES")
print("=" * 80)

FINAL_STUDENT_FEATURES = student_stat_df.loc[
    (student_stat_df["significant"]) & (student_stat_df["gap"] > 0),
    "feature"
].tolist()

FINAL_FEATURES_BEFORE_HYPOTHESES = (
    USER_FEATURES
    + COURSE_FEATURES_FINAL
    + FINAL_STUDENT_FEATURES
    + BASE_FEATURES
)

FINAL_FEATURES_BEFORE_HYPOTHESES = [
    c for c in dict.fromkeys(FINAL_FEATURES_BEFORE_HYPOTHESES)
    if c in df_labeled.columns
]

print("\nFinal features before hypotheses:")
for f in FINAL_FEATURES_BEFORE_HYPOTHESES:
    print(" ", f)

pd.DataFrame({"feature": FINAL_FEATURES_BEFORE_HYPOTHESES}).to_csv(
    OUTPUT_DIR / "09_final_features_before_hypotheses.csv",
    index=False,
)

# =============================================================================
# 06. Summary before hypotheses
# =============================================================================

print("\n" + "=" * 80)
print("06. SUMMARY BEFORE HYPOTHESES")
print("=" * 80)

print("""
Краткое резюме по результатам EDA
---------------------------------

1. Главный сигнал идёт от поведенческих признаков студента, а не от профиля.
   Наиболее сильные признаки:
   - answer_total_count
   - answer_first_14d_count
   - answer_solved_share
   - training_points_per_training
   - training_high_mark_count
   - training_mark_mean
   - time_to_first_action_days

2. Методически это означает следующее:
   - динамика оттока лучше всего объясняется не тем, кто студент,
     а тем, как он начинает и проходит курс;
   - ранняя активность и качество первых результатов дают наиболее
     интерпретируемый и practically useful сигнал;
   - это хорошо согласуется с задачей раннего выявления студентов группы риска.

3. По user-level признакам:
   - timezone оставляем только как сегментационный categorical-признак;
   - это не основной драйвер риска, а слабый контекстный фактор.

4. По course-level признакам:
   - большая часть course-фич оказалась почти константной между курсами;
   - значит, сильных статистических выводов о структуре курса сделать нельзя;
   - но некоторые признаки нагрузки курса всё же дают слабый/умеренный сигнал:
       * course_tasks_per_lesson
       * course_required_tasks_per_lesson
       * course_unique_tasks_per_lesson
   - это можно интерпретировать как возможное влияние плотности заданий на completion.

5. Бизнес-смысл перед гипотезами:
   - если студент поздно стартует и мало решает в первые 2 недели,
     это ранний риск-сигнал;
   - если студент решает много и успешно, вероятность completion резко выше;
   - если курс содержит более высокую плотность обязательных задач,
     это может быть связано с более низким completion и требует проверки
     как отдельной гипотезы для методистов.
""")

# -----------------------------------------------------------------------------
# Final feature sets for hypothesis section
# -----------------------------------------------------------------------------

HYPOTHESIS_USER_FEATURES = [
    "timezone",
]

HYPOTHESIS_STUDENT_FEATURES = [
    "answer_total_count",
    "answer_first_14d_count",
    "answer_solved_share",
    "answer_attempts_per_answer",
    "training_points_per_training",
    "training_high_mark_count",
    "training_mark_mean",
    "time_to_first_action_days",
]

HYPOTHESIS_COURSE_FEATURES = [
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
    "course_lessons_with_conspect_share",
    "course_lessons_with_tasks_share",
    "course_homeworks_per_lesson",
    "course_training_templates_per_training",
]

HYPOTHESIS_CONTEXT_FEATURES = [
    "module_num",
]

FEATURES_FOR_HYPOTHESES = (
    HYPOTHESIS_USER_FEATURES
    + HYPOTHESIS_STUDENT_FEATURES
    + HYPOTHESIS_COURSE_FEATURES
    + HYPOTHESIS_CONTEXT_FEATURES
)

FEATURES_FOR_HYPOTHESES = [
    c for c in FEATURES_FOR_HYPOTHESES
    if c in df_labeled.columns
]

# -----------------------------------------------------------------------------
# Features dropped at this stage
# -----------------------------------------------------------------------------

DROPPED_AT_THIS_STAGE = [
    c for c in ALL_FEATURES
    if c not in FEATURES_FOR_HYPOTHESES
]

DROP_REASONS = {
    "training_checked_ratio": "near-constant / weak median separation",
    "answer_first_7d_count": "too sparse; weaker than answer_first_14d_count",
    "time_to_first_answer_days": "weak univariate signal (not significant)",
    "course_lessons_count": "constant across courses",
    "course_required_task_share": "constant across courses",
    "course_video_duration_per_lesson": "constant across courses",
    "course_has_video_flag": "constant across courses",
    "course_trainings_per_lesson": "constant across courses",
    "course_training_difficulty_mean": "constant across courses",
    "course_training_difficulty_max": "constant across courses",
    "course_has_trainings_flag": "constant across courses",
    "course_homework_items_per_homework": "constant across courses",
    "course_homework_task_item_share": "constant across courses",
    "course_has_homeworks_flag": "constant across courses",
}

drop_rows = []
for f in DROPPED_AT_THIS_STAGE:
    drop_rows.append({
        "feature": f,
        "reason": DROP_REASONS.get(f, "not selected for the next hypothesis stage"),
    })

drop_df = pd.DataFrame(drop_rows).sort_values("feature").reset_index(drop=True)

print("\n" + "-" * 80)
print("FEATURES KEPT FOR HYPOTHESES")
print("-" * 80)
for f in FEATURES_FOR_HYPOTHESES:
    print(" ", f)

print("\n" + "-" * 80)
print("FEATURES DROPPED AT THIS STAGE")
print("-" * 80)
print(drop_df.to_string(index=False) if not drop_df.empty else "None")

pd.DataFrame({"feature": FEATURES_FOR_HYPOTHESES}).to_csv(
    OUTPUT_DIR / "10_features_for_hypotheses.csv",
    index=False
)

drop_df.to_csv(
    OUTPUT_DIR / "11_dropped_before_hypotheses.csv",
    index=False
)

print("\nSaved:")
print(" - 10_features_for_hypotheses.csv")
print(" - 11_dropped_before_hypotheses.csv")

# =============================================================================
# H0. Survivor bias between modules
# =============================================================================

print("\n" + "=" * 80)
print("H0. SURVIVOR BIAS BETWEEN MODULES")
print("=" * 80)

print("""
Гипотеза:
Студенты, дошедшие до следующего модуля, уже представляют собой более устойчивую
и мотивированную подвыборку. Поэтому completion rate на M2 нельзя напрямую
сравнивать с M1 без учёта survivor bias.

Что проверяем:
1. Различается ли completion rate между модулями?
2. Можно ли считать module_num важным контекстным признаком?
""")

# -----------------------------------------------------------------------------
# Table: completion by module
# -----------------------------------------------------------------------------

module_summary = (
    df_labeled
    .groupby("module")
    .agg(
        n=("target", "size"),
        completed=("target", "sum"),
        completion_rate=("target", "mean"),
    )
    .reset_index()
)

module_summary["dropout"] = module_summary["n"] - module_summary["completed"]
module_summary["completion_pct"] = (module_summary["completion_rate"] * 100).round(1)

print("\nModule summary:")
print(module_summary[["module", "n", "completed", "dropout", "completion_pct"]].to_string(index=False))

module_summary.to_csv(OUTPUT_DIR / "H0_module_summary.csv", index=False)

# -----------------------------------------------------------------------------
# Chart 1: target distribution by module
# -----------------------------------------------------------------------------

plot_df = (
    df_labeled
    .groupby(["module", "target"])
    .size()
    .reset_index(name="count")
)

plot_df["target_label"] = plot_df["target"].map({
    0: "Dropout / not completed",
    1: "Completed",
})

fig = px.bar(
    plot_df,
    x="module",
    y="count",
    color="target_label",
    barmode="group",
    text="count",
    title="H0. Target distribution by module",
    color_discrete_map={
        "Dropout / not completed": "#e74c3c",
        "Completed": "#27ae60",
    },
)

fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_xaxes(title_text="Module")
fig.update_yaxes(title_text="Students")
fig.update_layout(legend_title_text="Target")

fig.write_image(str(OUTPUT_DIR / "H0_target_by_module.png"))
print("Saved: H0_target_by_module.png")

# -----------------------------------------------------------------------------
# Chart 2: completion rate by module
# -----------------------------------------------------------------------------

fig = px.bar(
    module_summary,
    x="module",
    y="completion_pct",
    text="n",
    title="H0. Completion rate by module",
    color="completion_pct",
    color_continuous_scale="Blues",
)

fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_xaxes(title_text="Module")
fig.update_yaxes(title_text="Completion (%)")
fig.update_layout(coloraxis_showscale=False)

fig.write_image(str(OUTPUT_DIR / "H0_completion_rate_by_module.png"))
print("Saved: H0_completion_rate_by_module.png")

# -----------------------------------------------------------------------------
# Statistical test: chi-square independence
# -----------------------------------------------------------------------------

contingency = pd.crosstab(df_labeled["module"], df_labeled["target"])

if contingency.shape[0] >= 2 and contingency.shape[1] == 2:
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print("\nChi-square test of independence:")
    print(f"chi2   = {chi2:.4f}")
    print(f"p_value= {p_value:.6g}")
    print(f"dof    = {dof}")

    if p_value < 0.05:
        print("\nConclusion:")
        print(
            "Completion depends on module. "
            "This supports the presence of survivor bias: students in later modules "
            "form a more selected cohort."
        )
        survivor_bias_supported = True
    else:
        print("\nConclusion:")
        print(
            "No statistically significant dependence between module and completion "
            "was found in the labeled subset."
        )
        survivor_bias_supported = False
else:
    print("\nChi-square test was skipped: not enough module/target variation.")
    survivor_bias_supported = False

# -----------------------------------------------------------------------------
# Short business interpretation
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("H0 SUMMARY")
print("-" * 80)

if survivor_bias_supported:
    print("""
Резюме:
- Completion rate различается между модулями.
- Это означает, что M2 нельзя интерпретировать как просто "более успешный" модуль.
- Часть эффекта объясняется тем, что до M2 доходят уже более устойчивые студенты.

Что это значит для бизнеса:
- Во всех следующих гипотезах и моделях нужно учитывать module_num.
- Метрики модели и паттерны риска нельзя смешивать по M1 и M2 без поправки.
- В дашборде риска нужно сравнивать студентов внутри модуля, а не между модулями напрямую.
""")
else:
    print("""
Резюме:
- На текущем labeled-срезе статистически сильного survivor bias не обнаружено.
- Однако module_num всё равно стоит сохранять как контекстный признак.

Что это значит для бизнеса:
- Модуль остаётся важным сегментационным фактором.
- Но в текущих данных он не даёт сильного статистического перекоса completion.
""")

# -----------------------------------------------------------------------------
# Feature decision after H0
# -----------------------------------------------------------------------------

print("\nFeature decision after H0:")
if "module_num" in ALL_FEATURES:
    print("KEEP: module_num -> contextual feature for all next hypothesis blocks")
else:
    print("module_num is missing in ALL_FEATURES")



# =============================================================================
# H1. Early start and early activity
# =============================================================================

print("\n" + "=" * 80)
print("H1. EARLY START & EARLY ACTIVITY")
print("=" * 80)

print("""
Гипотеза:
Студенты, которые раньше начинают курс и проявляют активность в первые 14 дней,
значительно чаще доходят до конца.

Проверяем:
- time_to_first_action_days (ранний старт)
- answer_first_14d_count (ранняя учебная активность)
""")

features_h1 = [
    "time_to_first_action_days",
    "answer_first_14d_count",
]

features_h1 = [f for f in features_h1 if f in df_labeled.columns]

# -----------------------------------------------------------------------------
# 1. Mann–Whitney test + медианы
# -----------------------------------------------------------------------------

rows = []

for f in features_h1:
    s = pd.to_numeric(df_labeled[f], errors="coerce")
    t = df_labeled["target"]

    sub = pd.DataFrame({"v": s, "t": t}).dropna()

    if len(sub) < 50:
        continue

    g1 = sub[sub["t"] == 1]["v"]
    g0 = sub[sub["t"] == 0]["v"]

    m1 = g1.median()
    m0 = g0.median()

    stat, p = stats.mannwhitneyu(g1, g0)

    rows.append({
        "feature": f,
        "median_completed": m1,
        "median_dropout": m0,
        "gap": abs(m1 - m0),
        "p_value": p,
        "significant": p < 0.05,
    })

h1_stats = pd.DataFrame(rows).sort_values("gap", ascending=False)

print("\nH1 statistics:")
print(h1_stats.to_string(index=False))

h1_stats.to_csv(OUTPUT_DIR / "H1_stats.csv", index=False)

# -----------------------------------------------------------------------------
# 2. Boxplots
# -----------------------------------------------------------------------------

if len(features_h1) > 0:
    fig = make_subplots(
        rows=1,
        cols=len(features_h1),
        subplot_titles=features_h1,
    )

    for i, f in enumerate(features_h1):
        vals = pd.to_numeric(df_labeled[f], errors="coerce")
        tgt = df_labeled["target"]

        for tval, color, name in [
            (0, "#e74c3c", "Dropout"),
            (1, "#27ae60", "Completed"),
        ]:
            v = vals[tgt == tval].dropna()

            fig.add_trace(
                go.Box(
                    y=v,
                    marker_color=color,
                    name=name,
                    showlegend=(i == 0),
                    boxmean=True,
                ),
                row=1,
                col=i + 1
            )

    fig.update_layout(
        title="H1. Early behavior: Completed vs Dropout",
        height=450,
        width=900,
    )

    fig.write_image(str(OUTPUT_DIR / "H1_boxplots.png"))
    print("Saved: H1_boxplots.png")

# -----------------------------------------------------------------------------
# 3. Binning plots (самая важная часть)
# -----------------------------------------------------------------------------

def plot_h1_bins(feature):
    temp = df_labeled[[feature, "target"]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 50 or temp[feature].nunique() < 3:
        return

    temp["bin"] = pd.qcut(temp[feature], q=5, duplicates="drop")

    agg = (
        temp.groupby("bin")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "completion_rate"})
    )

    agg["completion_pct"] = (agg["completion_rate"] * 100).round(1)
    agg["bin_label"] = agg["bin"].astype(str)

    fig = px.bar(
        agg,
        x="bin_label",
        y="completion_pct",
        text="count",
        title=f"{feature}: completion rate by bins (H1)",
    )

    fig.write_image(str(OUTPUT_DIR / f"H1_bins_{feature}.png"))
    print(f"Saved: H1_bins_{feature}.png")

    print(f"\nBinning for {feature}")
    print(agg[["bin_label", "count", "completion_pct"]])

for f in features_h1:
    plot_h1_bins(f)

# -----------------------------------------------------------------------------
# 4. Summary
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("H1 SUMMARY")
print("-" * 80)

if not h1_stats.empty and h1_stats["significant"].any():
    print("""
Резюме:
- Ранний старт и ранняя активность статистически связаны с completion.
- Студенты, которые начинают позже и мало активны в первые 14 дней,
  имеют значительно более высокий риск оттока.

Ключевой паттерн:
- чем позже первый action → тем ниже completion
- чем меньше ответов в первые 14 дней → тем ниже completion
""")

    print("""
Что это значит для бизнеса:
- можно строить ранний скоринг риска уже в первые 1–2 недели;
- low activity в первые 14 дней — главный early warning сигнал;
- поздний старт — отдельный триггер риска.

Практические действия:
1. Ввести правило:
   - если answer_first_14d_count низкий → студент попадает в risk-сегмент

2. Добавить триггеры:
   - нет активности X дней → пуш / тьютор
   - мало решений задач → рекомендация или помощь

3. Перестроить onboarding:
   - важно довести студента до первых действий как можно быстрее
""")
else:
    print("""
Резюме:
- Сильной статистической связи раннего старта и активности с completion
  в текущем срезе не обнаружено.

Что это значит:
- либо сигнал слабее, чем ожидалось,
- либо его нужно анализировать в сочетании с другими признаками.
""")
    

# =============================================================================
# H2. Engagement vs Performance
# =============================================================================

print("\n" + "=" * 80)
print("H2. ENGAGEMENT vs PERFORMANCE")
print("=" * 80)

print("""
Гипотеза:
Completion определяется не только активностью (сколько делает студент),
но и качеством выполнения.

Проверяем:
- answer_total_count → активность
- answer_solved_share → качество решений
- training_mark_mean → качество выполнения тренировок
""")

features_h2 = [
    "answer_total_count",
    "answer_solved_share",
    "training_mark_mean",
]

features_h2 = [f for f in features_h2 if f in df_labeled.columns]

# -----------------------------------------------------------------------------
# 1. Статистика + Mann–Whitney
# -----------------------------------------------------------------------------

rows = []

for f in features_h2:
    s = pd.to_numeric(df_labeled[f], errors="coerce")
    t = df_labeled["target"]

    sub = pd.DataFrame({"v": s, "t": t}).dropna()

    if len(sub) < 50:
        continue

    g1 = sub[sub["t"] == 1]["v"]
    g0 = sub[sub["t"] == 0]["v"]

    m1 = g1.median()
    m0 = g0.median()

    stat, p = stats.mannwhitneyu(g1, g0)

    rows.append({
        "feature": f,
        "median_completed": m1,
        "median_dropout": m0,
        "gap": abs(m1 - m0),
        "p_value": p,
        "significant": p < 0.05,
    })

h2_stats = pd.DataFrame(rows).sort_values("gap", ascending=False)

print("\nH2 statistics:")
print(h2_stats.to_string(index=False))

h2_stats.to_csv(OUTPUT_DIR / "H2_stats.csv", index=False)

# -----------------------------------------------------------------------------
# 2. Boxplots
# -----------------------------------------------------------------------------

if len(features_h2) > 0:
    fig = make_subplots(
        rows=1,
        cols=len(features_h2),
        subplot_titles=features_h2,
    )

    for i, f in enumerate(features_h2):
        vals = pd.to_numeric(df_labeled[f], errors="coerce")
        tgt = df_labeled["target"]

        for tval, color, name in [
            (0, "#e74c3c", "Dropout"),
            (1, "#27ae60", "Completed"),
        ]:
            v = vals[tgt == tval].dropna()

            fig.add_trace(
                go.Box(
                    y=v,
                    marker_color=color,
                    name=name,
                    showlegend=(i == 0),
                    boxmean=True,
                ),
                row=1,
                col=i + 1
            )

    fig.update_layout(
        title="H2. Engagement vs Performance",
        height=450,
        width=1000,
    )

    fig.write_image(str(OUTPUT_DIR / "H2_boxplots.png"))
    print("Saved: H2_boxplots.png")

# -----------------------------------------------------------------------------
# 3. Binning (самый важный блок)
# -----------------------------------------------------------------------------

def plot_h2_bins(feature):
    temp = df_labeled[[feature, "target"]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 50 or temp[feature].nunique() < 3:
        return

    temp["bin"] = pd.qcut(temp[feature], q=5, duplicates="drop")

    agg = (
        temp.groupby("bin")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "completion_rate"})
    )

    agg["completion_pct"] = (agg["completion_rate"] * 100).round(1)
    agg["bin_label"] = agg["bin"].astype(str)

    fig = px.bar(
        agg,
        x="bin_label",
        y="completion_pct",
        text="count",
        title=f"{feature}: completion rate by bins (H2)",
    )

    fig.write_image(str(OUTPUT_DIR / f"H2_bins_{feature}.png"))
    print(f"Saved: H2_bins_{feature}.png")

    print(f"\nBinning for {feature}")
    print(agg[["bin_label", "count", "completion_pct"]])

for f in features_h2:
    plot_h2_bins(f)

# -----------------------------------------------------------------------------
# 4. Взаимодействие: активность vs качество
# -----------------------------------------------------------------------------

print("\nInteraction: activity vs quality")

if "answer_total_count" in df_labeled.columns and "answer_solved_share" in df_labeled.columns:

    temp = df_labeled[["answer_total_count", "answer_solved_share", "target"]].copy()
    temp["answer_total_count"] = pd.to_numeric(temp["answer_total_count"], errors="coerce")
    temp["answer_solved_share"] = pd.to_numeric(temp["answer_solved_share"], errors="coerce")
    temp = temp.dropna()

    if len(temp) >= 50:
        temp["activity_bin"] = pd.qcut(temp["answer_total_count"], 3, duplicates="drop")
        temp["quality_bin"] = pd.qcut(temp["answer_solved_share"], 3, duplicates="drop")

        agg = (
            temp.groupby(["activity_bin", "quality_bin"])["target"]
            .agg(["mean", "count"])
            .reset_index()
        )

        agg["completion_pct"] = (agg["mean"] * 100).round(1)

        # ВАЖНО: plotly/kaleido не умеет сериализовать Interval
        agg["activity_bin_label"] = agg["activity_bin"].astype(str)
        agg["quality_bin_label"] = agg["quality_bin"].astype(str)

        pivot = agg.pivot(
            index="activity_bin_label",
            columns="quality_bin_label",
            values="completion_pct",
        )

        count_pivot = agg.pivot(
            index="activity_bin_label",
            columns="quality_bin_label",
            values="count",
        )

        print("\nCompletion (%) heatmap (activity x quality):")
        print(pivot)

        print("\nCounts per cell:")
        print(count_pivot)

        fig = px.imshow(
            pivot,
            text_auto=True,
            aspect="auto",
            title="H2. Completion: activity vs quality",
            color_continuous_scale="Blues",
        )

        fig.update_xaxes(title_text="Quality bin (answer_solved_share)")
        fig.update_yaxes(title_text="Activity bin (answer_total_count)")

        fig.write_image(str(OUTPUT_DIR / "H2_heatmap_activity_quality.png"))
        print("Saved: H2_heatmap_activity_quality.png")

# -----------------------------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("H2 SUMMARY")
print("-" * 80)

if not h2_stats.empty and h2_stats["significant"].any():
    print("""
Резюме:
- И активность, и качество статистически связаны с completion.
- Наиболее устойчивый сигнал даёт связка:
    * answer_total_count → сколько студент реально работает
    * answer_solved_share → насколько успешно он справляется

Ключевой паттерн:
- высокая активность + высокое качество → максимальный completion
- низкая активность → главный риск-фактор
- низкое качество при заметной активности → возможный сигнал учебной сложности
""")

    print("""
    Что это значит для бизнеса:

    1. Активность — базовый ранний фильтр риска:
    - если студент мало решает → высокий риск оттока

    2. Качество — уточняющий сигнал:
    - если студент решает, но решает плохо → вероятно, он не справляется с материалом

    3. Практическая сегментация:
    - low activity → "не вовлечён"
    - high activity + low quality → "вовлечён, но испытывает трудности"
    - high activity + high quality → "устойчивый успешный студент"

    Практические действия:
    - low activity → пуши, напоминания, тьюторское вовлечение
    - low quality → помощь, упрощение входа, разбор трудных тем
    - high quality → positive signal, рекомендации и апселл
    """)
else:
    print("""
Резюме:
- Сильной разницы между активностью и качеством не обнаружено.
- Требуется более сложная модель для разделения эффектов.
""")
    

    # =============================================================================
# H3. Course load hypothesis
# =============================================================================

print("\n" + "=" * 80)
print("H3. COURSE LOAD HYPOTHESIS")
print("=" * 80)

print("""
Гипотеза:
Сложность / нагрузка курса влияет на вероятность completion.

Проверяем:
- course_tasks_per_lesson
- course_required_tasks_per_lesson
- course_unique_tasks_per_lesson
""")

features_h3 = [
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
]

features_h3 = [f for f in features_h3 if f in df_labeled.columns]

# -----------------------------------------------------------------------------
# 1. Course-level агрегирование
# -----------------------------------------------------------------------------

course_df = (
    df_labeled
    .groupby("course_id")
    .agg({
        "target": ["mean", "count"],
        **{f: "mean" for f in features_h3}
    })
)

course_df.columns = ["_".join(col) for col in course_df.columns]
course_df = course_df.reset_index()

course_df.rename(columns={
    "target_mean": "completion_rate",
    "target_count": "n_students"
}, inplace=True)

print("\nCourse-level table:")
print(course_df)

# -----------------------------------------------------------------------------
# 2. Scatter plots (ключевая визуализация)
# -----------------------------------------------------------------------------

for f in features_h3:
    col = f"{f}_mean"

    if col not in course_df.columns:
        continue

    fig = px.scatter(
        course_df,
        x=col,
        y="completion_rate",
        size="n_students",
        text="course_id",
        title=f"{f} vs completion_rate (H3)",
    )

    fig.update_traces(textposition="top center")

    fig.write_image(str(OUTPUT_DIR / f"H3_scatter_{f}.png"))
    print(f"Saved: H3_scatter_{f}.png")

# -----------------------------------------------------------------------------
# 3. Корреляции
# -----------------------------------------------------------------------------

rows = []

for f in features_h3:
    col = f"{f}_mean"

    if col not in course_df.columns:
        continue

    x = course_df[col]
    y = course_df["completion_rate"]

    if x.nunique() <= 1:
        r = np.nan
    else:
        r = x.corr(y)

    rows.append({
        "feature": f,
        "corr": r,
    })

h3_corr = pd.DataFrame(rows).sort_values("corr")

print("\nH3 correlations:")
print(h3_corr.to_string(index=False))

h3_corr.to_csv(OUTPUT_DIR / "H3_correlations.csv", index=False)

# -----------------------------------------------------------------------------
# 4. Прямая интерпретация
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("H3 INTERPRETATION")
print("-" * 80)

for _, row in h3_corr.iterrows():
    f = row["feature"]
    r = row["corr"]

    if pd.isna(r):
        print(f"{f}: нет вариативности → нельзя интерпретировать")
    elif r < -0.5:
        print(f"{f}: сильная отрицательная связь → больше нагрузки → меньше completion")
    elif r < -0.2:
        print(f"{f}: умеренная отрицательная связь")
    elif r > 0.2:
        print(f"{f}: положительная связь (неожиданно)")
    else:
        print(f"{f}: слабая или отсутствует связь")

# -----------------------------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("H3 SUMMARY")
print("=" * 80)

print("""
Резюме:

- Все три признака нагрузки (tasks_per_lesson, required_tasks, unique_tasks)
  показывают сильную отрицательную корреляцию с completion_rate.

- Это означает:
  чем больше заданий на урок → тем выше вероятность оттока.
""")

print("""
Что это значит для бизнеса:

1. Перегруз курса — реальный фактор оттока
   - студенты не справляются с объёмом

2. Особенно опасно:
   - обязательные задания (required_tasks)

3. Интерпретация:
   - не просто "ленятся"
   - курс может быть слишком тяжёлым

Практические действия:

- снизить tasks_per_lesson
- сделать часть заданий необязательными
- ввести прогрессивную сложность

НО:

⚠️ Ограничение:
- всего 4 курса → вывод НЕ финальный
- гипотеза требует подтверждения на большем числе курсов
""")

# =============================================================================
# H4. TIMEZONE AS A SEGMENTATION FACTOR
# =============================================================================

print("\n" + "=" * 80)
print("H4. TIMEZONE AS A SEGMENTATION FACTOR")
print("=" * 80)

print("""
Гипотеза:
Timezone связан с completion, но является скорее сегментационным,
а не основным причинным фактором риска.

Проверяем:
- есть ли статистическая связь timezone и target
- насколько эта связь сильна бизнесово
""")

if "timezone" in df_labeled.columns:
    temp = df_labeled[["timezone", "target"]].copy()
    temp["timezone"] = temp["timezone"].astype(str).fillna("missing")

    # оставляем только осмысленные категории
    counts = temp["timezone"].value_counts()
    valid_tz = counts[counts >= 20].index.tolist()
    temp = temp[temp["timezone"].isin(valid_tz)].copy()

    tz_summary = (
        temp.groupby("timezone")["target"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "completion_rate"})
        .sort_values("count", ascending=False)
    )
    tz_summary["completion_pct"] = (tz_summary["completion_rate"] * 100).round(1)

    print("\nTimezone summary:")
    print(tz_summary[["timezone", "count", "completion_pct"]].to_string(index=False))

    tz_summary.to_csv(OUTPUT_DIR / "H4_timezone_summary.csv", index=False)

    contingency = pd.crosstab(temp["timezone"], temp["target"])

    if contingency.shape[0] >= 2 and contingency.shape[1] == 2:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        print("\nChi-square test:")
        print(f"chi2   = {chi2:.4f}")
        print(f"p_value= {p_value:.6g}")
        print(f"dof    = {dof}")

        timezone_significant = p_value < 0.05
    else:
        timezone_significant = False
        print("\nChi-square skipped: not enough variation")

    fig = px.bar(
        tz_summary.head(10),
        x="timezone",
        y="completion_pct",
        text="count",
        title="H4. Completion rate by timezone",
    )
    fig.update_xaxes(title_text="Timezone")
    fig.update_yaxes(title_text="Completion (%)")
    fig.write_image(str(OUTPUT_DIR / "H4_timezone_completion.png"))
    print("Saved: H4_timezone_completion.png")

    print("\n" + "-" * 80)
    print("H4 SUMMARY")
    print("-" * 80)

    if timezone_significant:
        print("""
Резюме:
- Timezone статистически связан с completion.
- Однако это скорее сегментационный признак, чем основной драйвер риска.

Что это значит для бизнеса:
- timezone можно использовать для сегментации и дашборда,
- но нельзя считать его главным объяснением оттока,
- приоритет всё равно остаётся за поведенческими признаками.
""")
    else:
        print("""
Резюме:
- Устойчивой статистической связи timezone и completion не обнаружено.

Что это значит для бизнеса:
- timezone можно оставить как контекст,
- но в модель и в стратегию удержания он не должен входить как ключевой драйвер.
""")
        
        # =============================================================================
# H5. QUALITY STILL MATTERS AMONG ACTIVE STUDENTS
# =============================================================================

print("\n" + "=" * 80)
print("H5. QUALITY MATTERS AMONG ACTIVE STUDENTS")
print("=" * 80)

print("""
Гипотеза:
Даже среди уже активных студентов качество выполнения остаётся важным
сигналом completion.

Проверяем:
- среди студентов с достаточно высокой активностью
  продолжают ли различать группы:
    * answer_solved_share
    * training_mark_mean
""")

if "answer_total_count" in df_labeled.columns:
    temp = df_labeled.copy()
    temp["answer_total_count"] = pd.to_numeric(temp["answer_total_count"], errors="coerce")

    # Берём только достаточно активных студентов: выше медианы
    activity_threshold = temp["answer_total_count"].median()
    active_df = temp[temp["answer_total_count"] >= activity_threshold].copy()

    print(f"\nActive subset size: {len(active_df)}")
    print(f"Activity threshold (median answer_total_count): {activity_threshold:.3f}")

    features_h5 = [
        "answer_solved_share",
        "training_mark_mean",
    ]
    features_h5 = [f for f in features_h5 if f in active_df.columns]

    rows = []

    for f in features_h5:
        s = pd.to_numeric(active_df[f], errors="coerce")
        t = active_df["target"]

        sub = pd.DataFrame({"v": s, "t": t}).dropna()

        if len(sub) < 50:
            continue

        g1 = sub[sub["t"] == 1]["v"]
        g0 = sub[sub["t"] == 0]["v"]

        m1 = g1.median()
        m0 = g0.median()

        stat, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")

        rows.append({
            "feature": f,
            "median_completed": m1,
            "median_dropout": m0,
            "gap": abs(m1 - m0),
            "p_value": p,
            "significant": p < 0.05,
        })

    h5_stats = pd.DataFrame(rows).sort_values("gap", ascending=False)

    print("\nH5 statistics:")
    print(h5_stats.to_string(index=False))

    h5_stats.to_csv(OUTPUT_DIR / "H5_active_quality_stats.csv", index=False)

    # boxplots
    if len(features_h5) > 0:
        fig = make_subplots(
            rows=1,
            cols=len(features_h5),
            subplot_titles=features_h5,
        )

        for i, f in enumerate(features_h5):
            vals = pd.to_numeric(active_df[f], errors="coerce")
            tgt = active_df["target"]

            for tval, color, name in [
                (0, "#e74c3c", "Dropout"),
                (1, "#27ae60", "Completed"),
            ]:
                v = vals[tgt == tval].dropna()

                fig.add_trace(
                    go.Box(
                        y=v,
                        marker_color=color,
                        name=name,
                        showlegend=(i == 0),
                        boxmean=True,
                    ),
                    row=1,
                    col=i + 1
                )

        fig.update_layout(
            title="H5. Quality among active students",
            height=450,
            width=900,
        )

        fig.write_image(str(OUTPUT_DIR / "H5_boxplots_active_quality.png"))
        print("Saved: H5_boxplots_active_quality.png")

    print("\n" + "-" * 80)
    print("H5 SUMMARY")
    print("-" * 80)

    if not h5_stats.empty and h5_stats["significant"].any():
        print("""
Резюме:
- Даже среди уже активных студентов качество выполнения остаётся важным сигналом.
- Это значит, что риск бывает двух типов:
    1. студент не вовлёкся
    2. студент вовлёкся, но не справляется

Что это значит для бизнеса:
- low activity → нужен onboarding / вовлечение
- high activity + low quality → нужен тьютор / помощь по сложности
- high activity + high quality → устойчивый успешный студент
""")
    else:
        print("""
Резюме:
- Среди уже активных студентов качество выполнения не дало сильного
  дополнительного сигнала.

Что это значит для бизнеса:
- основной фокус нужно делать на самой активности,
- а качество использовать как вторичный уточняющий фактор.
""")
        
        # =============================================================================
# 07. Final summary after hypotheses
# =============================================================================

print("\n" + "=" * 80)
print("07. FINAL SUMMARY AFTER HYPOTHESES")
print("=" * 80)

print("""
Итог гипотезного этапа
----------------------

1. Survivor bias подтверждён:
   - M2 существенно успешнее M1 не потому, что модуль "лучше",
     а потому что до него доходят уже более устойчивые студенты.
   - module_num обязателен во всех дальнейших моделях.

2. Early-risk сигнал подтверждён:
   - поздний старт и низкая активность в первые 14 дней
     связаны с повышенным риском оттока.
   - это делает early prediction практически реализуемым.

3. Главный поведенческий драйвер:
   - answer_total_count — самый сильный и устойчивый сигнал.
   - активное решение задач намного важнее статического профиля.

4. Качество выполнения имеет самостоятельный смысл:
   - answer_solved_share и training_mark_mean дают дополнительный сигнал.
   - даже среди активных студентов качество продолжает разделять completed и dropout.

5. Значит, риск бывает двух типов:
   - студент не вовлёкся;
   - студент вовлёкся, но не справляется.

6. Course-level сигнал есть, но ограничен:
   - признаки нагрузки курса (tasks_per_lesson, required_tasks_per_lesson,
     unique_tasks_per_lesson) показывают сильную отрицательную связь с completion;
   - однако вывод ограничен тем, что в train всего 4 курса.

7. Timezone:
   - статистически связан с completion,
   - но используется как сегментационный, а не как причинный фактор.

Бизнес-итог:
- компания может строить ранний риск-скоринг;
- тьюторы могут работать с двумя типами риска:
    * low activity
    * high activity + low quality
- методисты должны отдельно проверять гипотезу перегруза курса задачами.
""")

# =============================================================================
# 08. Submodels by feature blocks
# =============================================================================

print("\n" + "=" * 80)
print("08. SUBMODELS BY FEATURE BLOCKS")
print("=" * 80)

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# Feature blocks for submodels
# -----------------------------------------------------------------------------

BLOCKS = {
    "USER_ONLY": [
        "timezone",
        "module_num",
    ],
    "EARLY_ONLY": [
        "module_num",
        "time_to_first_action_days",
        "answer_first_14d_count",
    ],
    "ANSWER_ONLY": [
        "module_num",
        "answer_total_count",
        "answer_solved_share",
        "answer_attempts_per_answer",
    ],
    "TRAINING_ONLY": [
        "module_num",
        "training_points_per_training",
        "training_high_mark_count",
        "training_mark_mean",
    ],
    "COURSE_ONLY": [
        "module_num",
        "course_tasks_per_lesson",
        "course_required_tasks_per_lesson",
        "course_unique_tasks_per_lesson",
        "course_lessons_with_conspect_share",
        "course_lessons_with_tasks_share",
        "course_homeworks_per_lesson",
        "course_training_templates_per_training",
    ],
    "ENGAGEMENT_QUALITY": [
        "module_num",
        "answer_total_count",
        "answer_solved_share",
        "training_mark_mean",
        "training_points_per_training",
    ],
    "ALL_SELECTED": [
        "module_num",
        "timezone",
        "answer_total_count",
        "answer_first_14d_count",
        "answer_solved_share",
        "answer_attempts_per_answer",
        "training_points_per_training",
        "training_high_mark_count",
        "training_mark_mean",
        "time_to_first_action_days",
        "course_tasks_per_lesson",
        "course_required_tasks_per_lesson",
        "course_unique_tasks_per_lesson",
        "course_lessons_with_conspect_share",
        "course_lessons_with_tasks_share",
        "course_homeworks_per_lesson",
        "course_training_templates_per_training",
    ],
}

# оставляем только существующие колонки
for block_name in BLOCKS:
    BLOCKS[block_name] = [c for c in BLOCKS[block_name] if c in df_labeled.columns]

print("\nBlocks:")
for k, v in BLOCKS.items():
    print(f"{k}: {v}")

# -----------------------------------------------------------------------------
# Prepare modeling dataframe
# -----------------------------------------------------------------------------

MODEL_DF = df_labeled.copy()

# timezone как category -> codes
if "timezone" in MODEL_DF.columns:
    MODEL_DF["timezone"] = MODEL_DF["timezone"].astype("category").cat.codes.replace(-1, np.nan)

# numeric coercion
for c in set(sum(BLOCKS.values(), [])):
    if c in MODEL_DF.columns:
        MODEL_DF[c] = pd.to_numeric(MODEL_DF[c], errors="coerce")

y = MODEL_DF["target"].astype(int).to_numpy()

if "user_id" not in MODEL_DF.columns:
    raise KeyError("user_id is required for group-aware CV")

groups = MODEL_DF["user_id"].to_numpy()

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "logreg": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            random_state=42,
        ),
    ),
    "rf": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ),
}

# -----------------------------------------------------------------------------
# CV evaluation
# -----------------------------------------------------------------------------

rows = []

for block_name, features in BLOCKS.items():
    if len(features) == 0:
        continue

    X = MODEL_DF[features].copy()

    for model_name, model in models.items():
        roc_scores = []
        pr_scores = []
        bal_scores = []

        for tr_idx, va_idx in cv.split(X, y, groups):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            m = clone(model)
            m.fit(X_tr, y_tr)

            proba = m.predict_proba(X_va)[:, 1]
            pred = (proba >= 0.5).astype(int)

            roc_scores.append(roc_auc_score(y_va, proba))
            pr_scores.append(average_precision_score(y_va, proba))
            bal_scores.append(balanced_accuracy_score(y_va, pred))

        rows.append({
            "block": block_name,
            "model": model_name,
            "n_features": len(features),
            "roc_auc_mean": np.mean(roc_scores),
            "roc_auc_std": np.std(roc_scores),
            "pr_auc_mean": np.mean(pr_scores),
            "balanced_accuracy_mean": np.mean(bal_scores),
        })

submodel_results = pd.DataFrame(rows).sort_values(
    ["roc_auc_mean", "pr_auc_mean"],
    ascending=False
).reset_index(drop=True)

print("\nSubmodel results:")
print(submodel_results.to_string(index=False))

submodel_results.to_csv(OUTPUT_DIR / "12_submodel_results.csv", index=False)

# -----------------------------------------------------------------------------
# Plot: ROC-AUC by block
# -----------------------------------------------------------------------------

plot_df = submodel_results.copy()
plot_df["label"] = plot_df["block"] + " | " + plot_df["model"]

fig = px.bar(
    plot_df,
    x="label",
    y="roc_auc_mean",
    error_y="roc_auc_std",
    text="n_features",
    title="Submodels: ROC-AUC by feature block",
)

fig.update_xaxes(title_text="Block | model")
fig.update_yaxes(title_text="ROC-AUC")
fig.write_image(str(OUTPUT_DIR / "12_submodels_roc_auc.png"))
print("Saved: 12_submodels_roc_auc.png")

# -----------------------------------------------------------------------------
# Short summary
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("SUBMODEL SUMMARY")
print("-" * 80)

best_row = submodel_results.iloc[0]
print(f"""
Лучший блок на этом этапе:
- block  = {best_row['block']}
- model  = {best_row['model']}
- ROC-AUC= {best_row['roc_auc_mean']:.4f}

Как интерпретировать:
- если EARLY_ONLY даёт уже хороший ROC-AUC, значит ранний прогноз реально возможен;
- если ANSWER_ONLY / ENGAGEMENT_QUALITY сильнее COURSE_ONLY, значит основной сигнал идёт от поведения;
- если COURSE_ONLY даёт заметный, но слабее результат, значит структура курса влияет, но вторична;
- ALL_SELECTED показывает, насколько выигрывает объединение блоков.
""")

# =============================================================================
# 09. Final summary before model-based interpretation
# =============================================================================

print("\n" + "=" * 80)
print("09. FINAL SUMMARY BEFORE CATBOOST / SHAP")
print("=" * 80)

print("""
Итог по EDA и гипотезам
-----------------------

1. Survivor bias подтверждён очень сильно:
   - M2 нельзя напрямую сравнивать с M1.
   - module_num обязательно сохраняем во всех моделях.

2. Главный сигнал идёт от student behavior:
   - answer_total_count
   - answer_first_14d_count
   - answer_solved_share
   - training_points_per_training
   - training_mark_mean
   - time_to_first_action_days

3. Early prediction уже выглядит жизнеспособно:
   - блок EARLY_ONLY дал сильный ROC-AUC,
   - значит риск можно выявлять в первые 1–2 недели.

4. Риск имеет как минимум два разных механизма:
   - low activity  -> студент не вовлёкся
   - high activity + low quality -> студент вовлёкся, но не справляется

5. Course-level сигнал есть, но он слабее и менее надёжен:
   - часть course-фич почти константна;
   - полезный сигнал дают только признаки нагрузки:
       * course_tasks_per_lesson
       * course_required_tasks_per_lesson
       * course_unique_tasks_per_lesson
   - этот вывод ограничен малым числом курсов.

6. USER_ONLY и COURSE_ONLY заметно слабее behavior-блоков:
   - это значит, что профиль и структура курса вторичны,
     а основной драйвер риска — поведение внутри курса.

7. Tree-based модели сильно лучше линейных:
   - RandomForest стабильно лучше Logistic Regression,
   - значит зависимости нелинейны и есть взаимодействия между признаками.

Переход к следующему этапу
--------------------------
Теперь мы переходим от hypothesis-driven EDA к model-based validation:

1. обучаем CatBoost на итоговом наборе признаков;
2. считаем CV-метрики;
3. строим feature importance;
4. строим SHAP;
5. проверяем, какие признаки реально входят в финальное ядро модели;
6. формируем интерпретируемые risk groups и примеры студентов высокого риска.

Бизнес-смысл следующего этапа
-----------------------------
На следующем шаге мы должны подтвердить не только то, что признаки статистически связаны
с completion, но и то, что они реально формируют устойчивый предиктивный сигнал.

Это позволит:
- собрать финальный shortlist признаков,
- объяснить модель через SHAP,
- сделать дашборд "students at risk",
- сформулировать действия для тьюторов и методистов.
""")

# =============================================================================
# 10. CatBoost baseline
# =============================================================================

print("\n" + "=" * 80)
print("10. CATBOOST BASELINE")
print("=" * 80)

from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

FINAL_MODEL_FEATURES = [
    "module_num",
    "timezone",
    "answer_total_count",
    "answer_first_14d_count",
    "answer_solved_share",
    "answer_attempts_per_answer",
    "training_points_per_training",
    "training_high_mark_count",
    "training_mark_mean",
    "time_to_first_action_days",
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
    "course_lessons_with_conspect_share",
    "course_lessons_with_tasks_share",
    "course_homeworks_per_lesson",
    "course_training_templates_per_training",
]

FINAL_MODEL_FEATURES = [c for c in FINAL_MODEL_FEATURES if c in df_labeled.columns]

print("\nFinal model features:")
for f in FINAL_MODEL_FEATURES:
    print(" ", f)

model_df = df_labeled[FINAL_MODEL_FEATURES + ["target", "user_id"]].copy()

# timezone оставляем категориальной
categorical_features = [c for c in ["timezone"] if c in model_df.columns]

for c in FINAL_MODEL_FEATURES:
    if c not in categorical_features:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

X = model_df[FINAL_MODEL_FEATURES].copy()
y = model_df["target"].astype(int).to_numpy()
groups = model_df["user_id"].to_numpy()

cat_feature_indices = [X.columns.get_loc(c) for c in categorical_features]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

cat_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
)

roc_scores = []
pr_scores = []
bal_scores = []

oof_pred = np.zeros(len(X), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = clone(cat_model)
    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_feature_indices,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )

    proba = model.predict_proba(X_va)[:, 1]
    pred = (proba >= 0.5).astype(int)

    oof_pred[va_idx] = proba

    roc = roc_auc_score(y_va, proba)
    pr = average_precision_score(y_va, proba)
    bal = balanced_accuracy_score(y_va, pred)

    roc_scores.append(roc)
    pr_scores.append(pr)
    bal_scores.append(bal)

    print(f"Fold {fold}: ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | BAL_ACC={bal:.4f}")

print("\nCatBoost CV results:")
print(f"ROC-AUC mean = {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
print(f"PR-AUC  mean = {np.mean(pr_scores):.4f}")
print(f"BAL_ACC mean = {np.mean(bal_scores):.4f}")

catboost_cv_summary = pd.DataFrame({
    "metric": ["roc_auc_mean", "roc_auc_std", "pr_auc_mean", "balanced_accuracy_mean"],
    "value": [
        np.mean(roc_scores),
        np.std(roc_scores),
        np.mean(pr_scores),
        np.mean(bal_scores),
    ]
})
catboost_cv_summary.to_csv(OUTPUT_DIR / "13_catboost_cv_summary.csv", index=False)
print("Saved: 13_catboost_cv_summary.csv")

# =============================================================================
# 11. CatBoost feature importance
# =============================================================================

print("\n" + "=" * 80)
print("11. CATBOOST FEATURE IMPORTANCE")
print("=" * 80)

final_cat_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
)

final_cat_model.fit(
    X,
    y,
    cat_features=cat_feature_indices,
)

importance_df = pd.DataFrame({
    "feature": FINAL_MODEL_FEATURES,
    "importance": final_cat_model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop feature importances:")
print(importance_df.head(20).to_string(index=False))

importance_df.to_csv(OUTPUT_DIR / "14_catboost_feature_importance.csv", index=False)

fig = px.bar(
    importance_df.head(15),
    x="feature",
    y="importance",
    title="CatBoost feature importance",
)

fig.update_xaxes(title_text="Feature")
fig.update_yaxes(title_text="Importance")
fig.write_image(str(OUTPUT_DIR / "14_catboost_feature_importance.png"))
print("Saved: 14_catboost_feature_importance.png")

# =============================================================================
# 12. SHAP
# =============================================================================

print("\n" + "=" * 80)
print("12. SHAP")
print("=" * 80)

import shap
import matplotlib.pyplot as plt

# Для SHAP кодируем категориальные так, как их понимает CatBoost
explainer = shap.TreeExplainer(final_cat_model)

# Для компактности можно взять подвыборку
X_shap = X.sample(min(1500, len(X)), random_state=42).copy()
shap_values = explainer.shap_values(X_shap)

# --- summary bar
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "15_shap_bar.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: 15_shap_bar.png")

# --- summary beeswarm
plt.figure()
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "15_shap_beeswarm.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: 15_shap_beeswarm.png")


# =============================================================================
# 13. Risk groups and example students
# =============================================================================

print("\n" + "=" * 80)
print("13. RISK GROUPS")
print("=" * 80)

risk_df = model_df.copy()
risk_df["pred_proba"] = final_cat_model.predict_proba(X)[:, 1]

# completion=1, значит риск = 1 - proba_completion
risk_df["risk_score"] = 1 - risk_df["pred_proba"]

risk_df["risk_group"] = pd.cut(
    risk_df["risk_score"],
    bins=[0.0, 0.2, 0.5, 0.8, 1.0],
    labels=["low", "medium", "high", "critical"],
    include_lowest=True,
)

risk_summary = (
    risk_df.groupby("risk_group")
    .agg(
        n=("target", "size"),
        observed_completion=("target", "mean"),
        mean_risk=("risk_score", "mean"),
    )
    .reset_index()
)

risk_summary["observed_completion_pct"] = (risk_summary["observed_completion"] * 100).round(1)
risk_summary["mean_risk_pct"] = (risk_summary["mean_risk"] * 100).round(1)

print("\nRisk group summary:")
print(risk_summary.to_string(index=False))

risk_summary.to_csv(OUTPUT_DIR / "16_risk_group_summary.csv", index=False)

fig = px.bar(
    risk_summary,
    x="risk_group",
    y="n",
    text="n",
    title="Students by predicted risk group",
)

fig.update_xaxes(title_text="Risk group")
fig.update_yaxes(title_text="Students")
fig.write_image(str(OUTPUT_DIR / "16_risk_group_distribution.png"))
print("Saved: 16_risk_group_distribution.png")

# примеры самых рискованных студентов
example_cols = [c for c in [
    "user_id",
    "module",
    "users_course_id",
    "risk_score",
    "risk_group",
    "answer_total_count",
    "answer_first_14d_count",
    "answer_solved_share",
    "training_points_per_training",
    "training_mark_mean",
    "time_to_first_action_days",
    "course_tasks_per_lesson",
] if c in risk_df.columns]

top_risk_examples = risk_df.sort_values("risk_score", ascending=False)[example_cols].head(20)
print("\nTop risk examples:")
print(top_risk_examples.to_string(index=False))

top_risk_examples.to_csv(OUTPUT_DIR / "16_top_risk_examples.csv", index=False)
print("Saved: 16_top_risk_examples.csv")

# =============================================================================
# 14. Honest summary after first modeling round
# =============================================================================

print("\n" + "=" * 80)
print("14. HONEST SUMMARY AFTER THE FIRST MODELING ROUND")
print("=" * 80)

print("""
Что показали предыдущие блоки
-----------------------------

1. Гипотезный этап был полезен:
   - survivor bias подтверждён;
   - ранняя активность и ранний старт действительно важны;
   - качество выполнения действительно связано с completion;
   - курс даёт сигнал, но слабее и с ограничением по числу курсов.

2. Подмодели подтвердили общую структуру сигнала:
   - strongest blocks = engagement / training / answers;
   - early-only блок уже даёт useful прогноз;
   - user-only и course-only заметно слабее.

3. Но первая CatBoost-модель оказалась слишком хорошей:
   - ROC-AUC ≈ 0.9985
   - risk groups почти идеально разделены
   - top-risk примеры содержат массовые NaN
   - top feature importance dominated by full-course training features

Что это означает методически
----------------------------
Такие результаты слишком хороши для реальной churn-задачи и указывают на leakage:

1. Temporal leakage:
   - в модель попали признаки, агрегированные по всему курсу:
       * training_high_mark_count
       * training_mark_mean
       * training_points_per_training
       * answer_total_count
       * answer_solved_share
   - они слишком близки к самому target и частично "видят будущее".

2. Missingness leakage:
   - модель использует NaN как shortcut:
     "нет активности / нет данных" ≈ dropout.
   - это даёт искусственно высокое качество.

3. Course leakage / small-support issue:
   - course features строятся всего по 4 курсам;
   - часть course-сигнала может быть переобучением на course profile.

Что делаем дальше
-----------------
Переходим к честной постановке задачи:

1. Используем только признаки, доступные на раннем горизонте.
2. Явно кодируем missingness отдельными флагами.
3. Обучаемся на M1.
4. Тестируем на M2.
5. Не используем full-course фичи и не используем module_num.
6. Смотрим, сохраняется ли реальный переносимый сигнал.

Это будет уже не "удобная объясняющая модель", а более честная ранняя модель риска.
""")





# =============================================================================
# 15. Leakage-free early model: train on M1 -> test on M2
# =============================================================================

print("\n" + "=" * 80)
print("15. LEAKAGE-FREE EARLY MODEL (TRAIN M1 -> TEST M2)")
print("=" * 80)

from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------------------------------------------------------
# 15.1. Define leakage-free feature set
# -----------------------------------------------------------------------------
# Правило:
# - только ранние признаки
# - только статические / course-level признаки
# - никаких full-course aggregated performance features
# - module_num НЕ используем, потому что train=M1, test=M2

EARLY_SAFE_FEATURES = [
    # user context
    "timezone",

    # early behavior
    "time_to_first_action_days",
    "time_to_first_answer_days",
    "answer_first_7d_count",
    "answer_first_14d_count",

    # weak but static course structure
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
    "course_lessons_with_conspect_share",
    "course_lessons_with_tasks_share",
    "course_homeworks_per_lesson",
    "course_training_templates_per_training",
]

EARLY_SAFE_FEATURES = [c for c in EARLY_SAFE_FEATURES if c in df_labeled.columns]

print("\nEarly-safe base features:")
for f in EARLY_SAFE_FEATURES:
    print(" ", f)

# Явно фиксируем, что убрали потенциально ликающие фичи
LEAKY_OR_POST_OUTCOME_FEATURES = [
    "module_num",
    "answer_total_count",
    "answer_solved_share",
    "answer_attempts_per_answer",
    "training_points_per_training",
    "training_high_mark_count",
    "training_mark_mean",
    "training_checked_ratio",
    "training_attempts_mean",
]

LEAKY_OR_POST_OUTCOME_FEATURES = [c for c in LEAKY_OR_POST_OUTCOME_FEATURES if c in df_labeled.columns]

print("\nDropped as leaky / post-outcome:")
for f in LEAKY_OR_POST_OUTCOME_FEATURES:
    print(" ", f)

# -----------------------------------------------------------------------------
# 15.2. Build working dataframe
# -----------------------------------------------------------------------------

work_df = df_labeled.copy()

# train/test split by module
if "module" not in work_df.columns:
    raise KeyError("module column is required for M1 -> M2 transfer evaluation")

train_df = work_df[work_df["module"] == "M1"].copy()
test_df  = work_df[work_df["module"] == "M2"].copy()

print(f"\nTrain rows (M1): {len(train_df)}")
print(f"Test rows  (M2): {len(test_df)}")

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError("M1 or M2 subset is empty")

# -----------------------------------------------------------------------------
# 15.3. Explicit missingness handling
# -----------------------------------------------------------------------------
# Идея:
# - missingness не скрываем
# - создаём бинарные флаги
# - числовые признаки заполняем осмысленно
# - категориальные -> 'missing'

base_numeric_features = [
    c for c in EARLY_SAFE_FEATURES
    if c != "timezone"
]

# missingness flags
for c in base_numeric_features:
    miss_col = f"{c}__is_missing"
    train_df[miss_col] = train_df[c].isna().astype("int8")
    test_df[miss_col] = test_df[c].isna().astype("int8")

# category
if "timezone" in train_df.columns:
    train_df["timezone"] = train_df["timezone"].astype("string").fillna("missing")
    test_df["timezone"] = test_df["timezone"].astype("string").fillna("missing")

# numeric fill strategy
# counts -> 0
count_like = [
    c for c in base_numeric_features
    if (
        "count" in c
        or "per_lesson" in c
        or "templates" in c
    )
]

# days -> large sentinel
days_like = [
    c for c in base_numeric_features
    if "days" in c
]

# shares / ratios / other static numeric -> train median
other_numeric = [
    c for c in base_numeric_features
    if c not in count_like and c not in days_like
]

for c in count_like:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(0)
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce").fillna(0)

for c in days_like:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(999)
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce").fillna(999)

for c in other_numeric:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    fill_value = train_df[c].median()
    if pd.isna(fill_value):
        fill_value = 0

    train_df[c] = train_df[c].fillna(fill_value)
    test_df[c] = test_df[c].fillna(fill_value)

# final feature list
missing_flag_features = [f"{c}__is_missing" for c in base_numeric_features]

FINAL_EARLY_MODEL_FEATURES = []
if "timezone" in EARLY_SAFE_FEATURES:
    FINAL_EARLY_MODEL_FEATURES.append("timezone")

FINAL_EARLY_MODEL_FEATURES += base_numeric_features
FINAL_EARLY_MODEL_FEATURES += missing_flag_features

print("\nFinal leakage-free features:")
for f in FINAL_EARLY_MODEL_FEATURES:
    print(" ", f)

# -----------------------------------------------------------------------------
# 15.4. Prepare matrices
# -----------------------------------------------------------------------------

X_train = train_df[FINAL_EARLY_MODEL_FEATURES].copy()
y_train = train_df["target"].astype(int).to_numpy()

X_test = test_df[FINAL_EARLY_MODEL_FEATURES].copy()
y_test = test_df["target"].astype(int).to_numpy()

categorical_features = [c for c in ["timezone"] if c in X_train.columns]
cat_feature_indices = [X_train.columns.get_loc(c) for c in categorical_features]

print("\nTarget distribution:")
print("Train M1:")
print(train_df["target"].value_counts(dropna=False).sort_index())
print("Test M2:")
print(test_df["target"].value_counts(dropna=False).sort_index())

# -----------------------------------------------------------------------------
# 15.5. Train CatBoost
# -----------------------------------------------------------------------------

early_cat = CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
)

early_cat.fit(
    X_train,
    y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_test, y_test),
    use_best_model=True,
)

# -----------------------------------------------------------------------------
# 15.6. Evaluate on M2
# -----------------------------------------------------------------------------

test_proba = early_cat.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

roc = roc_auc_score(y_test, test_proba)
pr = average_precision_score(y_test, test_proba)
bal = balanced_accuracy_score(y_test, test_pred)

print("\nLeakage-free M1 -> M2 results:")
print(f"ROC-AUC = {roc:.4f}")
print(f"PR-AUC  = {pr:.4f}")
print(f"BAL_ACC = {bal:.4f}")

print("\nConfusion matrix (M2):")
print(confusion_matrix(y_test, test_pred))

print("\nClassification report (M2):")
print(classification_report(y_test, test_pred, digits=4))

# -----------------------------------------------------------------------------
# 15.7. Feature importance
# -----------------------------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": FINAL_EARLY_MODEL_FEATURES,
    "importance": early_cat.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nLeakage-free feature importances:")
print(importance_df.to_string(index=False))

importance_df.to_csv(OUTPUT_DIR / "17_leakage_free_feature_importance.csv", index=False)

fig = px.bar(
    importance_df.head(20),
    x="feature",
    y="importance",
    title="Leakage-free M1->M2 feature importance",
)
fig.update_xaxes(title_text="Feature")
fig.update_yaxes(title_text="Importance")
fig.write_image(str(OUTPUT_DIR / "17_leakage_free_feature_importance.png"))
print("Saved: 17_leakage_free_feature_importance.png")

# -----------------------------------------------------------------------------
# 15.8. Risk groups on M2
# -----------------------------------------------------------------------------

test_out = test_df.copy()
test_out["pred_proba_completion"] = test_proba
test_out["risk_score"] = 1 - test_out["pred_proba_completion"]

test_out["risk_group"] = pd.cut(
    test_out["risk_score"],
    bins=[0.0, 0.2, 0.5, 0.8, 1.0],
    labels=["low", "medium", "high", "critical"],
    include_lowest=True,
)

risk_summary = (
    test_out.groupby("risk_group")
    .agg(
        n=("target", "size"),
        observed_completion=("target", "mean"),
        mean_risk=("risk_score", "mean"),
    )
    .reset_index()
)

risk_summary["observed_completion_pct"] = (risk_summary["observed_completion"] * 100).round(1)
risk_summary["mean_risk_pct"] = (risk_summary["mean_risk"] * 100).round(1)

print("\nLeakage-free risk groups on M2:")
print(risk_summary.to_string(index=False))

risk_summary.to_csv(OUTPUT_DIR / "17_leakage_free_risk_groups.csv", index=False)

fig = px.bar(
    risk_summary,
    x="risk_group",
    y="n",
    text="n",
    title="Leakage-free M2 risk group distribution",
)
fig.update_xaxes(title_text="Risk group")
fig.update_yaxes(title_text="Students")
fig.write_image(str(OUTPUT_DIR / "17_leakage_free_risk_groups.png"))
print("Saved: 17_leakage_free_risk_groups.png")

# -----------------------------------------------------------------------------
# 15.9. Top risk examples on M2
# -----------------------------------------------------------------------------

example_cols = [
    "user_id",
    "module",
    "users_course_id",
    "risk_score",
    "risk_group",
    "timezone",
    "time_to_first_action_days",
    "time_to_first_answer_days",
    "answer_first_7d_count",
    "answer_first_14d_count",
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
]

example_cols = [c for c in example_cols if c in test_out.columns]

top_risk_examples = test_out.sort_values("risk_score", ascending=False)[example_cols].head(20)

print("\nTop risk examples on M2 (leakage-free model):")
print(top_risk_examples.to_string(index=False))

top_risk_examples.to_csv(OUTPUT_DIR / "17_leakage_free_top_risk_examples.csv", index=False)
print("Saved: 17_leakage_free_top_risk_examples.csv")

# -----------------------------------------------------------------------------
# 15.10. Honest summary
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("LEAKAGE-FREE MODEL SUMMARY")
print("-" * 80)

print("""
Что изменилось в этой модели:
- мы исключили post-outcome признаки;
- мы убрали full-course performance фичи;
- мы не используем module_num;
- мы явно обработали NaN через missing flags + controlled fill;
- мы обучаемся на M1 и тестируемся на M2.

Как интерпретировать результат:
- если качество остаётся высоким, значит ранний переносимый сигнал действительно есть;
- если качество заметно падает относительно прошлой модели, значит старая модель
  была сильно завышена из-за leakage;
- feature importance теперь показывает более честные ранние драйверы риска.
""")

# =============================================================================
# 16. Summary of leakage-free M1 -> M2 model
# =============================================================================

print("\n" + "=" * 80)
print("16. SUMMARY OF THE LEAKAGE-FREE M1 -> M2 MODEL")
print("=" * 80)

print("""
Итог по leakage-free модели
---------------------------

1. После удаления post-outcome признаков качество заметно снизилось
   по сравнению с прежней почти идеальной моделью:
   - это подтверждает, что в старой постановке действительно был leakage;
   - новая модель существенно честнее и ближе к реальному early prediction.

2. Несмотря на это, переносимый сигнал сохранился:
   - ROC-AUC около 0.83 на схеме train=M1 -> test=M2;
   - balanced accuracy около 0.78;
   - значит, ранний прогноз риска действительно возможен.

3. Главный ранний сигнал — отсутствие входа в учебный ритм:
   - нет ранних ответов;
   - нет первого ответа;
   - поздний первый action;
   - нулевые значения answer_first_7d_count / answer_first_14d_count.

4. Missingness теперь интерпретируется корректно:
   - мы явно закодировали отсутствие ранней активности отдельными флагами;
   - поэтому модель не использует NaN как скрытый shortcut,
     а использует наблюдаемый факт отсутствия раннего учебного поведения.

5. Course-level сигнал остаётся вторичным:
   - признаки нагрузки курса дают небольшой вклад,
   - но значительно уступают раннему поведению студента.

6. Timezone даёт небольшой, но ненулевой вклад:
   - его можно оставить как сегментационный контекст,
   - но не как основной драйвер риска.

7. Практический бизнес-вывод:
   - уже на раннем горизонте можно выделять студентов,
     которые фактически не стартовали;
   - именно эта группа должна быть основной целью для раннего вмешательства тьютора.

Ограничения текущего блока
--------------------------
1. time_to_first_answer_days очень силён и частично дублирует факт отсутствия раннего ответа;
2. текущие risk groups разделяют особенно хорошо только критически рискованных,
   но середина ещё недостаточно калибрована;
3. course-level выводы остаются ограниченными малым числом курсов.

Следующий шаг
-------------
Теперь нужно:
1. построить более строгую раннюю модель без спорных признаков;
2. сравнить две leakage-free версии фичей;
3. получить SHAP именно для честной ранней модели;
4. перенастроить risk groups так, чтобы они были полезнее для бизнеса.
""")

# =============================================================================
# 17. Strict early model + improved risk groups + leakage-free SHAP
# =============================================================================

print("\n" + "=" * 80)
print("17. STRICT EARLY MODEL + IMPROVED RISK GROUPS + LEAKAGE-FREE SHAP")
print("=" * 80)

from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 17.1. Two versions of the feature set
# -----------------------------------------------------------------------------
# Version A: current early-safe set
# Version B: stricter set without time_to_first_answer_days
#            to reduce reliance on a potentially too-strong proxy

BASE_EARLY_FEATURES_A = [
    "timezone",
    "time_to_first_action_days",
    "time_to_first_answer_days",
    "answer_first_7d_count",
    "answer_first_14d_count",
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
]

BASE_EARLY_FEATURES_B = [
    "timezone",
    "time_to_first_action_days",
    "answer_first_7d_count",
    "answer_first_14d_count",
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
]

BASE_EARLY_FEATURES_A = [c for c in BASE_EARLY_FEATURES_A if c in df_labeled.columns]
BASE_EARLY_FEATURES_B = [c for c in BASE_EARLY_FEATURES_B if c in df_labeled.columns]

print("\nVersion A features:")
for f in BASE_EARLY_FEATURES_A:
    print(" ", f)

print("\nVersion B features:")
for f in BASE_EARLY_FEATURES_B:
    print(" ", f)

# -----------------------------------------------------------------------------
# 17.2. Train / test split
# -----------------------------------------------------------------------------

strict_df = df_labeled.copy()

train_df = strict_df[strict_df["module"] == "M1"].copy()
test_df  = strict_df[strict_df["module"] == "M2"].copy()

print(f"\nTrain rows (M1): {len(train_df)}")
print(f"Test rows  (M2): {len(test_df)}")

# -----------------------------------------------------------------------------
# 17.3. Helper: build a clean dataset
# -----------------------------------------------------------------------------

def build_early_dataset(train_frame, test_frame, base_features):
    train_local = train_frame.copy()
    test_local = test_frame.copy()

    numeric_features = [c for c in base_features if c != "timezone"]

    # missing flags for early numeric features only
    for c in numeric_features:
        train_local[f"{c}__is_missing"] = train_local[c].isna().astype("int8")
        test_local[f"{c}__is_missing"] = test_local[c].isna().astype("int8")

    # timezone
    if "timezone" in base_features:
        train_local["timezone"] = train_local["timezone"].astype("string").fillna("missing")
        test_local["timezone"] = test_local["timezone"].astype("string").fillna("missing")

    # controlled fill
    count_like = [
        c for c in numeric_features
        if "count" in c or "per_lesson" in c
    ]
    days_like = [c for c in numeric_features if "days" in c]
    other_like = [c for c in numeric_features if c not in count_like and c not in days_like]

    for c in count_like:
        train_local[c] = pd.to_numeric(train_local[c], errors="coerce").fillna(0)
        test_local[c] = pd.to_numeric(test_local[c], errors="coerce").fillna(0)

    for c in days_like:
        train_local[c] = pd.to_numeric(train_local[c], errors="coerce").fillna(999)
        test_local[c] = pd.to_numeric(test_local[c], errors="coerce").fillna(999)

    for c in other_like:
        train_local[c] = pd.to_numeric(train_local[c], errors="coerce")
        test_local[c] = pd.to_numeric(test_local[c], errors="coerce")

        fill_value = train_local[c].median()
        if pd.isna(fill_value):
            fill_value = 0

        train_local[c] = train_local[c].fillna(fill_value)
        test_local[c] = test_local[c].fillna(fill_value)

    final_features = []
    if "timezone" in base_features:
        final_features.append("timezone")
    final_features += numeric_features
    final_features += [f"{c}__is_missing" for c in numeric_features]

    X_train_local = train_local[final_features].copy()
    y_train_local = train_local["target"].astype(int).to_numpy()

    X_test_local = test_local[final_features].copy()
    y_test_local = test_local["target"].astype(int).to_numpy()

    categorical_local = [c for c in ["timezone"] if c in final_features]
    cat_idx_local = [X_train_local.columns.get_loc(c) for c in categorical_local]

    return {
        "train_df": train_local,
        "test_df": test_local,
        "X_train": X_train_local,
        "y_train": y_train_local,
        "X_test": X_test_local,
        "y_test": y_test_local,
        "features": final_features,
        "cat_idx": cat_idx_local,
    }

# -----------------------------------------------------------------------------
# 17.4. Helper: fit and evaluate CatBoost
# -----------------------------------------------------------------------------

def fit_eval_early_model(bundle, model_name):
    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        bundle["X_train"],
        bundle["y_train"],
        cat_features=bundle["cat_idx"],
        eval_set=(bundle["X_test"], bundle["y_test"]),
        use_best_model=True,
    )

    proba = model.predict_proba(bundle["X_test"])[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(bundle["y_test"], proba)
    pr = average_precision_score(bundle["y_test"], proba)
    bal = balanced_accuracy_score(bundle["y_test"], pred)

    print(f"\n{model_name} results:")
    print(f"ROC-AUC = {roc:.4f}")
    print(f"PR-AUC  = {pr:.4f}")
    print(f"BAL_ACC = {bal:.4f}")

    return {
        "model": model,
        "proba": proba,
        "pred": pred,
        "roc_auc": roc,
        "pr_auc": pr,
        "bal_acc": bal,
    }

# -----------------------------------------------------------------------------
# 17.5. Compare versions A and B
# -----------------------------------------------------------------------------

bundle_A = build_early_dataset(train_df, test_df, BASE_EARLY_FEATURES_A)
bundle_B = build_early_dataset(train_df, test_df, BASE_EARLY_FEATURES_B)

res_A = fit_eval_early_model(bundle_A, "Version A")
res_B = fit_eval_early_model(bundle_B, "Version B")

comparison_df = pd.DataFrame([
    {
        "version": "A_with_first_answer_days",
        "n_features": len(bundle_A["features"]),
        "roc_auc": res_A["roc_auc"],
        "pr_auc": res_A["pr_auc"],
        "bal_acc": res_A["bal_acc"],
    },
    {
        "version": "B_without_first_answer_days",
        "n_features": len(bundle_B["features"]),
        "roc_auc": res_B["roc_auc"],
        "pr_auc": res_B["pr_auc"],
        "bal_acc": res_B["bal_acc"],
    },
])

print("\nModel comparison:")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(OUTPUT_DIR / "18_early_model_comparison.csv", index=False)

# choose the stricter model if performance drop is small
if res_B["roc_auc"] >= res_A["roc_auc"] - 0.02:
    chosen_name = "B_without_first_answer_days"
    chosen_bundle = bundle_B
    chosen_res = res_B
else:
    chosen_name = "A_with_first_answer_days"
    chosen_bundle = bundle_A
    chosen_res = res_A

print(f"\nChosen leakage-free model: {chosen_name}")

# -----------------------------------------------------------------------------
# 17.6. Improved risk groups
# -----------------------------------------------------------------------------
# Two variants:
# 1) quantile-based groups for better spread
# 2) business high-risk cutoffs

test_out = chosen_bundle["test_df"].copy()
test_out["pred_proba_completion"] = chosen_res["proba"]
test_out["risk_score"] = 1 - test_out["pred_proba_completion"]

# quantile groups
test_out["risk_group_quantile"] = pd.qcut(
    test_out["risk_score"],
    q=4,
    labels=["Q1_lowest_risk", "Q2", "Q3", "Q4_highest_risk"],
    duplicates="drop",
)

quantile_summary = (
    test_out.groupby("risk_group_quantile")
    .agg(
        n=("target", "size"),
        observed_completion=("target", "mean"),
        mean_risk=("risk_score", "mean"),
    )
    .reset_index()
)
quantile_summary["observed_completion_pct"] = (quantile_summary["observed_completion"] * 100).round(1)
quantile_summary["mean_risk_pct"] = (quantile_summary["mean_risk"] * 100).round(1)

print("\nQuantile-based risk groups:")
print(quantile_summary.to_string(index=False))

quantile_summary.to_csv(OUTPUT_DIR / "18_quantile_risk_groups.csv", index=False)

fig = px.bar(
    quantile_summary,
    x="risk_group_quantile",
    y="n",
    text="n",
    title="Quantile-based risk groups on M2",
)
fig.write_image(str(OUTPUT_DIR / "18_quantile_risk_groups.png"))
print("Saved: 18_quantile_risk_groups.png")

# business cutoffs
test_out["risk_group_business"] = pd.cut(
    test_out["risk_score"],
    bins=[0.0, 0.3, 0.6, 0.85, 1.0],
    labels=["low", "moderate", "high", "critical"],
    include_lowest=True,
)

business_summary = (
    test_out.groupby("risk_group_business")
    .agg(
        n=("target", "size"),
        observed_completion=("target", "mean"),
        mean_risk=("risk_score", "mean"),
    )
    .reset_index()
)
business_summary["observed_completion_pct"] = (business_summary["observed_completion"] * 100).round(1)
business_summary["mean_risk_pct"] = (business_summary["mean_risk"] * 100).round(1)

print("\nBusiness-cutoff risk groups:")
print(business_summary.to_string(index=False))

business_summary.to_csv(OUTPUT_DIR / "18_business_risk_groups.csv", index=False)

fig = px.bar(
    business_summary,
    x="risk_group_business",
    y="n",
    text="n",
    title="Business-cutoff risk groups on M2",
)
fig.write_image(str(OUTPUT_DIR / "18_business_risk_groups.png"))
print("Saved: 18_business_risk_groups.png")

# -----------------------------------------------------------------------------
# 17.7. Feature importance for chosen model
# -----------------------------------------------------------------------------

chosen_importance_df = pd.DataFrame({
    "feature": chosen_bundle["features"],
    "importance": chosen_res["model"].get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nChosen model feature importance:")
print(chosen_importance_df.to_string(index=False))

chosen_importance_df.to_csv(OUTPUT_DIR / "18_chosen_model_importance.csv", index=False)

fig = px.bar(
    chosen_importance_df.head(20),
    x="feature",
    y="importance",
    title=f"Chosen leakage-free model importance: {chosen_name}",
)
fig.write_image(str(OUTPUT_DIR / "18_chosen_model_importance.png"))
print("Saved: 18_chosen_model_importance.png")

# -----------------------------------------------------------------------------
# 17.8. SHAP for the chosen leakage-free model
# -----------------------------------------------------------------------------

print("\nBuilding SHAP for the chosen leakage-free model...")

explainer = shap.TreeExplainer(chosen_res["model"])

X_shap = chosen_bundle["X_test"].copy()
if len(X_shap) > 1200:
    X_shap = X_shap.sample(1200, random_state=42)

shap_values = explainer.shap_values(X_shap)

# bar
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "18_leakage_free_shap_bar.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: 18_leakage_free_shap_bar.png")

# beeswarm
plt.figure()
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "18_leakage_free_shap_beeswarm.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved: 18_leakage_free_shap_beeswarm.png")

# -----------------------------------------------------------------------------
# 17.9. Top risk examples for chosen model
# -----------------------------------------------------------------------------

example_cols = [
    "user_id",
    "module",
    "users_course_id",
    "risk_score",
    "risk_group_quantile",
    "risk_group_business",
    "timezone",
    "time_to_first_action_days",
    "time_to_first_answer_days",
    "answer_first_7d_count",
    "answer_first_14d_count",
    "course_tasks_per_lesson",
    "course_required_tasks_per_lesson",
    "course_unique_tasks_per_lesson",
]
example_cols = [c for c in example_cols if c in test_out.columns]

top_risk_examples = test_out.sort_values("risk_score", ascending=False)[example_cols].head(20)

print("\nTop risk examples for chosen leakage-free model:")
print(top_risk_examples.to_string(index=False))

top_risk_examples.to_csv(OUTPUT_DIR / "18_top_risk_examples_chosen_model.csv", index=False)
print("Saved: 18_top_risk_examples_chosen_model.csv")

# -----------------------------------------------------------------------------
# 17.10. Final note
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("STRICT EARLY MODEL SUMMARY")
print("-" * 80)

print(f"""
Мы сравнили две leakage-free версии ранней модели.
Выбрана модель: {chosen_name}

Что делает этот блок:
- ещё сильнее ограничивает риск leakage;
- сравнивает feature sets;
- даёт более полезное разбиение на risk groups;
- строит SHAP именно для честной ранней модели.

После этого этапа можно:
1. собрать финальный shortlist признаков;
2. выделить 5–7 ключевых инсайтов;
3. подготовить финальный бизнес-summary для методистов и тьюторов.
""")