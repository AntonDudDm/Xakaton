"""
eda_linear_v1.py
Линейное EDA (без утечек, интерпретируемые фичи)
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

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda_linear_v1"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Load
# =============================================================================

df = pl.read_parquet(DATA_DIR / "df_train.parquet")
df_infer = pl.read_parquet(DATA_DIR / "df_infer.parquet")

print(f"df_train : {df.shape}")
print(f"df_infer : {df_infer.shape}")

df_pd = df.to_pandas()

# =============================================================================
# Feature blocks
# =============================================================================

USER_FEATURES = [
    #"d_wk_region_id", все предлставители с одного региона
    "timezone",
    #"wk_gender", почи все наны
]

COURSE_FEATURES = [
    "course_groups_actual_duration_sum",
    "course_groups_with_video_count",
    "course_homework_item_count",
    "course_has_required_tasks_flag",
    "course_lesson_number_max",
    "course_lessons_survival_share",
    "course_lessons_scratch_share",
    "course_lessons_with_conspect_share",
    "course_lessons_with_tasks_share",
    "course_training_difficulty",
    "course_trainings_count",
    "course_video_duration_mean",
    "course_unique_homework_types",
]

TRAINING_FEATURES = [
    "training_attempts_mean",
    "training_checked_ratio",
    "training_high_mark_count",
    "training_mark_mean",
    "training_olympiad_type_count",
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

ALL_FEATURES = (
    USER_FEATURES
    + COURSE_FEATURES
    + TRAINING_FEATURES
    + ANSWER_FEATURES
    + TIME_FEATURES
    + BASE_FEATURES
)



# оставляем только существующие
ALL_FEATURES = [c for c in ALL_FEATURES if c in df_pd.columns]

print("\nSelected features:")
for f in ALL_FEATURES:
    print("  ", f)

# =============================================================================
# Convert types
# =============================================================================

# categorical
for col in USER_FEATURES:
    if col in df_pd.columns:
        df_pd[col] = df_pd[col].astype("category")

# numeric
for col in ALL_FEATURES:
    if col not in USER_FEATURES:
        df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")

# labeled subset
df_labeled = df_pd[df_pd["target"].notna()].copy()

print("\nFinal dataset shape:", df_labeled.shape)

# =============================================================================
# 01. Univariate analysis: Completed vs Dropout
# =============================================================================

print("\n" + "=" * 80)
print("01. UNIVARIATE ANALYSIS")
print("=" * 80)

def to_num(df, col):
    s = pd.to_numeric(df[col], errors="coerce")
    mask = s.notna() & df["target"].notna()
    return s[mask], df.loc[mask, "target"]


# -----------------------------------------------------------------------------
# 1. Numeric features: boxplots + median gap
# -----------------------------------------------------------------------------

# =============================================================================
# Clean numeric features
# =============================================================================

EXCLUDE_FEATURES = [
    "module_num",
]

numeric_features = []
for c in ALL_FEATURES:
    if c in USER_FEATURES or c in EXCLUDE_FEATURES:
        continue

    s = pd.to_numeric(df_labeled[c], errors="coerce")

    if s.notna().sum() < 50:
        continue
    if s.nunique(dropna=True) <= 1:
        continue

    # убираем почти константные признаки
    top_freq = s.value_counts(normalize=True, dropna=True).iloc[0]
    if top_freq >= 0.95:
        print(f"Skip near-constant feature: {c}")
        continue

    numeric_features.append(c)

print("\nNumeric features used:")
print(numeric_features)


# =============================================================================
# Ranking by median gap
# =============================================================================

median_gaps = {}
stat_results = []

for c in numeric_features:
    vals = pd.to_numeric(df_labeled[c], errors="coerce")
    tgt = df_labeled["target"]

    sub = pd.DataFrame({"v": vals, "t": tgt}).dropna()

    if len(sub) < 50:
        continue

    g1 = sub[sub["t"] == 1]["v"]
    g0 = sub[sub["t"] == 0]["v"]

    m1 = g1.median()
    m0 = g0.median()

    median_gaps[c] = abs(m1 - m0)

    stat, p = stats.mannwhitneyu(g1, g0)

    stat_results.append({
        "feature": c,
        "median_completed": m1,
        "median_dropout": m0,
        "gap": abs(m1 - m0),
        "p_value": p,
    })

stat_df = pd.DataFrame(stat_results).sort_values("gap", ascending=False)

# Оставляем только реально различающиеся признаки для визуализации
stat_df = stat_df[
    (stat_df["gap"] > 0)
].copy()

# Добавим удобную колонку значимости
stat_df["significant"] = stat_df["p_value"] < 0.05

print("\nTop features:")
print(stat_df.head(15))

stat_df.to_csv(OUTPUT_DIR / "01_univariate_stats.csv", index=False)

# -----------------------------------------------------------------------------
# Boxplots for top features
# -----------------------------------------------------------------------------

top_features = stat_df.head(12)["feature"].tolist()

if len(top_features) > 0:
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
        c = i % n_cols + 1

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
                col=c
            )

    fig.update_layout(
        title="Top informative numeric features: Completed vs Dropout",
        height=max(700, 320 * n_rows),
        width=1150,
        legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
    )

    fig.write_image(str(OUTPUT_DIR / "02_boxplots_extended.png"))
    print("Saved: 02_boxplots_extended.png")
else:
    print("No features passed boxplot filtering.")

# -----------------------------------------------------------------------------
# 2. Categorical features
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("Categorical features analysis")
print("-" * 80)

for col in USER_FEATURES:
    if col not in df_labeled.columns:
        continue

    temp = df_labeled[[col, "target"]].copy()
    temp[col] = temp[col].astype(str).fillna("missing")

    agg = temp.groupby(col)["target"].agg(["count", "mean"]).reset_index()
    agg["completion_pct"] = (agg["mean"] * 100).round(1)

    # Берём только осмысленные категории
    agg = agg[agg["count"] >= 20].copy()
    agg = agg.sort_values("count", ascending=False).head(10)

    if agg.empty:
        print(f"\nFeature: {col} -> no categories with count >= 20")
        continue

    print(f"\nFeature: {col}")
    print(agg[[col, "count", "completion_pct"]])

    fig = px.bar(
        agg,
        x=col,
        y="completion_pct",
        title=f"Completion rate by {col}",
        text="count",
    )
    fig.write_image(str(OUTPUT_DIR / f"01_cat_{col}.png"))
    print(f"Saved: 01_cat_{col}.png")

# =============================================================================
# Completion rate vs bins
# =============================================================================

def plot_binned_feature(df, feature, bins=5):
    temp = df[[feature, "target"]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 50 or temp[feature].nunique() < 2:
        print(f"Skip binning for {feature}: not enough data")
        return None

    temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")

    agg = (
        temp.groupby("bin")["target"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "completion_rate"})
    )

    agg["completion_pct"] = (agg["completion_rate"] * 100).round(1)

    # Plotly/Kaleido не умеет serializовать pandas.Interval
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

    fig.write_image(str(OUTPUT_DIR / f"bins_{feature}.png"))
    print(f"Saved: bins_{feature}.png")

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
            print(agg)


print("\n" + "="*80)
print("UNIVARIATE INSIGHTS")
print("="*80)

print("""
1. Поведенческие признаки сильнее пользовательских:
   - answer_total_count и answer_first_14d_count дают один из самых сильных сигналов.
   - Это означает, что активное решение задач важнее статического профиля студента.

2. Качество выполнения тоже важно:
   - training_points_per_training, training_high_mark_count, training_mark_mean
   - Completed не только активнее, но и успешнее справляются с тренировками.

3. Ранний старт связан с успешностью:
   - time_to_first_action_days ниже у Completed
   - Поздний старт можно рассматривать как ранний риск-сигнал.

4. В курсовом дизайне есть интересный сигнал:
   - course_groups_with_video_count различает Completed и Dropout
   - Это повод отдельно проверить гипотезу о влиянии видеонагруженности курса.

5. User-сегментация пока слабая:
   - timezone можно оставить для сегментации
   - region и gender в текущем виде не дают полезного сигнала.
""")

# =============================================================================
# 03. Course-level analysis
# =============================================================================

print("\n" + "=" * 80)
print("03. COURSE-LEVEL ANALYSIS")
print("=" * 80)

if "course_id" not in df_labeled.columns:
    raise KeyError("course_id is required for course-level analysis")

course_features = [
    c for c in COURSE_FEATURES
    if c in df_labeled.columns
]

print(f"\nCourse features requested: {len(course_features)}")
print(course_features)

# -----------------------------------------------------------------------------
# Aggregate to course level
# -----------------------------------------------------------------------------

course_df = (
    df_labeled
    .groupby("course_id")
    .agg({
        "target": ["mean", "count"],
        **{c: "mean" for c in course_features}
    })
)

course_df.columns = ["_".join(col).strip() for col in course_df.columns.values]
course_df = course_df.reset_index()

course_df.rename(columns={
    "target_mean": "completion_rate",
    "target_count": "n_students"
}, inplace=True)

print(f"\nCourses before min-size filter: {len(course_df)}")

# фильтр на минимальный размер курса
course_df = course_df[course_df["n_students"] >= 20].copy()

print(f"Courses after min-size filter (n_students >= 20): {len(course_df)}")

print("\nCourse-level dataset preview:")
print(course_df.head())

# -----------------------------------------------------------------------------
# Variability diagnostics
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("COURSE FEATURE VARIABILITY")
print("-" * 80)

variability_rows = []

for c in course_features:
    col = f"{c}_mean"

    if col not in course_df.columns:
        continue

    s = pd.to_numeric(course_df[col], errors="coerce")

    variability_rows.append({
        "feature": c,
        "n_courses_non_null": int(s.notna().sum()),
        "n_unique": int(s.nunique(dropna=True)),
        "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
        "min": float(s.min()) if s.notna().sum() > 0 else np.nan,
        "max": float(s.max()) if s.notna().sum() > 0 else np.nan,
    })

variability_df = pd.DataFrame(variability_rows)

if variability_df.empty:
    print("No course-level variability table could be built.")
else:
    variability_df = variability_df.sort_values(
        ["n_unique", "std"],
        ascending=[True, True]
    ).reset_index(drop=True)

    print(variability_df.to_string(index=False))
    variability_df.to_csv(OUTPUT_DIR / "03_course_feature_variability.csv", index=False)

# -----------------------------------------------------------------------------
# Split features by variability quality
# -----------------------------------------------------------------------------

constant_course_features = []
low_variability_course_features = []
valid_course_features = []

for _, row in variability_df.iterrows():
    f = row["feature"]
    n_unique = row["n_unique"]

    if n_unique <= 1:
        constant_course_features.append(f)
    elif n_unique == 2:
        low_variability_course_features.append(f)
    else:
        valid_course_features.append(f)

print("\nConstant course features (drop from scatter/corr):")
print(constant_course_features if constant_course_features else "None")

print("\nLow-variability course features (keep only as descriptive, not strong evidence):")
print(low_variability_course_features if low_variability_course_features else "None")

print("\nValid course features for scatter/correlation:")
print(valid_course_features if valid_course_features else "None")

# -----------------------------------------------------------------------------
# Scatter plots only for features with enough variability
# -----------------------------------------------------------------------------

if len(valid_course_features) == 0:
    print("\nNo course features with sufficient variability for scatter analysis.")
else:
    for f in valid_course_features:
        col = f"{f}_mean"

        if col not in course_df.columns:
            continue

        fig = px.scatter(
            course_df,
            x=col,
            y="completion_rate",
            size="n_students",
            hover_name="course_id",
            title=f"{f} vs completion_rate (course level)",
        )

        fig.write_image(str(OUTPUT_DIR / f"course_{f}.png"))
        print(f"Saved: course_{f}.png")

# -----------------------------------------------------------------------------
# Correlations only where variance exists
# -----------------------------------------------------------------------------

corrs = []

for f in course_features:
    col = f"{f}_mean"

    if col not in course_df.columns:
        continue

    s = pd.to_numeric(course_df[col], errors="coerce")

    if s.nunique(dropna=True) <= 1:
        corrs.append({
            "feature": f,
            "corr": np.nan,
            "status": "constant"
        })
        continue

    r = s.corr(course_df["completion_rate"])

    corrs.append({
        "feature": f,
        "corr": r,
        "status": "ok" if pd.notna(r) else "nan_after_corr"
    })

corr_df = pd.DataFrame(corrs)

if not corr_df.empty:
    corr_df = corr_df.sort_values(
        by=["status", "corr"],
        ascending=[True, False],
        na_position="last"
    ).reset_index(drop=True)

print("\nCourse feature correlations:")
print(corr_df.to_string(index=False))

corr_df.to_csv(OUTPUT_DIR / "03_course_feature_correlations.csv", index=False)

# -----------------------------------------------------------------------------
# Final conclusion: what course features do we keep
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("COURSE-LEVEL CONCLUSION")
print("=" * 80)

print("""
В course-level анализе мы оцениваем не поведение конкретного студента,
а влияние самой структуры курса на completion_rate.

Важно:
- если у признака почти нет вариативности между курсами, он не может дать устойчивый сигнал;
- такие признаки нельзя интерпретировать через scatter/correlation;
- это ограничение данных, а не ошибка признака как идеи.
""")

course_features_to_keep = []
course_features_to_keep_descriptive = []
course_features_to_drop = []

for f in course_features:
    if f in valid_course_features:
        course_features_to_keep.append(f)
    elif f in low_variability_course_features:
        course_features_to_keep_descriptive.append(f)
    else:
        course_features_to_drop.append(f)

print("\nKeep for analytical interpretation (enough variability):")
print(course_features_to_keep if course_features_to_keep else "None")

print("\nKeep only as descriptive course metadata (low variability):")
print(course_features_to_keep_descriptive if course_features_to_keep_descriptive else "None")

print("\nDrop from course-level evidence block (constant / non-informative):")
print(course_features_to_drop if course_features_to_drop else "None")

print("\nShort interpretation:")
if len(course_features_to_keep) == 0:
    print(
        "Most course features have too little variability across courses. "
        "This means we cannot make strong statistical claims about course structure in this dataset."
    )
else:
    print(
        "A small subset of course features shows enough variability to be inspected. "
        "These features may support weak-to-moderate hypotheses about course design and completion."
    )

# Итоговый список course-фич, которые оставляем дальше
COURSE_FEATURES_FINAL = course_features_to_keep + course_features_to_keep_descriptive

print("\nFinal course features kept for next EDA blocks:")
print(COURSE_FEATURES_FINAL if COURSE_FEATURES_FINAL else "None")

pd.DataFrame({
    "course_features_final": COURSE_FEATURES_FINAL
}).to_csv(OUTPUT_DIR / "03_course_features_final.csv", index=False)


# # =============================================================================
# # 01. Target distribution by module
# # Проверка survivor bias между M1 и M2
# # =============================================================================

# print("\n" + "=" * 80)
# print("01. TARGET DISTRIBUTION BY MODULE — SURVIVOR BIAS CHECK")
# print("=" * 80)

# # Только labeled train-часть: M1 + M2
# cr_by_module = (
#     df.filter(pl.col("target").is_not_null())
#     .group_by("module")
#     .agg(
#         pl.len().alias("n"),
#         (pl.col("target").sum() / pl.len() * 100).round(1).alias("completion_pct"),
#         ((pl.col("target") == 0).sum()).alias("dropouts"),
#         ((pl.col("target") == 1).sum()).alias("completed"),
#     )
#     .sort("module")
# )

# print("\n── Completion rate by module (train only) ──")
# print(cr_by_module)

# # Сохраним таблицу
# cr_by_module.write_csv(OUTPUT_DIR / "01_completion_by_module_table.csv")

# cr_pd = cr_by_module.to_pandas()

# # Текстовый вывод для консоли / md
# if len(cr_pd) >= 2:
#     m1_rate = cr_pd.loc[cr_pd["module"] == "M1", "completion_pct"]
#     m2_rate = cr_pd.loc[cr_pd["module"] == "M2", "completion_pct"]

#     if len(m1_rate) > 0 and len(m2_rate) > 0:
#         diff = float(m2_rate.iloc[0] - m1_rate.iloc[0])

#         print("\n── Interpretation ──")
#         print(
#             f"M2 completion is {'higher' if diff > 0 else 'lower'} than M1 by {diff:.1f} p.p."
#         )
#         print(
#             "This is consistent with survivor bias: students who reached M2 are already a more selected cohort."
#         )
#     else:
#         print("\nInterpretation: one of the modules is missing in the labeled subset.")
# else:
#     print("\nInterpretation: not enough modules in labeled data to assess survivor bias.")

# subtitle = "  |  ".join(
#     f"{row['module']}: {row['completion_pct']}% completed (n={row['n']})"
#     for _, row in cr_pd.iterrows()
# )

# plot_df = (
#     df.group_by(["module", "target"])
#     .agg(pl.len().alias("count"))
#     .sort(["module", "target"])
#     .with_columns(
#         pl.when(pl.col("target").is_null())
#         .then(pl.lit("M3 (infer)"))
#         .otherwise(pl.col("target").cast(pl.Int64).cast(pl.Utf8))
#         .alias("target_label")
#     )
#     .to_pandas()
# )

# fig = px.bar(
#     plot_df,
#     x="module",
#     y="count",
#     color="target_label",
#     barmode="group",
#     title=(
#         "Target distribution by module<br>"
#         f"<span style='font-size:14px;font-weight:normal'>{subtitle}</span>"
#     ),
#     color_discrete_map={
#         "1": "#27ae60",         # completed
#         "0": "#e74c3c",         # dropout / not completed
#         "M3 (infer)": "#bdc3c7" # unlabeled inference module
#     },
#     labels={
#         "module": "Module",
#         "count": "Rows",
#         "target_label": "Target",
#     },
#     text="count",
# )

# fig.update_traces(
#     texttemplate="%{text}",
#     textposition="outside",
#     cliponaxis=False,
# )

# fig.update_xaxes(title_text="Module")
# fig.update_yaxes(title_text="Rows")
# fig.update_layout(
#     legend_title_text="Target",
#     margin=dict(t=95, l=60, r=40, b=60),
# )

# fig.write_image(str(OUTPUT_DIR / "01_target_by_module.png"), width=900, height=520)
# print("Saved: 01_target_by_module.png")
# print("Saved: 01_completion_by_module_table.csv")