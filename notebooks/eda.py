"""
eda_v4.py — EDA + group-aware CV + inference для предсказания dropout.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import plotly.colors as plc
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Пути ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

df       = pl.read_parquet(DATA_DIR / "df_train.parquet")
df_infer = pl.read_parquet(DATA_DIR / "df_infer.parquet")
print(f"df_train : {df.shape}")
print(f"df_infer : {df_infer.shape}")

# ── Признаки для модели ────────────────────────────────────────────────────
# Только rate/intensity (не накапливаются со временем) + структурные.
# Обоснование выбора — см. eda_notes.md.
# SAFE_HIST = [c for c in df.columns
#              if c.startswith("hist_m1_") or c.startswith("hist_m2_")]

FINAL_FEATURES = [c for c in [
    "time_training_intensity_per_observed_day",   # действий/день
    "time_media_intensity_per_observed_day",       # медиа/день
    "time_to_first_action_days",                   # дней до первого входа
    # *SAFE_HIST,                                    # итоги предыдущих модулей
    "module_num",                                  # survivor bias M1→M2
    "course_id"
] if c in df.columns]

print(f"\nFINAL_FEATURES : {len(FINAL_FEATURES)}")
# print(f"  hist_*       : {len(SAFE_HIST)}")
# print(f"  rate/struct  : {len(FINAL_FEATURES) - len(SAFE_HIST)}")

# ── Вспомогательные функции ────────────────────────────────────────────────
def to_num(df_pd: pd.DataFrame, col: str):
    """Числовые значения колонки + согласованная маска ненулевых target."""
    s    = pd.to_numeric(df_pd[col], errors="coerce")
    mask = s.notna() & df_pd["target"].notna()
    return s[mask], df_pd.loc[mask, "target"]

def find_col(df_pd: pd.DataFrame, keywords: list) -> str | None:
    """Первая колонка, содержащая хотя бы одно ключевое слово."""
    for kw in keywords:
        hits = [c for c in df_pd.columns if kw in c and c != "target"]
        if hits:
            return hits[0]
    return None

# Labeled subset для всех stat-тестов
df_labeled_pd = df.filter(pl.col("target").is_not_null()).to_pandas()
df_m2_pd      = df.filter(pl.col("module") == "M2").to_pandas()

# ══════════════════════════════════════════════════════════════════════════
# 01. Target distribution по модулям — проверка survivor bias
# ══════════════════════════════════════════════════════════════════════════
cr_by_module = (
    df.filter(pl.col("target").is_not_null())
    .group_by("module")
    .agg(
        pl.len().alias("n"),
        (pl.col("target").sum() / pl.len() * 100).round(1).alias("completion_pct"),
        (pl.col("target") == 0).sum().alias("dropouts"),
    )
    .sort("module")
)
print("\n── Completion rate by module ──")
print(cr_by_module)

subtitle = "  |  ".join(
    f"{row['module']}: {row['completion_pct']}% завершили"
    for _, row in cr_by_module.to_pandas().iterrows()
)
fig = px.bar(
    df.group_by(["module", "target"])
      .agg(pl.len().alias("count"))
      .sort(["module", "target"])
      .with_columns(pl.col("target").cast(pl.Utf8).fill_null("M3 (infer)").alias("target_label"))
      .to_pandas(),
    x="module", y="count", color="target_label", barmode="group",
    title=(f"Target distribution by module<br>"
           f"<span style='font-size:14px;font-weight:normal'>"
           f"{subtitle} — Survivor bias подтверждён</span>"),
    color_discrete_map={"1": "#27ae60", "0": "#e74c3c", "M3 (infer)": "#bdc3c7"},
    labels={"target_label": "target"}, text="count",
)
fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
fig.update_xaxes(title_text="Module")
fig.update_yaxes(title_text="Rows")
fig.write_image(str(OUTPUT_DIR / "01_target_by_module.png"), width=900, height=520)
print("Saved: 01_target_by_module.png")

# ══════════════════════════════════════════════════════════════════════════
# 02. Completion rate по course_id
#     Подтверждает необходимость course_id в модели вместо COURSE_CONST
# ══════════════════════════════════════════════════════════════════════════
if "course_id" in df.columns:
    course_cr = (
        df.filter(pl.col("target").is_not_null())
        .group_by("course_id")
        .agg(
            pl.len().alias("n"),
            (pl.col("target").sum() / pl.len() * 100).round(1).alias("completion_pct"),
        )
        .filter(pl.col("n") >= 20)
        .sort("completion_pct", descending=True)
    )
    if len(course_cr) > 0:
        course_pd = course_cr.to_pandas()
        course_pd["course_id"] = course_pd["course_id"].astype(str)
        fig = px.bar(
            course_pd, x="course_id", y="completion_pct",
            title=("Completion rate by course_id<br>"
                   "<span style='font-size:14px;font-weight:normal'>"
                   "Разные курсы — разная сложность → course_id в модель</span>"),
            text=course_pd["n"].apply(lambda x: f"n={x}"),
            color="completion_pct", color_continuous_scale="Blues",
            range_y=[0, course_pd["completion_pct"].max() + 15],
        )
        fig.update_traces(textposition="inside", textfont_size=11, cliponaxis=False)
        fig.update_xaxes(title_text="course_id")
        fig.update_yaxes(title_text="Completion (%)")
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=85))
        fig.write_image(str(OUTPUT_DIR / "02_completion_by_course.png"), width=900, height=480)
        print("Saved: 02_completion_by_course.png")

# ══════════════════════════════════════════════════════════════════════════
# 03. Boxplots: top-9 SAFE features по разрыву медиан
# ══════════════════════════════════════════════════════════════════════════
valid_safe = [
    c for c in FINAL_FEATURES
    if c in df_labeled_pd.columns
    and pd.to_numeric(df_labeled_pd[c], errors="coerce").notna().sum() >= 50
    and pd.to_numeric(df_labeled_pd[c], errors="coerce").nunique() > 1 and c not in ['module_num', 'course_id']
]

median_gaps = {}
for c in valid_safe:
    vals, tgt = to_num(df_labeled_pd, c)
    sub = pd.DataFrame({"v": vals, "t": tgt}).dropna()
    if len(sub) < 50:
        continue
    m1 = sub.loc[sub["t"] == 1, "v"].median()
    m0 = sub.loc[sub["t"] == 0, "v"].median()
    if not (np.isnan(m1) or np.isnan(m0)):
        median_gaps[c] = abs(m1 - m0)

top9 = sorted(median_gaps, key=median_gaps.get, reverse=True)[:9]
print(f"\nTop by median gap: {top9}")

if top9:
    n_cols = 3
    n_rows = (len(top9) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[c.replace("_", " ")[:26] for c in top9],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )
    for i, col in enumerate(top9):
        r, ci = i // n_cols + 1, i % n_cols + 1
        vals, tgt = to_num(df_labeled_pd, col)
        for tval, color, name in [(0, "#e74c3c", "Dropout"), (1, "#27ae60", "Completed")]:
            v = vals[tgt == tval].dropna()
            fig.add_trace(
                go.Box(y=v, name=name, marker_color=color,
                       showlegend=(i == 0), legendgroup=name, boxmean=True),
                row=r, col=ci,
            )
    fig.update_layout(
        title_text="Top SAFE features: Completed vs Dropout",
        height=900, width=1150,
        legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
    )
    fig.write_image(str(OUTPUT_DIR / "03_safe_boxplots.png"), width=1150, height=900)
    print("Saved: 03_safe_boxplots.png")


# ══════════════════════════════════════════════════════════════════════════
# 04. Point-biserial correlation (SAFE features, NaN-safe)
#     Защита: nunique < 2 → пропуск; np.isnan(r) → пропуск
# ══════════════════════════════════════════════════════════════════════════
pb_results = []
for c in valid_safe:
    vals, tgt = to_num(df_labeled_pd, c)
    sub = pd.DataFrame({"v": vals, "t": tgt}).dropna()
    if len(sub) < 100 or sub["v"].nunique() < 2:
        continue
    r, p = stats.pointbiserialr(sub["t"].astype(int), sub["v"])
    if np.isnan(r):
        continue
    pb_results.append({"feature": c, "r": round(r, 4), "p_value": round(p, 6)})

if pb_results:
    pb_df = (
        pd.DataFrame(pb_results)
        .assign(abs_r=lambda x: x["r"].abs())
        .sort_values("abs_r", ascending=False)
        .head(25)
        .sort_values("r")
    )
    pb_df["label"] = pb_df["feature"].str.replace("_", " ").str[:30]

    r_min, r_max = pb_df["r"].min(), pb_df["r"].max()
    r_range = r_max - r_min
    bar_colors = (
        ["#888888"] * len(pb_df)
        if r_range == 0
        else [
            plc.sample_colorscale("RdBu", float((v - r_min) / r_range))[0]
            for v in pb_df["r"]
        ]
    )

    fig = go.Figure(go.Bar(
        x=pb_df["r"], y=pb_df["label"], orientation="h",
        marker_color=bar_colors,
        customdata=pb_df[["p_value"]].values,
        hovertemplate="r=%{x:.4f}<br>p=%{customdata[0]:.6f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_xaxes(title_text="r (point-biserial)")
    fig.update_yaxes(title_text="", tickfont=dict(size=11))
    fig.update_layout(
        title="Point-biserial r с target (SAFE features)",
        height=max(400, len(pb_df) * 28 + 100),
        margin=dict(l=215, r=60, t=60, b=50),
    )
    fig.write_image(str(OUTPUT_DIR / "04_pb_corr.png"), width=950,
                    height=max(400, len(pb_df) * 28 + 100))
    print("Saved: 04_pb_corr.png")

# ══════════════════════════════════════════════════════════════════════════
# 05. Hypothesis tests (Mann-Whitney U, α=0.05)
#     H1/H2/H4 — на всём labeled; H6 — только M2 (hist_m1 = null для M1)
#     H3/H5 исключены: признаки имеют признаки мягкой утечки (см. eda_notes.md)
# ══════════════════════════════════════════════════════════════════════════
hypotheses = [
    ("H1: Интенсивность тренировок/день → dropout",
     find_col(df_labeled_pd, ["training_intensity_per_observed"]),
     df_labeled_pd),
    ("H2: Доля просмотренных медиа → dropout",
     find_col(df_labeled_pd, ["media_view_fraction"]),
     df_labeled_pd),
    ("H4: Скорость первого входа → dropout",
     find_col(df_labeled_pd, ["time_to_first_action", "first_login"]),
     df_labeled_pd),
    # ("H6 [M2]: Оценка
]

hyp_rows = []
for hyp_name, col, data in hypotheses:
    if col is None or col not in data.columns:
        hyp_rows.append(dict(hypothesis=hyp_name, feature="— not found —",
                             med_comp="—", med_drop="—", p_value="—", sig="—"))
        continue
    sub = data[[col, "target"]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 50 or sub[col].nunique() < 2:
        hyp_rows.append(dict(hypothesis=hyp_name, feature=col,
                             med_comp="—", med_drop="—", p_value="n<50/const", sig="—"))
        continue
    g1 = sub.loc[sub["target"] == 1, col]
    g0 = sub.loc[sub["target"] == 0, col]
    _, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    hyp_rows.append(dict(
        hypothesis=hyp_name, feature=col,
        med_comp=round(g1.median(), 3), med_drop=round(g0.median(), 3),
        p_value=f"{p:.5f}", sig="✓" if p < 0.05 else "✗",
    ))

hyp_df = pd.DataFrame(hyp_rows)
print("\n── Hypothesis tests (Mann-Whitney U) ──")
print(hyp_df.to_string(index=False))

fig = go.Figure(data=[go.Table(
    columnwidth=[260, 220, 95, 95, 80, 40],
    header=dict(
        values=["Hypothesis", "Feature", "Med(comp)", "Med(drop)", "p-value", "Sig"],
        fill_color="#2c3e50", font=dict(color="white", size=11),
        align="left", height=34,
    ),
    cells=dict(
        values=[hyp_df[c] for c in
                ["hypothesis", "feature", "med_comp", "med_drop", "p_value", "sig"]],
        fill_color=[["#f4f6f7" if i % 2 == 0 else "#eaf0fb" for i in range(len(hyp_df))]],
        align="left", height=30, font=dict(size=11),
    ),
)])
fig.update_layout(
    title="Hypothesis tests (Mann-Whitney U, α=0.05)",
    height=260, margin=dict(l=10, r=10, t=55, b=10),
)
fig.write_image(str(OUTPUT_DIR / "05_hypothesis_tests.png"), width=1050, height=260)
print("Saved: 05_hypothesis_tests.png")

# ══════════════════════════════════════════════════════════════════════════
# 07. Group-aware CV + финальная модель + inference
#     Группировка по user_id (НЕ users_course_id — он уникален и ломает CV)
# ══════════════════════════════════════════════════════════════════════════
GROUP_COL = "user_id"

df_train_model = (
    df.filter(pl.col("target").is_not_null())
    .select(FINAL_FEATURES + ["target", GROUP_COL])
    .to_pandas()
)

for c in FINAL_FEATURES:
    df_train_model[c] = pd.to_numeric(df_train_model[c], errors="coerce")

if df_train_model["course_id"].dtype == object:
    le = LabelEncoder()
    df_train_model["course_id"] = le.fit_transform(
        df_train_model["course_id"].astype(str)
    )

X_train = df_train_model[FINAL_FEATURES]
y_train = df_train_model["target"].astype(int).to_numpy()
groups  = df_train_model[GROUP_COL].to_numpy()

print(f"\nUnique groups (users): {pd.Series(groups).nunique()}")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "random_forest": make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-999),
        RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    ),
    "logistic_regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="saga", random_state=42,
        ),
    ),
}

cv_scores: dict[str, list[float]] = {name: [] for name in models}
for tr_idx, va_idx in cv.split(X_train, y_train, groups):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    for name, model in models.items():
        m = clone(model)
        m.fit(X_tr, y_tr)
        cv_scores[name].append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))

print("\n── CV results (StratifiedGroupKFold-5, ROC-AUC) ──")
for name, scores in cv_scores.items():
    print(f"  {name}: mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  "
          f"folds={[round(s, 4) for s in scores]}")

best_name  = max(cv_scores, key=lambda k: np.mean(cv_scores[k]))
best_model = clone(models[best_name])
best_model.fit(X_train, y_train)
print(f"Best model: {best_name}")

X_infer   = df_infer.select(FINAL_FEATURES).to_pandas()
for c in FINAL_FEATURES:
    X_infer[c] = pd.to_numeric(X_infer[c], errors="coerce")
if "course_id" in X_infer.columns and X_infer["course_id"].dtype == object:
    X_infer["course_id"] = le.transform(X_infer["course_id"].astype(str))

infer_proba = best_model.predict_proba(X_infer)[:, 1]

infer_out = (
    df_infer
    .select([c for c in ["user_id", "users_course_id", "module"] if c in df_infer.columns])
    .with_columns(pl.Series("pred_proba", infer_proba))
)
infer_out.write_parquet(OUTPUT_DIR / "predictions.parquet")
print(f"Saved: predictions.parquet")

print(f"\nDone. Outputs → {OUTPUT_DIR}/")