## Цель проекта
Проект собирает технически согласованный датасет уровня `users_course_id` для анализа прохождения курсов в LMS.
Пайплайн в `notebooks/` строит master-таблицу признаков из сырых CSV, затем готовит train/infer-выгрузки и выполняет EDA, кросс-валидацию и inference.

## Структура проекта
```text
.
├── data/
│   ├── raw/
│   │   └── *.csv
│   ├── preprocessing/
│   │   ├── final_user_course_features.csv
│   │   ├── M1_M4.csv
│   │   ├── df_train.parquet
│   │   ├── df_infer.parquet
│   │   └── figures/
│   └── eda_output/
│       ├── *.png
│       └── predictions.parquet
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── leave_only_M1_M4.ipynb
│   ├── target_preparation.ipynb
│   ├── eda.py
│   └── old_preprocessing.ipynb
├── scripts/
└── requirements.txt
```

## Структура данных
- `data/raw/`: сырые CSV LMS. По коду используются реестры пользователей и записей на курсы, структура курсов (`lessons`, `lesson_tasks`, `trainings`, `groups`, `homeworks`, `homework_items`), пользовательские логи (`user_lessons`, `user_trainings`, `user_answers`, `wk_users_courses_actions`, `wk_media_view_sessions`, `user_access_histories`, `user_award_badges`) и модульные таблицы `stats__module_1..4`.
- `data/preprocessing/`: выходы подготовки признаков и таргета: `final_user_course_features.csv`, `M1_M4.csv`, `df_train.parquet`, `df_infer.parquet`, а также служебные рисунки в `figures/`.
- `data/eda_output/`: выходы `notebooks/eda.py`: графики EDA и `predictions.parquet`.

## Порядок запуска ноутбуков
1. `notebooks/preprocessing.ipynb`  
   Загружает сырые таблицы из `data/raw`, нормализует ключи и типы, строит базовую сущность `users_course_id`, агрегирует feature-блоки по пользователю, курсу и `user-course`, объединяет их в master-таблицу и добавляет time-based признаки.
2. `notebooks/leave_only_M1_M4.ipynb`  
   Читает `data/preprocessing/final_user_course_features.csv`, оставляет участников программы по наличию `stats_m1_row_count` / `stats_m2_row_count` / `stats_m3_row_count`, проставляет `module` и сохраняет результат в `data/preprocessing/M1_M4.csv`.
3. `notebooks/target_preparation.ipynb`  
   Читает `M1_M4.csv`, строит `target` для `M1` и `M2` через `stats_m1_module_completed_flag` / `stats_m2_module_completed_flag`, формирует `hist_m1_*` и `hist_m2_*`, удаляет leakage-признаки и сохраняет `data/preprocessing/df_train.parquet` и `data/preprocessing/df_infer.parquet`.
4. `notebooks/eda.py`  
   Читает `df_train.parquet` и `df_infer.parquet`, выполняет EDA по размеченной выборке, проверяет гипотезы, сравнивает модели через `StratifiedGroupKFold`, обучает лучшую модель и сохраняет inference в `data/eda_output/predictions.parquet` вместе с графиками.

`notebooks/old_preprocessing.ipynb` присутствует в репозитории, но по текущему коду не входит в основную цепочку: он отдельно читает raw-таблицы и не формирует артефакты, которые затем читают следующие шаги.

## Результат
Финальный артефакт полной цепочки — `data/eda_output/predictions.parquet`.
Он содержит inference-оценки для выборки `M3`; промежуточной основой для него служат `data/preprocessing/df_train.parquet` и `data/preprocessing/df_infer.parquet`.
