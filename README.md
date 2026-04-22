## Цель проекта
Проект собирает согласованный датасет уровня `users_course_id` для анализа прохождения курсов в LMS и раннего прогнозирования риска оттока.
Пайплайн в `notebooks/` строит master-таблицу признаков из сырых CSV, затем готовит train/infer-выгрузки, выполняет EDA, проверку гипотез, сравнение блоков признаков, leakage-aware / leakage-free моделирование и формирует итоговый parquet с предсказаниями.

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
│       ├── *.csv
│       └── prediction.parquet
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── leave_only_M1_M4.ipynb
│   ├── target_preparation.ipynb
│   └── eda.py
├── scripts/
└── requirements.txt
```

## Структура данных

* `data/raw/`: сырые CSV LMS. В коде используются таблицы базовой сущности и доступа к курсам (`users_courses`, `user_access_histories`), профиль пользователя (`users`), структура курса (`lessons`, `lesson_tasks`, `trainings`, `groups`, `homeworks`, `homework_items`), пользовательское поведение (`user_lessons`, `user_trainings`, `user_answers`, `wk_users_courses_actions`, `wk_media_view_sessions`, `user_activity_histories`) и модульные таблицы `stats__module_1..4`.
* `data/preprocessing/`: результаты сборки признаков и подготовки таргета: `final_user_course_features.csv`, `M1_M4.csv`, `df_train.parquet`, `df_infer.parquet`, а также служебные рисунки в `figures/`.
* `data/eda_output/`: результаты основного EDA / modeling-скрипта: графики, таблицы метрик, feature importance, SHAP, risk groups и итоговый `prediction.parquet`.

## Порядок запуска

1. `notebooks/preprocessing.ipynb`
   Загружает сырые таблицы из `data/raw`, нормализует ключи и типы, строит базовую сущность уровня `users_course_id`, агрегирует feature-блоки по пользователю, курсу и user-course, объединяет их в master-таблицу и добавляет time-based признаки.

2. `notebooks/leave_only_M1_M4.ipynb`
   Читает `data/preprocessing/final_user_course_features.csv`, оставляет строки с модульной историей (`M1–M4`), формирует рабочий срез программы, проставляет `module` и сохраняет результат в `data/preprocessing/M1_M4.csv`.

3. `notebooks/target_preparation.ipynb`
   Читает `M1_M4.csv`, строит `target` для размеченных модулей (`M1`, `M2`) через модульные флаги завершения, формирует исторические признаки `hist_m1_*` / `hist_m2_*`, удаляет hard leakage-признаки и сохраняет `data/preprocessing/df_train.parquet` и `data/preprocessing/df_infer.parquet`.

4. `notebooks/eda.py`
   Читает `df_train.parquet` и `df_infer.parquet`, выполняет:

   * EDA по размеченной выборке;
   * проверку гипотез;
   * сравнение подмоделей по блокам признаков;
   * baseline-моделирование;
   * более строгую leakage-free постановку;
   * интерпретацию через feature importance / SHAP;
   * формирование risk groups;
   * экспорт итогового inference-файла `data/eda_output/prediction.parquet`.

## Что делает финальный EDA / modeling script

Скрипт последовательно проходит несколько этапов:

* отбор и первичная интерпретация признаков;
* проверка survivor bias между модулями;
* анализ ранней активности, качества выполнения и course-level нагрузки;
* сравнение моделей по feature-блокам;
* baseline-модель для диагностики общего сигнала;
* переход к более честной leakage-free ранней модели;
* сравнение строгих ранних версий;
* построение risk groups;
* сохранение финальных предсказаний для infer-выборки.

## Результат

Финальный артефакт полной цепочки — `data/eda_output/prediction.parquet`.

Он содержит inference-предсказания для `df_infer.parquet`: идентификаторы пользователя / user-course, риск-скор, риск-группу и ключевые ранние признаки, используемые для интерпретации результата.

## Короткий итог

Проект строит полный пайплайн от сырых LMS-логов до интерпретируемой ранней модели риска оттока.
Основной сигнал идёт от раннего поведения студента, а итоговый результат — parquet с предсказаниями, пригодный для аналитики, дашборда и тьюторского сопровождения.