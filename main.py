"""
Прогнозирование показателей надёжности ЖД — CatBoost и ETS.

Шаги:
1.   Генерация синтетических помесячных данных (годовые итоги 2024 года)
1.5. Оптимизация гиперпараметров CatBoost (Optuna)
2.   Прогноз CatBoost
3.   Прогноз ETS (Holt-Winters AAA)
4.   Сравнение CatBoost и ETS по SMAPE
5.   Построение HTML-графиков
"""

import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from catboost_model import (
    run_catboost_forecast, ALL_PLOT_INDICATORS, TARGET_INDICATORS,
    LAGS, ROLLING_WINDOWS,
)
from ets_model import run_ets_forecast
from generation_config import (
    ANNUAL_DATA_PATH, ANNUAL_YEAR,
    START_YEAR, END_YEAR,
    INDICATOR_CONFIG, SEASONAL_PROFILES,
    OUTPUT_CSV, SEED,
)
from model_optimizer import run_optimization_all, load_best_params
from series_generator import generate_monthly_data
from plotting import plot_predictions

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==================================================
# УПРАВЛЕНИЕ ШАГАМИ
#
# Установи флаг в False, чтобы пропустить шаг и
# загрузить готовый результат из CSV-файла.
# ==================================================
RUN_GENERATION   = False   # Шаг 1:   генерация monthly_df (данные уже есть)
RUN_OPTIMIZATION = False   # Шаг 1.5: оптимизация гиперпараметров CatBoost
RUN_FORECAST     = False   # Шаг 2:   обучение CatBoost и прогноз
RUN_ETS          = False    # Шаг 3:   обучение ETS и прогноз
RUN_COMPARISON   = False    # Шаг 4:   сравнение ETS vs CatBoost по SMAPE
RUN_PLOTS        = True   # Шаг 5:   построение HTML-графиков CatBoost

# Метод оптимизации: "grid" (полный перебор) или "optuna" (байесовский поиск)
OPTIMIZE_METHOD   = "optuna"
OPTIMIZE_N_TRIALS = 20      # число испытаний: 20 вместо 50 → в ~2.5× быстрее
OPTIMIZE_N_SPLITS = 3       # число фолдов CV: 3 вместо 5  → в ~1.7× быстрее

# Визуализация обучения CatBoost (см. habr.com/ru/companies/otus/articles/527554/)
PLOT_TRAINING  = True    # кривые train/val RMSE → catboost_results/training_curves/
PLOT_TREE      = True    # структура дерева SVG  → catboost_results/trees/
PLOT_TREE_IDX  = 0       # индекс дерева для plot_tree (0 = первое дерево)
PLOT_SHAP      = True    # значимость факторов (SHAP) → catboost_results/shap/

MONTHLY_CSV  = OUTPUT_CSV                                   # synthetic_data/synthetic_monthly.csv
FORECAST_CSV = Path("catboost_results/catboost_forecasts.csv")
ETS_CSV      = Path("ets_results/ets_forecasts.csv")
PLOTS_DIR    = Path("catboost_results/plots")

forecast_years = list(range(END_YEAR + 1, END_YEAR + 3))  # [2025, 2026]


# --------------------------------------------------
# 1. Генерация синтетических помесячных данных
#
#    Источник: годовые итоги по каждой дороге за 2024 год
#              (examples/example_year.xlsx)
#    Результат: monthly_df — 16 дорог × N лет × 12 месяцев
#               сохраняется в synthetic_data/synthetic_monthly.csv
# --------------------------------------------------
if RUN_GENERATION:
    monthly_df = generate_monthly_data(
        annual_data_path=ANNUAL_DATA_PATH,
        start_year=START_YEAR,               # 2018
        end_year=END_YEAR,                   # 2024
        annual_year=ANNUAL_YEAR,             # 2024 — год привязки к реальным данным
        indicator_config=INDICATOR_CONFIG,
        seasonal_profiles=SEASONAL_PROFILES,
        seed=SEED,
        output_csv=MONTHLY_CSV,
    )
    print(f"[Шаг 1] Данные сгенерированы: {monthly_df.shape[0]} строк "
          f"({monthly_df['ROAD_NAME'].nunique()} дорог × "
          f"{monthly_df['YEAR'].nunique()} лет × 12 месяцев)")
else:
    monthly_df = pd.read_csv(MONTHLY_CSV, encoding="utf-8-sig")
    print(f"[Шаг 1] Пропущен. Загружено из {MONTHLY_CSV}: {monthly_df.shape[0]} строк")


# --------------------------------------------------
# 1.5. Оптимизация гиперпараметров CatBoost
#
#    Результат — словарь {target: best_params_dict} —
#    передаётся в run_catboost_forecast как
#    catboost_params_per_target.
#
#    Если RUN_OPTIMIZATION=False, модуль пытается
#    загрузить ранее сохранённые параметры из JSON-кеша
#    (catboost_results/best_params/<target>.json).
# --------------------------------------------------
if RUN_OPTIMIZATION:
    best_params_per_target = run_optimization_all(
        monthly_df=monthly_df,
        method=OPTIMIZE_METHOD,
        n_trials=OPTIMIZE_N_TRIALS,
        n_splits=OPTIMIZE_N_SPLITS,
        lags=LAGS,                   # [1, 2, 3, 6, 12, 24, 36] — синхронизировано с моделью
        rolling_windows=ROLLING_WINDOWS,
        test_year=END_YEAR,          # 2024 — тот же год оценки, что в run_catboost_forecast
        seed=SEED,
        outdir="catboost_results",
        force_refit=True,
        # road_norm=True по умолчанию — z-score нормализация, синхронизирована с run_catboost_forecast
    )
    print(f"[Шаг 1.5] Оптимизация завершена для {len(best_params_per_target)} показателей")
else:
    best_params_per_target = {}
    for t in TARGET_INDICATORS:
        cached = load_best_params("catboost_results", t)
        if cached:
            best_params_per_target[t] = cached

    if best_params_per_target:
        print(f"[Шаг 1.5] Пропущен. Загружены кешированные параметры "
              f"для {len(best_params_per_target)} показателей")
    else:
        print("[Шаг 1.5] Пропущен. Используются параметры CatBoost по умолчанию.")


# --------------------------------------------------
# 2. Прогноз CatBoost
# --------------------------------------------------
if RUN_FORECAST:
    forecast_df = run_catboost_forecast(
        monthly_df=monthly_df,
        forecast_years=forecast_years,
        outdir="catboost_results",
        lags=[1, 2, 3, 6, 12, 24, 36],
        rolling_windows=[3, 6, 12],
        test_year=END_YEAR,                  # 2024 — год оценки качества
        random_seed=SEED,
        catboost_params_per_target=best_params_per_target or None,
        plot_training=PLOT_TRAINING,
        plot_tree=PLOT_TREE,
        plot_tree_idx=PLOT_TREE_IDX,
        plot_shap=PLOT_SHAP,
    )
    print(f"[Шаг 2] Прогноз готов: {forecast_df.shape[0]} строк "
          f"(годы {forecast_years[0]}–{forecast_years[-1]})")
else:
    forecast_df = pd.read_csv(FORECAST_CSV, encoding="utf-8-sig")
    print(f"[Шаг 2] Пропущен. Загружено из {FORECAST_CSV}: {forecast_df.shape[0]} строк")


# --------------------------------------------------
# 3. Прогноз ETS (Holt-Winters AAA)
#
#    Для каждого из 7 прямых показателей и каждой дороги
#    обучается отдельная ETS-модель с fallback AAA → AA → A.
#    Производные вычисляются из прогнозов.
#    Доверительный интервал: ± 1.96 σ (реальные остатки теста).
# --------------------------------------------------
if RUN_ETS:
    print(f"\n[Шаг 3] ETS прогноз (годы {forecast_years[0]}–{forecast_years[-1]})...")
    ets_df = run_ets_forecast(
        monthly_df=monthly_df,
        forecast_years=forecast_years,
        outdir="ets_results",
        test_year=END_YEAR,          # 2024 — год оценки качества
        plot_html=True,
        plots_dir="ets_results/plots",
    )
    print(f"[Шаг 3] ETS готов: {ets_df.shape[0]} строк "
          f"(годы {forecast_years[0]}–{forecast_years[-1]})")
else:
    if ETS_CSV.exists():
        ets_df = pd.read_csv(ETS_CSV, encoding="utf-8-sig")
        print(f"[Шаг 3] Пропущен. Загружено из {ETS_CSV}: {ets_df.shape[0]} строк")
    else:
        ets_df = None
        print(f"[Шаг 3] Пропущен. Файл {ETS_CSV} не найден.")


# --------------------------------------------------
# 4. Сравнение ETS vs CatBoost по SMAPE
#
#    Результаты сохраняются в compare_resuls/:
#      - smape_comparison_detailed.csv      (indicator × road, COUNT+SUM агрегированы)
#      - smape_comparison_by_indicator.csv  (сводный по показателю)
#      - smape_comparison_split.csv         (indicator × road, COUNT и SUM раздельно)
# --------------------------------------------------
if RUN_COMPARISON:
    from smape_comparison import build_smape_comparison
    print(f"\n[Шаг 4] Сравнение ETS vs CatBoost по SMAPE...")
    try:
        build_smape_comparison(
            ets_metrics_path="ets_results/ets_metrics.csv",
            catboost_metrics_path="catboost_results/catboost_metrics.csv",
            detailed_out="compare_resuls/smape_comparison_detailed.csv",
            summary_out="compare_resuls/smape_comparison_by_indicator.csv",
            split_out="compare_resuls/smape_comparison_split.csv",
        )
    except FileNotFoundError as e:
        print(f"[Шаг 4] Пропущен: {e}")
else:
    print("[Шаг 4] Пропущен.")


# --------------------------------------------------
# 5. Визуализация CatBoost — интерактивные HTML-графики
#
#    Для каждой дороги и каждого целевого показателя
#    строится HTML-файл с:
#      - историческими данными (тёмная линия)
#      - прогнозом CatBoost (синяя пунктирная линия)
#      - доверительным интервалом 95 % (синяя заливка)
#      - вертикальной границей история / прогноз
#
#    ETS-графики строятся внутри run_ets_forecast (шаг 3).
# --------------------------------------------------
if RUN_PLOTS:
    roads = monthly_df["ROAD_NAME"].unique()
    print(f"\n[Шаг 5] Построение графиков CatBoost -> {PLOTS_DIR}/")

    for indicator in ALL_PLOT_INDICATORS:
        for road in roads:
            road_slug = road.replace(" ", "_").replace("-", "_")
            outpath = PLOTS_DIR / indicator / f"{road_slug}.html"
            plot_predictions(
                monthly_df=monthly_df,
                forecast_df=forecast_df,
                indicator=indicator,
                road=road,
                outpath=outpath,
            )

    print(f"[OK] Графики CatBoost сохранены: {len(ALL_PLOT_INDICATORS)} показателей × "
          f"{len(roads)} дорог = {len(ALL_PLOT_INDICATORS) * len(roads)} файлов")
else:
    print("[Шаг 5] Пропущен!")