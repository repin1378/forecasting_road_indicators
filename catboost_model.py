"""
catboost_model.py

CatBoost-прогнозирование показателей надёжности ЖД.

Архитектура
-----------
• Входные данные — плоский DataFrame формата generate_monthly_data:
  ROAD_NAME, YEAR, MONTH + все показатели.

• Категориальные признаки (передаются CatBoost через cat_features):
    ROAD_NAME  — название дороги
    YEAR       — год
    MONTH      — номер месяца

• Целевые переменные (для каждой обучается отдельная модель):
    TRAIN_KM, LOSS_12_COUNT, LOSS_12_SUM,
    LOSS_3_COUNT, LOSS_3_SUM, LOSS_TECH_COUNT, LOSS_TECH_SUM

• Производные показатели (вычисляются из прогнозов, не прогнозируются):
    LOSS_COUNT_TOTAL = LOSS_12_COUNT + LOSS_3_COUNT + LOSS_TECH_COUNT
    LOSS_SUM_TOTAL   = LOSS_12_SUM   + LOSS_3_SUM   + LOSS_TECH_SUM
    SPECIFIC_LOSS    = LOSS_SUM_TOTAL / TRAIN_KM * 1 000 000

• Глобальная модель: обучается на всех дорогах сразу — ROAD_NAME
  как категориальный признак даёт перенос знаний между дорогами.

• Рекурсивный прогноз: для каждого будущего месяца лаги вычисляются
  по уже предсказанным значениям предыдущих шагов.

Использование
-------------
    from catboost_model import run_catboost_forecast
    from generation_config import ANNUAL_DATA_PATH, ANNUAL_YEAR, ...

    monthly_df = generate_monthly_data(...)
    forecast_df = run_catboost_forecast(
        monthly_df=monthly_df,
        forecast_years=[2025, 2026],
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from catboost import CatBoostRegressor, Pool


# ============================================================
# КОНСТАНТЫ
# ============================================================

TARGET_INDICATORS: List[str] = [
    "TRAIN_KM",
    "LOSS_12_COUNT", "LOSS_12_SUM",
    "LOSS_3_COUNT",  "LOSS_3_SUM",
    "LOSS_TECH_COUNT", "LOSS_TECH_SUM",
]

# Производные показатели — вычисляются из прогнозов, не прогнозируются отдельно
DERIVED_INDICATORS: List[str] = [
    "LOSS_COUNT_TOTAL", "LOSS_SUM_TOTAL", "SPECIFIC_LOSS",
]

# Все показатели для визуализации (прямые + производные)
ALL_PLOT_INDICATORS: List[str] = TARGET_INDICATORS + DERIVED_INDICATORS

# Ширина доверительного интервала (доля от прогноза): ±5 %
CI_HALF_WIDTH: float = 0.05

# Категориальные признаки (передаются CatBoost через cat_features)
CAT_FEATURES: List[str] = ["ROAD_NAME", "YEAR", "MONTH"]

# Счётчики — после прогноза округляются до целых
COUNT_COLS: List[str] = [
    "LOSS_12_COUNT", "LOSS_3_COUNT", "LOSS_TECH_COUNT", "LOSS_COUNT_TOTAL",
]

LAGS: List[int] = [1, 2, 3, 6, 12]
ROLLING_WINDOWS: List[int] = [3, 6, 12]

# Параметры CatBoost по умолчанию (см. статьи по настройке)
DEFAULT_CB_PARAMS: dict = {
    "loss_function":      "RMSE",
    "iterations":         2000,
    "learning_rate":      0.03,
    "depth":              6,
    "l2_leaf_reg":        3.0,
    "random_strength":    1.0,
    "bagging_temperature": 1.0,
    "use_best_model":     True,
    "early_stopping_rounds": 50,
    "verbose":            False,
}


# ============================================================
# ПРОИЗВОДНЫЕ ПОКАЗАТЕЛИ
# ============================================================

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет LOSS_COUNT_TOTAL, LOSS_SUM_TOTAL, SPECIFIC_LOSS из прогнозов."""
    df = df.copy()
    df["LOSS_COUNT_TOTAL"] = (
        df["LOSS_12_COUNT"] + df["LOSS_3_COUNT"] + df["LOSS_TECH_COUNT"]
    )
    df["LOSS_SUM_TOTAL"] = (
        df["LOSS_12_SUM"] + df["LOSS_3_SUM"] + df["LOSS_TECH_SUM"]
    )
    safe_km = df["TRAIN_KM"].replace(0, np.nan)
    df["SPECIFIC_LOSS"] = df["LOSS_SUM_TOTAL"] / safe_km * 1_000_000
    return df


# ============================================================
# МЕТРИКИ
# ============================================================

def mape_percent(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape_percent(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _add_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int],
    windows: List[int],
) -> pd.DataFrame:
    """
    Добавляет к df временны́е признаки для target_col.

    Признаки строятся внутри каждой дороги (groupby ROAD_NAME) — гарантирует
    отсутствие утечки между дорогами.

    Признаки:
        month_sin / month_cos — циклическое кодирование сезонности
        t                     — линейный индекс времени (тренд)
        lag_{k}               — значение за k месяцев назад
        roll_mean_{w}         — скользящее среднее за w месяцев (сдвиг 1)
        roll_std_{w}          — скользящее СКО за w месяцев (сдвиг 1)
        diff_1 / diff_12      — разности 1-го и 12-го порядка
    """
    df = df.sort_values(["ROAD_NAME", "YEAR", "MONTH"]).reset_index(drop=True)

    # Циклическое кодирование месяца
    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)

    # Линейный тренд (порядковый номер месяца внутри дороги)
    df["t"] = df.groupby("ROAD_NAME").cumcount()

    # Признаки строятся отдельно для каждой дороги
    for road, gdf in df.groupby("ROAD_NAME", sort=False):
        idx = gdf.index
        vals = gdf[target_col]

        # Лаги
        for lag in lags:
            df.loc[idx, f"lag_{lag}"] = vals.shift(lag).values

        # Скользящие статистики по предыдущим значениям (shift=1, нет утечки)
        shifted = vals.shift(1)
        for w in windows:
            df.loc[idx, f"roll_mean_{w}"] = (
                shifted.rolling(w, min_periods=1).mean().values
            )
            df.loc[idx, f"roll_std_{w}"] = (
                shifted.rolling(w, min_periods=1).std(ddof=0).fillna(0).values
            )

        # Разности
        df.loc[idx, "diff_1"]  = vals.diff(1).values
        df.loc[idx, "diff_12"] = vals.diff(12).values

    return df


def _feature_cols(lags: List[int], windows: List[int]) -> List[str]:
    """Возвращает упорядоченный список признаков (без целевой переменной)."""
    cols = list(CAT_FEATURES)                        # ROAD_NAME, YEAR, MONTH
    cols += ["month_sin", "month_cos", "t"]          # сезонность + тренд
    cols += [f"lag_{l}"       for l in lags]         # лаги
    cols += [f"roll_mean_{w}" for w in windows]      # скользящие средние
    cols += [f"roll_std_{w}"  for w in windows]      # скользящие СКО
    cols += ["diff_1", "diff_12"]                    # разности
    return cols


# ============================================================
# РЕКУРСИВНЫЙ ПРОГНОЗ
# ============================================================

def _recursive_forecast(
    model: CatBoostRegressor,
    history_df: pd.DataFrame,
    target_col: str,
    forecast_years: List[int],
    lags: List[int],
    windows: List[int],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Рекурсивный прогноз для всех дорог на forecast_years.

    На каждом шаге (year, month):
      1. Строки с NaN-target добавляются к растущей истории.
      2. _add_features пересчитывает лаги с учётом уже предсказанных значений.
      3. Прогнозы записываются обратно в историю для следующего шага.

    Возвращает DataFrame: ROAD_NAME, YEAR, MONTH, target_col
    """
    roads = history_df["ROAD_NAME"].unique()
    work = history_df[["ROAD_NAME", "YEAR", "MONTH", target_col]].copy()
    results: List[pd.DataFrame] = []

    for year in sorted(forecast_years):
        for month in range(1, 13):
            # Строки прогноза: все дороги, один месяц
            future = pd.DataFrame({
                "ROAD_NAME": roads,
                "YEAR":      year,
                "MONTH":     month,
                target_col:  np.nan,
            })

            # Объединяем историю + будущую строку, строим признаки
            combined = pd.concat([work, future], ignore_index=True)
            combined = _add_features(combined, target_col, lags, windows)

            # Отбираем строки прогноза по NaN-target
            pred_mask = combined[target_col].isna()
            X_pred = combined.loc[pred_mask, feature_cols]

            # Тип ROAD_NAME должен быть строкой для CatBoost
            X_pred = X_pred.copy()
            X_pred["ROAD_NAME"] = X_pred["ROAD_NAME"].astype(str)

            # NaN в лаговых признаках (начало истории) → 0
            X_pred = X_pred.fillna(0)

            preds = np.maximum(model.predict(X_pred), 0.0)

            # Записываем прогнозы
            future[target_col] = preds
            work = pd.concat(
                [work, future[["ROAD_NAME", "YEAR", "MONTH", target_col]]],
                ignore_index=True,
            )
            results.append(future)

    return pd.concat(results, ignore_index=True)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_catboost_forecast(
    monthly_df: pd.DataFrame,
    forecast_years: List[int],
    outdir: str = "catboost_results",
    lags: List[int] = None,
    rolling_windows: List[int] = None,
    test_year: Optional[int] = None,
    random_seed: int = 42,
    catboost_params: Optional[dict] = None,
    catboost_params_per_target: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    Прогнозирование всех целевых показателей для всех дорог.

    Параметры
    ----------
    monthly_df               : DataFrame из generate_monthly_data
    forecast_years           : список лет прогноза, напр. [2025, 2026]
    outdir                   : директория для CSV и метрик
    lags                     : список лагов (по умолчанию LAGS)
    rolling_windows          : размеры окон (по умолчанию ROLLING_WINDOWS)
    test_year                : год тестирования (по умолчанию max(YEAR))
    random_seed              : зерно воспроизводимости
    catboost_params          : глобальное переопределение параметров CatBoost
                               (применяется ко всем показателям)
    catboost_params_per_target : словарь {target: params_dict} с оптимальными
                               параметрами для каждого показателя отдельно.
                               Берётся из run_optimization_all() / load_best_params().
                               Имеет приоритет над catboost_params.

    Возвращает
    ----------
    pd.DataFrame с прогнозами в формате:
        ROAD_NAME, YEAR, MONTH, TRAIN_KM, LOSS_12_COUNT, LOSS_12_SUM,
        LOSS_3_COUNT, LOSS_3_SUM, LOSS_TECH_COUNT, LOSS_TECH_SUM,
        LOSS_COUNT_TOTAL, LOSS_SUM_TOTAL, SPECIFIC_LOSS

    Доверительные интервалы (±5 % от прогноза) рассчитываются при
    визуализации в plotting.plot_predictions — отдельных моделей не требуется.
    """
    lags = lags or LAGS
    rolling_windows = rolling_windows or ROLLING_WINDOWS

    base_params = {**DEFAULT_CB_PARAMS, "random_seed": random_seed}
    if catboost_params:
        base_params.update(catboost_params)

    base_out = Path(outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    if test_year is None:
        test_year = int(monthly_df["YEAR"].max())

    # Убеждаемся, что ROAD_NAME — строка (CatBoost требует str для cat)
    monthly_df = monthly_df.copy()
    monthly_df["ROAD_NAME"] = monthly_df["ROAD_NAME"].astype(str)

    feat_cols = _feature_cols(lags, rolling_windows)

    forecast_parts: Dict[str, pd.DataFrame] = {}   # target → прогнозный DataFrame
    metrics_rows: List[dict] = []

    for target in TARGET_INDICATORS:
        # Параметры: база → глобальный override → per-target override
        params = base_params.copy()
        if catboost_params_per_target and target in catboost_params_per_target:
            params.update(catboost_params_per_target[target])

        print(f"  [{target}] обучение модели...")

        # ── Признаки на полных данных ──────────────────────
        full_feat = _add_features(monthly_df, target, lags, rolling_windows)

        train_feat = full_feat[full_feat["YEAR"] <  test_year].dropna(subset=feat_cols + [target])
        test_feat  = full_feat[full_feat["YEAR"] == test_year].fillna(0)

        X_train = train_feat[feat_cols]
        y_train = train_feat[target].values

        X_test  = test_feat[feat_cols]
        y_test  = test_feat[target].values

        # ── CatBoost Pool (категориальные признаки по имени) ──
        cb_cat = [c for c in CAT_FEATURES if c in feat_cols]

        train_pool = Pool(X_train, label=y_train, cat_features=cb_cat)
        eval_pool  = Pool(X_test,  label=y_test,  cat_features=cb_cat)

        # ── Основная модель ────────────────────────────────
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=eval_pool)

        # ── Метрики на тесте ───────────────────────────────
        pred_test = np.maximum(model.predict(eval_pool), 0.0)

        for road in monthly_df["ROAD_NAME"].unique():
            mask = test_feat["ROAD_NAME"] == road
            if mask.sum() == 0:
                continue
            metrics_rows.append({
                "indicator":  target,
                "road":       road,
                "MAPE_%":     mape_percent(y_test[mask.values], pred_test[mask.values]),
                "SMAPE_%":    smape_percent(y_test[mask.values], pred_test[mask.values]),
                "n_trees":    model.tree_count_,
                "test_year":  test_year,
            })

        # ── Рекурсивный прогноз ────────────────────────────
        forecast_part = _recursive_forecast(
            model=model,
            history_df=monthly_df[["ROAD_NAME", "YEAR", "MONTH", target]],
            target_col=target,
            forecast_years=forecast_years,
            lags=lags,
            windows=rolling_windows,
            feature_cols=feat_cols,
        )
        forecast_parts[target] = forecast_part.set_index(
            ["ROAD_NAME", "YEAR", "MONTH"]
        )[target]

        print(f"  [{target}] готово. Деревьев: {model.tree_count_}")

    # ── Сборка итогового DataFrame ─────────────────────────
    idx = pd.MultiIndex.from_product(
        [sorted(monthly_df["ROAD_NAME"].unique()), forecast_years, range(1, 13)],
        names=["ROAD_NAME", "YEAR", "MONTH"],
    )
    out = pd.DataFrame(index=idx)

    for target in TARGET_INDICATORS:
        out[target] = forecast_parts[target]

    out = out.reset_index()

    # Неотрицательность
    for col in TARGET_INDICATORS:
        out[col] = out[col].clip(lower=0)

    # Округление счётчиков
    for col in [c for c in COUNT_COLS if c in out.columns]:
        out[col] = out[col].round().astype(int)

    # Производные показатели
    out = compute_derived(out)

    # Доверительные интервалы ± CI_HALF_WIDTH для всех показателей
    for col in ALL_PLOT_INDICATORS:
        if col in out.columns:
            out[f"{col}_lower_95"] = (out[col] * (1.0 - CI_HALF_WIDTH)).clip(lower=0)
            out[f"{col}_upper_95"] =  out[col] * (1.0 + CI_HALF_WIDTH)

    # Порядок столбцов: сначала значения, затем CI-пары рядом с каждым показателем
    id_cols = ["ROAD_NAME", "YEAR", "MONTH"]
    value_ci_cols = []
    for col in ALL_PLOT_INDICATORS:
        value_ci_cols.append(col)
        value_ci_cols.append(f"{col}_lower_95")
        value_ci_cols.append(f"{col}_upper_95")
    out = out[id_cols + value_ci_cols].sort_values(["ROAD_NAME", "YEAR", "MONTH"]).reset_index(drop=True)

    # ── Сохранение ─────────────────────────────────────────
    forecast_path = base_out / "catboost_forecasts.csv"
    metrics_path  = base_out / "catboost_metrics.csv"

    out.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Прогнозы сохранены: {forecast_path}")
    print(f"✅ Метрики сохранены:   {metrics_path}")
    print(f"   Строк в прогнозе:   {len(out)}")

    return out
