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
from typing import Dict, List, Optional, Tuple

from catboost import CatBoostRegressor, Pool
from catboost_viz import save_training_curves, save_tree, save_shap_report


# ============================================================
# КОНСТАНТЫ
# ============================================================

TARGET_INDICATORS: List[str] = [
    "TRAIN_KM",
    "LOSS_12_COUNT", "LOSS_12_SUM",
    "LOSS_3_COUNT",  "LOSS_3_SUM",
    "LOSS_TECH_COUNT", "LOSS_TECH_SUM",
]

# Показатели, обучаемые одиночной моделью (loss_function="RMSE")
SINGLE_TARGETS: List[str] = ["TRAIN_KM"]

# Группы для совместного обучения (loss_function="MultiRMSE").
# COUNT и SUM одного типа отказов тесно коррелированы — совместная модель
# учитывает их взаимосвязь и предотвращает противоречия в прогнозах.
MULTI_TARGET_GROUPS: Dict[str, List[str]] = {
    "LOSS_12":   ["LOSS_12_COUNT",   "LOSS_12_SUM"],
    "LOSS_3":    ["LOSS_3_COUNT",    "LOSS_3_SUM"],
    "LOSS_TECH": ["LOSS_TECH_COUNT", "LOSS_TECH_SUM"],
}

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

LAGS: List[int] = [1, 2, 3, 6, 12, 24, 36]
ROLLING_WINDOWS: List[int] = [3, 6, 12]

# Лаги для кросс-индикаторных признаков (короче, чтобы не раздувать матрицу)
CROSS_LAGS: List[int] = [1, 3, 6, 12]

# TRAIN_KM используется как признак-предиктор для LOSS_* моделей:
# объём работ напрямую влияет на число и сумму потерь от отказов.
CROSS_INDICATORS: List[str] = ["TRAIN_KM"]

# Минимальное std при z-score нормализации (защита от деления на 0 у малых/постоянных рядов).
# Значение 1.0 нейтрально: при std < 1 нормализация фактически превращается в сдвиг на mean.
ROAD_NORM_MIN_STD: float = 1.0

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

def mape_percent(y_true, y_pred, cap: float = 200.0) -> float:
    """
    MAPE (%), устойчивый к нулям.

    Вместо деления на абсолютное значение (взрывается при y_true ≈ 0)
    используем знаменатель = max(|y_true|, 1% от среднего ряда).
    Вклад каждой точки ограничен значением cap (по умолчанию 200 %),
    чтобы единственный нулевой месяц не искажал среднее.

    При y_true ≡ 0 (все нули) возвращает 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Порог = 1 % от среднего ненулевого значения ряда (но не меньше 1e-6)
    mean_abs = np.abs(y_true[y_true != 0]).mean() if np.any(y_true != 0) else 1.0
    floor    = max(mean_abs * 0.01, 1e-6)
    denom    = np.maximum(np.abs(y_true), floor)
    per_pt   = np.minimum(np.abs((y_true - y_pred) / denom) * 100.0, cap)
    return float(np.mean(per_pt))


def smape_percent(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


# ============================================================
# Z-SCORE НОРМАЛИЗАЦИЯ ПО ДОРОГАМ
# ============================================================

def _compute_road_stats(
    df: pd.DataFrame,
    target_col: str,
    road_col: str = "ROAD_NAME",
    min_std: float = ROAD_NORM_MIN_STD,
) -> Dict[str, Tuple[float, float]]:
    """
    Вычисляет (mean, std) таргета отдельно по каждой дороге.

    Назначение: z-score нормализация перед обучением — решает проблему
    «Калининград vs Московская», когда разница масштабов на порядок
    приводит к тому, что глобальная модель усредняет несопоставимые ряды.

    min_std: нижняя граница std (защита от деления на 0 для постоянных рядов).
    Вычисляется только по ненулевым ненулевым наблюдениям train-части —
    данные test/forecast не участвуют (нет утечки).
    """
    stats: Dict[str, Tuple[float, float]] = {}
    for road, group in df.groupby(road_col):
        vals = group[target_col].dropna().values.astype(float)
        mean = float(vals.mean()) if len(vals) > 0 else 0.0
        std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        stats[str(road)] = (mean, max(std, min_std))
    return stats


def _normalize_target(
    y: np.ndarray,
    roads: np.ndarray,
    road_stats: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """
    Применяет z-score нормализацию: y_norm = (y - mean[road]) / std[road].

    Векторизованная реализация — без Python-цикла по строкам.
    Дороги, отсутствующие в road_stats, получают (mean=0, std=1) — нейтральное значение.
    """
    roads = np.asarray(roads, dtype=str)
    means = np.array([road_stats.get(r, (0.0, 1.0))[0] for r in roads])
    stds  = np.array([road_stats.get(r, (0.0, 1.0))[1] for r in roads])
    return (np.asarray(y, dtype=float) - means) / stds


def _denormalize_target(
    y_norm: np.ndarray,
    roads: np.ndarray,
    road_stats: Dict[str, Tuple[float, float]],
    clip_zero: bool = True,
) -> np.ndarray:
    """
    Обратная трансформация: y = y_norm * std[road] + mean[road].

    clip_zero=True: отрицательные значения обнуляются (счётчики и суммы ≥ 0).
    """
    roads = np.asarray(roads, dtype=str)
    means = np.array([road_stats.get(r, (0.0, 1.0))[0] for r in roads])
    stds  = np.array([road_stats.get(r, (0.0, 1.0))[1] for r in roads])
    y = np.asarray(y_norm, dtype=float) * stds + means
    return np.maximum(y, 0.0) if clip_zero else y


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
        road_mean / road_std  — исторический масштаб дороги по показателю
                                (вычисляется по ненулевым наблюдениям, без NaN)
        month_sin / month_cos — циклическое кодирование сезонности
        t                     — линейный индекс времени (тренд)
        lag_{k}               — значение за k месяцев назад
        roll_mean_{w}         — скользящее среднее за w месяцев (сдвиг 1)
        roll_std_{w}          — скользящее СКО за w месяцев (сдвиг 1)
        diff_1 / diff_12      — разности 1-го и 12-го порядка

    road_mean / road_std — ключевые признаки для глобальной модели:
    сообщают CatBoost, что Калининградская дорога работает на порядок меньшем
    масштабе, чем Приволжская, и это нормально, а не аномалия.
    Вычисляются только по наблюдаемым (не-NaN) значениям — в рекурсивном
    прогнозе будущие строки имеют NaN, но получают корректный исторический σ.
    """
    df = df.sort_values(["ROAD_NAME", "YEAR", "MONTH"]).reset_index(drop=True)

    # ── Масштабные признаки дороги ─────────────────────────────
    # Считаем только по ненулевым ненулевым наблюдениям (исторические строки).
    observed = df[df[target_col].notna()]
    road_mean = observed.groupby("ROAD_NAME")[target_col].mean()
    road_std  = observed.groupby("ROAD_NAME")[target_col].std().fillna(0)
    df["road_mean"] = df["ROAD_NAME"].map(road_mean).fillna(0)
    df["road_std"]  = df["ROAD_NAME"].map(road_std).fillna(0)

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

        # YoY-коэффициент: lag_1 / lag_13 = прошлый месяц / тот же месяц год назад.
        # shift(13) вместо shift(12) исключает утечку текущего значения.
        # fillna(1.0) — нейтральное значение при нехватке истории.
        lag1  = vals.shift(1)
        lag13 = vals.shift(13).replace(0, np.nan)
        df.loc[idx, "yoy_ratio"] = (lag1 / lag13).fillna(1.0).values

    return df


def _feature_cols(lags: List[int], windows: List[int]) -> List[str]:
    """Возвращает упорядоченный список признаков (без целевой переменной)."""
    cols = list(CAT_FEATURES)                        # ROAD_NAME, YEAR, MONTH
    cols += ["road_mean", "road_std"]                # масштаб дороги
    cols += ["month_sin", "month_cos", "t"]          # сезонность + тренд
    cols += [f"lag_{l}"       for l in lags]         # лаги
    cols += [f"roll_mean_{w}" for w in windows]      # скользящие средние
    cols += [f"roll_std_{w}"  for w in windows]      # скользящие СКО
    cols += ["diff_1", "diff_12", "yoy_ratio"]       # разности + YoY-коэффициент
    return cols


# ============================================================
# FEATURE ENGINEERING — MULTI-TARGET
# ============================================================

def _add_features_multi(
    df: pd.DataFrame,
    targets: List[str],
    lags: List[int],
    windows: List[int],
    cross_cols: Optional[List[str]] = None,
    cross_lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Строит матрицу признаков для нескольких целевых переменных одновременно.

    Для каждого таргета создаются признаки с суффиксом _{TARGET}:
        road_mean_{T} / road_std_{T} — масштаб дороги по каждому таргету
        lag_k_{T}, roll_mean_w_{T}, diff_12_{T} …
        yoy_ratio_{T}               — lag_1 / lag_13 (YoY-коэффициент без утечки)

    cross_cols (напр. ["TRAIN_KM"]): дополнительные столбцы, для которых строятся
        только лаги (cross_lags). df должен содержать эти столбцы.
        Переносят информацию из смежных показателей: объём работы TRAIN_KM →
        ожидаемое число и сумму потерь от отказов.

    Общие структурные признаки (month_sin, month_cos, t) добавляются один раз.
    """
    cross_cols = cross_cols or []
    cross_lags = cross_lags or []

    df = df.sort_values(["ROAD_NAME", "YEAR", "MONTH"]).reset_index(drop=True)

    # ── Масштабные признаки дороги (по каждому таргету) ────────
    for target in targets:
        observed = df[df[target].notna()]
        road_mean = observed.groupby("ROAD_NAME")[target].mean()
        road_std  = observed.groupby("ROAD_NAME")[target].std().fillna(0)
        df[f"road_mean_{target}"] = df["ROAD_NAME"].map(road_mean).fillna(0)
        df[f"road_std_{target}"]  = df["ROAD_NAME"].map(road_std).fillna(0)

    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    df["t"] = df.groupby("ROAD_NAME").cumcount()

    for road, gdf in df.groupby("ROAD_NAME", sort=False):
        idx = gdf.index

        # ── Основные признаки по каждому таргету ───────────────
        for target in targets:
            vals = gdf[target]
            for lag in lags:
                df.loc[idx, f"lag_{lag}_{target}"] = vals.shift(lag).values
            shifted = vals.shift(1)
            for w in windows:
                df.loc[idx, f"roll_mean_{w}_{target}"] = (
                    shifted.rolling(w, min_periods=1).mean().values
                )
                df.loc[idx, f"roll_std_{w}_{target}"] = (
                    shifted.rolling(w, min_periods=1).std(ddof=0).fillna(0).values
                )
            df.loc[idx, f"diff_1_{target}"]  = vals.diff(1).values
            df.loc[idx, f"diff_12_{target}"] = vals.diff(12).values

            # YoY-коэффициент (lag_1 / lag_13): рост последнего месяца к
            # тому же месяцу прошлого года. fillna(1.0) — нейтральное значение.
            lag1  = vals.shift(1)
            lag13 = vals.shift(13).replace(0, np.nan)
            df.loc[idx, f"yoy_ratio_{target}"] = (lag1 / lag13).fillna(1.0).values

        # ── Кросс-индикаторные лаги (напр. TRAIN_KM) ───────────
        for ccol in cross_cols:
            if ccol not in gdf.columns:
                continue
            cvals = gdf[ccol]
            for lag in cross_lags:
                df.loc[idx, f"lag_{lag}_{ccol}"] = cvals.shift(lag).values

    return df


def _feature_cols_multi(
    targets: List[str],
    lags: List[int],
    windows: List[int],
    cross_cols: Optional[List[str]] = None,
    cross_lags: Optional[List[int]] = None,
) -> List[str]:
    """Возвращает список признаков для многоцелевой модели."""
    cross_cols = cross_cols or []
    cross_lags = cross_lags or []

    cols = list(CAT_FEATURES)               # ROAD_NAME, YEAR, MONTH
    cols += ["month_sin", "month_cos", "t"] # сезонность + тренд
    for target in targets:
        cols += [f"road_mean_{target}", f"road_std_{target}"]   # масштаб дороги
        cols += [f"lag_{l}_{target}"        for l in lags]
        cols += [f"roll_mean_{w}_{target}"  for w in windows]
        cols += [f"roll_std_{w}_{target}"   for w in windows]
        cols += [f"diff_1_{target}", f"diff_12_{target}"]
        cols += [f"yoy_ratio_{target}"]                         # YoY-коэффициент
    for ccol in cross_cols:
        cols += [f"lag_{l}_{ccol}" for l in cross_lags]         # кросс-индикаторы
    return cols


# ============================================================
# РЕКУРСИВНЫЙ ПРОГНОЗ — MULTI-TARGET
# ============================================================

def _recursive_forecast_multi(
    model,
    history_df: pd.DataFrame,
    targets: List[str],
    forecast_years: List[int],
    lags: List[int],
    windows: List[int],
    feature_cols: List[str],
    cross_cols: Optional[List[str]] = None,
    cross_lags: Optional[List[int]] = None,
    cross_future_df: Optional[pd.DataFrame] = None,
    road_stats_multi: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> pd.DataFrame:
    """
    Рекурсивный прогноз для MultiRMSE-модели (несколько таргетов одновременно).

    На каждом шаге (year, month):
      1. Строки с NaN-таргетами добавляются к растущей истории.
      2. _add_features_multi пересчитывает лаги по уже предсказанным значениям,
         включая лаги кросс-индикаторов из cross_future_df (напр. TRAIN_KM).
      3. Оба таргета предсказываются одновременно — модель учитывает их связь.
      4. Предсказания денормализуются (z-score → исходный масштаб) перед
         записью в историю — история всегда хранится в исходном масштабе,
         что гарантирует корректность лагов на следующем шаге.

    cross_future_df (опционально): DataFrame с колонками
        (ROAD_NAME, YEAR, MONTH, *cross_cols) для лет forecast_years.
        Передаётся прогноз TRAIN_KM, полученный на предыдущем шаге.

    Возвращает DataFrame: ROAD_NAME, YEAR, MONTH, *targets
    """
    cross_cols = cross_cols or []
    cross_lags = cross_lags or []

    roads = history_df["ROAD_NAME"].unique()

    # work хранит историю в исходном масштабе (после денормализации)
    hist_keep = ["ROAD_NAME", "YEAR", "MONTH"] + targets
    for ccol in cross_cols:
        if ccol in history_df.columns:
            hist_keep.append(ccol)
    work = history_df[[c for c in hist_keep if c in history_df.columns]].copy()
    results: List[pd.DataFrame] = []

    for year in sorted(forecast_years):
        for month in range(1, 13):
            future_data: dict = {"ROAD_NAME": roads, "YEAR": year, "MONTH": month}
            for t in targets:
                future_data[t] = np.nan

            # Кросс-признаки для будущей строки: берём из cross_future_df
            for ccol in cross_cols:
                if cross_future_df is not None and ccol in cross_future_df.columns:
                    mask_cf = (
                        (cross_future_df["YEAR"]  == year) &
                        (cross_future_df["MONTH"] == month)
                    )
                    cf_slice = cross_future_df.loc[mask_cf].set_index("ROAD_NAME")
                    future_data[ccol] = [
                        cf_slice.at[r, ccol] if r in cf_slice.index else np.nan
                        for r in roads
                    ]
                else:
                    future_data[ccol] = np.nan

            future = pd.DataFrame(future_data)

            combined = pd.concat([work, future], ignore_index=True)
            combined = _add_features_multi(
                combined, targets, lags, windows, cross_cols, cross_lags
            )

            pred_mask = combined[targets[0]].isna()
            X_pred = combined.loc[pred_mask, feature_cols].copy()
            X_pred["ROAD_NAME"] = X_pred["ROAD_NAME"].astype(str)
            X_pred = X_pred.fillna(0)

            # predict → shape (n_roads, n_targets) для MultiRMSE
            raw_preds = model.predict(X_pred)
            preds = np.zeros_like(raw_preds, dtype=float)
            road_arr = future["ROAD_NAME"].values
            for i, t in enumerate(targets):
                if road_stats_multi and t in road_stats_multi:
                    preds[:, i] = _denormalize_target(
                        raw_preds[:, i], road_arr, road_stats_multi[t]
                    )
                else:
                    preds[:, i] = np.maximum(raw_preds[:, i], 0.0)

            for i, t in enumerate(targets):
                future[t] = preds[:, i]

            work_append_cols = ["ROAD_NAME", "YEAR", "MONTH"] + targets
            for ccol in cross_cols:
                if ccol in future.columns:
                    work_append_cols.append(ccol)
            work = pd.concat(
                [work, future[[c for c in work_append_cols if c in future.columns]]],
                ignore_index=True,
            )
            results.append(future[["ROAD_NAME", "YEAR", "MONTH"] + targets])

    return pd.concat(results, ignore_index=True)


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
    road_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Рекурсивный прогноз для всех дорог на forecast_years.

    На каждом шаге (year, month):
      1. Строки с NaN-target добавляются к растущей истории.
      2. _add_features пересчитывает лаги с учётом уже предсказанных значений.
      3. Прогнозы денормализуются (z-score → исходный масштаб) и записываются
         обратно в историю — work DataFrame хранит исходные значения,
         что гарантирует корректность лагов на следующем шаге.

    road_stats: словарь {road: (mean, std)} из _compute_road_stats.
                None → денормализация не применяется (прогноз без нормировки).

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

            raw_preds = model.predict(X_pred)
            if road_stats is not None:
                # Денормализация z-score → исходный масштаб
                preds = _denormalize_target(raw_preds, future["ROAD_NAME"].values, road_stats)
            else:
                preds = np.maximum(raw_preds, 0.0)

            # Записываем прогнозы в исходном масштабе (history всегда original-scale)
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
    plot_training: bool = False,
    plot_tree: bool = False,
    plot_tree_idx: int = 0,
    plot_shap: bool = False,
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
    plot_training              : True — сохранять кривые train/val RMSE по итерациям
                               как интерактивный HTML в catboost_results/training_curves/.
                               Аналог model.fit(..., plot=True) в Jupyter.
    plot_tree                  : True — сохранять структуру дерева ансамбля как SVG
                               в catboost_results/trees/. Требует graphviz.
                               Аналог model.plot_tree(tree_idx=0) в Jupyter.
    plot_tree_idx              : индекс дерева для plot_tree (по умолчанию 0).
    plot_shap                  : True — сохранять SHAP-отчёт о значимости факторов
                               в catboost_results/shap/. Один файл на модель.

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

    # Хранилища тестовых прогнозов для вычисления метрик производных показателей
    # (LOSS_COUNT_TOTAL, LOSS_SUM_TOTAL, SPECIFIC_LOSS)
    _test_pred_store:   Dict[str, np.ndarray] = {}
    _test_actual_store: Dict[str, np.ndarray] = {}
    _test_roads_ref:    Optional[np.ndarray]  = None

    # ── ОДИНОЧНЫЕ МОДЕЛИ (RMSE) ────────────────────────────────
    for target in SINGLE_TARGETS:
        params = base_params.copy()
        if catboost_params_per_target and target in catboost_params_per_target:
            params.update(catboost_params_per_target[target])

        print(f"  [{target}] обучение модели (RMSE)...")

        # ── Признаки на полных данных ──────────────────────
        full_feat = _add_features(monthly_df, target, lags, rolling_windows)

        train_feat = full_feat[full_feat["YEAR"] <  test_year].dropna(subset=feat_cols + [target])
        test_feat  = full_feat[full_feat["YEAR"] == test_year].fillna(0)

        X_train = train_feat[feat_cols]
        y_train = train_feat[target].values
        roads_train = train_feat["ROAD_NAME"].values

        X_test  = test_feat[feat_cols]
        y_test  = test_feat[target].values   # исходный масштаб — для метрик
        roads_test = test_feat["ROAD_NAME"].values

        # ── Z-score нормализация по дорогам ───────────────────────
        # Калининград и другие дороги нетипичного масштаба получают
        # равный вес — модель обучается на безразмерных отклонениях.
        road_stats = _compute_road_stats(train_feat, target)
        y_train_norm = _normalize_target(y_train, roads_train, road_stats)
        y_test_norm  = _normalize_target(y_test,  roads_test,  road_stats)

        # ── CatBoost Pool (категориальные признаки по имени) ──────
        cb_cat = [c for c in CAT_FEATURES if c in feat_cols]

        train_pool = Pool(X_train, label=y_train_norm, cat_features=cb_cat)
        eval_pool  = Pool(X_test,  label=y_test_norm,  cat_features=cb_cat)

        # ── Основная модель ────────────────────────────────
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=eval_pool)

        if plot_training:
            p = save_training_curves(model, indicator=target, outdir=base_out / "training_curves")
            if p: print(f"  [{target}] кривые обучения: {p}")
        if plot_tree:
            p = save_tree(model, indicator=target, outdir=base_out / "trees", tree_idx=plot_tree_idx)
            if p: print(f"  [{target}] дерево (tree_idx={plot_tree_idx}): {p}")
        if plot_shap:
            p = save_shap_report(model, X_test, feat_cols, indicator=target, outdir=base_out / "shap")
            if p: print(f"  [{target}] SHAP-отчёт: {p}")

        # ── Метрики в исходном масштабе (денормализация) ──────────
        pred_test_norm = model.predict(eval_pool)
        pred_test = _denormalize_target(pred_test_norm, roads_test, road_stats)

        # Сохраняем для вычисления метрик производных показателей
        _test_pred_store[target]   = pred_test
        _test_actual_store[target] = y_test
        if _test_roads_ref is None:
            _test_roads_ref = roads_test

        for road in monthly_df["ROAD_NAME"].unique():
            mask = test_feat["ROAD_NAME"] == road
            if mask.sum() == 0:
                continue
            metrics_rows.append({
                "indicator": target, "road": road,
                "MAPE_%":    mape_percent(y_test[mask.values], pred_test[mask.values]),
                "SMAPE_%":   smape_percent(y_test[mask.values], pred_test[mask.values]),
                "n_trees":   model.tree_count_, "test_year": test_year,
            })

        # ── Рекурсивный прогноз ────────────────────────────
        forecast_part = _recursive_forecast(
            model=model,
            history_df=monthly_df[["ROAD_NAME", "YEAR", "MONTH", target]],
            target_col=target, forecast_years=forecast_years,
            lags=lags, windows=rolling_windows, feature_cols=feat_cols,
            road_stats=road_stats,
        )
        forecast_parts[target] = forecast_part.set_index(["ROAD_NAME", "YEAR", "MONTH"])[target]
        print(f"  [{target}] готово. Деревьев: {model.tree_count_}")

    # ── СОВМЕСТНЫЕ МОДЕЛИ (MultiRMSE) ──────────────────────────
    for group_name, targets in MULTI_TARGET_GROUPS.items():
        params = base_params.copy()
        params["loss_function"] = "MultiRMSE"
        if catboost_params_per_target and group_name in catboost_params_per_target:
            params.update(catboost_params_per_target[group_name])
        params["loss_function"] = "MultiRMSE"   # гарантируем после override

        print(f"  [{group_name}] обучение модели (MultiRMSE + cross: {CROSS_INDICATORS}): {targets}...")

        # ── Кросс-признаки: добавляем TRAIN_KM в обучающую выборку ─
        multi_cols_for_feat = ["ROAD_NAME", "YEAR", "MONTH"] + targets + CROSS_INDICATORS
        multi_feat_cols = _feature_cols_multi(
            targets, lags, rolling_windows,
            cross_cols=CROSS_INDICATORS, cross_lags=CROSS_LAGS,
        )
        full_feat = _add_features_multi(
            monthly_df[[c for c in multi_cols_for_feat if c in monthly_df.columns]].copy(),
            targets, lags, rolling_windows,
            cross_cols=CROSS_INDICATORS, cross_lags=CROSS_LAGS,
        )

        train_feat = full_feat[full_feat["YEAR"] <  test_year].dropna(subset=multi_feat_cols + targets)
        test_feat  = full_feat[full_feat["YEAR"] == test_year].fillna(0)

        X_train = train_feat[multi_feat_cols]
        X_test  = test_feat[multi_feat_cols]
        y_train = train_feat[targets].values   # (n, 2) — исходный масштаб
        y_test  = test_feat[targets].values    # (n, 2) — исходный масштаб
        roads_train = train_feat["ROAD_NAME"].values
        roads_test  = test_feat["ROAD_NAME"].values

        # ── Z-score нормализация — отдельно для каждого таргета ───
        road_stats_multi: Dict[str, Dict[str, Tuple[float, float]]] = {}
        y_train_norm = np.zeros_like(y_train, dtype=float)
        y_test_norm  = np.zeros_like(y_test,  dtype=float)
        for ti, t in enumerate(targets):
            stats = _compute_road_stats(train_feat, t)
            road_stats_multi[t] = stats
            y_train_norm[:, ti] = _normalize_target(y_train[:, ti], roads_train, stats)
            y_test_norm[:, ti]  = _normalize_target(y_test[:, ti],  roads_test,  stats)

        cb_cat = [c for c in CAT_FEATURES if c in multi_feat_cols]
        train_pool = Pool(X_train, label=y_train_norm, cat_features=cb_cat)
        eval_pool  = Pool(X_test,  label=y_test_norm,  cat_features=cb_cat)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=eval_pool)

        if plot_training:
            p = save_training_curves(model, indicator=group_name, outdir=base_out / "training_curves")
            if p: print(f"  [{group_name}] кривые обучения: {p}")
        if plot_tree:
            p = save_tree(model, indicator=group_name, outdir=base_out / "trees", tree_idx=plot_tree_idx)
            if p: print(f"  [{group_name}] дерево (tree_idx={plot_tree_idx}): {p}")
        if plot_shap:
            p = save_shap_report(
                model, X_test, multi_feat_cols,
                indicator=group_name, outdir=base_out / "shap", targets=targets,
            )
            if p: print(f"  [{group_name}] SHAP-отчёт: {p}")

        # Метрики по каждому таргету в исходном масштабе (денормализация)
        raw_pred_test = model.predict(eval_pool)           # нормированное пространство
        for ti, target in enumerate(targets):
            pred_col = _denormalize_target(
                raw_pred_test[:, ti], roads_test, road_stats_multi[target]
            )

            # Сохраняем для метрик производных показателей
            _test_pred_store[target]   = pred_col
            _test_actual_store[target] = y_test[:, ti]
            if _test_roads_ref is None:
                _test_roads_ref = roads_test

            for road in monthly_df["ROAD_NAME"].unique():
                mask = test_feat["ROAD_NAME"] == road
                if mask.sum() == 0:
                    continue
                metrics_rows.append({
                    "indicator": target, "road": road,
                    "MAPE_%":    mape_percent(y_test[mask.values, ti], pred_col[mask.values]),
                    "SMAPE_%":   smape_percent(y_test[mask.values, ti], pred_col[mask.values]),
                    "n_trees":   model.tree_count_, "test_year": test_year,
                })

        # ── Рекурсивный прогноз: TRAIN_KM из уже готового прогноза ─
        cross_future_df = None
        if "TRAIN_KM" in forecast_parts:
            cross_future_df = forecast_parts["TRAIN_KM"].reset_index()

        fp_multi = _recursive_forecast_multi(
            model=model,
            history_df=monthly_df[
                [c for c in multi_cols_for_feat if c in monthly_df.columns]
            ].copy(),
            targets=targets, forecast_years=forecast_years,
            lags=lags, windows=rolling_windows, feature_cols=multi_feat_cols,
            cross_cols=CROSS_INDICATORS, cross_lags=CROSS_LAGS,
            cross_future_df=cross_future_df,
            road_stats_multi=road_stats_multi,
        )
        indexed = fp_multi.set_index(["ROAD_NAME", "YEAR", "MONTH"])
        for target in targets:
            forecast_parts[target] = indexed[target]
        print(f"  [{group_name}] готово. Деревьев: {model.tree_count_}")

    # ── Метрики производных показателей ──────────────────────────────────────────
    # LOSS_COUNT_TOTAL, LOSS_SUM_TOTAL, SPECIFIC_LOSS — вычисляются из прогнозов
    # тест-периода; SMAPE сравнивает производные фактические и прогнозные значения.
    if len(_test_pred_store) == len(TARGET_INDICATORS) and _test_roads_ref is not None:
        _pred_df   = pd.DataFrame({"ROAD_NAME": _test_roads_ref})
        _actual_df = pd.DataFrame({"ROAD_NAME": _test_roads_ref})
        for _t in TARGET_INDICATORS:
            _pred_df[_t]   = _test_pred_store[_t]
            _actual_df[_t] = _test_actual_store[_t]
        _pred_derived   = compute_derived(_pred_df)
        _actual_derived = compute_derived(_actual_df)
        for _ind in DERIVED_INDICATORS:
            for road in monthly_df["ROAD_NAME"].unique():
                _mask = _test_roads_ref == road
                if _mask.sum() == 0:
                    continue
                metrics_rows.append({
                    "indicator": _ind, "road": road,
                    "MAPE_%":    mape_percent(
                        _actual_derived[_ind].values[_mask],
                        _pred_derived[_ind].values[_mask]),
                    "SMAPE_%":   smape_percent(
                        _actual_derived[_ind].values[_mask],
                        _pred_derived[_ind].values[_mask]),
                    "n_trees":   0, "test_year": test_year,
                })

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

    # CI счётчиков — целые числа (lower/upper не могут быть дробными)
    for col in COUNT_COLS:
        for suffix in ("_lower_95", "_upper_95"):
            ci_col = f"{col}{suffix}"
            if ci_col in out.columns:
                out[ci_col] = out[ci_col].round().astype(int)

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
