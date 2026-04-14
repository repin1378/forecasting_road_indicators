"""
series_generator.py

Генерация синтетических временных рядов показателей ЖД.

Основная функция: generate_monthly_data
  — читает годовые итоги из Excel (example_year.xlsx),
  — разбивает каждый показатель по месяцам с сезонностью, трендом и шумом,
  — для базового года (ANNUAL_YEAR) масштабирует месячные значения так,
    чтобы их сумма точно совпадала с годовым итогом,
  — сохраняет результат в CSV.

Функция generate_all_series сохранена для обратной совместимости с main.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


# ======================================================
# НОВАЯ ГЕНЕРАЦИЯ: месячные данные по дорогам
# ======================================================

def _normalize_profile(profile: dict) -> dict:
    """Нормирует профиль сезонности: сумма по 12 месяцам = 12."""
    total = sum(profile.values())
    return {m: v / total * 12 for m, v in profile.items()}


def generate_monthly_data(
    annual_data_path: "str | Path",
    start_year: int,
    end_year: int,
    annual_year: int,
    indicator_config: dict,
    seasonal_profiles: dict,
    seed: int = 42,
    output_csv: "Optional[str | Path]" = None,
) -> pd.DataFrame:
    """
    Генерирует помесячные значения показателей для всех дорог за диапазон лет.

    Параметры
    ----------
    annual_data_path : путь к Excel-файлу с годовыми итогами (example_year.xlsx).
        Обязательные столбцы: ROAD_NAME + по одному на каждый ключ indicator_config.
    start_year, end_year : диапазон генерации (включительно).
    annual_year : год, к которому привязаны фактические годовые данные.
        Для этого года месячные значения масштабируются до точного совпадения
        с годовым итогом.
    indicator_config : словарь вида
        {
          "INDICATOR_NAME": {
              "seasonal": "<ключ из seasonal_profiles>",
              "trend_pct": float,   # изменение в год (0.01 = +1%/год)
              "noise_pct": float,   # СКО шума / среднемесячное значение
              "is_count":  bool,    # True → округлять до целых
          }, ...
        }
    seasonal_profiles : словарь профилей сезонности {name: {1: w1, ..., 12: w12}}.
    seed : зерно генератора.
    output_csv : если задан, сохраняет DataFrame в CSV (UTF-8 BOM).

    Возвращает
    ----------
    pd.DataFrame со столбцами:
        ROAD_NAME, YEAR, MONTH,
        TRAIN_KM,
        LOSS_12_COUNT, LOSS_12_SUM,
        LOSS_3_COUNT,  LOSS_3_SUM,
        LOSS_TECH_COUNT, LOSS_TECH_SUM,
        LOSS_COUNT_TOTAL, LOSS_SUM_TOTAL,
        SPECIFIC_LOSS
    """
    rng = np.random.default_rng(seed)

    annual_df = pd.read_excel(annual_data_path)
    base_indicators = list(indicator_config.keys())

    norm_profiles = {
        name: _normalize_profile(profile)
        for name, profile in seasonal_profiles.items()
    }

    rows: list = []

    for _, road_row in annual_df.iterrows():
        road_name = road_row["ROAD_NAME"]

        for year in range(start_year, end_year + 1):
            year_offset = year - annual_year

            for month in range(1, 13):
                row: dict = {"ROAD_NAME": road_name, "YEAR": year, "MONTH": month}

                for ind in base_indicators:
                    annual_val = float(road_row[ind])
                    cfg = indicator_config[ind]
                    profile = norm_profiles[cfg["seasonal"]]

                    # Годовое значение с учётом тренда
                    year_annual = annual_val * (1.0 + cfg["trend_pct"]) ** year_offset

                    # Базовое месячное значение по сезонному профилю
                    monthly_base = year_annual / 12.0 * profile[month]

                    # Случайный шум
                    noise = rng.normal(0.0, cfg["noise_pct"] * abs(monthly_base))
                    row[ind] = max(0.0, monthly_base + noise)

                rows.append(row)

    df = pd.DataFrame(rows)

    # --------------------------------------------------
    # Масштабирование базового года до точных годовых итогов
    # --------------------------------------------------
    ref_mask = df["YEAR"] == annual_year
    for ind in base_indicators:
        for road_name, grp in df[ref_mask].groupby("ROAD_NAME"):
            month_sum = grp[ind].sum()
            if month_sum > 0:
                annual_val = float(
                    annual_df.loc[annual_df["ROAD_NAME"] == road_name, ind].iloc[0]
                )
                df.loc[ref_mask & (df["ROAD_NAME"] == road_name), ind] *= annual_val / month_sum

    # --------------------------------------------------
    # Округление счётчиков
    # --------------------------------------------------
    for ind, cfg in indicator_config.items():
        if cfg.get("is_count"):
            df[ind] = df[ind].round().astype(int)

    # --------------------------------------------------
    # Производные показатели
    # --------------------------------------------------
    df["LOSS_COUNT_TOTAL"] = (
        df["LOSS_12_COUNT"] + df["LOSS_3_COUNT"] + df["LOSS_TECH_COUNT"]
    )
    df["LOSS_SUM_TOTAL"] = (
        df["LOSS_12_SUM"] + df["LOSS_3_SUM"] + df["LOSS_TECH_SUM"]
    )
    df["SPECIFIC_LOSS"] = df["LOSS_SUM_TOTAL"] / df["TRAIN_KM"] * 1_000_000

    # --------------------------------------------------
    # Порядок столбцов и сортировка
    # --------------------------------------------------
    column_order = [
        "ROAD_NAME", "YEAR", "MONTH",
        "TRAIN_KM",
        "LOSS_12_COUNT", "LOSS_12_SUM",
        "LOSS_3_COUNT", "LOSS_3_SUM",
        "LOSS_TECH_COUNT", "LOSS_TECH_SUM",
        "LOSS_COUNT_TOTAL", "LOSS_SUM_TOTAL",
        "SPECIFIC_LOSS",
    ]
    df = df[column_order].sort_values(["ROAD_NAME", "YEAR", "MONTH"]).reset_index(drop=True)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"CSV сохранён: {output_csv}  [{len(df)} строк]")

    return df


# ======================================================
# УСТАРЕВШАЯ ГЕНЕРАЦИЯ (обратная совместимость с main.py)
# ======================================================

# ======================================================
# БАЗОВАЯ ГЕНЕРАЦИЯ
# ======================================================

def _generate_base_series(
    dates: pd.DatetimeIndex,
    base: float,
    trend: float,
    seasonal_amp: float,
    noise_scale: float,
    allow_spikes: bool = False,
    num_forced_outliers: int = 0,
) -> pd.Series:
    """
    Генерация временного ряда с трендом, сезонностью и шумом.
    Все параметры трактуются как ГОДОВЫЕ.
    """

    n = len(dates)
    t = np.arange(n)

    # -------------------------------
    # Год → месяц
    # -------------------------------
    base_m = base / 12.0
    trend_m = trend / 12.0

    # -------------------------------
    # Аддитивная структура
    # -------------------------------
    series = base_m + trend_m * t
    series += seasonal_amp * np.sin(2 * np.pi * t / 12)
    series += np.random.normal(0, noise_scale, size=n)

    series = pd.Series(series, index=dates)

    # -------------------------------
    # Режимные изменения (структурные сдвиги)
    # -------------------------------
    years = series.index.year
    multiplier = np.ones(n)

    multiplier[years >= 2022] *= 1.15
    multiplier *= 1.0 + 0.02 * np.clip(years - 2022, 0, None)

    series *= multiplier

    # -------------------------------
    # Принудительные выбросы
    # -------------------------------
    if allow_spikes and num_forced_outliers > 0:
        idx = np.random.choice(n, size=num_forced_outliers, replace=False)
        series.iloc[idx] *= np.random.uniform(1.5, 2.5, size=len(idx))

    # -------------------------------
    # ФИЗИЧЕСКОЕ ОГРАНИЧЕНИЕ
    # -------------------------------
    series = series.clip(lower=0)

    return series


# ======================================================
# ГЕНЕРАЦИЯ ВСЕХ ПОКАЗАТЕЛЕЙ
# ======================================================

def generate_all_series(
    gen_params: Dict,
    start: str,
    end: str,
    allow_spikes: bool = False,
    num_forced_outliers: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Генерирует все показатели по всем дорогам.
    """

    dates = pd.date_range(start, end, freq="MS")

    indicators = set()
    for road_params in gen_params.values():
        indicators.update(road_params.keys())

    result = {ind: pd.DataFrame(index=dates) for ind in indicators}

    for road, params in gen_params.items():

        # --- базовые ряды ---
        tmp = {}

        for ind, cfg in params.items():
            if isinstance(cfg, dict):
                tmp[ind] = _generate_base_series(
                    dates=dates,
                    allow_spikes=allow_spikes,
                    num_forced_outliers=num_forced_outliers,
                    **cfg,
                )

        # --- агрегированные показатели ---
        if "loss_total" in params:
            expr = params["loss_total"]
            tmp["loss_total"] = eval(expr, {}, tmp).clip(lower=0)

        if "specific" in params:
            expr = params["specific"]
            specific = eval(expr, {}, tmp)

            # нелинейный структурный множитель
            years = dates.year
            structure_factor = 1.0 + 0.15 * np.tanh((years - 2021) / 2)
            specific = specific * structure_factor

            tmp["specific"] = specific.clip(lower=0)

        # --- запись в итог ---
        for ind, series in tmp.items():
            result[ind][road] = series

    return result
