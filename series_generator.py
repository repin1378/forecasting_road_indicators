"""
series_generator.py

Генерация синтетических временных рядов
для показателей перевозочного процесса ЖД.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


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
