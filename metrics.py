# metrics.py
"""
Модуль расчёта метрик для ETS-модели:
- MAPE - Средняя абсолютная процентная ошибка
- SMAPE - Симметричная средняя абсолютная процентная ошибка
- AIC - Информационный критерий Акаике
- resid_std (std остатков) - Стандартное отклонение остатков
"""

import numpy as np
import pandas as pd


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)"""
    return np.mean(
        np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))
    ) * 100


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (%)"""
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100


def compute_metrics(y_test, pred_test, fit_full):
    """
    Возвращает словарь метрик:
    - MAPE_test_%
    - SMAPE_test_%
    - AIC
    - resid_std
    """

    mape_val = float(mape(y_test, pred_test))
    smape_val = float(smape(y_test, pred_test))

    aic_val = getattr(fit_full, "aic", np.nan)
    resid_std = float(np.std(fit_full.resid))

    return {
        "MAPE_test_%": mape_val,
        "SMAPE_test_%": smape_val,
        "AIC": aic_val,
        "resid_std": resid_std,
    }

def mape_percent(y_true, y_pred, cap: float = 200.0) -> float:
    """
    MAPE (%), устойчивый к нулям.

    Знаменатель = max(|y_true|, 1 % от среднего ряда), что исключает взрыв
    при y_true ≈ 0 (характерно для малых дорог с редкими инцидентами).
    Вклад каждой точки ограничен значением cap (200 % по умолчанию).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_abs = np.abs(y_true[y_true != 0]).mean() if np.any(y_true != 0) else 1.0
    floor    = max(mean_abs * 0.01, 1e-6)
    denom    = np.maximum(np.abs(y_true), floor)
    per_pt   = np.minimum(np.abs((y_true - y_pred) / denom) * 100.0, cap)
    return float(np.mean(per_pt))


def smape_percent(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)