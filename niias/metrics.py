"""Metrics used by the NIIAS CatBoost forecasting pipeline."""

from __future__ import annotations

import numpy as np


def mape_percent(y_true, y_pred, cap: float = 200.0) -> float:
    """Robust MAPE, capped per point and stable around zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mean_abs = np.abs(y_true[y_true != 0]).mean() if np.any(y_true != 0) else 1.0
    floor = max(mean_abs * 0.01, 1e-6)
    denom = np.maximum(np.abs(y_true), floor)
    per_point = np.minimum(np.abs((y_true - y_pred) / denom) * 100.0, cap)
    return float(np.mean(per_point))


def smape_percent(y_true, y_pred, eps: float = 1e-9) -> float:
    """Symmetric MAPE in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return float("nan")

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if mask.any() else float("nan")


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))) if mask.any() else float("nan")


def compute_metric_row(indicator: str, y_true, y_pred, model: str, test_year: int) -> dict:
    return {
        "indicator": indicator,
        "model": model,
        "MAPE_%": mape_percent(y_true, y_pred),
        "SMAPE_%": smape_percent(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "test_year": test_year,
    }

