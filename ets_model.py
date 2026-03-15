"""
ets_model.py

ETS (AAA) прогнозирование временных рядов
с fallback-стратегией и глобальной агрегацией результатов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from metrics import mape_percent, smape_percent
from plotting import plot_ets_forecast


# ======================================================
# Вспомогательные функции
# ======================================================

def _fit_ets_with_fallback(series: pd.Series):
    """
    Надёжный ETS fallback:
    - AAA → AA → A
    - проверка converged флага
    """

    # -------------------------
    # Вариант 1: ETS(A,A,A)
    # -------------------------
    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)

        if getattr(fit, "mle_retvals", {}).get("converged", True):
            return fit, "AAA"
    except Exception:
        pass

    # -------------------------
    # Вариант 2: ETS(A,A)
    # -------------------------
    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)

        if getattr(fit, "mle_retvals", {}).get("converged", True):
            return fit, "AA"
    except Exception:
        pass

    # -------------------------
    # Вариант 3: ETS(A)
    # -------------------------
    model = ExponentialSmoothing(
        series,
        trend=None,
        seasonal=None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)

    return fit, "A"


def _aic_from_fit(fit) -> float:
    """Извлекает AIC из модели."""
    try:
        return float(fit.aic)
    except Exception:
        return np.nan


# ======================================================
# Основная функция ETS
# ======================================================

def run_ets_forecast(
    data_bundle: Dict[str, pd.DataFrame],
    indicator: str,
    roads: List[str],
    horizon: int = 12,
    outdir: str = "ets_results",
):

    base_out = Path(outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    outdir = base_out / indicator
    outdir.mkdir(parents=True, exist_ok=True)

    data = data_bundle[indicator]

    future_idx = pd.date_range(
        data.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    forecasts_local = []
    metrics_local = []

    for road in roads:
        y = data[road]

        # -----------------------------
        # Train / test
        # -----------------------------
        y_train = y.iloc[:-horizon]
        y_test = y.iloc[-horizon:]

        # -----------------------------
        # Обучение ETS с fallback
        # -----------------------------
        fit, model_type = _fit_ets_with_fallback(y_train)

        # -----------------------------
        # Прогноз (тест)
        # -----------------------------
        pred_test = fit.forecast(horizon)
        residuals = y_test.values - pred_test.values

        # -----------------------------
        # Прогноз (будущее)
        # -----------------------------
        fit_full, _ = _fit_ets_with_fallback(y)
        pred_future = fit_full.forecast(horizon)

        # доверительный интервал (приближённый)
        sigma = np.std(residuals, ddof=0)
        lower_95 = pred_future - 1.96 * sigma
        upper_95 = pred_future + 1.96 * sigma

        plot_ets_forecast(
            data=y,
            future_idx=future_idx,
            forecast=pred_future,
            lower=lower_95,
            upper=upper_95,
            road=road,
            indicator=indicator,
            outpath=outdir / f"forecast_{indicator}_{road}.png",
        )

        forecasts_df = pd.DataFrame({
            "indicator": indicator,
            "road": road,
            "date": future_idx,
            "forecast": pred_future.values,
            "lower_95": lower_95.values,
            "upper_95": upper_95.values,
        })

        metrics_row = {
            "indicator": indicator,
            "road": road,
            "model_type": model_type,
            "MAPE_test_%": mape_percent(y_test, pred_test),
            "SMAPE_test_%": smape_percent(y_test, pred_test),
            "AIC": _aic_from_fit(fit),
            "resid_std": float(np.std(residuals, ddof=0)),
        }

        forecasts_local.append(forecasts_df)
        metrics_local.append(metrics_row)

    # ==================================================
    # ЛОКАЛЬНЫЕ CSV (по индикатору)
    # ==================================================

    forecasts_df = pd.concat(forecasts_local, ignore_index=True)
    forecasts_df = forecasts_df.drop_duplicates(
        subset=["indicator", "road", "date"]
    )

    metrics_df = pd.DataFrame(metrics_local)
    metrics_df = metrics_df.drop_duplicates(
        subset=["indicator", "road"]
    )

    forecasts_df.to_csv(
        outdir / "forecasts_with_ci.csv",
        index=False,
    )
    metrics_df.to_csv(
        outdir / "ets_metrics.csv",
        index=False,
    )

    # ==================================================
    # ГЛОБАЛЬНЫЕ CSV (по всем индикаторам)
    # ==================================================

    # --- прогнозы ---
    f_all = base_out / "ets_forecasts_all_indicators.csv"
    forecasts_df.to_csv(
        f_all,
        mode="a",
        header=not f_all.exists(),
        index=False,
    )

    # --- метрики ---
    m_all = base_out / "ets_metrics_all_indicators.csv"
    metrics_df.to_csv(
        m_all,
        mode="a",
        header=not m_all.exists(),
        index=False,
    )

    print(f"✅ ETS прогноз выполнен для '{indicator}'")
