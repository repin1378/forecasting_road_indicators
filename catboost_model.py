"""
catboost_model.py

CatBoost + Feature Engineering для прогнозирования временных рядов.
Реализована защита от дубликатов при сохранении результатов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from catboost import CatBoostRegressor
from plotting import plot_catboost_forecast


# ======================================================
# Метрики
# ======================================================

def mape_percent(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape_percent(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def aic_proxy(residuals: np.ndarray, n_features: int) -> float:
    """
    Квази-AIC для ML-моделей (proxy).
    Используется только для анализа сложности.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n == 0:
        return np.nan
    sigma2 = np.mean(residuals ** 2)
    return float(n * np.log(sigma2 + 1e-9) + 2 * n_features)


# ======================================================
# Feature Engineering
# ======================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    m = df["date"].dt.month
    df["month"] = m
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["year"] = df["date"].dt.year
    return df


def make_supervised_frame(
    y: pd.Series,
    lags: List[int],
    rolling_windows: List[int],
    exog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    df = pd.DataFrame({"date": y.index, "y": y.values})
    df = add_calendar_features(df)

    for l in lags:
        df[f"lag_{l}"] = df["y"].shift(l)

    df["diff_1"] = df["y"].diff(1)
    df["diff_12"] = df["y"].diff(12)

    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std(ddof=0)

    if exog is not None:
        ex = exog.loc[y.index]
        for c in ex.columns:
            df[f"exog_{c}"] = ex[c].values

    return df.dropna().reset_index(drop=True)


# ======================================================
# Рекурсивный прогноз
# ======================================================

def recursive_forecast(
    model: CatBoostRegressor,
    y_history: pd.Series,
    future_idx: pd.DatetimeIndex,
    lags: List[int],
    rolling_windows: List[int],
    exog_future: Optional[pd.DataFrame],
    feature_cols: List[str],
) -> pd.Series:

    history = y_history.copy()
    preds = []

    for d in future_idx:
        row = pd.DataFrame({"date": [d]})
        row = add_calendar_features(row)

        for l in lags:
            row[f"lag_{l}"] = history.iloc[-l]

        row["diff_1"] = history.iloc[-1] - history.iloc[-2]
        row["diff_12"] = history.iloc[-1] - history.iloc[-12] if len(history) >= 12 else 0.0

        for w in rolling_windows:
            win = history.iloc[-w:]
            row[f"roll_mean_{w}"] = win.mean()
            row[f"roll_std_{w}"] = win.std(ddof=0)

        if exog_future is not None:
            for c in exog_future.columns:
                row[f"exog_{c}"] = exog_future.loc[d, c]

        X = row[feature_cols]
        y_hat = float(model.predict(X)[0])
        preds.append(y_hat)

        history = pd.concat([history, pd.Series([y_hat], index=[d])])

    return pd.Series(preds, index=future_idx)


# ======================================================
# Основная функция CatBoost
# ======================================================

def run_catboost_forecast(
    data_bundle: Dict[str, pd.DataFrame],
    indicator: str,
    roads: List[str],
    horizon: int = 12,
    outdir: str = "catboost_results",
    lags: List[int] = [1, 2, 3, 6, 12, 18, 24, 36],
    rolling_windows: List[int] = [3, 6, 12, 24],
    exog_indicators: Optional[List[str]] = None,
    use_prediction_intervals: bool = True,
    random_seed: int = 42,
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
        # Экзогенные признаки
        # -----------------------------
        exog_df = exog_future = None
        if exog_indicators:
            exog_df = pd.DataFrame({
                name: data_bundle[name][road]
                for name in exog_indicators
            })
            exog_future = pd.DataFrame(
                {c: exog_df[c].iloc[-1] for c in exog_df.columns},
                index=future_idx,
            )

        # -----------------------------
        # Train / test
        # -----------------------------
        y_train = y.iloc[:-horizon]
        y_test = y.iloc[-horizon:]
        ex_train = exog_df.loc[y_train.index] if exog_df is not None else None

        train_df = make_supervised_frame(
            y=y_train,
            lags=lags,
            rolling_windows=rolling_windows,
            exog=ex_train,
        )

        X_train = train_df.drop(columns=["date", "y"])
        y_train_target = train_df["y"].values
        feature_cols = list(X_train.columns)

        # -----------------------------
        # Основная модель
        # -----------------------------
        model = CatBoostRegressor(
            loss_function="MAE",
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            random_seed=random_seed,
            verbose=False,
        )
        model.fit(X_train, y_train_target)

        # -----------------------------
        # Квантильные модели
        # -----------------------------
        q_low = q_high = None
        if use_prediction_intervals:
            q_low = CatBoostRegressor(
                loss_function="Quantile:alpha=0.05",
                iterations=model.tree_count_,
                learning_rate=model.learning_rate_,
                depth=model.get_params()["depth"],
                random_seed=random_seed,
                verbose=False,
            )
            q_high = CatBoostRegressor(
                loss_function="Quantile:alpha=0.95",
                iterations=model.tree_count_,
                learning_rate=model.learning_rate_,
                depth=model.get_params()["depth"],
                random_seed=random_seed,
                verbose=False,
            )
            q_low.fit(X_train, y_train_target)
            q_high.fit(X_train, y_train_target)

        # -----------------------------
        # Прогноз (тест)
        # -----------------------------
        pred_test = recursive_forecast(
            model,
            y_train,
            y_test.index,
            lags,
            rolling_windows,
            exog_df.loc[y_test.index] if exog_df is not None else None,
            feature_cols,
        )

        residuals = y_test.values - pred_test.values

        # -----------------------------
        # Прогноз (будущее)
        # -----------------------------
        pred_future = recursive_forecast(
            model,
            y,
            future_idx,
            lags,
            rolling_windows,
            exog_future,
            feature_cols,
        )

        lower_95 = upper_95 = None
        if q_low is not None and q_high is not None:
            lower_95 = recursive_forecast(
                q_low, y, future_idx, lags, rolling_windows, exog_future, feature_cols
            )
            upper_95 = recursive_forecast(
                q_high, y, future_idx, lags, rolling_windows, exog_future, feature_cols
            )

        plot_catboost_forecast(
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
        })

        if lower_95 is not None:
            forecasts_df["lower_95"] = lower_95.values
            forecasts_df["upper_95"] = upper_95.values

        metrics_row = {
            "indicator": indicator,
            "road": road,
            "model_type": "CatBoost",
            "MAPE_test_%": mape_percent(y_test, pred_test),
            "SMAPE_test_%": smape_percent(y_test, pred_test),
            "AIC": aic_proxy(residuals, len(feature_cols)),
            "resid_std": float(np.std(residuals, ddof=0)),
            "features_n": len(feature_cols),
        }

        forecasts_local.append(forecasts_df)
        metrics_local.append(metrics_row)

    # ==================================================
    # УДАЛЕНИЕ ДУБЛИКАТОВ ПЕРЕД СОХРАНЕНИЕМ
    # ==================================================

    forecasts_df = pd.concat(forecasts_local, ignore_index=True)
    forecasts_df = forecasts_df.drop_duplicates(
        subset=["indicator", "road", "date"]
    )

    metrics_df = pd.DataFrame(metrics_local)
    metrics_df = metrics_df.drop_duplicates(
        subset=["indicator", "road", "model_type"]
    )

    # -----------------------------
    # Сохранение
    # -----------------------------
    f_path = base_out / "catboost_forecasts_all_indicators.csv"
    m_path = base_out / "catboost_metrics_all_indicators.csv"

    forecasts_df.to_csv(f_path, mode="a", header=not f_path.exists(), index=False)
    metrics_df.to_csv(m_path, mode="a", header=not m_path.exists(), index=False)

    print(f"✅ CatBoost прогноз выполнен для '{indicator}'")
