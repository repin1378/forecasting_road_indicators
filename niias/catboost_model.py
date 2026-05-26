"""CatBoost forecasting for NIIAS monthly data.

Input workbook format:
    niias/examples/niias_data.xlsx
    columns: Дата, ОТС, ПЧ_ОТС

The implementation mirrors the root CatBoost pipeline steps for this simpler
single-series dataset: Excel normalization, lag/rolling features, a held-out
test year, recursive future forecast, confidence bands, metrics, and a
seasonal naive baseline for SMAPE comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:  # pragma: no cover - handled when training is requested
    CatBoostRegressor = None
    Pool = None

try:
    from .metrics import compute_metric_row
except ImportError:  # pragma: no cover - allows running as a script dependency
    from metrics import compute_metric_row


DATE_COL = "Дата"
BASE_DIR = Path(__file__).resolve().parent
RAW_TARGET_MAP: Dict[str, str] = {
    "ОТС": "OTS",
    "ПЧ_ОТС": "PCH_OTS",
}
TARGET_INDICATORS: List[str] = list(RAW_TARGET_MAP.values())
TARGET_TITLES: Dict[str, str] = {
    "OTS": "ОТС",
    "PCH_OTS": "ПЧ_ОТС",
}

CAT_FEATURES: List[str] = ["YEAR", "MONTH"]
LAGS: List[int] = [1, 2, 3, 6, 12, 24, 36]
ROLLING_WINDOWS: List[int] = [3, 6, 12]
CI_HALF_WIDTH: float = 0.05

DEFAULT_CB_PARAMS: dict = {
    "loss_function": "RMSE",
    "iterations": 1200,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "use_best_model": True,
    "early_stopping_rounds": 50,
    "allow_writing_files": False,
    "verbose": False,
}


def _month_start(value) -> pd.Timestamp:
    """Convert 'YYYY-MM', 'MM.YYYY' or any date-like value to month start."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_period("M").to_timestamp()
    text = str(value).strip()
    if "." in text and "-" not in text:
        left, right = text.split(".", 1)
        if len(left) <= 2 and len(right) == 4:
            return pd.Timestamp(year=int(right), month=int(left), day=1)
    return pd.to_datetime(text).to_period("M").to_timestamp()


def filter_by_date_range(
    df: pd.DataFrame,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Return rows inside inclusive monthly date bounds."""
    out = df.copy()
    start_dt = _month_start(start) if start is not None else out["DATE"].min()
    end_dt = _month_start(end) if end is not None else out["DATE"].max()
    return (
        out[(out["DATE"] >= start_dt) & (out["DATE"] <= end_dt)]
        .sort_values("DATE")
        .reset_index(drop=True)
    )


def load_niias_data(
    path: str | Path | None = None,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Read and normalize the NIIAS Excel file to DATE/YEAR/MONTH/targets."""
    path = Path(path) if path is not None else BASE_DIR / "examples" / "niias_data.xlsx"
    df = pd.read_excel(path)
    required = [DATE_COL, *RAW_TARGET_MAP.keys()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}. Found: {df.columns.tolist()}")

    df = df[required].rename(columns={DATE_COL: "DATE", **RAW_TARGET_MAP}).copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["YEAR"] = df["DATE"].dt.year.astype(int)
    df["MONTH"] = df["DATE"].dt.month.astype(int)
    for target in TARGET_INDICATORS:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    df = (
        df[["DATE", "YEAR", "MONTH", *TARGET_INDICATORS]]
        .dropna(subset=["DATE"])
        .sort_values("DATE")
        .drop_duplicates(subset=["YEAR", "MONTH"], keep="last")
        .reset_index(drop=True)
    )
    return filter_by_date_range(df, start=start, end=end)


def _add_features(df: pd.DataFrame, target_col: str, lags: List[int], windows: List[int]) -> pd.DataFrame:
    """Add time, lag, rolling, differencing and YoY features for one target."""
    df = df.sort_values(["YEAR", "MONTH"]).reset_index(drop=True).copy()
    observed = df[target_col].dropna()
    df["series_mean"] = float(observed.mean()) if len(observed) else 0.0
    df["series_std"] = float(observed.std(ddof=1)) if len(observed) > 1 else 0.0
    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    df["t"] = np.arange(len(df))

    vals = df[target_col]
    for lag in lags:
        df[f"lag_{lag}"] = vals.shift(lag)

    shifted = vals.shift(1)
    for window in windows:
        df[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
        df[f"roll_std_{window}"] = shifted.rolling(window, min_periods=1).std(ddof=0).fillna(0)

    df["diff_1"] = vals.diff(1)
    df["diff_12"] = vals.diff(12)
    lag1 = vals.shift(1)
    lag13 = vals.shift(13).replace(0, np.nan)
    df["yoy_ratio"] = (lag1 / lag13).fillna(1.0)
    return df


def _feature_cols(lags: List[int], windows: List[int]) -> List[str]:
    cols = list(CAT_FEATURES)
    cols += ["series_mean", "series_std", "month_sin", "month_cos", "t"]
    cols += [f"lag_{lag}" for lag in lags]
    cols += [f"roll_mean_{window}" for window in windows]
    cols += [f"roll_std_{window}" for window in windows]
    cols += ["diff_1", "diff_12", "yoy_ratio"]
    return cols


def _make_future_dates(
    last_date: pd.Timestamp,
    forecast_years: Optional[List[int]] = None,
    forecast_start=None,
    forecast_end=None,
) -> pd.DataFrame:
    if forecast_start is not None or forecast_end is not None:
        if forecast_start is None or forecast_end is None:
            raise ValueError("forecast_start and forecast_end must be provided together")
        start = _month_start(forecast_start)
        end = _month_start(forecast_end)
        if end < start:
            raise ValueError(f"forecast_end must be >= forecast_start: {forecast_start}..{forecast_end}")
        dates = pd.date_range(start, end, freq="MS")
        return pd.DataFrame({"DATE": dates, "YEAR": dates.year, "MONTH": dates.month})

    if forecast_years:
        rows = [
            {"DATE": pd.Timestamp(year=year, month=month, day=1), "YEAR": year, "MONTH": month}
            for year in sorted(forecast_years)
            for month in range(1, 13)
        ]
        return pd.DataFrame(rows)

    start = (last_date + pd.offsets.MonthBegin(1)).normalize()
    dates = pd.date_range(start, periods=24, freq="MS")
    return pd.DataFrame({"DATE": dates, "YEAR": dates.year, "MONTH": dates.month})


def _recursive_forecast(
    model: CatBoostRegressor,
    history_df: pd.DataFrame,
    target_col: str,
    future_dates: pd.DataFrame,
    lags: List[int],
    windows: List[int],
    feature_cols: List[str],
) -> pd.DataFrame:
    work = history_df[["DATE", "YEAR", "MONTH", target_col]].copy()
    results: List[pd.DataFrame] = []

    for _, row in future_dates.sort_values("DATE").iterrows():
        future = pd.DataFrame([{
            "DATE": row["DATE"],
            "YEAR": int(row["YEAR"]),
            "MONTH": int(row["MONTH"]),
            target_col: np.nan,
        }])
        combined = pd.concat([work, future], ignore_index=True)
        featured = _add_features(combined, target_col, lags, windows)
        X_pred = featured.loc[featured[target_col].isna(), feature_cols].tail(1).fillna(0)
        pred = max(float(model.predict(X_pred)[0]), 0.0)
        future[target_col] = pred
        work = pd.concat([work, future], ignore_index=True)
        results.append(future)

    return pd.concat(results, ignore_index=True)


def _seasonal_naive_test(full_feat: pd.DataFrame, target: str, test_year: int) -> tuple[np.ndarray, np.ndarray]:
    test = full_feat[full_feat["YEAR"] == test_year].copy()
    return test[target].to_numpy(dtype=float), test["lag_12"].to_numpy(dtype=float)


def _seasonal_naive_actual(
    actual_df: pd.DataFrame,
    target: str,
    forecast_dates: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    full_feat = _add_features(actual_df, target, LAGS, ROLLING_WINDOWS)
    test = forecast_dates[["DATE"]].merge(full_feat[["DATE", target, "lag_12"]], on="DATE", how="left")
    return test[target].to_numpy(dtype=float), test["lag_12"].to_numpy(dtype=float)


def _write_xlsx(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def run_catboost_forecast(
    monthly_df: pd.DataFrame,
    forecast_years: Optional[List[int]] = None,
    outdir: str | Path = "niias/catboost_results",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    test_year: Optional[int] = None,
    model_start=None,
    model_end=None,
    forecast_start=None,
    forecast_end=None,
    random_seed: int = 42,
    catboost_params: Optional[dict] = None,
    catboost_params_per_target: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """Train one CatBoost model per target and save forecasts plus metrics."""
    if CatBoostRegressor is None or Pool is None:
        raise ImportError(
            "CatBoost is required for training. Install it in the Python environment: "
            "pip install catboost"
        )

    lags = lags or LAGS
    rolling_windows = rolling_windows or ROLLING_WINDOWS
    feature_cols = _feature_cols(lags, rolling_windows)

    base_out = Path(outdir)
    base_out.mkdir(parents=True, exist_ok=True)
    actual_df = monthly_df.sort_values("DATE").reset_index(drop=True)
    train_df = filter_by_date_range(actual_df, start=model_start, end=model_end)
    if train_df.empty:
        raise ValueError(f"No training data in model date range: {model_start}..{model_end}")
    if test_year is None:
        test_year = int(train_df["YEAR"].max())

    requested_future_dates = _make_future_dates(
        train_df["DATE"].max(),
        forecast_years=forecast_years,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )
    if not requested_future_dates.empty and requested_future_dates["DATE"].min() <= train_df["DATE"].max():
        raise ValueError(
            "Forecast period must start after the model training period. "
            f"model_end={train_df['DATE'].max().date()}, "
            f"forecast_start={requested_future_dates['DATE'].min().date()}"
        )
    internal_start = train_df["DATE"].max() + pd.offsets.MonthBegin(1)
    internal_dates = pd.date_range(internal_start, requested_future_dates["DATE"].max(), freq="MS")
    future_dates = pd.DataFrame({
        "DATE": internal_dates,
        "YEAR": internal_dates.year,
        "MONTH": internal_dates.month,
    })

    run_info = pd.DataFrame([{
        "model_start": train_df["DATE"].min(),
        "model_end": train_df["DATE"].max(),
        "model_points": len(train_df),
        "forecast_start": requested_future_dates["DATE"].min(),
        "forecast_end": requested_future_dates["DATE"].max(),
        "forecast_points": len(requested_future_dates),
        "source_points": len(actual_df),
    }])
    forecast_parts: List[pd.DataFrame] = [requested_future_dates.copy()]
    metric_rows: List[dict] = []
    baseline_rows: List[dict] = []

    base_params = {**DEFAULT_CB_PARAMS, "random_seed": random_seed}
    if catboost_params:
        base_params.update(catboost_params)

    for target in TARGET_INDICATORS:
        params = base_params.copy()
        if catboost_params_per_target and target in catboost_params_per_target:
            params.update(catboost_params_per_target[target])

        print(f"  [{target}] training CatBoost...")
        full_feat = _add_features(train_df, target, lags, rolling_windows)
        train_feat = full_feat.dropna(subset=feature_cols + [target])
        actual_on_forecast = requested_future_dates[["DATE"]].merge(
            actual_df[["DATE", target]], on="DATE", how="left"
        )
        has_actual = actual_on_forecast[target].notna().any()
        if train_feat.empty:
            raise ValueError(f"Not enough data for train target={target} in model range")

        X_train = train_feat[feature_cols].fillna(0)
        y_train = train_feat[target].to_numpy(dtype=float)

        cat_cols = [c for c in CAT_FEATURES if c in feature_cols]
        train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)

        params["use_best_model"] = False
        params["early_stopping_rounds"] = None
        model = CatBoostRegressor(**params)
        model.fit(train_pool)

        forecast_target_all = _recursive_forecast(
            model=model,
            history_df=train_df[["DATE", "YEAR", "MONTH", target]],
            target_col=target,
            future_dates=future_dates,
            lags=lags,
            windows=rolling_windows,
            feature_cols=feature_cols,
        )[["DATE", "YEAR", "MONTH", target]]
        forecast_target = requested_future_dates[["DATE", "YEAR", "MONTH"]].merge(
            forecast_target_all[["DATE", target]], on="DATE", how="left"
        )
        if has_actual:
            scored = forecast_target[["DATE", target]].merge(
                actual_df[["DATE", target]].rename(columns={target: f"{target}_actual"}),
                on="DATE",
                how="left",
            ).dropna(subset=[f"{target}_actual"])
            metric = compute_metric_row(
                target,
                scored[f"{target}_actual"].to_numpy(dtype=float),
                scored[target].to_numpy(dtype=float),
                "CatBoost",
                int(requested_future_dates["YEAR"].min()),
            )
            metric["n_trees"] = model.tree_count_
            metric["model_points"] = len(train_df)
            metric["forecast_start"] = requested_future_dates["DATE"].min()
            metric["forecast_end"] = requested_future_dates["DATE"].max()
            metric_rows.append(metric)

            naive_true, naive_pred = _seasonal_naive_actual(actual_df, target, requested_future_dates)
            baseline = compute_metric_row(
                target,
                naive_true,
                naive_pred,
                "SeasonalNaive",
                int(requested_future_dates["YEAR"].min()),
            )
            baseline["model_points"] = len(train_df)
            baseline["forecast_start"] = requested_future_dates["DATE"].min()
            baseline["forecast_end"] = requested_future_dates["DATE"].max()
            baseline_rows.append(baseline)

        forecast_parts.append(forecast_target[[target]])
        print(f"  [{target}] done. Trees: {model.tree_count_}")

    out = pd.concat(forecast_parts, axis=1)
    for target in TARGET_INDICATORS:
        out[target] = out[target].clip(lower=0)
        out[f"{target}_lower_95"] = (out[target] * (1.0 - CI_HALF_WIDTH)).clip(lower=0)
        out[f"{target}_upper_95"] = out[target] * (1.0 + CI_HALF_WIDTH)

    order = ["DATE", "YEAR", "MONTH"]
    for target in TARGET_INDICATORS:
        order += [target, f"{target}_lower_95", f"{target}_upper_95"]
    out = out[order]

    forecast_path = base_out / "catboost_forecasts.xlsx"
    metrics_path = base_out / "catboost_metrics.xlsx"
    baseline_path = base_out / "seasonal_naive_metrics.xlsx"
    metrics_df = pd.DataFrame(metric_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    _write_xlsx(forecast_path, {"forecast": out, "run_info": run_info})
    _write_xlsx(metrics_path, {"metrics": metrics_df, "run_info": run_info})
    _write_xlsx(baseline_path, {"metrics": baseline_df, "run_info": run_info})

    print(f"Forecasts saved: {forecast_path}")
    print(f"CatBoost metrics saved: {metrics_path}")
    print(f"Baseline metrics saved: {baseline_path}")
    return out
