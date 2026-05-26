"""NIIAS CatBoost forecasting entry point.

Steps:
1. Load and normalize niias/examples/niias_data.xlsx.
1.5. Optionally optimize CatBoost hyperparameters.
2. Train CatBoost models and make a recursive forecast.
3. Compare CatBoost SMAPE with a seasonal naive baseline.
4. Build interactive HTML plots.
"""

from __future__ import annotations

from pathlib import Path

try:
    from .catboost_model import (
        LAGS,
        ROLLING_WINDOWS,
        TARGET_INDICATORS,
        load_niias_data,
        run_catboost_forecast,
    )
    from .model_optimizer import load_best_params, run_optimization_all
    from .plotting import plot_predictions
    from .smape_comparison import build_smape_comparison
except ImportError:  # pragma: no cover - supports `python niias/main.py`
    from catboost_model import (
        LAGS,
        ROLLING_WINDOWS,
        TARGET_INDICATORS,
        load_niias_data,
        run_catboost_forecast,
    )
    from model_optimizer import load_best_params, run_optimization_all
    from plotting import plot_predictions
    from smape_comparison import build_smape_comparison


RUN_OPTIMIZATION = True
RUN_FORECAST = False
RUN_COMPARISON = False
RUN_PLOTS = False

OPTIMIZE_METHOD = "optuna"
OPTIMIZE_N_TRIALS = 50
OPTIMIZE_N_SPLITS = 3

BASE_DIR = Path(__file__).resolve().parent
DATA_XLSX = BASE_DIR / "examples" / "niias_data.xlsx"
OUTDIR = BASE_DIR / "catboost_results"
FORECAST_XLSX = OUTDIR / "catboost_forecasts.xlsx"
PLOTS_DIR = OUTDIR / "plots"
SEED = 42

# Inclusive monthly ranges. The example below trains on 72 points
# from 11.2017 through 10.2023 and forecasts 11.2023 through 12.2024.
MODEL_START = "11.2017"
MODEL_END = "10.2023"
FORECAST_START = "11.2023"
FORECAST_END = "12.2024"


def main() -> None:
    monthly_df = load_niias_data(DATA_XLSX)
    print(
        f"[Step 1] Loaded {DATA_XLSX}: {len(monthly_df)} rows, "
        f"{monthly_df['DATE'].min().date()}..{monthly_df['DATE'].max().date()}"
    )

    if RUN_OPTIMIZATION:
        best_params_per_target = run_optimization_all(
            monthly_df=monthly_df,
            method=OPTIMIZE_METHOD,
            n_trials=OPTIMIZE_N_TRIALS,
            n_splits=OPTIMIZE_N_SPLITS,
            lags=LAGS,
            rolling_windows=ROLLING_WINDOWS,
            model_start=MODEL_START,
            model_end=MODEL_END,
            seed=SEED,
            outdir=OUTDIR,
            force_refit=True,
        )
        print(f"[Step 1.5] Optimization complete: {len(best_params_per_target)} targets")
    else:
        best_params_per_target = {
            target: cached
            for target in TARGET_INDICATORS
            if (cached := load_best_params(OUTDIR, target))
        }
        print(f"[Step 1.5] Optimization skipped. Cached params: {len(best_params_per_target)}")

    if RUN_FORECAST:
        forecast_df = run_catboost_forecast(
            monthly_df=monthly_df,
            outdir=OUTDIR,
            lags=LAGS,
            rolling_windows=ROLLING_WINDOWS,
            model_start=MODEL_START,
            model_end=MODEL_END,
            forecast_start=FORECAST_START,
            forecast_end=FORECAST_END,
            random_seed=SEED,
            catboost_params_per_target=best_params_per_target or None,
        )
        print(f"[Step 2] Forecast ready: {len(forecast_df)} rows")
    else:
        forecast_df = load_niias_data(DATA_XLSX)
        if FORECAST_XLSX.exists():
            forecast_df = __import__("pandas").read_excel(FORECAST_XLSX, sheet_name="forecast")
            forecast_df["DATE"] = __import__("pandas").to_datetime(forecast_df["DATE"])
        print(f"[Step 2] Forecast skipped. Loaded {FORECAST_XLSX}")

    if RUN_COMPARISON:
        build_smape_comparison(
            baseline_metrics_path=OUTDIR / "seasonal_naive_metrics.xlsx",
            catboost_metrics_path=OUTDIR / "catboost_metrics.xlsx",
            detailed_out=BASE_DIR / "compare_results" / "smape_comparison_detailed.xlsx",
            summary_out=BASE_DIR / "compare_results" / "smape_comparison_by_indicator.xlsx",
        )
        print("[Step 3] SMAPE comparison complete")

    if RUN_PLOTS:
        for indicator in TARGET_INDICATORS:
            plot_predictions(
                monthly_df=monthly_df,
                forecast_df=forecast_df,
                indicator=indicator,
                outpath=PLOTS_DIR / f"{indicator}.html",
            )
        print(f"[Step 4] Plots saved: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
