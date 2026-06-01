"""Hyperparameter optimization for the NIIAS CatBoost models."""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:  # pragma: no cover - handled when optimization is requested
    CatBoostRegressor = None
    Pool = None

try:
    from .catboost_model import (
        CAT_FEATURES,
        DEFAULT_CB_PARAMS,
        LAGS,
        PCH_MODEL_RATIO,
        PCH_MODEL_TYPES,
        ROLLING_WINDOWS,
        TARGET_INDICATORS,
        _build_features,
        _feature_cols,
        _model_target_col,
        _recursive_forecast,
        _split_model_config,
        _target_cross_cols,
        add_ratio_targets,
        filter_by_date_range,
    )
    from .metrics import smape_percent
except ImportError:  # pragma: no cover
    from catboost_model import (
        CAT_FEATURES,
        DEFAULT_CB_PARAMS,
        LAGS,
        PCH_MODEL_RATIO,
        PCH_MODEL_TYPES,
        ROLLING_WINDOWS,
        TARGET_INDICATORS,
        _build_features,
        _feature_cols,
        _model_target_col,
        _recursive_forecast,
        _split_model_config,
        _target_cross_cols,
        add_ratio_targets,
        filter_by_date_range,
    )
    from metrics import smape_percent


GRID_PARAM_SPACE: Dict[str, list] = {
    "learning_rate": [0.01, 0.03, 0.1],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [1, 3, 5],
    "iterations": [400, 800],
}

OPTUNA_PARAM_SPACE: Dict[str, dict] = {
    "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
    "depth": {"type": "int", "low": 3, "high": 8},
    "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
    "iterations": {"type": "int", "low": 300, "high": 1500},
    "random_strength": {"type": "float", "low": 0.1, "high": 5.0},
    "bagging_temperature": {"type": "float", "low": 0.0, "high": 2.0},
}

BACKTEST_HORIZON = 14
LAG_CANDIDATES: List[List[int]] = [
    [1, 2, 3, 6, 12],
    [1, 2, 3, 6, 12, 24],
    [1, 2, 3, 6, 12, 24, 36],
    [1, 2, 3, 6, 12, 13, 24],
    [1, 2, 3, 6, 12, 13, 24, 36],
]
ROLLING_WINDOW_CANDIDATES: List[List[int]] = [
    [3, 6],
    [3, 6, 12],
    [2, 3, 6, 12],
    [3, 4, 6, 12],
]


def _encode_int_list(values: List[int]) -> str:
    return ",".join(str(int(v)) for v in values)


def _decode_int_list(value: str | List[int]) -> List[int]:
    if isinstance(value, list):
        return sorted({int(v) for v in value})
    return sorted({int(part) for part in str(value).split(",") if part})


def _time_series_splits(n_periods: int, n_splits: int):
    """Yield expanding-window splits over ordered period positions."""
    n_splits = min(n_splits, max(1, n_periods - 1))
    test_size = max(1, n_periods // (n_splits + 1))
    for split_idx in range(n_splits):
        train_end = n_periods - test_size * (n_splits - split_idx)
        val_start = train_end
        val_end = min(val_start + test_size, n_periods)
        if train_end <= 0 or val_start >= val_end:
            continue
        yield np.arange(0, train_end), np.arange(val_start, val_end)


def _backtest_period_splits(
    periods: pd.DataFrame,
    n_splits: int,
    max_lag: int,
    horizon: int = BACKTEST_HORIZON,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Build rolling-origin backtest folds with contiguous validation periods."""
    periods = periods.sort_values(["YEAR", "MONTH"]).reset_index(drop=True)
    n_periods = len(periods)
    min_train_periods = max_lag + 1
    if n_periods <= min_train_periods + 1:
        return []

    feasible_horizon = max(1, (n_periods - min_train_periods) // max(n_splits, 1))
    horizon = min(horizon, feasible_horizon)
    if horizon <= 0:
        return []

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for split_idx in range(n_splits):
        val_end = n_periods - 1 - (n_splits - 1 - split_idx) * horizon
        val_start = val_end - horizon + 1
        train_end = val_start - 1
        if train_end + 1 < min_train_periods or val_start < 0 or val_start > val_end:
            continue
        train_periods = periods.iloc[: train_end + 1].copy()
        val_periods = periods.iloc[val_start : val_end + 1].copy()
        splits.append((train_periods, val_periods))
    return splits


def _build_train_data(
    monthly_df: pd.DataFrame,
    target: str,
    lags: List[int],
    windows: List[int],
    test_year: Optional[int] = None,
    model_start=None,
    model_end=None,
) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    if model_start is not None or model_end is not None:
        monthly_df = filter_by_date_range(monthly_df, start=model_start, end=model_end)
    elif test_year is not None:
        monthly_df = monthly_df[monthly_df["YEAR"] < test_year].copy()
    monthly_df = add_ratio_targets(monthly_df)
    model_target = _model_target_col(target)
    cross_cols = _target_cross_cols(target)
    feat_cols = _feature_cols(lags, windows, cross_cols)
    full = _build_features(monthly_df, model_target, lags, windows, cross_cols)
    train = (
        full.dropna(subset=feat_cols + [model_target])
        .sort_values(["YEAR", "MONTH"])
        .reset_index(drop=True)
    )
    X_train = train[feat_cols].fillna(0)
    y_train = train[model_target].to_numpy(dtype=float)
    cat_cols = [c for c in CAT_FEATURES if c in feat_cols]
    return X_train, y_train, cat_cols


def _backtest_smape(
    params: dict,
    monthly_df: pd.DataFrame,
    target: str,
    lags: List[int],
    windows: List[int],
    n_splits: int,
    seed: int,
    trial=None,
    horizon: int = BACKTEST_HORIZON,
) -> float:
    if CatBoostRegressor is None or Pool is None:
        raise ImportError(
            "CatBoost is required for optimization. Install it in the Python environment: "
            "pip install catboost"
        )

    cb_params_only, trial_lags, trial_windows, pch_model_type = _split_model_config(
        params,
        lags,
        windows,
    )
    df = add_ratio_targets(monthly_df).sort_values("DATE").reset_index(drop=True)
    periods = df[["YEAR", "MONTH"]].drop_duplicates().sort_values(["YEAR", "MONTH"])
    splits = _backtest_period_splits(periods, n_splits=n_splits, max_lag=max(trial_lags), horizon=horizon)
    if not splits:
        return float("inf")

    model_target = _model_target_col(target, pch_model_type)
    cross_cols = _target_cross_cols(target)
    feat_cols = _feature_cols(trial_lags, trial_windows, cross_cols)
    cat_cols = [c for c in CAT_FEATURES if c in feat_cols]
    scores: List[float] = []

    for fold, (train_periods, val_periods) in enumerate(splits):
        train_keys = list(zip(train_periods["YEAR"], train_periods["MONTH"]))
        val_keys = list(zip(val_periods["YEAR"], val_periods["MONTH"]))
        train_df = df[df.set_index(["YEAR", "MONTH"]).index.isin(train_keys)].copy()
        val_df = df[df.set_index(["YEAR", "MONTH"]).index.isin(val_keys)].copy()
        if train_df.empty or val_df.empty:
            continue

        full_feat = _build_features(train_df, model_target, trial_lags, trial_windows, cross_cols)
        train_feat = full_feat.dropna(subset=feat_cols + [model_target]).sort_values(["YEAR", "MONTH"])
        if train_feat.empty:
            continue

        cb_params = {
            **DEFAULT_CB_PARAMS,
            **cb_params_only,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "use_best_model": False,
            "early_stopping_rounds": None,
        }
        ots_forecast_df = None
        if target == "PCH_OTS":
            ots_feat_cols = _feature_cols(trial_lags, trial_windows, [])
            ots_full_feat = _build_features(train_df, "OTS", trial_lags, trial_windows, [])
            ots_train_feat = ots_full_feat.dropna(subset=ots_feat_cols + ["OTS"]).sort_values(["YEAR", "MONTH"])
            if ots_train_feat.empty:
                continue
            ots_model = CatBoostRegressor(**cb_params)
            ots_model.fit(Pool(
                ots_train_feat[ots_feat_cols].fillna(0),
                ots_train_feat["OTS"].to_numpy(dtype=float),
                cat_features=[c for c in CAT_FEATURES if c in ots_feat_cols],
            ))
            ots_forecast_df = _recursive_forecast(
                model=ots_model,
                history_df=train_df[["DATE", "YEAR", "MONTH", "OTS"]],
                target_col="OTS",
                future_dates=val_df[["DATE", "YEAR", "MONTH"]].sort_values("DATE").reset_index(drop=True),
                lags=trial_lags,
                windows=trial_windows,
                feature_cols=ots_feat_cols,
            )[["DATE", "OTS"]]

        model = CatBoostRegressor(**cb_params)

        X_train = train_feat[feat_cols].fillna(0)
        y_train = train_feat[model_target].to_numpy(dtype=float)
        model.fit(Pool(X_train, y_train, cat_features=cat_cols))

        future_dates = val_df[["DATE", "YEAR", "MONTH"]].sort_values("DATE").reset_index(drop=True)
        cross_future_df = None
        if cross_cols:
            if target == "PCH_OTS" and ots_forecast_df is not None and "OTS" in cross_cols:
                cross_future_df = ots_forecast_df.copy()
            else:
                cross_future_df = val_df[["DATE", *cross_cols]].copy()

        pred_model = _recursive_forecast(
            model=model,
            history_df=train_df[["DATE", "YEAR", "MONTH", model_target] + cross_cols],
            target_col=model_target,
            future_dates=future_dates,
            lags=trial_lags,
            windows=trial_windows,
            feature_cols=feat_cols,
            cross_cols=cross_cols,
            cross_future_df=cross_future_df,
        )

        if target == "PCH_OTS" and pch_model_type == PCH_MODEL_RATIO:
            pred_df = pred_model[["DATE", model_target]].merge(
                ots_forecast_df.rename(columns={"OTS": "OTS_pred"}),
                on="DATE",
                how="left",
            ).merge(
                val_df[["DATE", target]], on="DATE", how="left"
            )
            y_pred = (pred_df[model_target] * pred_df["OTS_pred"]).to_numpy(dtype=float)
            y_true = pred_df[target].to_numpy(dtype=float)
        else:
            pred_df = pred_model[["DATE", model_target]].rename(
                columns={model_target: f"{target}_pred"}
            ).merge(
                val_df[["DATE", target]].rename(columns={target: f"{target}_actual"}),
                on="DATE",
                how="left",
            )
            y_pred = pred_df[f"{target}_pred"].to_numpy(dtype=float)
            y_true = pred_df[f"{target}_actual"].to_numpy(dtype=float)

        score = smape_percent(y_true, y_pred)
        scores.append(score)

        if trial is not None:
            import optuna
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(scores)) if scores else float("inf")


def optimize_model_parameters_grid(
    monthly_df: pd.DataFrame,
    target: str,
    lags: List[int],
    windows: List[int],
    param_space: Optional[Dict[str, list]] = None,
    n_splits: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    param_space = param_space or GRID_PARAM_SPACE
    best_score = float("inf")
    best_params: Dict[str, object] = {}
    keys = list(param_space.keys())
    for combo in product(*param_space.values()):
        params = dict(zip(keys, combo))
        score = _backtest_smape(params, monthly_df, target, lags, windows, n_splits, seed)
        if score < best_score:
            best_score = score
            best_params = params.copy()
        print(f"  grid SMAPE={score:.4f} best={best_score:.4f} params={params}")
    return best_params


def optimize_model_parameters_optuna(
    monthly_df: pd.DataFrame,
    target: str,
    lags: List[int],
    windows: List[int],
    param_space: Optional[Dict[str, dict]] = None,
    n_splits: int = 3,
    n_trials: int = 20,
    seed: int = 42,
) -> tuple[Dict[str, object], object]:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    param_space = param_space or OPTUNA_PARAM_SPACE

    def objective(trial) -> float:
        params: Dict[str, object] = {}
        for name, cfg in param_space.items():
            if cfg["type"] == "float":
                params[name] = trial.suggest_float(name, cfg["low"], cfg["high"], log=cfg.get("log", False))
            elif cfg["type"] == "int":
                params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
            else:
                raise ValueError(f"Unsupported param type: {cfg['type']}")
        params["lags"] = _decode_int_list(trial.suggest_categorical(
            "lags",
            [_encode_int_list(v) for v in LAG_CANDIDATES],
        ))
        params["rolling_windows"] = _decode_int_list(trial.suggest_categorical(
            "rolling_windows",
            [_encode_int_list(v) for v in ROLLING_WINDOW_CANDIDATES],
        ))
        if target == "PCH_OTS":
            params["pch_model_type"] = trial.suggest_categorical("pch_model_type", PCH_MODEL_TYPES)
        return _backtest_smape(params, monthly_df, target, lags, windows, n_splits, seed, trial=trial)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_params)
    if "lags" in best_params:
        best_params["lags"] = _decode_int_list(best_params["lags"])
    if "rolling_windows" in best_params:
        best_params["rolling_windows"] = _decode_int_list(best_params["rolling_windows"])
    if target != "PCH_OTS":
        best_params.pop("pch_model_type", None)
    return best_params, study


def save_best_params(params: dict, outdir: str | Path, target: str, meta: Optional[dict] = None) -> Path:
    out_dir = Path(outdir) / "best_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target}.json"
    payload = {"params": params}
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_best_params(outdir: str | Path, target: str) -> Optional[dict]:
    path = Path(outdir) / "best_params" / f"{target}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("params")


def run_optimization(
    monthly_df: pd.DataFrame,
    target: str,
    method: str = "optuna",
    n_trials: int = 20,
    n_splits: int = 3,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    test_year: Optional[int] = None,
    model_start=None,
    model_end=None,
    seed: int = 42,
    outdir: str | Path = "niias/catboost_results",
    force_refit: bool = False,
    param_space: Optional[dict] = None,
) -> dict:
    if CatBoostRegressor is None or Pool is None:
        raise ImportError(
            "CatBoost is required for optimization. Install it in the Python environment: "
            "pip install catboost"
        )

    if not force_refit:
        cached = load_best_params(outdir, target)
        if cached is not None:
            print(f"  [{target}] cached params: {cached}")
            return cached

    lags = lags or LAGS
    windows = rolling_windows or ROLLING_WINDOWS
    optimization_df = monthly_df.copy()
    if model_start is not None or model_end is not None:
        optimization_df = filter_by_date_range(optimization_df, start=model_start, end=model_end)
    elif test_year is not None:
        optimization_df = optimization_df[optimization_df["YEAR"] < test_year].copy()
    optimization_df = add_ratio_targets(optimization_df)

    X_train, y_train, cat_cols = _build_train_data(
        optimization_df, target, lags, windows
    )
    print(
        f"[Optimization] {target}: rows={len(X_train)} features={X_train.shape[1]} "
        f"method={method} objective=backtest_SMAPE"
    )
    started = time.time()

    if method == "grid":
        best_params = optimize_model_parameters_grid(
            monthly_df=optimization_df,
            target=target,
            lags=lags,
            windows=windows,
            param_space=param_space,
            n_splits=n_splits,
            seed=seed,
        )
        best_score = _backtest_smape(best_params, optimization_df, target, lags, windows, n_splits, seed)
        meta = {
            "method": "grid",
            "objective": "backtest_SMAPE",
            "n_splits": n_splits,
            "backtest_horizon": BACKTEST_HORIZON,
            "optimized_features": False,
            "lags": lags,
            "rolling_windows": windows,
            "pch_model_type": best_params.get("pch_model_type", PCH_MODEL_RATIO) if target == "PCH_OTS" else None,
            "best_backtest_smape": round(float(best_score), 4),
        }
    elif method == "optuna":
        best_params, study = optimize_model_parameters_optuna(
            monthly_df=optimization_df,
            target=target,
            lags=lags,
            windows=windows,
            param_space=param_space,
            n_splits=n_splits,
            n_trials=n_trials,
            seed=seed,
        )
        meta = {
            "method": "optuna",
            "objective": "backtest_SMAPE",
            "n_trials": len(study.trials),
            "n_splits": n_splits,
            "backtest_horizon": BACKTEST_HORIZON,
            "optimized_features": True,
            "lags": best_params.get("lags", lags),
            "rolling_windows": best_params.get("rolling_windows", windows),
            "pch_model_type": best_params.get("pch_model_type", PCH_MODEL_RATIO) if target == "PCH_OTS" else None,
            "best_backtest_smape": round(float(study.best_value), 4),
        }
    else:
        raise ValueError("method must be 'grid' or 'optuna'")

    meta["elapsed_sec"] = round(time.time() - started, 1)
    save_best_params(best_params, outdir, target, meta)
    return best_params


def run_optimization_all(
    monthly_df: pd.DataFrame,
    targets: Optional[List[str]] = None,
    method: str = "optuna",
    n_trials: int = 20,
    n_splits: int = 3,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    test_year: Optional[int] = None,
    model_start=None,
    model_end=None,
    seed: int = 42,
    outdir: str | Path = "niias/catboost_results",
    force_refit: bool = False,
    param_space: Optional[dict] = None,
) -> Dict[str, dict]:
    targets = targets or TARGET_INDICATORS
    return {
        target: run_optimization(
            monthly_df=monthly_df,
            target=target,
            method=method,
            n_trials=n_trials,
            n_splits=n_splits,
            lags=lags,
            rolling_windows=rolling_windows,
            test_year=test_year,
            model_start=model_start,
            model_end=model_end,
            seed=seed,
            outdir=outdir,
            force_refit=force_refit,
            param_space=param_space,
        )
        for target in targets
    }
