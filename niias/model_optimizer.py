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
        ROLLING_WINDOWS,
        TARGET_INDICATORS,
        _add_features,
        _feature_cols,
        filter_by_date_range,
    )
except ImportError:  # pragma: no cover
    from catboost_model import (
        CAT_FEATURES,
        DEFAULT_CB_PARAMS,
        LAGS,
        ROLLING_WINDOWS,
        TARGET_INDICATORS,
        _add_features,
        _feature_cols,
        filter_by_date_range,
    )


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
    feat_cols = _feature_cols(lags, windows)
    full = _add_features(monthly_df, target, lags, windows)
    train = (
        full.dropna(subset=feat_cols + [target])
        .sort_values(["YEAR", "MONTH"])
        .reset_index(drop=True)
    )
    X_train = train[feat_cols].fillna(0)
    y_train = train[target].to_numpy(dtype=float)
    cat_cols = [c for c in CAT_FEATURES if c in feat_cols]
    return X_train, y_train, cat_cols


def _cv_rmse(params: dict, X: pd.DataFrame, y: np.ndarray, cat_cols: List[str], n_splits: int, seed: int, trial=None) -> float:
    if CatBoostRegressor is None or Pool is None:
        raise ImportError(
            "CatBoost is required for optimization. Install it in the Python environment: "
            "pip install catboost"
        )

    periods = X[["YEAR", "MONTH"]].drop_duplicates().sort_values(["YEAR", "MONTH"])
    scores: List[float] = []

    for fold, (tr_period_pos, val_period_pos) in enumerate(_time_series_splits(len(periods), n_splits)):
        tr_periods = periods.iloc[tr_period_pos]
        val_periods = periods.iloc[val_period_pos]
        period_index = X.set_index(["YEAR", "MONTH"]).index
        tr_mask = period_index.isin(list(zip(tr_periods["YEAR"], tr_periods["MONTH"])))
        val_mask = period_index.isin(list(zip(val_periods["YEAR"], val_periods["MONTH"])))
        if tr_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        cb_params = {
            **DEFAULT_CB_PARAMS,
            **params,
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        model = CatBoostRegressor(**cb_params)
        model.fit(
            Pool(X[tr_mask], y[tr_mask], cat_features=cat_cols),
            eval_set=Pool(X[val_mask], y[val_mask], cat_features=cat_cols),
        )
        pred = model.predict(X[val_mask])
        scores.append(float(np.sqrt(np.mean((y[val_mask] - pred) ** 2))))

        if trial is not None:
            import optuna
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(scores)) if scores else float("inf")


def optimize_model_parameters_grid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: List[str],
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
        score = _cv_rmse(params, X_train, y_train, cat_cols, n_splits, seed)
        if score < best_score:
            best_score = score
            best_params = params.copy()
        print(f"  grid RMSE={score:.4f} best={best_score:.4f} params={params}")
    return best_params


def optimize_model_parameters_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: List[str],
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
        return _cv_rmse(params, X_train, y_train, cat_cols, n_splits, seed, trial=trial)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), study


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
    X_train, y_train, cat_cols = _build_train_data(
        monthly_df, target, lags, windows, test_year, model_start=model_start, model_end=model_end
    )
    print(f"[Optimization] {target}: rows={len(X_train)} features={X_train.shape[1]} method={method}")
    started = time.time()

    if method == "grid":
        best_params = optimize_model_parameters_grid(X_train, y_train, cat_cols, param_space, n_splits, seed)
        meta = {"method": "grid", "n_splits": n_splits}
    elif method == "optuna":
        best_params, study = optimize_model_parameters_optuna(
            X_train, y_train, cat_cols, param_space, n_splits, n_trials, seed
        )
        meta = {
            "method": "optuna",
            "n_trials": len(study.trials),
            "n_splits": n_splits,
            "best_cv_rmse": round(float(study.best_value), 4),
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
