"""
model_optimizer.py

Оптимизация гиперпараметров CatBoost для прогнозирования показателей ЖД.

Два режима (параметр method):
  • "grid"   — перебор фиксированной сетки (аналог mlgu.ru/1004),
               CV-разбивка — TimeSeriesSplit (корректно для временных рядов).
  • "optuna" — байесовская оптимизация TPE (Optuna); в 5–20 раз быстрее
               grid при большом пространстве параметров.

Ключевые отличия от базовой реализации (mlgu.ru/1004):
  • TimeSeriesSplit вместо обычного K-Fold — нет утечки данных из будущего.
  • Панельные данные: разбивка по уникальным периодам (YEAR, MONTH),
    а не по строкам, — каждый fold содержит все дороги.
  • Optuna TPE + MedianPruner — ранняя остановка неперспективных испытаний.
  • Результаты сохраняются в JSON для повторного использования без переобучения.
  • Важность признаков — интерактивный HTML (Plotly).

Использование
-------------
    from model_optimizer import run_optimization, load_best_params

    # Оптимизация (выполняется один раз; результат кешируется в JSON)
    best = run_optimization(
        monthly_df=monthly_df,
        target="TRAIN_KM",
        method="optuna",    # или "grid"
        n_trials=50,        # только для optuna
        n_splits=5,
        outdir="catboost_results",
    )

    # Прогноз с оптимальными параметрами
    forecast_df = run_catboost_forecast(
        monthly_df=monthly_df,
        forecast_years=[2025, 2026],
        catboost_params=best,
    )
"""

from __future__ import annotations

import json
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit

# Импортируем вспомогательные функции из основного модуля
from catboost_model import (
    DEFAULT_CB_PARAMS,
    CAT_FEATURES,
    LAGS,
    ROLLING_WINDOWS,
    TARGET_INDICATORS,
    _add_features,
    _feature_cols,
)


# ============================================================
# ПРОСТРАНСТВО ПАРАМЕТРОВ
# ============================================================

#: Сетка для метода "grid" — полный перебор (аналог mlgu.ru/1004)
GRID_PARAM_SPACE: Dict[str, list] = {
    "learning_rate": [0.01, 0.03, 0.1],
    "depth":         [4, 6, 8],
    "l2_leaf_reg":   [1, 3, 5, 7],
    "iterations":    [500, 1000],
}

#: Диапазоны для метода "optuna" — байесовский поиск
OPTUNA_PARAM_SPACE: Dict[str, dict] = {
    "learning_rate":      {"type": "float", "low": 0.005, "high": 0.3,  "log": True},
    "depth":              {"type": "int",   "low": 4,     "high": 10},
    "l2_leaf_reg":        {"type": "float", "low": 1.0,   "high": 10.0, "log": True},
    "iterations":         {"type": "int",   "low": 200,   "high": 2000},
    "random_strength":    {"type": "float", "low": 0.1,   "high": 5.0},
    "bagging_temperature":{"type": "float", "low": 0.0,   "high": 2.0},
}


# ============================================================
# ПОДГОТОВКА ОБУЧАЮЩЕЙ ВЫБОРКИ
# ============================================================

def _build_train_data(
    monthly_df: pd.DataFrame,
    target: str,
    lags: List[int],
    windows: List[int],
    test_year: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Строит матрицу признаков для обучения.

    Возвращает (X_train, y_train, cat_cols).

    Данные сортируются по (YEAR, MONTH) — это важно для корректной
    работы TimeSeriesSplit на панельных данных: каждый fold содержит
    строки всех дорог за несколько последовательных периодов.
    """
    if test_year is None:
        test_year = int(monthly_df["YEAR"].max())

    df = monthly_df.copy()
    df["ROAD_NAME"] = df["ROAD_NAME"].astype(str)

    feat_cols = _feature_cols(lags, windows)
    full = _add_features(df, target, lags, windows)

    train = (
        full[full["YEAR"] < test_year]
        .dropna(subset=feat_cols + [target])
        .sort_values(["YEAR", "MONTH", "ROAD_NAME"])
        .reset_index(drop=True)
    )

    X_train = train[feat_cols].copy()
    y_train = train[target].values
    cat_cols = [c for c in CAT_FEATURES if c in feat_cols]

    return X_train, y_train, cat_cols


# ============================================================
# ВЫЧИСЛЕНИЕ RMSE НА КРОСС-ВАЛИДАЦИИ (TimeSeriesSplit)
# ============================================================

def _cv_rmse(
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
    n_splits: int,
    seed: int,
    trial=None,          # опционально: объект optuna.Trial для прунинга
) -> float:
    """
    Оценивает параметры через TimeSeriesSplit.

    Разбивка производится по уникальным периодам (YEAR, MONTH),
    а не по отдельным строкам — иначе одна дорога могла бы попасть
    в train, а другая — в val для одного и того же месяца.

    Возвращает среднее RMSE по фолдам.
    """
    # Уникальные периоды в порядке возрастания времени
    periods = X[["YEAR", "MONTH"]].drop_duplicates().sort_values(["YEAR", "MONTH"])
    period_idx = periods.index  # индексы строк периодов (по одной на каждый период в X)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores: List[float] = []

    for fold, (tr_period_pos, val_period_pos) in enumerate(tscv.split(period_idx)):
        # Периоды обучения и валидации
        tr_periods  = periods.iloc[tr_period_pos]
        val_periods = periods.iloc[val_period_pos]

        # Строки X, попадающие в каждый фолд
        tr_mask = X.set_index(["YEAR", "MONTH"]).index.isin(
            list(zip(tr_periods["YEAR"], tr_periods["MONTH"]))
        )
        val_mask = X.set_index(["YEAR", "MONTH"]).index.isin(
            list(zip(val_periods["YEAR"], val_periods["MONTH"]))
        )

        if tr_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        X_tr,  X_val  = X[tr_mask],  X[val_mask]
        y_tr,  y_val  = y[tr_mask],  y[val_mask]

        train_pool = Pool(X_tr,  y_tr,  cat_features=cat_cols)
        val_pool   = Pool(X_val, y_val, cat_features=cat_cols)

        cb_params = {
            **DEFAULT_CB_PARAMS,
            **params,
            "random_seed": seed,
            "verbose":     False,
        }

        model = CatBoostRegressor(**cb_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_pool, eval_set=val_pool)

        preds = np.maximum(model.predict(X_val), 0.0)
        rmse  = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        fold_scores.append(rmse)

        # Сообщаем Optuna промежуточный результат для прунинга
        if trial is not None:
            import optuna
            trial.report(np.mean(fold_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_scores)) if fold_scores else float("inf")


# ============================================================
# МЕТОД 1: GRID SEARCH (аналог mlgu.ru/1004 с TimeSeriesSplit)
# ============================================================

def optimize_model_parameters_grid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: List[str],
    param_space: Optional[Dict[str, list]] = None,
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Полный перебор сетки гиперпараметров с TimeSeriesSplit CV.

    Аналог GridSearchCV из sklearn, но:
      • корректная временна́я разбивка (TimeSeriesSplit),
      • прямая работа с CatBoost Pool и категориальными признаками.

    Параметры
    ----------
    X_train    : матрица признаков (обучение)
    y_train    : целевой вектор
    cat_cols   : список категориальных столбцов
    param_space: сетка параметров (по умолчанию GRID_PARAM_SPACE)
    n_splits   : число фолдов TimeSeriesSplit
    seed       : зерно воспроизводимости
    verbose    : выводить прогресс

    Возвращает
    ----------
    dict с лучшими гиперпараметрами
    """
    if param_space is None:
        param_space = GRID_PARAM_SPACE

    keys   = list(param_space.keys())
    values = list(param_space.values())
    combos = list(product(*values))
    total  = len(combos)

    if verbose:
        print(f"  Grid Search: {total} комбинаций × {n_splits} фолдов")

    best_score  = float("inf")
    best_params = {}
    t0 = time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        score  = _cv_rmse(params, X_train, y_train, cat_cols, n_splits, seed)

        if score < best_score:
            best_score  = score
            best_params = params.copy()

        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i)
            print(
                f"  [{i:3d}/{total}] RMSE={score:,.2f}  "
                f"best={best_score:,.2f}  "
                f"params={params}  "
                f"ETA {eta:.0f}s",
                flush=True,
            )

    if verbose:
        print(f"\n  Лучшие параметры (Grid): {best_params}")
        print(f"  Лучший CV RMSE: {best_score:,.4f}")

    return best_params


# ============================================================
# МЕТОД 2: OPTUNA (байесовская оптимизация TPE)
# ============================================================

def optimize_model_parameters_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: List[str],
    param_space: Optional[Dict[str, dict]] = None,
    n_splits: int = 5,
    n_trials: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Байесовская оптимизация гиперпараметров (Optuna TPE + MedianPruner).

    Преимущества перед Grid Search:
      • TPE (Tree-structured Parzen Estimator) концентрирует поиск
        в перспективных областях пространства параметров.
      • MedianPruner останавливает неперспективные испытания уже
        после первых фолдов — экономит время в 2–5 раз.
      • Непрерывное пространство (learning_rate, l2_leaf_reg и др.)
        исследуется эффективнее, чем дискретная сетка.

    Параметры
    ----------
    X_train    : матрица признаков (обучение)
    y_train    : целевой вектор
    cat_cols   : список категориальных столбцов
    param_space: описание пространства поиска (по умолчанию OPTUNA_PARAM_SPACE)
    n_splits   : число фолдов TimeSeriesSplit
    n_trials   : число испытаний Optuna
    seed       : зерно воспроизводимости
    verbose    : выводить прогресс Optuna

    Возвращает
    ----------
    dict с лучшими гиперпараметрами
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ImportError:
        raise ImportError(
            "Для метода 'optuna' установите пакет: pip install optuna"
        )

    if param_space is None:
        param_space = OPTUNA_PARAM_SPACE

    def objective(trial) -> float:
        params: dict = {}
        for name, cfg in param_space.items():
            t = cfg["type"]
            if t == "float":
                params[name] = trial.suggest_float(
                    name, cfg["low"], cfg["high"], log=cfg.get("log", False)
                )
            elif t == "int":
                params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
            elif t == "categorical":
                params[name] = trial.suggest_categorical(name, cfg["choices"])

        return _cv_rmse(
            params, X_train, y_train, cat_cols, n_splits, seed, trial=trial
        )

    optuna_verbosity = optuna.logging.INFO if verbose else optuna.logging.WARNING
    optuna.logging.set_verbosity(optuna_verbosity)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params.copy()
    best_score  = study.best_value

    if verbose:
        print(f"\n  Лучшие параметры (Optuna): {best_params}")
        print(f"  Лучший CV RMSE: {best_score:,.4f}")
        print(f"  Завершено испытаний: {len(study.trials)}")

    return best_params, study


# ============================================================
# ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================

def plot_feature_importance(
    model: CatBoostRegressor,
    feature_names: List[str],
    indicator: str,
    outpath: "Path | str",
    top_n: int = 20,
) -> None:
    """
    Строит горизонтальный bar-chart важности признаков (Plotly HTML).

    Параметры
    ----------
    model        : обученная CatBoostRegressor
    feature_names: список имён признаков (порядок совпадает с X)
    indicator    : название показателя (для заголовка)
    outpath      : путь для сохранения .html
    top_n        : сколько топ-признаков отображать
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [!] plotly не установлен — график важности не сохранён")
        return

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    importance = model.get_feature_importance()
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")         # для горизонтального bar снизу-вверх
    )

    fig = go.Figure(go.Bar(
        x=fi["importance"],
        y=fi["feature"],
        orientation="h",
        marker=dict(
            color=fi["importance"],
            colorscale="Blues",
            showscale=False,
        ),
        text=fi["importance"].round(2),
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Важность признаков</b><br><sup>{indicator}</sup>",
            font=dict(size=14),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(title="Важность (%)", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        margin=dict(l=160, r=60, t=80, b=50),
        width=900,
        height=max(350, top_n * 28),
    )

    fig.write_html(str(outpath), include_plotlyjs="cdn")


# ============================================================
# СОХРАНЕНИЕ / ЗАГРУЗКА РЕЗУЛЬТАТОВ
# ============================================================

def save_best_params(
    params: dict,
    outdir: str | Path,
    target: str,
    meta: Optional[dict] = None,
) -> Path:
    """
    Сохраняет лучшие параметры в <outdir>/best_params/<target>.json.

    Параметры
    ----------
    params  : словарь гиперпараметров
    outdir  : корневая директория результатов
    target  : название показателя (имя файла)
    meta    : дополнительные метаданные (score, method, n_trials, …)

    Возвращает путь к сохранённому файлу.
    """
    out_dir = Path(outdir) / "best_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{target}.json"

    payload = {"params": params}
    if meta:
        payload["meta"] = meta

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return fpath


def load_best_params(outdir: str | Path, target: str) -> Optional[dict]:
    """
    Загружает ранее сохранённые параметры из <outdir>/best_params/<target>.json.

    Возвращает dict параметров или None, если файл не найден.
    """
    fpath = Path(outdir) / "best_params" / f"{target}.json"
    if not fpath.exists():
        return None

    with open(fpath, encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("params")


# ============================================================
# ПУБЛИЧНЫЙ АЛИАС — совместимость со стилем mlgu.ru/1004
# ============================================================

def optimize_model_parameters(
    train_pool: "Pool",
    test_pool: "Pool",
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> "CatBoostRegressor":
    """
    Оптимизация гиперпараметров CatBoost — публичный интерфейс
    в стиле mlgu.ru/1004.

    В отличие от оригинальной реализации (sklearn GridSearchCV + обычный K-Fold):
      • используется TimeSeriesSplit — нет утечки данных из будущего,
      • категориальные признаки передаются через нативный CatBoost Pool,
      • не требует параметра n_jobs (CatBoost многопоточен внутри).

    Параметры
    ----------
    train_pool : CatBoost Pool с обучающими данными
    test_pool  : CatBoost Pool для ранней остановки
    n_splits   : число фолдов TimeSeriesSplit
    seed       : зерно воспроизводимости
    verbose    : выводить прогресс

    Возвращает
    ----------
    Обученный CatBoostRegressor с лучшими параметрами

    Пример
    ------
        from catboost import Pool
        from model_optimizer import optimize_model_parameters

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)

        model = optimize_model_parameters(train_pool, test_pool)
    """
    X = pd.DataFrame(
        train_pool.get_features(),
        columns=train_pool.get_feature_names() or [
            f"f{i}" for i in range(train_pool.num_col())
        ],
    )
    y = np.asarray(train_pool.get_label())
    cat_cols = list(train_pool.get_cat_feature_indices() or [])

    # Преобразуем числовые индексы кат.признаков в имена столбцов
    cat_col_names = [X.columns[i] for i in cat_cols] if cat_cols else []

    best_params = optimize_model_parameters_grid(
        X_train=X,
        y_train=y,
        cat_cols=cat_col_names,
        n_splits=n_splits,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Лучшие параметры: {best_params}")

    final_params = {**DEFAULT_CB_PARAMS, **best_params, "random_seed": seed, "verbose": False}
    model = CatBoostRegressor(**final_params)
    model.fit(train_pool, eval_set=test_pool)

    return model


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_optimization(
    monthly_df: pd.DataFrame,
    target: str = "TRAIN_KM",
    method: str = "optuna",
    n_trials: int = 50,
    n_splits: int = 5,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    test_year: Optional[int] = None,
    seed: int = 42,
    outdir: str | Path = "catboost_results",
    force_refit: bool = False,
    save_importance: bool = True,
    param_space: Optional[dict] = None,
) -> dict:
    """
    Полный цикл оптимизации гиперпараметров для одного показателя.

    Порядок работы:
      1. Если уже есть сохранённый JSON и force_refit=False — загружает его.
      2. Иначе строит обучающую выборку с признаками (лаги, скользящие, …).
      3. Запускает оптимизацию выбранным методом.
      4. Сохраняет результат в JSON.
      5. Дообучает финальную модель на полных данных и сохраняет
         важность признаков как HTML (если save_importance=True).

    Параметры
    ----------
    monthly_df      : DataFrame из generate_monthly_data
    target          : целевой показатель, напр. "TRAIN_KM"
    method          : "grid" или "optuna"
    n_trials        : число испытаний (только для method="optuna")
    n_splits        : число фолдов TimeSeriesSplit
    lags            : список лагов (по умолчанию LAGS)
    rolling_windows : размеры окон (по умолчанию ROLLING_WINDOWS)
    test_year       : последний год обучения (по умолчанию max(YEAR))
    seed            : зерно воспроизводимости
    outdir          : директория для сохранения результатов
    force_refit     : True — перезапустить оптимизацию, игнорируя кеш
    save_importance : True — сохранить HTML с важностью признаков
    param_space     : переопределение пространства поиска.
                      Для method="grid"   — Dict[str, list]  (аналог GRID_PARAM_SPACE).
                      Для method="optuna" — Dict[str, dict]  (аналог OPTUNA_PARAM_SPACE).
                      None — использовать значения по умолчанию из модуля.

    Возвращает
    ----------
    dict с лучшими гиперпараметрами (готов к передаче в catboost_params)
    """
    lags    = lags    or LAGS
    windows = rolling_windows or ROLLING_WINDOWS
    outdir  = Path(outdir)

    # ── Проверяем кеш ─────────────────────────────────────
    if not force_refit:
        cached = load_best_params(outdir, target)
        if cached is not None:
            print(f"  [{target}] Загружены кешированные параметры: {cached}")
            return cached

    # ── Строим обучающую выборку ───────────────────────────
    print(f"\n[Оптимизация] Показатель: {target}  Метод: {method}")
    X_train, y_train, cat_cols = _build_train_data(
        monthly_df, target, lags, windows, test_year
    )
    print(f"  Обучающих строк: {len(X_train)}  "
          f"Признаков: {X_train.shape[1]}  "
          f"Фолдов CV: {n_splits}")

    # ── Запуск оптимизации ────────────────────────────────
    t0 = time.time()
    study = None

    if method == "grid":
        best_params = optimize_model_parameters_grid(
            X_train, y_train, cat_cols,
            param_space=param_space,
            n_splits=n_splits, seed=seed,
        )
        meta = {"method": "grid", "n_splits": n_splits}

    elif method == "optuna":
        best_params, study = optimize_model_parameters_optuna(
            X_train, y_train, cat_cols,
            param_space=param_space,
            n_splits=n_splits, n_trials=n_trials, seed=seed,
        )
        meta = {
            "method":   "optuna",
            "n_trials": len(study.trials),
            "n_splits": n_splits,
            "best_cv_rmse": round(study.best_value, 4),
        }

    else:
        raise ValueError(f"method должен быть 'grid' или 'optuna', получено: {method!r}")

    elapsed = time.time() - t0
    print(f"  Время оптимизации: {elapsed:.1f}с")

    # ── Сохраняем параметры в JSON ─────────────────────────
    json_path = save_best_params(best_params, outdir, target, meta=meta)
    print(f"  Параметры сохранены: {json_path}")

    # ── Дообучаем финальную модель и строим важность признаков ──
    if save_importance:
        print(f"  Дообучение финальной модели для важности признаков...")
        final_params = {
            **DEFAULT_CB_PARAMS,
            **best_params,
            "random_seed":          seed,
            "verbose":              False,
            # eval_set не передаётся → отключаем зависящие от него параметры
            "use_best_model":       False,
            "early_stopping_rounds": None,
        }
        final_model = CatBoostRegressor(**final_params)
        final_model.fit(Pool(X_train, y_train, cat_features=cat_cols))

        fi_path = outdir / "feature_importance" / f"{target}.html"
        plot_feature_importance(
            model=final_model,
            feature_names=list(X_train.columns),
            indicator=target,
            outpath=fi_path,
        )
        print(f"  Важность признаков: {fi_path}")

    return best_params


# ============================================================
# ОПТИМИЗАЦИЯ ВСЕХ ПОКАЗАТЕЛЕЙ СРАЗУ
# ============================================================

def run_optimization_all(
    monthly_df: pd.DataFrame,
    targets: Optional[List[str]] = None,
    method: str = "optuna",
    n_trials: int = 50,
    n_splits: int = 5,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    test_year: Optional[int] = None,
    seed: int = 42,
    outdir: str | Path = "catboost_results",
    force_refit: bool = False,
    param_space: Optional[dict] = None,
) -> Dict[str, dict]:
    """
    Оптимизирует гиперпараметры для каждого показателя в targets.

    Поскольку каждый показатель имеет разную динамику (TRAIN_KM —
    практически монотонный тренд, LOSS_* — сезонный с шумом),
    оптимальные параметры могут существенно различаться.

    Параметры
    ----------
    targets     : список показателей (по умолчанию все TARGET_INDICATORS).
                  Можно передать подмножество для ускорения.
    param_space : переопределение пространства поиска для всех показателей.
                  Для method="grid"   — Dict[str, list].
                  Для method="optuna" — Dict[str, dict].
                  None — использовать GRID_PARAM_SPACE / OPTUNA_PARAM_SPACE.

    Возвращает
    ----------
    dict {target: best_params} — готов к итерации в run_catboost_forecast.
    """
    if targets is None:
        targets = TARGET_INDICATORS

    all_best: Dict[str, dict] = {}
    for target in targets:
        best = run_optimization(
            monthly_df=monthly_df,
            target=target,
            method=method,
            n_trials=n_trials,
            n_splits=n_splits,
            lags=lags,
            rolling_windows=rolling_windows,
            test_year=test_year,
            seed=seed,
            outdir=outdir,
            force_refit=force_refit,
            save_importance=True,
            param_space=param_space,
        )
        all_best[target] = best

    print(f"\n[OK] Оптимизация завершена для {len(all_best)} показателей")
    return all_best
