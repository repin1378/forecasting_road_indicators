"""
ets_model.py

ETS (Holt-Winters / ExponentialSmoothing) прогнозирование показателей надёжности ЖД.
Аналог catboost_model.py — единый формат входных данных, выходного CSV и HTML-графиков.

Показатели
----------
Прямые (ETS обучается отдельно для каждой дороги):
    TRAIN_KM, LOSS_12_COUNT, LOSS_12_SUM,
    LOSS_3_COUNT,  LOSS_3_SUM,
    LOSS_TECH_COUNT, LOSS_TECH_SUM

Производные (вычисляются из прогнозов):
    LOSS_COUNT_TOTAL = LOSS_12_COUNT + LOSS_3_COUNT + LOSS_TECH_COUNT
    LOSS_SUM_TOTAL   = LOSS_12_SUM   + LOSS_3_SUM   + LOSS_TECH_SUM
    SPECIFIC_LOSS    = LOSS_SUM_TOTAL / TRAIN_KM × 1 000 000

Fallback-стратегия
------------------
    ETS(A,A,A)  →  ETS(A,A)  →  ETS(A)

При ошибке или несходимости MLE переходит к более простой модели.

Доверительный интервал
----------------------
    ± 1.96 × σ, где σ = std(y_test − pred_test)
    Если σ = 0 (ряд константный) — используется ± 5 % от прогноза.

Выходные файлы (аналогичны CatBoost)
--------------------------------------
    {outdir}/ets_forecasts.csv              — прогнозы + CI (формат catboost_forecasts.csv)
    {outdir}/ets_metrics.csv                — MAPE, SMAPE, AIC, model_type
    {outdir}/ets_params/{indicator}.json    — α, β, γ, l₀, b₀, s₀..s₁₁ для всех дорог
    {outdir}/plots/{indicator}/{road}.html  — интерактивные HTML-графики
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ValueWarning,
                        message="No frequency information")

from metrics import mape_percent, smape_percent


# ============================================================
# КОНСТАНТЫ
# ============================================================

#: Показатели, для которых обучается отдельная ETS-модель на каждую дорогу
TARGET_INDICATORS: List[str] = [
    "TRAIN_KM",
    "LOSS_12_COUNT", "LOSS_12_SUM",
    "LOSS_3_COUNT",  "LOSS_3_SUM",
    "LOSS_TECH_COUNT", "LOSS_TECH_SUM",
]

#: Производные — вычисляются из прогнозов, не прогнозируются напрямую
DERIVED_INDICATORS: List[str] = [
    "LOSS_COUNT_TOTAL", "LOSS_SUM_TOTAL", "SPECIFIC_LOSS",
]

#: Все показатели для визуализации (прямые + производные)
ALL_PLOT_INDICATORS: List[str] = TARGET_INDICATORS + DERIVED_INDICATORS

#: Счётчики — округляются до целых после прогноза
COUNT_COLS: List[str] = [
    "LOSS_12_COUNT", "LOSS_3_COUNT", "LOSS_TECH_COUNT", "LOSS_COUNT_TOTAL",
]

#: Количество периодов сезонности
SEASONAL_PERIODS: int = 12

#: Резервная полуширина ДИ (±5 %) при σ = 0
_CI_FALLBACK_HALF_WIDTH: float = 0.05


# ============================================================
# ПРОИЗВОДНЫЕ ПОКАЗАТЕЛИ
# ============================================================

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет LOSS_COUNT_TOTAL, LOSS_SUM_TOTAL, SPECIFIC_LOSS из прогнозов."""
    df = df.copy()
    df["LOSS_COUNT_TOTAL"] = (
        df["LOSS_12_COUNT"] + df["LOSS_3_COUNT"] + df["LOSS_TECH_COUNT"]
    )
    df["LOSS_SUM_TOTAL"] = (
        df["LOSS_12_SUM"] + df["LOSS_3_SUM"] + df["LOSS_TECH_SUM"]
    )
    safe_km = df["TRAIN_KM"].replace(0, np.nan)
    df["SPECIFIC_LOSS"] = df["LOSS_SUM_TOTAL"] / safe_km * 1_000_000
    return df


# ============================================================
# ETS ОБУЧЕНИЕ С FALLBACK
# ============================================================

def _fit_ets_with_fallback(series: pd.Series):
    """
    Надёжное обучение ETS с fallback-стратегией:
        ETS(A,A,A)  →  ETS(A,A)  →  ETS(A)

    Переход к следующему варианту происходит при:
      - исключении во время обучения
      - флаге mle_retvals["converged"] == False (MLE не сошёлся)

    Параметры
    ----------
    series : pd.Series с DatetimeIndex, частота MS (начало месяца)

    Возвращает
    ----------
    (fit, model_type) — обученная модель и строка "AAA", "AA" или "A"
    """
    # ── Вариант 1: ETS(A,A,A) — тренд + сезонность ────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=SEASONAL_PERIODS,
                initialization_method="estimated",
            ).fit(optimized=True)
        if getattr(fit, "mle_retvals", {}).get("converged", True):
            return fit, "AAA"
    except Exception:
        pass

    # ── Вариант 2: ETS(A,A) — только тренд ────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
        if getattr(fit, "mle_retvals", {}).get("converged", True):
            return fit, "AA"
    except Exception:
        pass

    # ── Вариант 3: ETS(A) — только уровень ────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = ExponentialSmoothing(
            series,
            trend=None,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
    return fit, "A"


# ============================================================
# ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ МОДЕЛИ
# ============================================================

def _extract_ets_params(fit, model_type: str) -> dict:
    """
    Извлекает сглаживающие коэффициенты и начальные состояния из
    обученной модели ExponentialSmoothing.

    Возвращаемые ключи
    ------------------
    model_type  — "AAA", "AA" или "A"
    alpha       — коэффициент сглаживания уровня (α)
    beta        — коэффициент сглаживания тренда (β),  только AA/AAA
    gamma       — коэффициент сглаживания сезонности (γ), только AAA
    l0          — начальный уровень (l₀)
    b0          — начальный тренд  (b₀),  только AA/AAA
    s0..s11     — начальные сезонные компоненты,        только AAA
    aic         — информационный критерий Акаике
    """
    p = fit.params          # statsmodels dict с оптимизированными параметрами
    result: dict = {"model_type": model_type}

    result["alpha"] = float(p.get("smoothing_level", np.nan))

    if model_type in ("AA", "AAA"):
        result["beta"] = float(p.get("smoothing_trend",  np.nan))
        result["l0"]   = float(p.get("initial_level",    np.nan))
        result["b0"]   = float(p.get("initial_trend",    np.nan))

    if model_type == "AAA":
        result["gamma"] = float(p.get("smoothing_seasonal", np.nan))
        for i in range(SEASONAL_PERIODS):
            key = f"initial_seasons.{i}"
            if key in p:
                result[f"s{i}"] = float(p[key])

    if model_type == "A":
        result["l0"] = float(p.get("initial_level", np.nan))

    try:
        result["aic"] = float(fit.aic)
    except Exception:
        result["aic"] = None

    return result


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _road_series(monthly_df: pd.DataFrame, road: str, indicator: str) -> pd.Series:
    """
    Извлекает временной ряд одной дороги и показателя с DatetimeIndex (freq=MS).

    Используем pd.date_range с явным freq="MS" — это подавляет предупреждение
    statsmodels "No frequency information was provided, so inferred frequency MS
    will be used", которое возникает при передаче индекса без атрибута freq.
    """
    df = (
        monthly_df[monthly_df["ROAD_NAME"] == road]
        .sort_values(["YEAR", "MONTH"])
        .copy()
        .reset_index(drop=True)
    )
    first = df.iloc[0]
    idx = pd.date_range(
        start=f"{int(first['YEAR'])}-{int(first['MONTH']):02d}-01",
        periods=len(df),
        freq="MS",
    )
    return pd.Series(df[indicator].values, index=idx, name=indicator)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_ets_forecast(
    monthly_df: pd.DataFrame,
    forecast_years: List[int],
    outdir: str = "ets_results",
    test_year: Optional[int] = None,
    plot_html: bool = True,
    plots_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Прогнозирование всех целевых показателей для всех дорог методом ETS.

    Параметры
    ----------
    monthly_df    : DataFrame из generate_monthly_data
                    (ROAD_NAME, YEAR, MONTH, TRAIN_KM, LOSS_12_COUNT, …)
    forecast_years: список лет прогноза, напр. [2025, 2026]
    outdir        : директория для CSV, метрик, JSON
    test_year     : год оценки качества (по умолчанию max(YEAR))
    plot_html     : True — строить интерактивные HTML-графики
    plots_dir     : директория для HTML-файлов
                    (по умолчанию {outdir}/plots)

    Возвращает
    ----------
    pd.DataFrame в формате catboost_forecasts.csv:
        ROAD_NAME, YEAR, MONTH,
        TRAIN_KM,         TRAIN_KM_lower_95,         TRAIN_KM_upper_95,
        LOSS_12_COUNT,    LOSS_12_COUNT_lower_95,    LOSS_12_COUNT_upper_95,
        …
        SPECIFIC_LOSS,    SPECIFIC_LOSS_lower_95,    SPECIFIC_LOSS_upper_95

    Файлы на диске
    --------------
    {outdir}/ets_forecasts.csv              — прогнозы + CI 95 %
    {outdir}/ets_metrics.csv                — MAPE_test_%, SMAPE_test_%, AIC, …
    {outdir}/ets_params/{indicator}.json    — α, β, γ, l₀, b₀, s₀..s₁₁
    {outdir}/plots/{indicator}/{road}.html  — HTML-графики
    """
    base_out    = Path(outdir)
    plots_path  = Path(plots_dir) if plots_dir else base_out / "plots"
    params_path = base_out / "ets_params"
    base_out.mkdir(parents=True, exist_ok=True)
    params_path.mkdir(parents=True, exist_ok=True)

    if test_year is None:
        test_year = int(monthly_df["YEAR"].max())

    roads   = sorted(monthly_df["ROAD_NAME"].unique())
    horizon = len(forecast_years) * SEASONAL_PERIODS   # всего месяцев прогноза

    # ── Структуры накопления ───────────────────────────────────
    # forecast_store[indicator][(road, year, month)] = value
    forecast_store: Dict[str, dict] = {ind: {} for ind in TARGET_INDICATORS}
    ci_lo_store:    Dict[str, dict] = {ind: {} for ind in TARGET_INDICATORS}
    ci_hi_store:    Dict[str, dict] = {ind: {} for ind in TARGET_INDICATORS}
    metrics_rows:   List[dict]      = []
    # Хранилище тестовых прогнозов для вычисления метрик производных показателей
    _test_pred_store: Dict[str, Dict[str, np.ndarray]] = {}
    # params_store[indicator][road] = {alpha: ..., beta: ..., ...}
    params_store: Dict[str, dict] = {ind: {} for ind in TARGET_INDICATORS}

    # ── Цикл по показателям ────────────────────────────────────
    for indicator in TARGET_INDICATORS:
        print(f"  [ETS] {indicator}...")

        for road in roads:
            try:
                series = _road_series(monthly_df, road, indicator)

                # ── Разбиение train / test ─────────────────────
                y_train = series[series.index.year <  test_year]
                y_test  = series[series.index.year == test_year]

                if len(y_train) < SEASONAL_PERIODS + 2:
                    print(f"    [{road}] слишком мало точек обучения — пропуск")
                    continue

                # ── Обучение на train → метрики ────────────────
                fit_train, model_type = _fit_ets_with_fallback(y_train)

                if len(y_test) > 0:
                    pred_test = fit_train.forecast(len(y_test))
                    residuals = y_test.values - pred_test.values
                    sigma     = float(np.std(residuals, ddof=0))

                    # Сохраняем прогноз для последующего вычисления производных метрик
                    if indicator not in _test_pred_store:
                        _test_pred_store[indicator] = {}
                    _test_pred_store[indicator][road] = pred_test.values

                    metrics_rows.append({
                        "indicator":    indicator,
                        "road":         road,
                        "model_type":   model_type,
                        "MAPE_test_%":  mape_percent(y_test.values, pred_test.values),
                        "SMAPE_test_%": smape_percent(y_test.values, pred_test.values),
                        "AIC":          float(fit_train.aic)
                                        if hasattr(fit_train, "aic") else np.nan,
                        "resid_std":    sigma,
                        "test_year":    test_year,
                    })
                else:
                    sigma = 0.0

                # ── Обучение на полном ряду → прогноз ─────────
                fit_full, model_type_full = _fit_ets_with_fallback(series)
                pred_future = fit_full.forecast(horizon)          # horizon шагов

                # ── Доверительный интервал ─────────────────────
                # ± 1.96 σ, σ из тестового периода.
                # Если σ = 0 (ряд константный) — резервный ± 5 %.
                if sigma > 0:
                    lower = np.maximum(pred_future.values - 1.96 * sigma, 0.0)
                    upper =            pred_future.values + 1.96 * sigma
                else:
                    lower = np.maximum(pred_future.values * (1.0 - _CI_FALLBACK_HALF_WIDTH), 0.0)
                    upper =            pred_future.values * (1.0 + _CI_FALLBACK_HALF_WIDTH)

                # ── Сохранение параметров модели ───────────────
                params_store[indicator][road] = _extract_ets_params(fit_full, model_type_full)

                # ── Запись прогнозов в store ───────────────────
                for i, year in enumerate(sorted(forecast_years)):
                    for month in range(1, 13):
                        step = i * SEASONAL_PERIODS + (month - 1)
                        key  = (road, year, month)
                        forecast_store[indicator][key] = float(pred_future.values[step])
                        ci_lo_store[indicator][key]    = float(lower[step])
                        ci_hi_store[indicator][key]    = float(upper[step])

            except Exception as exc:
                print(f"    [{road}] ошибка при {indicator}: {exc}")
                continue

        # ── JSON параметров для данного показателя ─────────────
        params_json_path = params_path / f"{indicator}.json"
        with open(params_json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {road: params_store[indicator][road]
                 for road in sorted(params_store[indicator])},
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  [ETS] {indicator} готово → {params_json_path.name}")

    # ── Метрики производных показателей ──────────────────────────────────────────
    # LOSS_COUNT_TOTAL, LOSS_SUM_TOTAL, SPECIFIC_LOSS — из тестовых прогнозов ETS.
    # Для каждой дороги собираем прогнозы всех 7 прямых показателей и вычисляем
    # производные; SMAPE считается относительно фактических данных monthly_df.
    _derived_roads: set = set(roads)
    for _ind in TARGET_INDICATORS:
        _derived_roads &= set(_test_pred_store.get(_ind, {}).keys())

    if _derived_roads:
        _test_actual_base = (
            monthly_df[monthly_df["YEAR"] == test_year]
            .sort_values(["ROAD_NAME", "MONTH"])
        )
        for road in sorted(_derived_roads):
            _actual_road = _test_actual_base[_test_actual_base["ROAD_NAME"] == road]
            _n_actual    = len(_actual_road)
            _n_pred      = min(
                len(_test_pred_store[_ind][road]) for _ind in TARGET_INDICATORS
            )
            _n = min(_n_pred, _n_actual)
            if _n == 0:
                continue

            _pred_row   = {_ind: _test_pred_store[_ind][road][:_n]
                           for _ind in TARGET_INDICATORS}
            _actual_row = {_ind: _actual_road[_ind].values[:_n]
                           for _ind in TARGET_INDICATORS}

            _pred_df_rd   = pd.DataFrame(_pred_row)
            _actual_df_rd = pd.DataFrame(_actual_row)
            _pred_derived   = compute_derived(_pred_df_rd)
            _actual_derived = compute_derived(_actual_df_rd)

            for _d_ind in DERIVED_INDICATORS:
                metrics_rows.append({
                    "indicator":    _d_ind,
                    "road":         road,
                    "model_type":   "derived",
                    "MAPE_test_%":  mape_percent(
                        _actual_derived[_d_ind].values,
                        _pred_derived[_d_ind].values),
                    "SMAPE_test_%": smape_percent(
                        _actual_derived[_d_ind].values,
                        _pred_derived[_d_ind].values),
                    "AIC":          np.nan,
                    "resid_std":    np.nan,
                    "test_year":    test_year,
                })

    # ── Сборка итогового DataFrame ─────────────────────────────
    idx_tuples = [
        (road, year, month)
        for road in sorted(roads)
        for year in sorted(forecast_years)
        for month in range(1, 13)
    ]
    idx = pd.MultiIndex.from_tuples(idx_tuples, names=["ROAD_NAME", "YEAR", "MONTH"])
    out = pd.DataFrame(index=idx)

    for ind in TARGET_INDICATORS:
        out[ind]               = pd.Series(forecast_store[ind])
        out[f"{ind}_lower_95"] = pd.Series(ci_lo_store[ind])
        out[f"{ind}_upper_95"] = pd.Series(ci_hi_store[ind])

    out = out.reset_index()

    # Неотрицательность прямых показателей
    for col in TARGET_INDICATORS:
        out[col] = out[col].clip(lower=0)

    # Округление счётчиков
    for col in [c for c in COUNT_COLS if c in out.columns]:
        out[col] = out[col].round().astype(int)

    # Производные показатели
    out = compute_derived(out)

    # CI для производных (± 5 % — аналитический σ не применим к сумме)
    for col in DERIVED_INDICATORS:
        if col in out.columns:
            out[f"{col}_lower_95"] = (
                out[col] * (1.0 - _CI_FALLBACK_HALF_WIDTH)
            ).clip(lower=0)
            out[f"{col}_upper_95"] = out[col] * (1.0 + _CI_FALLBACK_HALF_WIDTH)

    # CI счётчиков — целые числа (lower/upper не могут быть дробными)
    for col in COUNT_COLS:
        for suffix in ("_lower_95", "_upper_95"):
            ci_col = f"{col}{suffix}"
            if ci_col in out.columns:
                out[ci_col] = out[ci_col].round().astype(int)

    # Порядок столбцов: значение + CI-пара рядом, как в catboost_forecasts.csv
    id_cols = ["ROAD_NAME", "YEAR", "MONTH"]
    value_ci_cols: List[str] = []
    for col in ALL_PLOT_INDICATORS:
        if col in out.columns:
            value_ci_cols += [col, f"{col}_lower_95", f"{col}_upper_95"]

    out = (
        out[id_cols + value_ci_cols]
        .sort_values(["ROAD_NAME", "YEAR", "MONTH"])
        .reset_index(drop=True)
    )

    # ── Сохранение CSV ─────────────────────────────────────────
    forecast_path = base_out / "ets_forecasts.csv"
    metrics_path  = base_out / "ets_metrics.csv"

    out.to_csv(forecast_path, index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] ETS прогнозы сохранены:  {forecast_path}")
    print(f"[OK] ETS метрики сохранены:    {metrics_path}")
    print(f"[OK] ETS параметры сохранены:  {params_path}/")
    print(f"     Строк в прогнозе: {len(out)}")

    # ── HTML-графики ───────────────────────────────────────────
    if plot_html:
        from plotting import plot_predictions   # локальный импорт — избегаем цикличности

        print(f"\n[ETS] Построение HTML-графиков → {plots_path}/")
        errors = 0

        for indicator in ALL_PLOT_INDICATORS:
            for road in roads:
                road_slug = road.replace(" ", "_").replace("-", "_")
                outpath   = plots_path / indicator / f"{road_slug}.html"
                try:
                    plot_predictions(
                        monthly_df=monthly_df,
                        forecast_df=out,
                        indicator=indicator,
                        road=road,
                        outpath=outpath,
                        model_name="ETS",
                        line_color="#E67E22",
                        fill_color="rgba(230, 126, 34, 0.13)",
                    )
                except Exception as exc:
                    errors += 1
                    print(f"    [{indicator}/{road}] график не построен: {exc}")

        n_ok = len(ALL_PLOT_INDICATORS) * len(roads) - errors
        print(
            f"[OK] Графики: {len(ALL_PLOT_INDICATORS)} показателей × "
            f"{len(roads)} дорог = {n_ok} файлов"
            + (f" ({errors} ошибок)" if errors else "")
        )

    return out
