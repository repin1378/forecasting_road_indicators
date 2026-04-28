"""
Визуализация прогнозов:
- ETS (классические методы)
- ML (машинное обучение, пример: CatBoost)
- plot_predictions — интерактивный HTML-график на Plotly

Особенности:
- без разрыва между историей и прогнозом
- месячная сетка
- человекочитаемые заголовки
- стрелки осей + временная шкала t
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go

from catboost_model import CI_HALF_WIDTH


# ======================================================
# ЧЕЛОВЕКОЧИТАЕМЫЕ НАЗВАНИЯ ПОКАЗАТЕЛЕЙ
# ======================================================

INDICATOR_TITLES = {
    # Новые (плоский формат)
    "TRAIN_KM": "Объём поездо-километровой работы",
    "LOSS_12_COUNT": "Количество отказов технических средств 1–2 категорий",
    "LOSS_12_SUM": "Потери поездо-часов из-за отказов 1–2 категорий",
    "LOSS_3_COUNT": "Количество отказов технических средств 3 категории",
    "LOSS_3_SUM": "Потери поездо-часов из-за отказов 3 категории",
    "LOSS_TECH_COUNT": "Количество технологических нарушений",
    "LOSS_TECH_SUM": "Потери поездо-часов из-за технологических нарушений",
    "LOSS_COUNT_TOTAL": "Общее количество отказов и нарушений",
    "LOSS_SUM_TOTAL": "Общие потери поездо-часов",
    "SPECIFIC_LOSS": (
        "Удельный показатель потерь поездо-часов "
        "на 1 млн. поездо-км"
    ),
    # Старые (для обратной совместимости)
    "train_km": "Объём поездо-километровой работы",
    "loss_12": "Потери поездо-часов (отказы 1–2 категорий)",
    "loss_3": "Потери поездо-часов (отказы 3 категории)",
    "loss_tech": "Потери поездо-часов (технологические нарушения)",
    "loss_total": "Общие потери поездо-часов",
    "specific": "Удельный показатель потерь на 1 млн. поездо-км",
}


# ======================================================
# ИНТЕРАКТИВНЫЙ PLOTLY-ГРАФИК (HTML)
# ======================================================

# Русские сокращения месяцев для подписей оси X
_RU_MONTHS = {
    1: "янв", 2: "фев", 3: "мар", 4: "апр",
    5: "май", 6: "июн", 7: "июл", 8: "авг",
    9: "сен", 10: "окт", 11: "ноя", 12: "дек",
}


def _make_date(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-{month:02d}-01")


def _quarterly_ticks(
    date_min: pd.Timestamp, date_max: pd.Timestamp
) -> tuple[list, list]:
    """Квартальные метки оси X с русскими названиями месяцев."""
    dates = pd.date_range(date_min, date_max, freq="QS")
    vals  = [str(d.date()) for d in dates]
    texts = [f"{_RU_MONTHS[d.month]} {d.year}" for d in dates]
    return vals, texts


def plot_predictions(
    monthly_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    indicator: str,
    road: str,
    outpath: "Path | str",
    *,
    model_name: str = "Градиентного Бустинга",
    line_color: str = "#1E64C8",
    fill_color: str = "rgba(30, 100, 200, 0.13)",
) -> None:
    """
    Строит детальный интерактивный HTML-график прогноза.

    Доверительный интервал 95 %:
      • Если в forecast_df есть столбцы {indicator}_lower_95 / {indicator}_upper_95,
        они используются напрямую (ETS — реальный σ из остатков теста).
      • Иначе CI рассчитывается как ± CI_HALF_WIDTH (5 %) от прогноза (CatBoost).

    На графике:
      • Тёмная линия + маркеры   — фактические данные
      • Цветная пунктир + маркеры — прогноз модели
      • Цветная заливка           — доверительный интервал 95 %
      • Красная вертикаль         — граница факт / прогноз
      • Квартальные метки         — русские названия месяцев (янв, апр, …)
      • Вспомогательная сетка     — каждый месяц

    Параметры
    ----------
    monthly_df  : исторические данные (ROAD_NAME, YEAR, MONTH, <indicator>)
    forecast_df : прогноз (ROAD_NAME, YEAR, MONTH, <indicator>,
                           [<indicator>_lower_95, <indicator>_upper_95])
    indicator   : название столбца, напр. "TRAIN_KM"
    road        : название дороги,  напр. "Октябрьская"
    outpath     : путь для сохранения .html
    model_name  : подпись модели в легенде (по умолчанию "CatBoost")
    line_color  : цвет линии прогноза (hex или rgba)
    fill_color  : цвет заливки доверительного интервала (rgba)
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # ── Исторические данные ────────────────────────────────
    hist = (
        monthly_df[monthly_df["ROAD_NAME"] == road]
        .sort_values(["YEAR", "MONTH"])
        .copy()
        .reset_index(drop=True)
    )
    hist["date"] = [_make_date(r.YEAR, r.MONTH) for _, r in hist.iterrows()]

    # ── Прогнозные данные ──────────────────────────────────
    fcast = (
        forecast_df[forecast_df["ROAD_NAME"] == road]
        .sort_values(["YEAR", "MONTH"])
        .copy()
        .reset_index(drop=True)
    )
    fcast["date"] = [_make_date(r.YEAR, r.MONTH) for _, r in fcast.iterrows()]

    last_date  = hist["date"].iloc[-1]
    last_value = float(hist[indicator].iloc[-1])

    # Прогноз присоединяется к последней точке истории (нет разрыва)
    fcast_dates  = pd.concat(
        [pd.Series([last_date]), fcast["date"]], ignore_index=True
    )
    fcast_values = pd.concat(
        [pd.Series([last_value]), fcast[indicator].clip(lower=0)],
        ignore_index=True,
    )

    # ── Доверительный интервал ─────────────────────────────
    # Приоритет: готовые столбцы lower_95/upper_95 из forecast_df
    # (ETS передаёт реальный σ; CatBoost — столбцы тоже есть, но CI_HALF_WIDTH)
    lower_col = f"{indicator}_lower_95"
    upper_col = f"{indicator}_upper_95"

    if (lower_col in fcast.columns
            and not fcast[lower_col].isna().all()):
        fcast_lower = pd.concat(
            [pd.Series([last_value]), fcast[lower_col].clip(lower=0)],
            ignore_index=True,
        )
        fcast_upper = pd.concat(
            [pd.Series([last_value]), fcast[upper_col].clip(lower=0)],
            ignore_index=True,
        )
        ci_label = "Доверительный интервал 95 %"
    else:
        fcast_lower = fcast_values * (1.0 - CI_HALF_WIDTH)
        fcast_upper = fcast_values * (1.0 + CI_HALF_WIDTH)
        # Первая точка (стык с историей) — нулевая ширина интервала
        fcast_lower.iloc[0] = last_value
        fcast_upper.iloc[0] = last_value
        ci_label = f"Доверительный интервал 95 % (±{int(CI_HALF_WIDTH * 100)} %)"

    # ── Диапазон Y-оси (не форсируем 0) ───────────────────
    all_y = list(hist[indicator]) + list(fcast_lower) + list(fcast_upper)
    y_min = max(0.0, min(all_y) * 0.92)
    y_max = max(all_y) * 1.08

    # ── Квартальные метки оси X ────────────────────────────
    tick_vals, tick_texts = _quarterly_ticks(
        hist["date"].iloc[0], fcast["date"].iloc[-1]
    )

    # ── Построение графика ─────────────────────────────────
    fig = go.Figure()

    # 1. Доверительный интервал 95 % (заливка — добавляется первой,
    #    чтобы оказаться под линиями)
    x_band = list(fcast_dates) + list(reversed(fcast_dates.tolist()))
    y_band = list(fcast_upper) + list(reversed(fcast_lower.tolist()))
    fig.add_trace(go.Scatter(
        x=x_band,
        y=y_band,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=ci_label,
        legendrank=3,
    ))

    # 2. Исторические данные — линия + маркеры
    fig.add_trace(go.Scatter(
        x=hist["date"],
        y=hist[indicator],
        mode="lines+markers",
        name="Факт",
        line=dict(color="#2b2b2b", width=1.6),
        marker=dict(size=4, color="#2b2b2b", symbol="circle"),
        hovertemplate="<b>Факт</b>  %{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
        legendrank=1,
    ))

    # 3. Прогноз — пунктир + маркеры
    fig.add_trace(go.Scatter(
        x=fcast_dates,
        y=fcast_values,
        mode="lines+markers",
        name=f"Прогноз {model_name}",
        line=dict(color=line_color, width=2.0, dash="dash"),
        marker=dict(size=5, color=line_color, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="<b>Прогноз</b>  %{x|%b %Y}<br>%{y:,.0f}<extra></extra>",
        legendrank=2,
    ))

    # 4. Вертикальная граница история / прогноз
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line=dict(color="crimson", width=1.5, dash="dot"),
        annotation_text="Начало прогноза",
        annotation_position="top right",
        annotation_font=dict(color="crimson", size=11),
    )

    # ── Оформление ─────────────────────────────────────────
    title_ind = INDICATOR_TITLES.get(indicator, indicator)

    fig.update_layout(
        title=dict(
            text=f"<b>{title_ind}</b><br><sup>{road}</sup>",
            font=dict(size=15, family="Arial"),
            x=0.02,
            xanchor="left",
        ),
        xaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickangle=-45,
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor="#dedede",
            gridwidth=1,
            minor=dict(
                dtick="M1",
                showgrid=True,
                gridcolor="#f0f0f0",
                gridwidth=1,
            ),
            showline=True,
            linecolor="#aaaaaa",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=title_ind, font=dict(size=11)),
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="#dedede",
            gridwidth=1,
            tickformat=",.0f",
            showline=True,
            linecolor="#aaaaaa",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=12),
            traceorder="normal",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            bordercolor="#cccccc",
        ),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=100, b=110),
        width=1300,
        height=580,
    )

    fig.write_html(str(outpath), include_plotlyjs="cdn")


# ======================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================

def _concat_last_point(data: pd.Series, future_idx, forecast):
    """Соединяет последнюю точку истории с прогнозом (без разрыва)."""
    x = [data.index[-1]] + list(future_idx)
    y = [data.iloc[-1]] + list(forecast)
    return x, y


def _decorate_axes(ax):
    """Стрелки осей и обозначение времени t."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(
        "",
        xy=(1.02, 0),
        xytext=(0, 0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
    )

    ax.annotate(
        "",
        xy=(0, 1.02),
        xytext=(0, 0),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
    )

    ax.text(
        1.02,
        -0.08,
        "t",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top",
    )


def _make_title(indicator: str, road: str) -> str:
    """Формирует человекочитаемый заголовок."""
    name = INDICATOR_TITLES.get(indicator, indicator)
    return f"{name} — {road}"


# ======================================================
# ETS
# ======================================================

def plot_ets_forecast(
    data: pd.Series,
    future_idx: pd.DatetimeIndex,
    forecast: pd.Series,
    lower: pd.Series | None,
    upper: pd.Series | None,
    road: str,
    indicator: str,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        data.index,
        data.values,
        label="Исторические данные",
        color="black",
        linewidth=1.6,
    )

    forecast_plot = forecast.clip(lower=0)
    lower_plot = lower.clip(lower=0) if lower is not None else None
    upper_plot = upper.clip(lower=0) if upper is not None else None

    fx, fy = _concat_last_point(data, future_idx, forecast_plot)
    ax.plot(fx, fy, label="Прогноз ETS", color="tab:blue", linewidth=2.2)

    if lower_plot is not None and upper_plot is not None:
        ix = [data.index[-1]] + list(future_idx)
        ax.fill_between(
            ix,
            [data.iloc[-1]] + list(lower_plot),
            [data.iloc[-1]] + list(upper_plot),
            alpha=0.25,
            color="tab:blue",
            label="Доверительный интервал (ETS)",
        )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    ax.set_xlim(data.index.min(), future_idx.max())
    ax.set_title(_make_title(indicator, road))
    ax.legend()

    _decorate_axes(ax)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ======================================================
# ML (CatBoost как пример)
# ======================================================

def plot_catboost_forecast(
    data: pd.Series,
    future_idx: pd.DatetimeIndex,
    forecast: pd.Series,
    lower: pd.Series | None,
    upper: pd.Series | None,
    road: str,
    indicator: str,
    outpath: Path,
):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        data.index,
        data.values,
        label="Исторические данные",
        color="black",
        linewidth=1.6,
    )

    forecast_plot = forecast.clip(lower=0)
    lower_plot = lower.clip(lower=0) if lower is not None else None
    upper_plot = upper.clip(lower=0) if upper is not None else None

    fx, fy = _concat_last_point(data, future_idx, forecast_plot)
    ax.plot(fx, fy, label="Прогноз ML", color="tab:orange", linewidth=2.2)

    if lower_plot is not None and upper_plot is not None:
        ix = [data.index[-1]] + list(future_idx)
        ax.fill_between(
            ix,
            [data.iloc[-1]] + list(lower_plot),
            [data.iloc[-1]] + list(upper_plot),
            alpha=0.25,
            color="tab:orange",
            label="Доверительный интервал (ML)",
        )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)

    ax.set_xlim(data.index.min(), future_idx.max())
    ax.set_title(_make_title(indicator, road))
    ax.legend()

    _decorate_axes(ax)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
