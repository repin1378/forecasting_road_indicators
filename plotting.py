"""
Визуализация прогнозов:
- ETS (классические методы)
- ML (машинное обучение, пример: CatBoost)

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


# ======================================================
# ЧЕЛОВЕКОЧИТАЕМЫЕ НАЗВАНИЯ ПОКАЗАТЕЛЕЙ
# ======================================================

INDICATOR_TITLES = {
    "train_km": "Объем поездо-километровой работы",
    "loss_12": (
        "Потери поездо-часов по причине отказов "
        "в работе технических средств 1 и 2 категорий"
    ),
    "loss_3": (
        "Потери поездо-часов по причине отказов "
        "в работе технических средств 3 категории"
    ),
    "loss_tech": (
        "Потери поездо-часов по причине технологических нарушений"
    ),
    "loss_total": (
        "Общие потери поездо-часов по причине отказов "
        "в работе технических средств и технологических нарушений"
    ),
    "specific": (
        "Удельный показатель общих потерь поездо-часов "
        "от отказов технических средств и технологических нарушений "
        "на 1 млн. поездо-км"
    ),
}


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
