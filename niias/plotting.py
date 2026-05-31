"""Plotly HTML charts for NIIAS forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .catboost_model import CI_HALF_WIDTH
except ImportError:  # pragma: no cover
    from catboost_model import CI_HALF_WIDTH


INDICATOR_DESCRIPTIONS_PATH = Path(__file__).with_name("indicator_descriptions.json")


def load_indicator_descriptions(path: str | Path | None = None) -> dict[str, str]:
    """Load human-readable indicator descriptions for chart titles."""
    desc_path = Path(path) if path is not None else INDICATOR_DESCRIPTIONS_PATH
    if not desc_path.exists():
        return {}
    with open(desc_path, encoding="utf-8") as f:
        return json.load(f)


def get_indicator_title(indicator: str, descriptions_path: str | Path | None = None) -> str:
    descriptions = load_indicator_descriptions(descriptions_path)
    return descriptions.get(indicator, indicator)


def _month_start(value) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value.to_period("M").to_timestamp()
    text = str(value).strip()
    if "." in text and "-" not in text:
        left, right = text.split(".", 1)
        if len(left) <= 2 and len(right) == 4:
            return pd.Timestamp(year=int(right), month=int(left), day=1)
    return pd.to_datetime(text).to_period("M").to_timestamp()


def plot_predictions(
    monthly_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    indicator: str,
    outpath: str | Path,
    *,
    model_name: str = "CatBoost",
    line_color: str = "#1E64C8",
    fill_color: str = "rgba(30, 100, 200, 0.13)",
    descriptions_path: str | Path | None = None,
    forecast_start=None,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required to build HTML charts. Install it in the Python environment: "
            "pip install plotly"
        ) from exc

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    indicator_title = get_indicator_title(indicator, descriptions_path)

    hist = monthly_df.sort_values("DATE").copy()
    fcast = forecast_df.sort_values("DATE").copy()
    hist["DATE"] = pd.to_datetime(hist["DATE"])
    fcast["DATE"] = pd.to_datetime(fcast["DATE"])

    boundary_date = _month_start(forecast_start) if forecast_start is not None else fcast["DATE"].min()
    if fcast.empty:
        raise ValueError("forecast_df is empty")

    fcast = fcast[fcast["DATE"] >= boundary_date].copy()
    if fcast.empty:
        raise ValueError(f"No forecast rows on or after forecast_start={boundary_date.date()}")

    hist_before_forecast = hist[hist["DATE"] < boundary_date]
    if hist_before_forecast.empty:
        anchor_date = fcast["DATE"].iloc[0]
        anchor_value = float(fcast[indicator].iloc[0])
        forecast_line_dates = fcast["DATE"]
        forecast_line_values = fcast[indicator].clip(lower=0)
    else:
        anchor_date = hist_before_forecast["DATE"].iloc[-1]
        anchor_value = float(hist_before_forecast[indicator].iloc[-1])
        forecast_line_dates = pd.concat([pd.Series([anchor_date]), fcast["DATE"]], ignore_index=True)
        forecast_line_values = pd.concat(
            [pd.Series([anchor_value]), fcast[indicator].clip(lower=0)],
            ignore_index=True,
        )

    lower_col = f"{indicator}_lower_95"
    upper_col = f"{indicator}_upper_95"
    if lower_col in fcast.columns and upper_col in fcast.columns:
        fcast_lower = fcast[lower_col].clip(lower=0).reset_index(drop=True)
        fcast_upper = fcast[upper_col].clip(lower=0).reset_index(drop=True)
        ci_label = "Диапазон прогноза (нижняя-верхняя граница)"
    else:
        fcast_values = fcast[indicator].clip(lower=0).reset_index(drop=True)
        fcast_lower = fcast_values * (1.0 - CI_HALF_WIDTH)
        fcast_upper = fcast_values * (1.0 + CI_HALF_WIDTH)
        ci_label = f"95% confidence interval (+/-{int(CI_HALF_WIDTH * 100)}%)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fcast["DATE"],
        y=fcast_lower,
        mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        hoverinfo="skip",
        showlegend=False,
        name="Нижняя граница",
    ))
    fig.add_trace(go.Scatter(
        x=fcast["DATE"],
        y=fcast_upper,
        mode="lines",
        fill="tonexty",
        fillcolor=fill_color,
        line=dict(color="rgba(0,0,0,0)", width=0),
        hoverinfo="skip",
        name=ci_label,
    ))
    fig.add_trace(go.Scatter(
        x=fcast["DATE"],
        y=fcast_lower,
        mode="lines",
        name="Нижняя граница прогноза",
        line=dict(color=line_color, width=1, dash="dot"),
        opacity=0.45,
    ))
    fig.add_trace(go.Scatter(
        x=fcast["DATE"],
        y=fcast_upper,
        mode="lines",
        name="Верхняя граница прогноза",
        line=dict(color=line_color, width=1, dash="dot"),
        opacity=0.45,
    ))
    fig.add_trace(go.Scatter(
        x=hist["DATE"],
        y=hist[indicator],
        mode="lines+markers",
        name="Факт",
        line=dict(color="#2b2b2b", width=1.7),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_line_dates,
        y=forecast_line_values,
        mode="lines+markers",
        name=f"Прогноз {model_name}",
        line=dict(color=line_color, width=2.0, dash="dash"),
        marker=dict(size=5, color=line_color, symbol="circle-open"),
    ))
    boundary_x = pd.to_datetime(boundary_date).to_pydatetime()
    fig.add_shape(
        type="line",
        x0=boundary_x,
        x1=boundary_x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="crimson", width=1.4, dash="dot"),
    )
    fig.add_annotation(
        x=boundary_x,
        y=1,
        xref="x",
        yref="paper",
        text="Начало прогноза",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="crimson", size=11),
    )

    fig.update_layout(
        title=dict(text=f"<b>{indicator_title}</b>", x=0.02, xanchor="left"),
        template="plotly_white",
        hovermode="x unified",
        width=1100,
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=90, b=60),
    )
    fig.update_xaxes(showgrid=True, dtick="M3", tickformat="%b %Y")
    fig.update_yaxes(title_text=indicator_title, showgrid=True, rangemode="tozero")
    fig.write_html(outpath, include_plotlyjs="cdn")
