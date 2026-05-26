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

    last_date = hist["DATE"].iloc[-1]
    last_value = float(hist[indicator].iloc[-1])
    fcast_dates = pd.concat([pd.Series([last_date]), fcast["DATE"]], ignore_index=True)
    fcast_values = pd.concat([pd.Series([last_value]), fcast[indicator].clip(lower=0)], ignore_index=True)

    lower_col = f"{indicator}_lower_95"
    upper_col = f"{indicator}_upper_95"
    if lower_col in fcast.columns and upper_col in fcast.columns:
        fcast_lower = pd.concat([pd.Series([last_value]), fcast[lower_col].clip(lower=0)], ignore_index=True)
        fcast_upper = pd.concat([pd.Series([last_value]), fcast[upper_col].clip(lower=0)], ignore_index=True)
        ci_label = "95% confidence interval"
    else:
        fcast_lower = fcast_values * (1.0 - CI_HALF_WIDTH)
        fcast_upper = fcast_values * (1.0 + CI_HALF_WIDTH)
        fcast_lower.iloc[0] = last_value
        fcast_upper.iloc[0] = last_value
        ci_label = f"95% confidence interval (+/-{int(CI_HALF_WIDTH * 100)}%)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fcast_dates) + list(reversed(fcast_dates.tolist())),
        y=list(fcast_upper) + list(reversed(fcast_lower.tolist())),
        fill="toself",
        fillcolor=fill_color,
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=ci_label,
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
        x=fcast_dates,
        y=fcast_values,
        mode="lines+markers",
        name=f"Прогноз {model_name}",
        line=dict(color=line_color, width=2.0, dash="dash"),
        marker=dict(size=5, color=line_color, symbol="circle-open"),
    ))
    fig.add_vline(
        x=last_date,
        line=dict(color="crimson", width=1.4, dash="dot"),
        annotation_text="Начало прогноза",
        annotation_position="top right",
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
