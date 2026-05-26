"""SMAPE comparison for NIIAS CatBoost vs seasonal naive baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .catboost_model import TARGET_TITLES
except ImportError:  # pragma: no cover
    from catboost_model import TARGET_TITLES


def _read_metrics(path: str | Path, value_name: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="metrics")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    smape_col = next((c for c in df.columns if "smape" in c.lower()), None)
    if smape_col is None:
        raise KeyError(f"SMAPE column not found in {path}. Columns: {df.columns.tolist()}")
    return df.rename(columns={smape_col: value_name})[["indicator", value_name]]


def build_smape_comparison(
    baseline_metrics_path: str | Path = "niias/catboost_results/seasonal_naive_metrics.xlsx",
    catboost_metrics_path: str | Path = "niias/catboost_results/catboost_metrics.xlsx",
    detailed_out: str | Path = "niias/compare_results/smape_comparison_detailed.xlsx",
    summary_out: str | Path = "niias/compare_results/smape_comparison_by_indicator.xlsx",
) -> pd.DataFrame:
    baseline = _read_metrics(baseline_metrics_path, "SMAPE_SeasonalNaive")
    catboost = _read_metrics(catboost_metrics_path, "SMAPE_CatBoost")
    out = baseline.merge(catboost, on="indicator", how="inner")
    out["indicator_name"] = out["indicator"].map(TARGET_TITLES).fillna(out["indicator"])
    out["Improvement_%"] = (
        (out["SMAPE_SeasonalNaive"] - out["SMAPE_CatBoost"])
        / out["SMAPE_SeasonalNaive"].replace(0, pd.NA)
        * 100.0
    )
    out = out[["indicator", "indicator_name", "SMAPE_SeasonalNaive", "SMAPE_CatBoost", "Improvement_%"]]

    detailed_out = Path(detailed_out)
    summary_out = Path(summary_out)
    detailed_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(detailed_out, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="comparison", index=False)
    with pd.ExcelWriter(summary_out, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="comparison", index=False)
    print(f"SMAPE comparison saved: {detailed_out}")
    return out
