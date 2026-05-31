"""SMAPE comparison for NIIAS CatBoost vs an external forecast model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INDICATOR_DESCRIPTIONS_PATH = Path(__file__).with_name("indicator_descriptions.json")


def _load_indicator_descriptions() -> dict[str, str]:
    if not INDICATOR_DESCRIPTIONS_PATH.exists():
        return {}
    with open(INDICATOR_DESCRIPTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _read_metrics(path: str | Path, model_name: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="metrics")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    metric_cols = ["MAPE_%", "SMAPE_%", "MAE", "RMSE"]
    missing = [c for c in ["indicator", *metric_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing metric columns in {path}: {missing}. Columns: {df.columns.tolist()}")

    out = df[["indicator", *metric_cols]].copy()
    return out.rename(columns={c: f"{c}_{model_name}" for c in metric_cols})


def _build_summary(detailed: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    rows = []
    for metric in ["MAPE_%", "SMAPE_%", "MAE", "RMSE"]:
        baseline_col = f"{metric}_{baseline_name}"
        catboost_col = f"{metric}_CatBoost"
        diff_col = f"{metric}_diff_CatBoost_minus_{baseline_name}"
        rows.append({
            "metric": metric,
            f"{baseline_name}_mean": detailed[baseline_col].mean(),
            "CatBoost_mean": detailed[catboost_col].mean(),
            f"diff_CatBoost_minus_{baseline_name}": detailed[diff_col].mean(),
            "CatBoost_better_count": int((detailed[catboost_col] < detailed[baseline_col]).sum()),
            f"{baseline_name}_better_count": int((detailed[baseline_col] < detailed[catboost_col]).sum()),
            "equal_count": int((detailed[baseline_col] == detailed[catboost_col]).sum()),
            "indicators_n": int(len(detailed)),
        })
    return pd.DataFrame(rows)


def build_smape_comparison(
    baseline_metrics_path: str | Path = "niias/catboost_results/niias_model_metrics.xlsx",
    catboost_metrics_path: str | Path = "niias/catboost_results/catboost_metrics.xlsx",
    detailed_out: str | Path = "niias/compare_results/smape_comparison_detailed.xlsx",
    summary_out: str | Path = "niias/compare_results/smape_comparison_by_indicator.xlsx",
    baseline_name: str = "NIIAS",
) -> pd.DataFrame:
    baseline = _read_metrics(baseline_metrics_path, baseline_name)
    catboost = _read_metrics(catboost_metrics_path, "CatBoost")
    out = baseline.merge(catboost, on="indicator", how="inner")
    indicator_descriptions = _load_indicator_descriptions()
    out["indicator_name"] = out["indicator"].map(indicator_descriptions).fillna(out["indicator"])

    for metric in ["MAPE_%", "SMAPE_%", "MAE", "RMSE"]:
        baseline_col = f"{metric}_{baseline_name}"
        catboost_col = f"{metric}_CatBoost"
        out[f"{metric}_diff_CatBoost_minus_{baseline_name}"] = out[catboost_col] - out[baseline_col]

    out["SMAPE_Improvement_%"] = (
        (out[f"SMAPE_%_{baseline_name}"] - out["SMAPE_%_CatBoost"])
        / out[f"SMAPE_%_{baseline_name}"].replace(0, pd.NA)
        * 100.0
    )
    out["winner_by_SMAPE"] = "equal"
    out.loc[out["SMAPE_%_CatBoost"] < out[f"SMAPE_%_{baseline_name}"], "winner_by_SMAPE"] = "CatBoost"
    out.loc[out[f"SMAPE_%_{baseline_name}"] < out["SMAPE_%_CatBoost"], "winner_by_SMAPE"] = baseline_name

    ordered_cols = [
        "indicator", "indicator_name",
        f"MAPE_%_{baseline_name}", "MAPE_%_CatBoost", f"MAPE_%_diff_CatBoost_minus_{baseline_name}",
        f"SMAPE_%_{baseline_name}", "SMAPE_%_CatBoost", f"SMAPE_%_diff_CatBoost_minus_{baseline_name}",
        f"MAE_{baseline_name}", "MAE_CatBoost", f"MAE_diff_CatBoost_minus_{baseline_name}",
        f"RMSE_{baseline_name}", "RMSE_CatBoost", f"RMSE_diff_CatBoost_minus_{baseline_name}",
        "SMAPE_Improvement_%", "winner_by_SMAPE",
    ]
    out = out[ordered_cols].sort_values("indicator").reset_index(drop=True)
    summary = _build_summary(out, baseline_name)

    detailed_out = Path(detailed_out)
    summary_out = Path(summary_out)
    detailed_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(detailed_out, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="comparison", index=False)
    with pd.ExcelWriter(summary_out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        out.to_excel(writer, sheet_name="by_indicator", index=False)
    print(f"SMAPE comparison saved: {detailed_out}")
    return out
