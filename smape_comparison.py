"""
smape_comparison.py

Сравнение ETS и CatBoost по SMAPE:
1) детально: indicator × road
2) агрегированно: по каждому indicator
"""

from pathlib import Path
import pandas as pd


def build_smape_comparison(
    ets_metrics_path: str | Path,
    catboost_metrics_path: str | Path,
    detailed_out: str | Path = "smape_comparison_detailed.csv",
    summary_out: str | Path = "smape_comparison_by_indicator.csv",
):
    """
    Создаёт:
    1) CSV с детальным сравнением ETS vs CatBoost (indicator × road)
    2) CSV с агрегированным выигрышем CatBoost по каждому indicator

    Improvement_% > 0 → CatBoost лучше
    """

    ets_metrics_path = Path(ets_metrics_path)
    catboost_metrics_path = Path(catboost_metrics_path)

    if not ets_metrics_path.exists():
        raise FileNotFoundError(f"ETS metrics not found: {ets_metrics_path}")
    if not catboost_metrics_path.exists():
        raise FileNotFoundError(
            f"CatBoost metrics not found: {catboost_metrics_path}"
        )

    # --------------------------------------------------
    # Чтение
    # --------------------------------------------------
    ets_df = pd.read_csv(ets_metrics_path)
    cb_df = pd.read_csv(catboost_metrics_path)

    ets_df = ets_df[
        ["indicator", "road", "SMAPE_test_%"]
    ].rename(columns={
        "SMAPE_test_%": "SMAPE_ETS"
    })

    cb_df = cb_df[
        ["indicator", "road", "SMAPE_test_%"]
    ].rename(columns={
        "SMAPE_test_%": "SMAPE_CatBoost"
    })

    # --------------------------------------------------
    # Детальное сравнение
    # --------------------------------------------------
    detailed = ets_df.merge(
        cb_df,
        on=["indicator", "road"],
        how="inner",
    )

    detailed["Improvement_%"] = (
        (detailed["SMAPE_ETS"] - detailed["SMAPE_CatBoost"])
        / detailed["SMAPE_ETS"]
        * 100.0
    )

    detailed = detailed.sort_values(
        ["indicator", "road"]
    ).reset_index(drop=True)

    detailed[["SMAPE_ETS", "SMAPE_CatBoost"]] = detailed[
        ["SMAPE_ETS", "SMAPE_CatBoost"]
    ].round(4)

    detailed["Improvement_%"] = detailed["Improvement_%"].round(2)

    # --- сохранение детального CSV ---
    detailed_out = Path(detailed_out)
    detailed_out.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(detailed_out, index=False)

    # --------------------------------------------------
    # АГРЕГИРОВАННОЕ сравнение по indicator
    # --------------------------------------------------
    summary = (
        detailed
        .groupby("indicator", as_index=False)
        .agg(
            roads_n=("road", "count"),
            SMAPE_ETS_mean=("SMAPE_ETS", "mean"),
            SMAPE_CatBoost_mean=("SMAPE_CatBoost", "mean"),
        )
    )

    summary["Improvement_%"] = (
        (summary["SMAPE_ETS_mean"] - summary["SMAPE_CatBoost_mean"])
        / summary["SMAPE_ETS_mean"]
        * 100.0
    )

    summary[["SMAPE_ETS_mean", "SMAPE_CatBoost_mean"]] = summary[
        ["SMAPE_ETS_mean", "SMAPE_CatBoost_mean"]
    ].round(4)

    summary["Improvement_%"] = summary["Improvement_%"].round(2)

    summary = summary.sort_values(
        "Improvement_%", ascending=False
    ).reset_index(drop=True)

    # --- сохранение агрегированного CSV ---
    summary_out = Path(summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    print("✅ SMAPE comparison completed")
    print(f"  • detailed: {detailed_out}")
    print(f"  • summary : {summary_out}")

    return detailed, summary


# ------------------------------------------------------
# CLI-запуск
# ------------------------------------------------------

if __name__ == "__main__":
    build_smape_comparison(
        ets_metrics_path="ets_results/ets_metrics_all_indicators.csv",
        catboost_metrics_path="catboost_results/catboost_metrics_all_indicators.csv",
    )
