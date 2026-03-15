import pandas as pd


def sanity_check_yearly_aggregation_to_csv(
    data_bundle: dict,
    gen_params: dict,
    roads: list,
    out_csv: str = "sanity_check_yearly.csv",
):
    """
    Агрегирует месячные синтетические данные до годового уровня,
    сравнивает с табличными значениями (GEN_PARAMS)
    и сохраняет результаты в CSV.

    Для удельного показателя (specific) табличное значение
    вычисляется из годовых табличных данных.
    """

    records = []

    for road in roads:

        # =====================================================
        # 1. Базовые показатели (поездо-км и потери)
        # =====================================================
        for indicator in ["train_km", "loss_12", "loss_3", "loss_tech"]:
            series = data_bundle[indicator][road]

            # агрегация до годового уровня
            yearly_sum = series.resample("YE").sum()
            synthetic_mean = yearly_sum.mean()

            table_value = gen_params[road][indicator]["base"]

            records.append({
                "road": road,
                "indicator": indicator,
                "table_value_year": table_value,
                "synthetic_year_mean": synthetic_mean,
                "relative_diff_pct":
                    100.0 * (synthetic_mean - table_value) / table_value,
            })

        # =====================================================
        # 2. Общие потери (табличные и синтетические)
        # =====================================================
        loss_total_table = (
            gen_params[road]["loss_12"]["base"]
            + gen_params[road]["loss_3"]["base"]
            + gen_params[road]["loss_tech"]["base"]
        )

        loss_total_year = (
            data_bundle["loss_total"][road]
            .resample("YE")
            .sum()
        )
        loss_total_synth_mean = loss_total_year.mean()

        records.append({
            "road": road,
            "indicator": "loss_total",
            "table_value_year": loss_total_table,
            "synthetic_year_mean": loss_total_synth_mean,
            "relative_diff_pct":
                100.0 * (loss_total_synth_mean - loss_total_table)
                / loss_total_table,
        })

        # =====================================================
        # 3. Удельный показатель (specific)
        # =====================================================
        train_km_table = gen_params[road]["train_km"]["base"]

        specific_table = loss_total_table / train_km_table * 1e6

        train_km_year = (
            data_bundle["train_km"][road]
            .resample("YE")
            .sum()
        )

        specific_synth = loss_total_year / train_km_year * 1e6
        specific_synth_mean = specific_synth.mean()

        records.append({
            "road": road,
            "indicator": "specific",
            "table_value_year": specific_table,
            "synthetic_year_mean": specific_synth_mean,
            "relative_diff_pct":
                100.0 * (specific_synth_mean - specific_table)
                / specific_table,
        })

    df = pd.DataFrame(records)

    df.to_csv(out_csv, index=False, float_format="%.6f")

    print(f"✅ Sanity-check сохранён в файл: {out_csv}")

    return df
