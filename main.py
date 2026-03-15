"""
ETS(A,A,A) — прогнозирование с доверительными интервалами (95%)

Шаги:
1. Генерация синтетических временных рядов (2022–2025)
2. Обучение ETS(A,A,A)
3. Прогноз на 12 месяцев 2026 года
4. Расчёт доверительных интервалов
5. Сохранение прогнозов, метрик и графиков
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from series_generator import generate_all_series
from generation_config import ROADS, DATES, GEN_PARAMS
from generation_config import SEED  # если нужно использовать seed явно
from ets_model import run_ets_forecast
# from ets_report import generate_ets_pdf_report
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sanity_check import sanity_check_yearly_aggregation_to_csv
from catboost_model import run_catboost_forecast
from smape_comparison import build_smape_comparison

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --------------------------------------------------
# 1. Генерация данных
# --------------------------------------------------
data_bundle = generate_all_series(
    gen_params=GEN_PARAMS,
    start="2018-01-01",
    end="2025-12-01",
    allow_spikes=True,
    num_forced_outliers=3,
)

ROADS = ["Окт", "Клнг", "Моск", "Горьк", "Сев", "С-Кав", "Ю-Вост", "Прив", "Кбш", "Сверд", "Ю-Ур", "З-Сиб", "Крас", "В-Сиб", "Заб", "Двост"]
INDICATORS = [
    "train_km",
    "loss_12",
    "loss_3",
    "loss_tech",
    "loss_total",
    "specific",
]


# --------------------------------------------------
# 2. ETS (baseline)
# --------------------------------------------------
for ind in INDICATORS:
    run_ets_forecast(
        data_bundle=data_bundle,
        indicator=ind,
        roads=ROADS,
        horizon=12,
        outdir="ets_results",
    )


# --------------------------------------------------
# 3. CatBoost (ML)
# --------------------------------------------------
for ind in INDICATORS:
    run_catboost_forecast(
        data_bundle=data_bundle,
        indicator=ind,
        roads=ROADS,
        horizon=12,
        outdir="catboost_results",
        exog_indicators=["train_km", "loss_total"] if ind == "specific" else None,
    )


print("✅ ETS и CatBoost рассчитаны. CSV и графики сохранены.")


build_smape_comparison(
    ets_metrics_path="ets_results/ets_metrics_all_indicators.csv",
    catboost_metrics_path="catboost_results/catboost_metrics_all_indicators.csv",
    detailed_out="compare_resuls/smape_comparison_detailed.csv",
    summary_out="compare_resuls/smape_comparison_by_indicator.csv",
)