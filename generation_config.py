# generation_config.py
"""
Конфигурация параметров генерации синтетических временных рядов.

Новая логика: базой служат фактические годовые итоги из example_year.xlsx.
Функция generate_monthly_data (series_generator.py) разбивает годовые значения
по месяцам с учётом сезонности, тренда и случайного шума.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# Зерно генератора
# -------------------------
SEED = 42
np.random.seed(SEED)

# -------------------------
# Путь к файлу годовых данных
# -------------------------
ANNUAL_DATA_PATH = Path(__file__).parent / "examples" / "example_year.xlsx"

# Год, к которому привязаны фактические данные
ANNUAL_YEAR = 2024

# -------------------------
# Диапазон генерации (включительно)
# -------------------------
START_YEAR = 2018
END_YEAR = 2024

# -------------------------
# Сезонные профили
#
# Ключ — номер месяца (1–12).
# Значения нормируются автоматически так, чтобы их сумма равнялась 12,
# а среднемесячное значение совпадало с annual_value / 12.
# -------------------------
SEASONAL_PROFILES = {
    # Объём перевозочной работы: сравнительно ровный,
    # небольшой подъём в летне-осенний период
    "flat": {
        1: 0.96, 2: 0.93, 3: 0.98, 4: 0.99, 5: 1.02, 6: 1.02,
        7: 1.03, 8: 1.04, 9: 1.03, 10: 1.02, 11: 1.00, 12: 0.98,
    },
    # Отказы и потери: максимум в декабре–феврале (морозы, снег),
    # минимум в июне–августе
    "winter_peak": {
        1: 1.20, 2: 1.15, 3: 1.05, 4: 0.97, 5: 0.90,
        6: 0.85, 7: 0.82, 8: 0.85, 9: 0.92, 10: 1.00, 11: 1.10, 12: 1.18,
    },
}

# -------------------------
# Конфигурация показателей
#
# seasonal  — ключ профиля из SEASONAL_PROFILES
# trend_pct — годовое изменение относительно базового года (доля: -0.01 = -1%/год)
# noise_pct — стандартное отклонение шума как доля от среднемесячного значения
# is_count  — True → округлять до целых после генерации
# -------------------------
INDICATOR_CONFIG = {
    "TRAIN_KM": dict(
        seasonal="flat",
        trend_pct=-0.005,
        noise_pct=0.015,
        is_count=False,
    ),
    "LOSS_12_COUNT": dict(
        seasonal="winter_peak",
        trend_pct=-0.015,
        noise_pct=0.08,
        is_count=True,
    ),
    "LOSS_12_SUM": dict(
        seasonal="winter_peak",
        trend_pct=-0.015,
        noise_pct=0.10,
        is_count=False,
    ),
    "LOSS_3_COUNT": dict(
        seasonal="winter_peak",
        trend_pct=0.010,
        noise_pct=0.07,
        is_count=True,
    ),
    "LOSS_3_SUM": dict(
        seasonal="winter_peak",
        trend_pct=0.010,
        noise_pct=0.10,
        is_count=False,
    ),
    "LOSS_TECH_COUNT": dict(
        seasonal="winter_peak",
        trend_pct=-0.010,
        noise_pct=0.06,
        is_count=True,
    ),
    "LOSS_TECH_SUM": dict(
        seasonal="winter_peak",
        trend_pct=-0.010,
        noise_pct=0.09,
        is_count=False,
    ),
}

# Путь к выходному CSV
OUTPUT_CSV = Path(__file__).parent / "synthetic_data" / "synthetic_monthly.csv"

# -------------------------
# Обратная совместимость с main.py
# -------------------------
ROADS = [
    "Октябрьская", "Калининградская", "Московская", "Горьковская",
    "Северная", "Северо-Кавказская", "Юго-Восточная", "Приволжская",
    "Куйбышевская", "Свердловская", "Южно-Уральская", "Западно-Сибирская",
    "Красноярская", "Восточно-Сибирская", "Забайкальская", "Дальневосточная",
]

DATES = pd.date_range("2022-01-01", periods=48, freq="MS")

GEN_PARAMS_LABEL = "Параметры для generate_all_series (main.py)"
GEN_PARAMS = {
    "Октябрьская": {
        "train_km":  dict(base=1.60e8, trend=-2.0e5, seasonal_amp=1.0e6, noise_scale=5.0e5),
        "loss_12":   dict(base=13000, trend=-20, seasonal_amp=800,  noise_scale=600),
        "loss_3":    dict(base=1100,  trend=5,   seasonal_amp=300,  noise_scale=200),
        "loss_tech": dict(base=14000, trend=-30, seasonal_amp=1200, noise_scale=900),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Калининградская": {
        "train_km":  dict(base=2.5e7, trend=5.0e4, seasonal_amp=3.0e5, noise_scale=2.0e5),
        "loss_12":   dict(base=1800,  trend=-3,   seasonal_amp=200,  noise_scale=120),
        "loss_3":    dict(base=200,   trend=1,    seasonal_amp=80,   noise_scale=50),
        "loss_tech": dict(base=2100,  trend=-4,   seasonal_amp=300,  noise_scale=180),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Московская": {
        "train_km":  dict(base=1.45e8, trend=-1.0e5, seasonal_amp=1.2e6, noise_scale=7.0e5),
        "loss_12":   dict(base=16000, trend=-25, seasonal_amp=1000, noise_scale=700),
        "loss_3":    dict(base=1500,  trend=3,   seasonal_amp=400,  noise_scale=250),
        "loss_tech": dict(base=18000, trend=-35, seasonal_amp=1500, noise_scale=1000),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Горьковская": {
        "train_km":  dict(base=9.0e7, trend=-5.0e4, seasonal_amp=8.0e5, noise_scale=4.0e5),
        "loss_12":   dict(base=9000,  trend=-10, seasonal_amp=600, noise_scale=400),
        "loss_3":    dict(base=900,   trend=2,   seasonal_amp=250, noise_scale=150),
        "loss_tech": dict(base=10000, trend=-15, seasonal_amp=800, noise_scale=600),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Северная": {
        "train_km":  dict(base=8.5e7, trend=-4.0e4, seasonal_amp=7.0e5, noise_scale=4.5e5),
        "loss_12":   dict(base=9500,  trend=-12, seasonal_amp=700, noise_scale=450),
        "loss_3":    dict(base=1000,  trend=2,   seasonal_amp=280, noise_scale=180),
        "loss_tech": dict(base=10500, trend=-18, seasonal_amp=900, noise_scale=650),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Северо-Кавказская": {
        "train_km":  dict(base=6.5e7, trend=3.0e4, seasonal_amp=9.0e5, noise_scale=5.0e5),
        "loss_12":   dict(base=8500,  trend=-5, seasonal_amp=1000, noise_scale=700),
        "loss_3":    dict(base=1200,  trend=3,  seasonal_amp=500,  noise_scale=300),
        "loss_tech": dict(base=9000,  trend=-8, seasonal_amp=1300, noise_scale=900),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Юго-Восточная": {
        "train_km":  dict(base=5.8e7, trend=-2.0e4, seasonal_amp=5.0e5, noise_scale=3.0e5),
        "loss_12":   dict(base=6000,  trend=-8, seasonal_amp=500, noise_scale=300),
        "loss_3":    dict(base=700,   trend=1,  seasonal_amp=200, noise_scale=120),
        "loss_tech": dict(base=6500,  trend=-10, seasonal_amp=600, noise_scale=350),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Приволжская": {
        "train_km":  dict(base=2.2e8, trend=4.0e5, seasonal_amp=2.0e6, noise_scale=1.0e6),
        "loss_12":   dict(base=24000, trend=-40, seasonal_amp=1600, noise_scale=1200),
        "loss_3":    dict(base=2200,  trend=6,   seasonal_amp=600,  noise_scale=400),
        "loss_tech": dict(base=26000, trend=-50, seasonal_amp=2000, noise_scale=1500),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Куйбышевская": {
        "train_km":  dict(base=9.8e7, trend=-6.0e4, seasonal_amp=8.0e5, noise_scale=5.0e5),
        "loss_12":   dict(base=11000, trend=-18, seasonal_amp=900, noise_scale=600),
        "loss_3":    dict(base=1000,  trend=2,   seasonal_amp=300, noise_scale=200),
        "loss_tech": dict(base=12000, trend=-25, seasonal_amp=1100, noise_scale=800),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Свердловская": {
        "train_km":  dict(base=1.05e8, trend=-5.0e4, seasonal_amp=9.0e5, noise_scale=6.0e5),
        "loss_12":   dict(base=12500, trend=-20, seasonal_amp=1000, noise_scale=700),
        "loss_3":    dict(base=1300,  trend=3,   seasonal_amp=400, noise_scale=250),
        "loss_tech": dict(base=13500, trend=-30, seasonal_amp=1300, noise_scale=900),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Южно-Уральская": {
        "train_km":  dict(base=8.8e7, trend=-4.0e4, seasonal_amp=7.5e5, noise_scale=4.5e5),
        "loss_12":   dict(base=10000, trend=-15, seasonal_amp=800, noise_scale=500),
        "loss_3":    dict(base=900,   trend=2,   seasonal_amp=300, noise_scale=200),
        "loss_tech": dict(base=11000, trend=-22, seasonal_amp=1000, noise_scale=700),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Западно-Сибирская": {
        "train_km":  dict(base=7.5e7, trend=-3.0e4, seasonal_amp=6.0e5, noise_scale=4.0e5),
        "loss_12":   dict(base=9000,  trend=-12, seasonal_amp=700, noise_scale=450),
        "loss_3":    dict(base=800,   trend=2,   seasonal_amp=250, noise_scale=150),
        "loss_tech": dict(base=9500,  trend=-18, seasonal_amp=850, noise_scale=600),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Красноярская": {
        "train_km":  dict(base=6.8e7, trend=-2.0e4, seasonal_amp=5.5e5, noise_scale=3.5e5),
        "loss_12":   dict(base=8500,  trend=-10, seasonal_amp=650, noise_scale=400),
        "loss_3":    dict(base=750,   trend=2,   seasonal_amp=220, noise_scale=140),
        "loss_tech": dict(base=9000,  trend=-15, seasonal_amp=800, noise_scale=550),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Восточно-Сибирская": {
        "train_km":  dict(base=9.2e7, trend=2.0e4, seasonal_amp=7.0e5, noise_scale=5.0e5),
        "loss_12":   dict(base=11000, trend=-8, seasonal_amp=900, noise_scale=600),
        "loss_3":    dict(base=1200,  trend=4, seasonal_amp=350, noise_scale=220),
        "loss_tech": dict(base=11500, trend=-12, seasonal_amp=1000, noise_scale=700),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Забайкальская": {
        "train_km":  dict(base=1.1e8, trend=-8.0e4, seasonal_amp=1.4e6, noise_scale=9.0e5),
        "loss_12":   dict(base=18000, trend=-35, seasonal_amp=2000, noise_scale=1500),
        "loss_3":    dict(base=1600,  trend=5,   seasonal_amp=600,  noise_scale=400),
        "loss_tech": dict(base=20000, trend=-45, seasonal_amp=2400, noise_scale=1800),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Дальневосточная": {
        "train_km":  dict(base=1.05e8, trend=-6.0e4, seasonal_amp=1.2e6, noise_scale=8.0e5),
        "loss_12":   dict(base=17000, trend=-30, seasonal_amp=1800, noise_scale=1300),
        "loss_3":    dict(base=1500,  trend=4,   seasonal_amp=500,  noise_scale=350),
        "loss_tech": dict(base=18500, trend=-38, seasonal_amp=2200, noise_scale=1600),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },
}