# generation_config.py
"""
Конфигурация параметров генерации временных рядов.
Используется в main.py и series_generator.py.
"""

import numpy as np
import pandas as pd

# -------------------------
# 1. Зерно генератора
# -------------------------
SEED = 42
np.random.seed(SEED)

# -------------------------
# 2. Набор дорог
# -------------------------
ROADS = [
    "Окт", "Клнг", "Моск", "Горьк", "Сев", "С-Кав", "Ю-Вост", "Прив", "Кбш",
    "Сверд", "Ю-Ур", "З-Сиб", "Крас", "В-Сиб", "Заб", "Двост", "Сеть"
]

# -------------------------
# 3. Даты (48 месяцев 2022–2025)
# 3.1 Последовательность календарных дат (MS = начало месяца)
# -------------------------
DATES = pd.date_range("2022-01-01", periods=48, freq="MS")

# -------------------------
# 4. Параметры генерации
#
# Для каждой дороги задаются:
#   train_km    — объем поездо-километровой работы
#   loss_12     — потери по отказам ТС 1–2 категорий
#   loss_3      — потери по отказам ТС 3 категории
#   loss_tech   — потери по технологическим нарушениям
#
# Производные показатели:
#   loss_total  = loss_12 + loss_3 + loss_tech
#   specific    = loss_total / train_km * 1e6
# -------------------------
GEN_PARAMS = {
    "Окт": {
        "train_km":  dict(base=1.60e8, trend=-2.0e5, seasonal_amp=1.0e6, noise_scale=5.0e5),
        "loss_12":   dict(base=13000, trend=-20, seasonal_amp=800,  noise_scale=600),
        "loss_3":    dict(base=1100,  trend=5,   seasonal_amp=300,  noise_scale=200),
        "loss_tech": dict(base=14000, trend=-30, seasonal_amp=1200, noise_scale=900),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Клнг": {
        "train_km":  dict(base=2.5e7, trend=5.0e4, seasonal_amp=3.0e5, noise_scale=2.0e5),
        "loss_12":   dict(base=1800,  trend=-3,   seasonal_amp=200,  noise_scale=120),
        "loss_3":    dict(base=200,   trend=1,    seasonal_amp=80,   noise_scale=50),
        "loss_tech": dict(base=2100,  trend=-4,   seasonal_amp=300,  noise_scale=180),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Моск": {
        "train_km":  dict(base=1.45e8, trend=-1.0e5, seasonal_amp=1.2e6, noise_scale=7.0e5),
        "loss_12":   dict(base=16000, trend=-25, seasonal_amp=1000, noise_scale=700),
        "loss_3":    dict(base=1500,  trend=3,   seasonal_amp=400,  noise_scale=250),
        "loss_tech": dict(base=18000, trend=-35, seasonal_amp=1500, noise_scale=1000),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Горьк": {
        "train_km":  dict(base=9.0e7, trend=-5.0e4, seasonal_amp=8.0e5, noise_scale=4.0e5),
        "loss_12":   dict(base=9000,  trend=-10, seasonal_amp=600, noise_scale=400),
        "loss_3":    dict(base=900,   trend=2,   seasonal_amp=250, noise_scale=150),
        "loss_tech": dict(base=10000, trend=-15, seasonal_amp=800, noise_scale=600),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Сев": {
        "train_km":  dict(base=8.5e7, trend=-4.0e4, seasonal_amp=7.0e5, noise_scale=4.5e5),
        "loss_12":   dict(base=9500,  trend=-12, seasonal_amp=700, noise_scale=450),
        "loss_3":    dict(base=1000,  trend=2,   seasonal_amp=280, noise_scale=180),
        "loss_tech": dict(base=10500, trend=-18, seasonal_amp=900, noise_scale=650),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "С-Кав": {
        "train_km":  dict(base=6.5e7, trend=3.0e4, seasonal_amp=9.0e5, noise_scale=5.0e5),
        "loss_12":   dict(base=8500,  trend=-5, seasonal_amp=1000, noise_scale=700),
        "loss_3":    dict(base=1200,  trend=3,  seasonal_amp=500,  noise_scale=300),
        "loss_tech": dict(base=9000,  trend=-8, seasonal_amp=1300, noise_scale=900),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Ю-Вост": {
        "train_km":  dict(base=5.8e7, trend=-2.0e4, seasonal_amp=5.0e5, noise_scale=3.0e5),
        "loss_12":   dict(base=6000,  trend=-8, seasonal_amp=500, noise_scale=300),
        "loss_3":    dict(base=700,   trend=1,  seasonal_amp=200, noise_scale=120),
        "loss_tech": dict(base=6500,  trend=-10, seasonal_amp=600, noise_scale=350),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Прив": {
        "train_km":  dict(base=2.2e8, trend=4.0e5, seasonal_amp=2.0e6, noise_scale=1.0e6),
        "loss_12":   dict(base=24000, trend=-40, seasonal_amp=1600, noise_scale=1200),
        "loss_3":    dict(base=2200,  trend=6,   seasonal_amp=600,  noise_scale=400),
        "loss_tech": dict(base=26000, trend=-50, seasonal_amp=2000, noise_scale=1500),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Кбш": {
        "train_km":  dict(base=9.8e7, trend=-6.0e4, seasonal_amp=8.0e5, noise_scale=5.0e5),
        "loss_12":   dict(base=11000, trend=-18, seasonal_amp=900, noise_scale=600),
        "loss_3":    dict(base=1000,  trend=2,   seasonal_amp=300, noise_scale=200),
        "loss_tech": dict(base=12000, trend=-25, seasonal_amp=1100, noise_scale=800),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Сверд": {
        "train_km":  dict(base=1.05e8, trend=-5.0e4, seasonal_amp=9.0e5, noise_scale=6.0e5),
        "loss_12":   dict(base=12500, trend=-20, seasonal_amp=1000, noise_scale=700),
        "loss_3":    dict(base=1300,  trend=3,   seasonal_amp=400, noise_scale=250),
        "loss_tech": dict(base=13500, trend=-30, seasonal_amp=1300, noise_scale=900),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Ю-Ур": {
        "train_km":  dict(base=8.8e7, trend=-4.0e4, seasonal_amp=7.5e5, noise_scale=4.5e5),
        "loss_12":   dict(base=10000, trend=-15, seasonal_amp=800, noise_scale=500),
        "loss_3":    dict(base=900,   trend=2,   seasonal_amp=300, noise_scale=200),
        "loss_tech": dict(base=11000, trend=-22, seasonal_amp=1000, noise_scale=700),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "З-Сиб": {
        "train_km":  dict(base=7.5e7, trend=-3.0e4, seasonal_amp=6.0e5, noise_scale=4.0e5),
        "loss_12":   dict(base=9000,  trend=-12, seasonal_amp=700, noise_scale=450),
        "loss_3":    dict(base=800,   trend=2,   seasonal_amp=250, noise_scale=150),
        "loss_tech": dict(base=9500,  trend=-18, seasonal_amp=850, noise_scale=600),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Крас": {
        "train_km":  dict(base=6.8e7, trend=-2.0e4, seasonal_amp=5.5e5, noise_scale=3.5e5),
        "loss_12":   dict(base=8500,  trend=-10, seasonal_amp=650, noise_scale=400),
        "loss_3":    dict(base=750,   trend=2,   seasonal_amp=220, noise_scale=140),
        "loss_tech": dict(base=9000,  trend=-15, seasonal_amp=800, noise_scale=550),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "В-Сиб": {
        "train_km":  dict(base=9.2e7, trend=2.0e4, seasonal_amp=7.0e5, noise_scale=5.0e5),
        "loss_12":   dict(base=11000, trend=-8, seasonal_amp=900, noise_scale=600),
        "loss_3":    dict(base=1200,  trend=4, seasonal_amp=350, noise_scale=220),
        "loss_tech": dict(base=11500, trend=-12, seasonal_amp=1000, noise_scale=700),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Заб": {
        "train_km":  dict(base=1.1e8, trend=-8.0e4, seasonal_amp=1.4e6, noise_scale=9.0e5),
        "loss_12":   dict(base=18000, trend=-35, seasonal_amp=2000, noise_scale=1500),
        "loss_3":    dict(base=1600,  trend=5,   seasonal_amp=600,  noise_scale=400),
        "loss_tech": dict(base=20000, trend=-45, seasonal_amp=2400, noise_scale=1800),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },

    "Двост": {
        "train_km":  dict(base=1.05e8, trend=-6.0e4, seasonal_amp=1.2e6, noise_scale=8.0e5),
        "loss_12":   dict(base=17000, trend=-30, seasonal_amp=1800, noise_scale=1300),
        "loss_3":    dict(base=1500,  trend=4,   seasonal_amp=500,  noise_scale=350),
        "loss_tech": dict(base=18500, trend=-38, seasonal_amp=2200, noise_scale=1600),
        "loss_total": "loss_12 + loss_3 + loss_tech",
        "specific":   "(loss_total / train_km) * 1e6",
    },
}