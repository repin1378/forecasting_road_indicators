# SARIMA/scale.py
"""
Входные данные:
    Массив длины n после обработки выбросов.

Выход:
    scaled_series - массив длины n, нормализованный делением на среднее.
    mean_value — среднее значение исходного ряда x^(1).
"""

import numpy as np


def scale_series(clean_series: np.ndarray):
    """
    clean_series : np.ndarray
        Ряд после обработки выбросов
    """

    clean_series = np.asarray(clean_series, dtype=float)

    # 1. Среднее значение (u1)
    mean_value = clean_series.mean()

    # 2. Нормализация масштаба
    scaled_series = clean_series / mean_value

    return scaled_series, mean_value
