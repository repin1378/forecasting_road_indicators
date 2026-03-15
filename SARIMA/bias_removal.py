# SARIMA/bias_removal.py
"""
Удаление смещения

Вход:
    diff_series - результат дифференцирования
    icase — номер схемы дифференцирования (0..4)
    nd — количество нулевых элементов в начале ряда

Выход:
    bias_removed_series — ряд данных (вещественные числа) в котором значения получены путем вычитания среднего значения ряда (результата дифференцирования)
    mean_u5 — среднее значение
"""

import numpy as np


def remove_bias(diff_series: np.ndarray, icase: int, nd: int):

    diff_series = np.asarray(diff_series, dtype=float)
    n = len(diff_series)

    # ---------------------------------------------------------
    # Icase = 2 или 4 — смещение НЕ удаляется
    # ---------------------------------------------------------
    if icase in (2, 4):
        return diff_series.copy(), 0.0

    # ---------------------------------------------------------
    # Icase = 0, 1, 3 — выполняем удаление смещения
    # ---------------------------------------------------------

    if nd >= n:
        raise ValueError("nd не может быть >= длины ряда")

    # 1. Среднее значение только по хвосту ряда
    mean_u5 = diff_series[nd:].mean()

    # 2. Построение x^(6) по правилу из Методики
    bias_removed = np.zeros(n)
    bias_removed[nd:] = diff_series[nd:] - mean_u5

    return bias_removed, float(mean_u5)