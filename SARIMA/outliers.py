# SARIMA/outliers.py
"""
Выходные данные:
    clean_series : np.ndarray
        Ряд, где выбросы заменены моделированными значениями.
    outlier_flags : np.ndarray
        Массив 0/1 — пометка выбросов.
    num_outliers : int
        Количество выбросов.
"""

import numpy as np
from scipy.stats import zscore

def detect_outliers_zscore(x: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Определение выбросов через Z-score.
    Возвращает массив флагов: 1 — выброс, 0 — норма.
    """
    z = zscore(x)
    return (np.abs(z) > threshold).astype(int)


def replace_outliers(x: np.ndarray, outlier_flags: np.ndarray) -> np.ndarray:
    """
    Замена выбросов на моделированные значения.
    Сейчас: среднее соседних значений.
    """
    clean_series = x.copy()
    n = len(x)

    for i in range(n):
        if outlier_flags[i] == 1:
            left = x[max(i - 1, 0)]
            right = x[min(i + 1, n - 1)]
            clean_series[i] = (left + right) / 2

    return clean_series


def process_outliers(x: np.ndarray, threshold: float = 3.0):
    """
    Главная функция шага обработки выбросов.

    Возвращает:
        clean_series   — очищенный ряд
        outlier_flags  — флаги выбросов (0/1)
        num_outliers   — количество выбросов
    """
    x = np.asarray(x, dtype=float)

    # Шаг 1: определяем выбросы
    outlier_flags = detect_outliers_zscore(x, threshold)

    # Шаг 2: заменяем выбросы
    clean_series = replace_outliers(x, outlier_flags)

    # Шаг 3: считаем количество выбросов
    num_outliers = int(outlier_flags.sum())

    return clean_series, outlier_flags, num_outliers
