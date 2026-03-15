# SARIMA/boxcox_transform.py
"""
Преобразование Бокса–Кокса.
Вход:
    normalized_series — нормализованный ряд
    lambda_value — параметр преобразования λ (0, 1 или 2)

Выход:
    boxcox_series — преобразованный ряд
    geometric_mean — среднее геометрическое ряда
"""

import numpy as np


def compute_geometric_mean(x: np.ndarray) -> float:
    """
    Вычисляет среднее геометрическое GM ряда
    """
    return np.exp(np.mean(np.log(x)))


def boxcox_transform(normalized_series: np.ndarray, lambda_value: float):
    """
    Выполняет преобразование Бокса–Кокса.
    """

    normalized_series = np.asarray(normalized_series, dtype=float)

    # 1. Среднее геометрическое GM
    geometric_mean = compute_geometric_mean(normalized_series)

    # 2. Преобразование
    if lambda_value == 0:
        # Логарифмический случай
        boxcox_series = geometric_mean * np.log(normalized_series)
    else:
        boxcox_series = (
            (normalized_series ** lambda_value) - 1.0
        ) / (lambda_value * (geometric_mean ** (lambda_value - 1)))

    return boxcox_series, geometric_mean
