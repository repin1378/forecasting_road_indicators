# SARIMA/check_input.py
"""
Шаг 1. Проверка на допустимые значения.
Вход:
    x : список/np.ndarray/pd.Series длины n
Проверяет:
    - n == 72
    - нет значений ≤ 0
Выход:
    f_err1 = 0 или 1
    err_msg = текст ошибки или None
"""

from typing import Sequence, Tuple, Optional
import numpy as np

EXPECTED_LENGTH = 72

ERROR_MESSAGE = (
    "Исходный ряд данных содержит неверное количество значений и/или "
    "отрицательные и/или нулевые значения!")

def check_input_series(
    x: Sequence[float],
    expected_length: int = EXPECTED_LENGTH
) -> Tuple[int, Optional[str]]:

    f_err1 = 0
    x_arr = np.asarray(x, dtype=float)

    # Проверка длины
    if x_arr.size != expected_length:
        return 1, ERROR_MESSAGE

    # Проверка значений
    if np.any(x_arr <= 0):
        return 1, ERROR_MESSAGE

    return 0, None


def assert_valid_series(
    x: Sequence[float],
    expected_length: int = EXPECTED_LENGTH
) -> None:
    f_err1, err_msg = check_input_series(x, expected_length)
    if f_err1 == 1:
        raise ValueError(err_msg)
