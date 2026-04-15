"""
smape_comparison.py

Сравнение ETS и CatBoost по SMAPE.

Особенности реализации:
  • ETS обучалась на агрегированных показателях (loss_12, loss_3, loss_tech),
    CatBoost — раздельно (LOSS_12_COUNT + LOSS_12_SUM, MultiRMSE).
    Для сравнения COUNT и SUM каждой группы усредняются в одно значение.
  • ETS-файл читается в кодировке cp1251 (сокращённые названия дорог).
  • Сокращения дорог и показателей разворачиваются в полные русские названия.

Выходные файлы
--------------
  detailed:  compare_resuls/smape_comparison_detailed.csv
             indicator, indicator_name, road, SMAPE_ETS, SMAPE_CatBoost, Improvement_%

  summary:   compare_resuls/smape_comparison_by_indicator.csv
             indicator, indicator_name, roads_n,
             SMAPE_ETS_mean, SMAPE_CatBoost_mean, Improvement_%
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# СПРАВОЧНИКИ НАЗВАНИЙ
# ============================================================

#: ETS-коды → канонический ключ показателя
ETS_TO_CANONICAL: dict[str, str] = {
    "train_km":   "TRAIN_KM",
    "loss_12":    "LOSS_12",
    "loss_3":     "LOSS_3",
    "loss_tech":  "LOSS_TECH",
    "loss_total": "LOSS_TOTAL",
    "specific":   "SPECIFIC_LOSS",
}

#: CatBoost-коды → канонический ключ
#  COUNT + SUM одной группы объединяются (среднее SMAPE при агрегации)
CB_TO_CANONICAL: dict[str, str] = {
    "TRAIN_KM":         "TRAIN_KM",
    "LOSS_12_COUNT":    "LOSS_12",
    "LOSS_12_SUM":      "LOSS_12",
    "LOSS_3_COUNT":     "LOSS_3",
    "LOSS_3_SUM":       "LOSS_3",
    "LOSS_TECH_COUNT":  "LOSS_TECH",
    "LOSS_TECH_SUM":    "LOSS_TECH",
    "LOSS_COUNT_TOTAL": "LOSS_TOTAL",
    "SPECIFIC_LOSS":    "SPECIFIC_LOSS",
}

#: Полные русские названия канонических показателей
INDICATOR_FULL_NAMES: dict[str, str] = {
    "TRAIN_KM": (
        "Объем поездо-километровой работы"
    ),
    "LOSS_12": (
        "Отказы в работе технических средств по 1, 2 категориям"
        " (количество и потери поездо-часов)"
    ),
    "LOSS_3": (
        "Отказы в работе технических средств по 3 категории"
        " (количество и потери поездо-часов)"
    ),
    "LOSS_TECH": (
        "Технологические нарушения"
        " (количество и потери поездо-часов)"
    ),
    "LOSS_TOTAL": (
        "Количество технических отказов и технологических нарушений (итого)"
    ),
    "SPECIFIC_LOSS": (
        "Удельный показатель общих потерь поездо-часов от отказов"
        " технических средств и технологических нарушений на 1 млн. поездо-км"
    ),
    # Отдельные показатели CatBoost (используются в детальном отчёте по CB)
    "TRAIN_KM_FULL":      "Объем поездо-километровой работы",
    "LOSS_12_COUNT_FULL": (
        "Количество отказов в работе технических средств по 1, 2 категориям"
    ),
    "LOSS_12_SUM_FULL": (
        "Потери поездо-часов из-за отказов в работе технических средств"
        " по 1, 2 категориям"
    ),
    "LOSS_3_COUNT_FULL": (
        "Количество отказов в работе технических средств по 3 категории"
    ),
    "LOSS_3_SUM_FULL": (
        "Потери поездо-часов из-за отказов в работе технических средств"
        " по 3 категории"
    ),
    "LOSS_TECH_COUNT_FULL": "Количество технологических нарушений",
    "LOSS_TECH_SUM_FULL": (
        "Потери поездо-часов из-за технологических нарушений"
    ),
    "LOSS_COUNT_TOTAL_FULL": (
        "Количество технических отказов и технологических нарушений"
    ),
    "LOSS_SUM_TOTAL_FULL": (
        "Потери поездо-часов из-за технических отказов"
        " и технологических нарушений"
    ),
    "SPECIFIC_LOSS_FULL": (
        "Удельный показатель общих потерь поездо-часов от отказов"
        " технических средств и технологических нарушений на 1 млн. поездо-км"
    ),
}

#: Полные названия показателей для CatBoost-кодов (для отдельных отчётов)
CB_INDICATOR_FULL_NAMES: dict[str, str] = {
    "TRAIN_KM":         "Объем поездо-километровой работы",
    "LOSS_12_COUNT":    (
        "Количество отказов в работе технических средств по 1, 2 категориям"
    ),
    "LOSS_12_SUM":      (
        "Потери поездо-часов из-за отказов в работе технических средств"
        " по 1, 2 категориям"
    ),
    "LOSS_3_COUNT":     (
        "Количество отказов в работе технических средств по 3 категории"
    ),
    "LOSS_3_SUM":       (
        "Потери поездо-часов из-за отказов в работе технических средств"
        " по 3 категории"
    ),
    "LOSS_TECH_COUNT":  "Количество технологических нарушений",
    "LOSS_TECH_SUM":    (
        "Потери поездо-часов из-за технологических нарушений"
    ),
    "LOSS_COUNT_TOTAL": (
        "Количество технических отказов и технологических нарушений"
    ),
    "LOSS_SUM_TOTAL":   (
        "Потери поездо-часов из-за технических отказов"
        " и технологических нарушений"
    ),
    "SPECIFIC_LOSS":    (
        "Удельный показатель общих потерь поездо-часов от отказов"
        " технических средств и технологических нарушений на 1 млн. поездо-км"
    ),
}

#: Сокращения дорог (ETS) → полные названия
ROAD_FULL_NAMES: dict[str, str] = {
    "Окт":    "Октябрьская",
    "Клнг":   "Калининградская",
    "Моск":   "Московская",
    "Горьк":  "Горьковская",
    "Сев":    "Северная",
    "С-Кав":  "Северо-Кавказская",
    "Ю-Вост": "Юго-Восточная",
    "Прив":   "Приволжская",
    "Кбш":    "Куйбышевская",
    "Сверд":  "Свердловская",
    "Ю-Ур":   "Южно-Уральская",
    "З-Сиб":  "Западно-Сибирская",
    "Крас":   "Красноярская",
    "В-Сиб":  "Восточно-Сибирская",
    "Заб":    "Забайкальская",
    "Двост":  "Дальневосточная",
}

# Порядок показателей в сводном отчёте
INDICATOR_ORDER: list[str] = [
    "TRAIN_KM",
    "LOSS_12",
    "LOSS_3",
    "LOSS_TECH",
    "LOSS_TOTAL",
    "SPECIFIC_LOSS",
]


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _read_ets(path: Path) -> pd.DataFrame:
    """
    Читает ETS-метрики, нормализует кодировку, имена дорог и показателей.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Нормализуем имя столбца SMAPE (в ETS-файле — SMAPE_test_%)
    smape_col = next(
        (c for c in df.columns if "smape" in c.lower()), None
    )
    if smape_col is None:
        raise KeyError(f"Столбец SMAPE не найден в {path}. Столбцы: {df.columns.tolist()}")
    df = df.rename(columns={smape_col: "SMAPE_ETS"})

    # Полные названия дорог
    df["road"] = df["road"].map(ROAD_FULL_NAMES).fillna(df["road"])

    # Канонические коды показателей
    df["indicator"] = df["indicator"].str.lower().map(ETS_TO_CANONICAL).fillna(df["indicator"])

    # ETS может содержать несколько строк на (indicator, road) —
    # разные тестовые периоды или типы моделей (AAA / AA / A).
    # Оставляем лучший результат (минимальный SMAPE) на каждую пару.
    df = (
        df.groupby(["indicator", "road"], as_index=False)
        .agg(SMAPE_ETS=("SMAPE_ETS", "min"))
    )

    return df[["indicator", "road", "SMAPE_ETS"]]


def _read_catboost(path: Path) -> pd.DataFrame:
    """
    Читает CatBoost-метрики, нормализует имена показателей до канонических,
    усредняя COUNT и SUM внутри каждой группы для одного road.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    smape_col = next(
        (c for c in df.columns if "smape" in c.lower()), None
    )
    if smape_col is None:
        raise KeyError(f"Столбец SMAPE не найден в {path}. Столбцы: {df.columns.tolist()}")
    df = df.rename(columns={smape_col: "SMAPE_CatBoost"})

    # Оставляем только показатели, участвующие в сравнении
    df = df[df["indicator"].isin(CB_TO_CANONICAL)].copy()
    df["indicator"] = df["indicator"].map(CB_TO_CANONICAL)

    # Усредняем COUNT + SUM для групповых показателей (например, LOSS_12)
    df = (
        df.groupby(["indicator", "road"], as_index=False)
        .agg(SMAPE_CatBoost=("SMAPE_CatBoost", "mean"))
    )

    return df


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def build_smape_comparison(
    ets_metrics_path: str | Path = "ets_results/ets_metrics_all_indicators.csv",
    catboost_metrics_path: str | Path = "catboost_results/catboost_metrics.csv",
    detailed_out: str | Path = "compare_resuls/smape_comparison_detailed.csv",
    summary_out: str | Path = "compare_resuls/smape_comparison_by_indicator.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строит два CSV-отчёта сравнения ETS и CatBoost по SMAPE.

    Детальный отчёт — indicator × road:
        indicator, indicator_name, road, SMAPE_ETS, SMAPE_CatBoost, Improvement_%

    Сводный отчёт — по показателю:
        indicator, indicator_name, roads_n,
        SMAPE_ETS_mean, SMAPE_CatBoost_mean, Improvement_%

    Improvement_% > 0 → CatBoost точнее (меньше SMAPE).

    Параметры
    ----------
    ets_metrics_path      : путь к CSV с метриками ETS
    catboost_metrics_path : путь к CSV с метриками CatBoost
    detailed_out          : путь для детального отчёта
    summary_out           : путь для сводного отчёта
    """
    ets_path = Path(ets_metrics_path)
    cb_path  = Path(catboost_metrics_path)

    if not ets_path.exists():
        raise FileNotFoundError(f"ETS-метрики не найдены: {ets_path}")
    if not cb_path.exists():
        raise FileNotFoundError(f"CatBoost-метрики не найдены: {cb_path}")

    ets_df = _read_ets(ets_path)
    cb_df  = _read_catboost(cb_path)

    # ── Детальное сравнение ───────────────────────────────────
    detailed = ets_df.merge(cb_df, on=["indicator", "road"], how="inner")

    detailed["Improvement_%"] = (
        (detailed["SMAPE_ETS"] - detailed["SMAPE_CatBoost"])
        / detailed["SMAPE_ETS"].replace(0, float("nan"))
        * 100.0
    ).round(2)

    detailed["SMAPE_ETS"]      = detailed["SMAPE_ETS"].round(4)
    detailed["SMAPE_CatBoost"] = detailed["SMAPE_CatBoost"].round(4)

    # Добавляем полные названия показателей
    detailed.insert(
        1, "indicator_name",
        detailed["indicator"].map(INDICATOR_FULL_NAMES).fillna(detailed["indicator"])
    )

    # Порядок строк: сначала по показателю (в INDICATOR_ORDER), затем по дороге
    indicator_rank = {k: i for i, k in enumerate(INDICATOR_ORDER)}
    detailed["_rank"] = detailed["indicator"].map(indicator_rank).fillna(99)
    detailed = (
        detailed.sort_values(["_rank", "road"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    # ── Сводный отчёт по показателю ──────────────────────────
    summary = (
        detailed
        .groupby(["indicator", "indicator_name"], as_index=False)
        .agg(
            roads_n          =("road",           "count"),
            SMAPE_ETS_mean   =("SMAPE_ETS",      "mean"),
            SMAPE_CatBoost_mean=("SMAPE_CatBoost","mean"),
        )
    )

    summary["Improvement_%"] = (
        (summary["SMAPE_ETS_mean"] - summary["SMAPE_CatBoost_mean"])
        / summary["SMAPE_ETS_mean"].replace(0, float("nan"))
        * 100.0
    ).round(2)

    summary["SMAPE_ETS_mean"]      = summary["SMAPE_ETS_mean"].round(4)
    summary["SMAPE_CatBoost_mean"] = summary["SMAPE_CatBoost_mean"].round(4)

    summary["_rank"] = summary["indicator"].map(indicator_rank).fillna(99)
    summary = (
        summary.sort_values("_rank")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    # ── Сохранение ────────────────────────────────────────────
    for df, path in [(detailed, Path(detailed_out)), (summary, Path(summary_out))]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    # ── Печать итогов ─────────────────────────────────────────
    print("[OK] Сравнение SMAPE (ETS vs CatBoost) завершено")
    print(f"   Детально:  {detailed_out}  ({len(detailed)} строк)")
    print(f"   Сводно:    {summary_out}  ({len(summary)} показателей)")
    print()
    print(summary[["indicator_name", "SMAPE_ETS_mean", "SMAPE_CatBoost_mean", "Improvement_%"]]
          .to_string(index=False))

    return detailed, summary


# ============================================================
# CLI-запуск
# ============================================================

if __name__ == "__main__":
    build_smape_comparison()
