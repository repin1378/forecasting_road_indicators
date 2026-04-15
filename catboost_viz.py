"""
catboost_viz.py

Визуализация процесса обучения и структуры моделей CatBoost.

По мотивам: https://habr.com/ru/companies/otus/articles/527554/

Две функции:
  save_training_curves(model, indicator, outdir)
      Сохраняет кривые train/val RMSE по итерациям как интерактивный HTML.
      Аналог запуска model.fit(..., plot=True) в Jupyter —
      но работает и в скрипте, сохраняет результат на диск.

  save_tree(model, indicator, outdir, tree_idx=0)
      Сохраняет структуру одного дерева ансамбля как SVG-файл.
      Аналог cat.plot_tree(tree_idx=0) в Jupyter.
      Требует: pip install graphviz  +  системный пакет graphviz.

Использование
-------------
    from catboost_viz import save_training_curves, save_tree

    model.fit(train_pool, eval_set=eval_pool)

    save_training_curves(model, indicator="TRAIN_KM",
                         outdir="catboost_results/training_curves")

    save_tree(model, indicator="TRAIN_KM",
              outdir="catboost_results/trees", tree_idx=0)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# АВТОМАТИЧЕСКОЕ ДОБАВЛЕНИЕ GRAPHVIZ В PATH (Windows)
#
# winget устанавливает graphviz в стандартный каталог, но PATH
# обновляется только в новом терминале. Добавляем путь программно
# при первом импорте модуля — без перезапуска и ручной настройки.
# ============================================================

_GRAPHVIZ_WIN_CANDIDATES = [
    r"C:\Program Files\Graphviz\bin",
    r"C:\Program Files (x86)\Graphviz\bin",
    r"C:\Graphviz\bin",
]

def _ensure_graphviz_on_path() -> None:
    """Добавляет каталог graphviz в PATH, если dot не найден (только Windows)."""
    if sys.platform != "win32":
        return
    # Проверяем, доступен ли dot уже
    import shutil
    if shutil.which("dot"):
        return
    for candidate in _GRAPHVIZ_WIN_CANDIDATES:
        if Path(candidate, "dot.exe").exists():
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            return

_ensure_graphviz_on_path()


# ============================================================
# КРИВЫЕ ОБУЧЕНИЯ
# ============================================================

def save_training_curves(
    model,
    indicator: str,
    outdir: str | Path,
    *,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    Строит и сохраняет кривые обучения CatBoost (train/val RMSE) как HTML.

    Использует model.get_evals_result() — работает в любом окружении
    (скрипт, Jupyter, CI). В Jupyter для интерактивности при обучении
    используйте model.fit(..., plot=True).

    Параметры
    ----------
    model     : обученный CatBoostRegressor / CatBoostClassifier
    indicator : название показателя (используется в заголовке и имени файла)
    outdir    : директория для сохранения .html
    title     : заголовок графика (по умолчанию — "Обучение: {indicator}")

    Возвращает путь к сохранённому HTML или None, если plotly не установлен.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  [!] plotly не установлен — кривые обучения не сохранены ({indicator})")
        return None

    evals = model.get_evals_result()
    # evals: {"learn": {"RMSE": [...]}, "validation": {"RMSE": [...]}}

    learn_rmse = evals.get("learn", {}).get("RMSE", [])
    val_rmse   = evals.get("validation", {}).get("RMSE", [])

    if not learn_rmse:
        print(f"  [!] get_evals_result() пуст — кривые обучения недоступны ({indicator})")
        return None

    iterations = list(range(1, len(learn_rmse) + 1))
    best_iter  = int(np.argmin(val_rmse)) + 1 if val_rmse else int(np.argmin(learn_rmse)) + 1
    best_val   = min(val_rmse) if val_rmse else min(learn_rmse)

    fig = go.Figure()

    # ── Train-кривая ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=iterations, y=learn_rmse,
        mode="lines",
        name="Train RMSE",
        line=dict(color="#2196F3", width=1.5),
    ))

    # ── Validation-кривая ────────────────────────────────
    if val_rmse:
        fig.add_trace(go.Scatter(
            x=iterations, y=val_rmse,
            mode="lines",
            name="Val RMSE",
            line=dict(color="#F44336", width=1.5),
        ))

        # Маркер лучшей итерации (early stopping)
        fig.add_trace(go.Scatter(
            x=[best_iter], y=[best_val],
            mode="markers",
            name=f"Best iter {best_iter}",
            marker=dict(color="#4CAF50", size=10, symbol="star"),
            hovertemplate=f"Iter: {best_iter}<br>Val RMSE: {best_val:,.4f}<extra></extra>",
        ))

        # Вертикальная линия на лучшей итерации
        fig.add_vline(
            x=best_iter,
            line=dict(color="#4CAF50", width=1, dash="dot"),
        )

    chart_title = title or f"Кривые обучения CatBoost — {indicator}"

    fig.update_layout(
        title=dict(
            text=f"<b>{chart_title}</b>"
                 + (f"<br><sup>Best iter: {best_iter} · Val RMSE: {best_val:,.2f}</sup>"
                    if val_rmse else ""),
            font=dict(size=14),
            x=0.02, xanchor="left",
        ),
        xaxis=dict(title="Итерация", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="RMSE", showgrid=True, gridcolor="#e0e0e0"),
        legend=dict(
            orientation="h", yanchor="top", y=0.98,
            xanchor="right", x=0.99,
        ),
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
        width=900, height=450,
    )

    outpath = Path(outdir) / f"{indicator}.html"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath), include_plotlyjs="cdn")
    return outpath


# ============================================================
# ВИЗУАЛИЗАЦИЯ ДЕРЕВА
# ============================================================

def save_tree(
    model,
    indicator: str,
    outdir: str | Path,
    *,
    tree_idx: int = 0,
    fmt: str = "svg",
) -> Optional[Path]:
    """
    Сохраняет структуру дерева CatBoost (plot_tree) в SVG/PNG файл.

    Аналог вызова cat.plot_tree(tree_idx=0) в Jupyter —
    но сохраняет результат на диск вместо отображения в ноутбуке.

    Требования
    ----------
    pip install graphviz          # Python-обёртка
    # Системный graphviz (dot):
    #   Windows: https://graphviz.org/download/
    #   Linux:   apt install graphviz
    #   macOS:   brew install graphviz

    Параметры
    ----------
    model     : обученный CatBoostRegressor / CatBoostClassifier
    indicator : название показателя (используется в имени файла)
    outdir    : директория для сохранения файла
    tree_idx  : индекс дерева в ансамбле (0 = первое дерево)
    fmt       : формат файла: "svg" (по умолчанию) или "png"

    Возвращает путь к сохранённому файлу или None при ошибке.

    Примечание
    ----------
    CatBoost строит симметричные (oblivious) деревья: на каждом уровне
    все листья разбиваются по одному и тому же условию. Поэтому дерево
    глубины 6 содержит 6 уровней с одинаковым сплитом на каждом ряду.
    """
    try:
        import graphviz  # noqa: F401  (нужен только для проверки наличия)
    except ImportError:
        print(
            f"  [!] Пакет graphviz не установлен — дерево не сохранено ({indicator}).\n"
            "      Установите: pip install graphviz\n"
            "      И системный graphviz: https://graphviz.org/download/"
        )
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        graph = model.plot_tree(tree_idx=tree_idx)
    except Exception as exc:
        print(f"  [!] model.plot_tree() завершился с ошибкой ({indicator}): {exc}")
        return None

    # graph — объект graphviz.Source; render() сохраняет файл на диск
    stem    = f"{indicator}_tree{tree_idx}"
    outfile = outdir / stem          # без расширения — graphviz добавит сам

    try:
        rendered = graph.render(
            filename=str(outfile),
            format=fmt,
            cleanup=True,            # удаляет промежуточный .gv файл
        )
        return Path(rendered)
    except Exception as exc:
        print(
            f"  [!] graphviz render завершился с ошибкой ({indicator}): {exc}\n"
            "      Убедитесь, что системный graphviz установлен и доступен в PATH."
        )
        return None
