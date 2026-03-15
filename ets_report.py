# ets_report.py
"""
Генерация PDF-отчёта по результатам ETS(A,A,A) прогнозирования.
Добавлена поддержка кириллицы через DejaVuSans.ttf.
Исправлена ошибка openssl_md5(... usedforsecurity=...).
"""

# --------------------------------------------------------------------
# 🔧 Patch: исправление ошибки openssl_md5(... usedforsecurity=...)
# --------------------------------------------------------------------
import hashlib

try:
    hashlib.md5(b"test", usedforsecurity=False)
except TypeError:
    _old_md5 = hashlib.md5

    def _md5_patched(data=b"", *args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _old_md5(data)

    hashlib.md5 = _md5_patched

# --------------------------------------------------------------------
# ReportLab imports
# --------------------------------------------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak
)
from reportlab.lib.units import cm

# --- Добавляем поддержку кириллицы ---
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))

import pandas as pd
from pathlib import Path


def generate_ets_pdf_report(
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    charts_dir: str = "ets_results",
    outfile: str = "ets_report.pdf"
):
    """
    Создаёт PDF отчёт по результатам прогнозирования ETS(A,A,A).
    """

    charts_path = Path(charts_dir)
    doc = SimpleDocTemplate(outfile, pagesize=A4)
    story = []

    # -------- Стили с поддержкой кириллицы ----------
    styles = getSampleStyleSheet()
    styles['Title'].fontName = 'DejaVu'
    styles['Heading1'].fontName = 'DejaVu'
    styles['Heading2'].fontName = 'DejaVu'
    styles['BodyText'].fontName = 'DejaVu'

    title_style = styles["Title"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    normal = styles["BodyText"]

    # --------------------------------------------------------------
    # 1. Заголовок
    # --------------------------------------------------------------
    story.append(Paragraph("Отчёт о прогнозировании ETS(A,A,A)", title_style))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Автоматически сформированный PDF-отчёт.", normal))
    story.append(Spacer(1, 0.7 * cm))

    # --------------------------------------------------------------
    # 2. Таблица метрик
    # --------------------------------------------------------------
    story.append(Paragraph("Сводная таблица метрик", h1))
    story.append(Spacer(1, 0.4 * cm))

    metrics_tbl_data = [
        ["Дорога", "MAPE (%)", "SMAPE (%)", "AIC", "Std остатков"]
    ]

    for road, row in metrics_df.iterrows():
        metrics_tbl_data.append([
            road,
            f"{row['MAPE_test_%']:.2f}",
            f"{row['SMAPE_test_%']:.2f}",
            f"{row['AIC']:.2f}",
            f"{row['resid_std']:.2f}",
        ])

    metrics_table = Table(metrics_tbl_data, colWidths=[3*cm]*5)
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 0), (-1, -1), "DejaVu", 9),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))

    story.append(metrics_table)
    story.append(Spacer(1, 1 * cm))

    # --------------------------------------------------------------
    # 3. Графики прогноза
    # --------------------------------------------------------------
    story.append(Paragraph("Графики прогнозов по дорогам", h1))
    story.append(Spacer(1, 0.4 * cm))

    roads = forecasts_df["road"].unique()

    for r in roads:
        img_path = charts_path / f"forecast_{r}.png"

        story.append(Paragraph(f"Дорога: {r}", h2))
        story.append(Spacer(1, 0.2 * cm))

        if img_path.exists():
            img = Image(str(img_path), width=15 * cm, height=7 * cm)
            story.append(img)
        else:
            story.append(Paragraph(f"⚠ График не найден: {img_path}", normal))

        story.append(Spacer(1, 0.8 * cm))
        story.append(PageBreak())

    # --------------------------------------------------------------
    # 4. Примеры строк прогнозов
    # --------------------------------------------------------------
    story.append(Paragraph("Примеры прогнозов (первые строки)", h1))
    story.append(Spacer(1, 0.4 * cm))

    preview = forecasts_df.head(12)
    tbl = [list(preview.columns)] + preview.values.tolist()

    table = Table(tbl)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONT", (0, 0), (-1, -1), "DejaVu", 8),
    ]))

    story.append(table)
    story.append(Spacer(1, 1 * cm))

    # --------------------------------------------------------------
    # 5. Сохранение PDF
    # --------------------------------------------------------------
    doc.build(story)
    print(f"📄 PDF отчёт успешно создан: {outfile}")