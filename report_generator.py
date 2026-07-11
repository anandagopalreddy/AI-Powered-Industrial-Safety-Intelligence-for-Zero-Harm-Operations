"""
Zero-Harm — Incident Report Generator
=========================================
Renders a one-page PDF incident report for a single zone assessment: risk
band, score, every triggered rule (rule-based and AI), the recommended
action, and a timestamp — the kind of artifact a safety officer would attach
to an audit trail or hand to a shift supervisor.
"""

import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

BAND_COLORS = {
    "LOW": colors.HexColor("#2ecc71"),
    "MODERATE": colors.HexColor("#f5a623"),
    "HIGH": colors.HexColor("#e8672c"),
    "CRITICAL": colors.HexColor("#e5393b"),
}


def build_incident_report_pdf(zone_name: str, hazard_class: str, assessment, gas_reading) -> bytes:
    """
    assessment: risk_engine.ZoneAssessment (zone_id, score, risk_level, triggers, recommended_action)
    gas_reading: data_simulator.GasReading for the same zone
    Returns raw PDF bytes, ready to stream as a download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleGreen", parent=styles["Title"], textColor=colors.HexColor("#1f5c3a"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#1f5c3a"))
    body_style = styles["BodyText"]
    dim_style = ParagraphStyle("Dim", parent=styles["BodyText"], textColor=colors.HexColor("#606a64"), fontSize=9)

    band_color = BAND_COLORS.get(assessment.risk_level, colors.grey)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    story = []
    story.append(Paragraph("Zero-Harm — Compound Risk Incident Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Generated: {now_str}", dim_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#d0d5d2")))
    story.append(Spacer(1, 12))

    header_table = Table(
        [
            ["Zone", zone_name],
            ["Zone ID", assessment.zone_id],
            ["Hazard Class", hazard_class],
            ["Gas Reading", f"{gas_reading.value_ppm:.1f} ppm ({gas_reading.gas_type})"],
        ],
        colWidths=[4 * cm, 10 * cm],
    )
    header_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Risk Assessment", h2_style))
    risk_table = Table(
        [["Compound Risk Score", f"{assessment.score} / 100"],
         ["Risk Band", assessment.risk_level]],
        colWidths=[4 * cm, 10 * cm],
    )
    risk_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 1), (1, 1), band_color),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Triggered Signals", h2_style))
    if assessment.triggers:
        for t in assessment.triggers:
            story.append(Paragraph(f"\u2022 {t}", body_style))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("No signals triggered — zone nominal.", body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Recommended Action", h2_style))
    action_table = Table([[assessment.recommended_action]], colWidths=[14 * cm])
    action_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f5f3")),
        ("BOX", (0, 0), (-1, -1), 0.75, band_color),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(action_table)
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#d0d5d2")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated automatically by the Zero-Harm Compound Risk Detection Engine. "
        "This report reflects the system's assessment at the moment of generation and "
        "should be reviewed alongside direct operator judgement, not in place of it.",
        dim_style,
    ))

    doc.build(story)
    return buffer.getvalue()
