"""
Reporting module.

Creates professional PDF reports for
StatVerify AI.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models import (
    AIExplanation,
    ReportData,
    StatisticalResult,
    VerificationResult,
)


# ==========================================================
# Report Formatter
# ==========================================================


class ReportFormatter:
    """
    Converts ReportData into printable sections.
    """

    @staticmethod
    def header(title: str) -> list:
        styles = getSampleStyleSheet()

        heading = styles["Heading1"]
        heading.alignment = TA_CENTER

        return [
            Paragraph(title, heading),
            Spacer(1, 20),
        ]

    # ------------------------------------------------------

    @staticmethod
    def statistics_table(
        statistics: StatisticalResult,
    ) -> Table | list:
        """
        Statistical Result table.
        """
        styles = getSampleStyleSheet()

        if statistics.additional_metrics.get("demographics"):
            elements = []
            elements.append(Paragraph("<b>Demographic Analysis Summary</b>", styles["Heading3"]))
            elements.append(Spacer(1, 10))

            rows = [["Demographic Variable", "Chi-square (χ²)", "p-value", "df"]]
            for var, metrics in statistics.additional_metrics["demographics"].items():
                rows.append([
                    var,
                    f"{metrics['chi2']:.2f}",
                    f"{metrics['p']:.3f}",
                    str(int(metrics['dof'])),
                ])
            
            t = Table(rows, colWidths=[160, 100, 100, 100])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            elements.append(t)
            return elements

        stat_name = "Statistic"
        p_val_name = "P Value"
        if statistics.test_name and "chi-square" in statistics.test_name.lower():
            stat_name = "Chi-square"
            p_val_name = "P-value"

        rows = [
            ["Metric", "Value"],
            ["Test", statistics.test_name],
            [stat_name, str(statistics.statistic)],
            [p_val_name, str(statistics.p_value)],
            ["Degrees of Freedom", str(statistics.degrees_of_freedom)],
            ["Effect Size", str(statistics.effect_size)],
            ["Interpretation", statistics.interpretation or ""],
        ]

        table = Table(
            rows,
            colWidths=[180, 280],
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),

                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),

                    ("TOPPADDING", (0, 1), (-1, -1), 6),

                    ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ]
            )
        )

        return table

    # ------------------------------------------------------

    @staticmethod
    def verification_table(
        verification: VerificationResult | None,
    ) -> Table | None:
        """
        Verification comparison table.
        """

        if verification is None:
            return None

        rows = [
            ["Metric", "Reported", "Calculated", "Match"]
        ]

        for item in verification.comparison.items:

            rows.append(
                [
                    item.metric,
                    str(item.reported),
                    str(item.calculated),
                    "YES" if item.matched else "NO",
                ]
            )

        table = Table(
            rows,
            colWidths=[140, 100, 100, 80],
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ]
            )
        )

        return table

    # ------------------------------------------------------

    @staticmethod
    def ai_section(
        ai: AIExplanation | None,
    ) -> list:
        """
        AI explanation section.
        """

        if ai is None:
            return []

        styles = getSampleStyleSheet()

        return [
            Paragraph("<b>AI Summary</b>", styles["Heading2"]),
            Paragraph(ai.summary, styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("<b>Explanation</b>", styles["Heading2"]),
            Paragraph(ai.explanation, styles["BodyText"]),
            Spacer(1, 10),
        ]
# ==========================================================
# Report Builder
# ==========================================================


class ReportBuilder:
    """
    Builds the complete report contents.

    Converts ReportData into a list of ReportLab
    flowables that can be written into a PDF.
    """

    def __init__(self):
        self.formatter = ReportFormatter()

    # ------------------------------------------------------

    def build(
        self,
        report: ReportData,
    ) -> list:
        """
        Build the complete report.
        """

        styles = getSampleStyleSheet()

        elements = []

        # --------------------------------------------------
        # Header
        # --------------------------------------------------

        elements.extend(
            self.formatter.header(
                "StatVerify AI\nStatistical Analysis Report"
            )
        )

        elements.append(
            Paragraph(
                f"<b>Generated:</b> "
                f"{datetime.now().strftime('%d %B %Y %H:%M')}",
                styles["BodyText"],
            )
        )

        elements.append(Spacer(1, 15))

        # --------------------------------------------------
        # Analysis Information
        # --------------------------------------------------

        elements.append(
            Paragraph(
                "<b>Analysis Information</b>",
                styles["Heading2"],
            )
        )

        info = [
            ["Mode", report.request.mode.value],
            ["Input Type", report.request.input_type.value],
            ["Source", report.request.source_name or "-"],
        ]

        info_table = Table(
            info,
            colWidths=[150, 300],
        )

        info_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        elements.append(info_table)

        elements.append(Spacer(1, 18))

        # --------------------------------------------------
        # Statistical Result
        # --------------------------------------------------

        elements.append(
            Paragraph(
                "<b>Statistical Result</b>",
                styles["Heading2"],
            )
        )

        stat_table = self.formatter.statistics_table(
            report.statistics
        )
        if isinstance(stat_table, list):
            elements.extend(stat_table)
        else:
            elements.append(stat_table)

        elements.append(Spacer(1, 18))

        # --------------------------------------------------
        # Verification
        # --------------------------------------------------

        verification_table = self.formatter.verification_table(
            report.verification
        )

        if verification_table is not None:

            elements.append(
                Paragraph(
                    "<b>Verification</b>",
                    styles["Heading2"],
                )
            )

            elements.append(verification_table)

            elements.append(Spacer(1, 18))

        # --------------------------------------------------
        # AI Section
        # --------------------------------------------------

        elements.extend(
            self.formatter.ai_section(
                report.ai
            )
        )

        # --------------------------------------------------
        # Footer
        # --------------------------------------------------

        elements.append(Spacer(1, 20))

        elements.append(
            Paragraph(
                "<font size='9'>"
                "Generated automatically by "
                "<b>StatVerify AI</b>"
                "</font>",
                styles["BodyText"],
            )
        )

        return elements

# ==========================================================
# PDF Exporter
# ==========================================================


class PDFExporter:
    """
    Exports ReportData as a professional PDF.
    """

    def __init__(
        self,
        output_directory: str = "exports",
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------

    def export(
        self,
        report: ReportData,
    ) -> Path:
        """
        Export report to PDF.
        """

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        pdf_path = (
            self.output_directory
            / f"report_{timestamp}.pdf"
        )

        document = SimpleDocTemplate(
            str(pdf_path),
            title="StatVerify AI Report",
            author="StatVerify AI",
        )

        builder = ReportBuilder()

        elements = builder.build(report)

        document.build(elements)

        return pdf_path


# ==========================================================
# Report Engine
# ==========================================================


class ReportEngine:
    """
    High-level report generation interface.

    Used by workflow.py.
    """

    def __init__(self):
        self.exporter = PDFExporter()

    # ------------------------------------------------------

    def generate(
        self,
        report: ReportData | dict,
    ) -> Path:
        """
        Generate the final PDF report.

        Returns
        -------
        pathlib.Path
            Path of generated PDF.
        """
        if isinstance(report, dict):
            req = report.get("request")
            stat = report.get("statistical_result")
            ver = report.get("verification_result")
            ai = report.get("ai_response")
            report_obj = report.get("report")
            if report_obj is None:
                report_obj = ReportData(
                    request=req,
                    statistics=stat or StatisticalResult(test_name="Unknown"),
                    verification=ver,
                    ai=ai,
                )
            report = report_obj

        return self.exporter.export(report)


# ==========================================================
# Singleton
# ==========================================================

report_engine = ReportEngine()