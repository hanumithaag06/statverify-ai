"""
Reporting module.

Creates professional PDF reports for
StatVerify AI.

Report structure
-----------------
1. Header
2. Analysis Information
3. Statistical Result  (+ Effect Size + CI)
4. Assumptions         (pass / warning / fail per assumption)
5. Verification        (reported vs calculated, in VERIFICATION mode)
6. AI Explanation
7. Footer
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models import (
    AIExplanation,
    AssumptionStatus,
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

        if getattr(statistics, "additional_results", None):
            elements = []

            has_breakdown = any(
                r.get("breakdown") for r in statistics.additional_results
            )

            n_tested = statistics.additional_metrics.get("variables_tested", len(statistics.additional_results))
            n_sig    = statistics.additional_metrics.get("significant_variables", 0)
            n_ns     = statistics.additional_metrics.get("non_significant_variables", n_tested - n_sig)

            summary_text = (
                f"<b>{n_tested} demographic variables analysed — "
                f"{n_sig} significant, {n_ns} non-significant.</b>"
            )
            elements.append(Paragraph(summary_text, styles["BodyText"]))
            elements.append(Spacer(1, 8))

            if not has_breakdown:
                # Fallback: no per-category counts available (e.g. group
                # sizes weren't detected from the source headers).
                rows = [["Variable", "χ²", "df", "p-value", "Result"]]

                for row in statistics.additional_results:
                    if "error" in row:
                        rows.append([row.get("variable", "?"), "Error", "-", "-", "–"])
                    else:
                        rows.append([
                            row.get("variable", "?"),
                            f"{row['chi_square']:.3f}" if row.get("chi_square") is not None else "-",
                            str(int(row["dof"])) if row.get("dof") is not None else "-",
                            f"{row['p_value']:.3f}" if row.get("p_value") is not None else "-",
                            "Sig *" if row.get("significant") else "NS",
                        ])

                t = Table(rows, colWidths=[155, 65, 45, 65, 50])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                elements.append(t)
                return elements

            # Full per-category breakdown, matching published-table format:
            # Variable | Sub-variable | Group A No./% | Group B No./% | χ² | df | p-value | Result
            rows = [["Variable", "Sub-variable", "Exp\nNo.", "Exp\n%", "Ctrl\nNo.", "Ctrl\n%", "χ²", "df", "p-value", "Result"]]

            span_commands = []
            highlight_commands = []
            current_row = 1

            for entry in statistics.additional_results:

                if "error" in entry:
                    rows.append([entry.get("variable", "?"), "-", "-", "-", "-", "-", "Error", "-", "-", "–"])
                    current_row += 1
                    continue

                breakdown = entry.get("breakdown") or [{}]
                groups = entry.get("groups", ["Experimental", "Control"])
                group_a = groups[0] if len(groups) > 0 else "Experimental"
                group_b = groups[1] if len(groups) > 1 else "Control"

                start_row = current_row

                for i, cat in enumerate(breakdown):
                    rows.append([
                        entry.get("variable", "?") if i == 0 else "",
                        cat.get("category", ""),
                        str(cat.get(f"{group_a}_n", "-")),
                        str(cat.get(f"{group_a}_pct", "-")),
                        str(cat.get(f"{group_b}_n", "-")),
                        str(cat.get(f"{group_b}_pct", "-")),
                        f"{entry['chi_square']:.3f}" if i == 0 else "",
                        str(entry["dof"]) if i == 0 else "",
                        f"{entry['p_value']:.3f}" if i == 0 else "",
                        ("Sig *" if entry.get("significant") else "NS") if i == 0 else "",
                    ])
                    current_row += 1

                end_row = current_row - 1

                if end_row > start_row:
                    for col in (0, 6, 7, 8, 9):
                        span_commands.append(("SPAN", (col, start_row), (col, end_row)))

                if entry.get("significant"):
                    highlight_commands.append(
                        ("BACKGROUND", (0, start_row), (-1, end_row), colors.lightyellow)
                    )
                    highlight_commands.append(
                        ("TEXTCOLOR", (9, start_row), (9, end_row), colors.darkred)
                    )

            t = Table(
                rows,
                colWidths=[85, 80, 35, 35, 35, 35, 40, 25, 45, 40],
                repeatRows=1,
            )
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        *span_commands,
                        *highlight_commands,
                    ]
                )
            )
            elements.append(t)
            return elements

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

        # ---- Name the key statistic by test type ----
        test_lower = (statistics.test_name or "").lower()
        if "chi" in test_lower:
            stat_name, stat_symbol = "Chi-square (χ²)", "χ²"
        elif "pearson" in test_lower or "spearman" in test_lower or "kendall" in test_lower:
            stat_name, stat_symbol = "Correlation Coefficient", "r / ρ / τ"
        elif "anova" in test_lower or "kruskal" in test_lower or "friedman" in test_lower:
            stat_name, stat_symbol = "Test Statistic (F / H / χ²)", ""
        elif "regression" in test_lower:
            stat_name, stat_symbol = "F Statistic", "F"
        else:
            stat_name, stat_symbol = "Test Statistic (t)", "t"

        def _fmt(value) -> str:
            """Format a numeric value or return em-dash for None."""
            if value is None:
                return "—"
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return str(value)

        # Confidence interval
        ci_text = "—"
        if statistics.confidence_interval:
            lo, hi = statistics.confidence_interval
            ci_text = f"[{_fmt(lo)},  {_fmt(hi)}]"

        rows = [
            ["Metric", "Value"],
            ["Test", statistics.test_name or "—"],
            [stat_name, _fmt(statistics.statistic)],
            ["p-value", _fmt(statistics.p_value)],
            ["Degrees of Freedom", _fmt(statistics.degrees_of_freedom)],
            ["Effect Size", _fmt(statistics.effect_size)],
            ["95% Confidence Interval", ci_text],
            ["Interpretation", statistics.interpretation or "—"],
        ]

        # Add key additional metrics (group means, R², slope, etc.)
        metrics = statistics.additional_metrics or {}
        metric_display = [
            ("group1_mean", "Group 1 Mean"),
            ("group2_mean", "Group 2 Mean"),
            ("mean_difference", "Mean Difference"),
            ("cohens_d", "Cohen's d"),
            ("cohens_d_magnitude", "Effect Magnitude"),
            ("eta_squared", "η² (Eta-Squared)"),
            ("eta_squared_magnitude", "Effect Magnitude"),
            ("r_squared", "R²"),
            ("adjusted_r_squared", "Adjusted R²"),
            ("rmse", "RMSE"),
            ("slope", "Regression Slope"),
            ("intercept", "Intercept"),
            ("rank_biserial_r", "Rank-Biserial r"),
        ]
        for key, label in metric_display:
            val = metrics.get(key)
            if val is not None:
                rows.append([label, str(val)])

        table = Table(
            rows,
            colWidths=[200, 260],
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
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
    def assumptions_section(
        statistics: StatisticalResult,
    ) -> list:
        """
        Assumptions check section.

        Renders a table with PASS / WARNING / FAIL per assumption.
        """

        assumptions = getattr(statistics, "assumptions", None)

        if not assumptions:
            return []

        styles = getSampleStyleSheet()

        elements = [
            Paragraph("<b>Assumption Checks</b>", styles["Heading2"]),
            Spacer(1, 6),
        ]

        rows = [["Assumption", "Status", "Details"]]

        for a in assumptions:
            if a.status == AssumptionStatus.PASS:
                status_text = "✓ PASS"
                row_color = colors.lightgreen
            elif a.status == AssumptionStatus.WARNING:
                status_text = "⚠ WARNING"
                row_color = colors.lightyellow
            else:
                status_text = "✗ FAIL"
                row_color = colors.lightcoral

            rows.append([a.name, status_text, a.message])

        t = Table(rows, colWidths=[130, 70, 260])

        # Build per-row background colors
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkslategray),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("WORDWRAP", (2, 1), (2, -1), True),
        ]

        for idx, a in enumerate(assumptions, start=1):
            if a.status == AssumptionStatus.PASS:
                bg = colors.Color(0.85, 0.95, 0.85)   # soft green
            elif a.status == AssumptionStatus.WARNING:
                bg = colors.Color(1.0, 0.97, 0.80)    # soft yellow
            else:
                bg = colors.Color(1.0, 0.80, 0.80)    # soft red
            style_cmds.append(("BACKGROUND", (0, idx), (-1, idx), bg))

        t.setStyle(TableStyle(style_cmds))
        elements.append(t)

        return elements

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

        elements.append(Spacer(1, 14))

        # --------------------------------------------------
        # Assumption Checks
        # --------------------------------------------------

        assumption_elements = self.formatter.assumptions_section(
            report.statistics
        )
        if assumption_elements:
            elements.extend(assumption_elements)
            elements.append(Spacer(1, 14))

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