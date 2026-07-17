"""
Validation engine for parsed research tables.

The validator never modifies data.

Its responsibility is to inspect the parsed table
and generate structured validation messages.

Errors stop the workflow.

Warnings allow the workflow to continue.

Information messages improve transparency.
"""

from __future__ import annotations

from src.config import settings
from src.models import (
    ParsedTable,
    ValidationMessage,
    ValidationReport,
    ValidationSeverity,
)


class DataValidator:
    """
    Validates parsed research tables.
    """

    def validate(
        self,
        table: ParsedTable,
    ) -> ValidationReport:

        report = ValidationReport()

        self._validate_table(table, report)

        self._validate_columns(table, report)

        self._validate_missing(table, report)

        self._validate_duplicates(table, report)

        return report

    # -------------------------------------------------

    def _validate_table(
        self,
        table: ParsedTable,
        report: ValidationReport,
    ) -> None:

        if table.rows == 0:

            report.messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                message="Table contains no rows.",
            )
        )

        if table.columns == 0:

            report.messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    message="Table contains no columns.",
                )
            )   

    # -------------------------------------------------

    def _validate_columns(
        self,
        table: ParsedTable,
        report: ValidationReport,
    ) -> None:

        for column in table.metadata:

            if column.is_identifier:

                report.messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.INFO,
                        column=column.name,
                        message="Identifier column detected.",
                    )
                )

            if column.is_constant:

                report.messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        column=column.name,
                        message="Constant column detected.",
                    )
                )

    # -------------------------------------------------

    def _validate_missing(
        self,
        table: ParsedTable,
        report: ValidationReport,
    ) -> None:

        for column in table.metadata:

            if column.total_values == 0:
                continue

            ratio = (
                column.missing_values /
                column.total_values
            )

            if ratio > settings.max_missing_ratio:

                report.messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        column=column.name,
                        message=(
                            f"{ratio:.0%} missing values."
                        ),
                    )
                )

    # -------------------------------------------------

    def _validate_duplicates(
        self,
        table: ParsedTable,
        report: ValidationReport,
    ) -> None:
        """
        Check for duplicate rows in the parsed dataframe.
    """

        duplicates = int(
            table.dataframe.duplicated().sum()
        )

        if duplicates > 0:
            report.messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    message=f"{duplicates} duplicate row(s) detected.",
                )
            )