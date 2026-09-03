"""
Validation engine for parsed research tables.

The validator never modifies data.

Its responsibility is to inspect the parsed table
and generate structured validation messages.

Errors   → stop the workflow.
Warnings → allow the workflow to continue with a note.
Info     → improve transparency.

Quality checks implemented
--------------------------
✓ Empty table (rows / columns)
✓ Minimum sample size (< 5 rows)
✓ Column-level identifier detection
✓ Constant columns
✓ Missing value ratio per column
✓ Duplicate rows
✓ Outlier detection (IQR-based) for numeric columns
✓ Data type consistency (numeric expected but text found)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import settings
from src.models import (
    ParsedTable,
    ValidationMessage,
    ValidationReport,
    ValidationSeverity,
    VariableType,
)


class DataValidator:
    """
    Validates parsed research tables.
    """

    # Minimum rows needed for any meaningful analysis
    MIN_SAMPLE_SIZE = 5

    def validate(
        self,
        table: ParsedTable,
    ) -> ValidationReport:

        report = ValidationReport()

        self._validate_table(table, report)
        self._validate_minimum_sample(table, report)
        self._validate_columns(table, report)
        self._validate_missing(table, report)
        self._validate_duplicates(table, report)
        self._validate_type_consistency(table, report)
        self._validate_outliers(table, report)

        return report

    # -------------------------------------------------
    # Table-level checks
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
    # Minimum sample size
    # -------------------------------------------------

    def _validate_minimum_sample(
        self,
        table: ParsedTable,
        report: ValidationReport,
    ) -> None:
        """
        Warn when the dataset is very small.
        Error when it is below the absolute minimum for any test.
        """

        n = table.rows

        if 0 < n < self.MIN_SAMPLE_SIZE:
            report.messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Dataset has only {n} rows. "
                        f"A minimum of {self.MIN_SAMPLE_SIZE} observations "
                        f"is required for reliable statistical analysis."
                    ),
                )
            )
        elif n < 30:
            report.messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Small sample size ({n} rows). "
                        f"Results may have low statistical power. "
                        f"Consider non-parametric alternatives if normality is in doubt."
                    ),
                )
            )
        else:
            report.messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    message=f"Sample size is adequate ({n} rows).",
                )
            )

    # -------------------------------------------------
    # Column-level checks
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
                        message=(
                            f"'{column.name}' appears to be an identifier column "
                            f"(high unique ratio). It will be excluded from analysis."
                        ),
                    )
                )

            if column.is_constant:

                report.messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        column=column.name,
                        message=(
                            f"'{column.name}' is a constant column "
                            f"(all values are identical). "
                            f"It cannot contribute statistical information."
                        ),
                    )
                )

    # -------------------------------------------------
    # Missing value ratio
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
                            f"'{column.name}' has {ratio:.0%} missing values "
                            f"({column.missing_values}/{column.total_values}). "
                            f"Threshold is {settings.max_missing_ratio:.0%}."
                        ),
                    )
                )

    # -------------------------------------------------
    # Duplicate rows
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
                    message=(
                        f"{duplicates} duplicate row(s) detected. "
                        f"Duplicate rows can inflate test statistics."
                    ),
                )
            )

    # -------------------------------------------------
    # Data type consistency
    # -------------------------------------------------

    def _validate_type_consistency(
        self,
        table: ParsedTable,
        report: ValidationReport,
    ) -> None:
        """
        Detect columns where the inferred type is numeric but
        a significant portion of values could not be parsed as numbers.
        """

        df = table.dataframe

        for column in table.metadata:

            if column.variable_type not in (
                VariableType.CONTINUOUS,
                VariableType.ORDINAL,
            ):
                continue

            series = df[column.name]
            non_null = series.dropna()

            if len(non_null) == 0:
                continue

            n_numeric = pd.to_numeric(non_null, errors="coerce").notna().sum()
            n_text = len(non_null) - n_numeric

            if n_text > 0:
                text_ratio = n_text / len(non_null)
                if text_ratio > 0.05:  # > 5% non-numeric values
                    report.messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.WARNING,
                            column=column.name,
                            message=(
                                f"'{column.name}' is classified as numeric but "
                                f"{n_text} value(s) ({text_ratio:.0%}) could not be "
                                f"parsed as numbers. Check for mixed data types."
                            ),
                        )
                    )

    # -------------------------------------------------
    # Outlier detection (IQR-based)
    # -------------------------------------------------

    def _validate_outliers(
        self,
        table: ParsedTable,
        report: ValidationReport,
    ) -> None:
        """
        IQR-based outlier detection for numeric columns.

        Values outside Q1 − 3×IQR or Q3 + 3×IQR (extreme outliers)
        trigger a WARNING.
        """

        # Skip outlier checks for summary tables — values are counts/percentages
        if bool(table.summary_metadata):
            return

        df = table.dataframe

        for column in table.metadata:

            if column.variable_type not in (
                VariableType.CONTINUOUS,
                VariableType.ORDINAL,
            ):
                continue

            series = pd.to_numeric(df[column.name], errors="coerce").dropna()

            if len(series) < 5:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            if iqr == 0:
                continue

            # Extreme outlier fence: 3 × IQR (Tukey)
            lower_fence = q1 - 3 * iqr
            upper_fence = q3 + 3 * iqr

            n_outliers = int(
                ((series < lower_fence) | (series > upper_fence)).sum()
            )

            if n_outliers > 0:
                report.messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        column=column.name,
                        message=(
                            f"'{column.name}' has {n_outliers} extreme outlier(s) "
                            f"(outside [{lower_fence:.2f}, {upper_fence:.2f}] "
                            f"using 3×IQR fence). "
                            f"Outliers can distort parametric test results."
                        ),
                    )
                )