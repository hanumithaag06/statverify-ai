"""
comparison.py

Comparison Engine.

Compares the statistical values reported in a research paper
against the values calculated by the Statistical Engine.

This module performs comparison only.
It does NOT perform any statistical calculations.
"""

from __future__ import annotations

from src.models import (
    ComparisonItem,
    ComparisonResult,
    StatisticalResult,
)
from src.tolerance import ToleranceChecker


class ComparisonEngine:
    """
    Compare reported statistical values with calculated values.
    """

    def __init__(self) -> None:
        self.tolerance = ToleranceChecker()

    # ---------------------------------------------------------

    def compare(
        self,
        reported: StatisticalResult,
        calculated: StatisticalResult,
    ) -> ComparisonResult:
        """
        Compare two StatisticalResult objects.
        """

        items = [
            self._compare_metric(
                "Statistic",
                reported.statistic,
                calculated.statistic,
            ),
            self._compare_metric(
                "P Value",
                reported.p_value,
                calculated.p_value,
            ),
            self._compare_metric(
                "Effect Size",
                reported.effect_size,
                calculated.effect_size,
            ),
            self._compare_metric(
                "Degrees of Freedom",
                reported.degrees_of_freedom,
                calculated.degrees_of_freedom,
            ),
        ]

        passed = all(item.matched for item in items)

        summary = (
            "All reported statistics match the calculated values."
            if passed
            else "Differences detected between reported and calculated values."
        )

        return ComparisonResult(
            passed=passed,
            items=items,
            summary=summary,
        )

    # ---------------------------------------------------------

    def _compare_metric(
        self,
        metric: str,
        reported: float | int | None,
        calculated: float | int | None,
    ) -> ComparisonItem:
        """
        Compare one statistical metric.
        """

        # Both missing → treat as matching
        if reported is None and calculated is None:
            return ComparisonItem(
                metric=metric,
                reported=None,
                calculated=None,
                matched=True,
                difference=None,
            )

        # One missing → mismatch
        if reported is None or calculated is None:
            return ComparisonItem(
                metric=metric,
                reported=reported,
                calculated=calculated,
                matched=False,
                difference=None,
            )

        reported_value = float(reported)
        calculated_value = float(calculated)

        matched = self.tolerance.is_close(
            reported_value,
            calculated_value,
        )

        difference = abs(
            reported_value - calculated_value
        )

        return ComparisonItem(
            metric=metric,
            reported=reported_value,
            calculated=calculated_value,
            matched=matched,
            difference=difference,
        )