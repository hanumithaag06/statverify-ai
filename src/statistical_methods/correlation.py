"""
correlation.py

Correlation statistical methods.

Contains:
    • Pearson Correlation
    • Spearman Correlation
    • Kendall Tau Correlation

Input:
    StatisticalDataset

Output:
    StatisticalResult

No dataframe manipulation occurs here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.config import settings
from src.models import (
    StatisticalDataset,
    StatisticalResult,
)


class CorrelationMethods:
    """
    Correlation statistical methods.
    """

    # =====================================================
    # Shared Validation
    # =====================================================

    @staticmethod
    def _validate(dataset: StatisticalDataset):

        x = np.asarray(dataset.x, dtype=float)
        y = np.asarray(dataset.y, dtype=float)

        if len(x) != len(y):
            return None, None, "Variables must contain the same number of observations."

        if len(x) < 3:
            return None, None, "At least three paired observations are required."

        return x, y, None

    # =====================================================
    # Pearson Correlation
    # =====================================================

    @staticmethod
    def pearson(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:

        x, y, error = CorrelationMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Pearson Correlation",
                interpretation=error,
            )

        statistic, p = pearsonr(x, y)

        interpretation = (
            "Statistically significant linear relationship."
            if p < settings.significance_level
            else "No statistically significant linear relationship."
        )

        return StatisticalResult(
            test_name="Pearson Correlation",
            statistic=float(statistic),
            p_value=float(p),
            interpretation=interpretation,
            additional_metrics={
                "sample_size": len(x),
                "correlation_type": "Pearson",
            },
        )

    # =====================================================
    # Spearman Correlation
    # =====================================================

    @staticmethod
    def spearman(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:

        x, y, error = CorrelationMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Spearman Correlation",
                interpretation=error,
            )

        statistic, p = spearmanr(x, y)

        interpretation = (
            "Statistically significant monotonic relationship."
            if p < settings.significance_level
            else "No statistically significant monotonic relationship."
        )

        return StatisticalResult(
            test_name="Spearman Correlation",
            statistic=float(statistic),
            p_value=float(p),
            interpretation=interpretation,
            additional_metrics={
                "sample_size": len(x),
                "correlation_type": "Spearman",
            },
        )

    # =====================================================
    # Kendall Tau
    # =====================================================

    @staticmethod
    def kendall(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:

        x, y, error = CorrelationMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Kendall Tau",
                interpretation=error,
            )

        statistic, p = kendalltau(x, y)

        interpretation = (
            "Statistically significant ordinal association."
            if p < settings.significance_level
            else "No statistically significant ordinal association."
        )

        return StatisticalResult(
            test_name="Kendall Tau",
            statistic=float(statistic),
            p_value=float(p),
            interpretation=interpretation,
            additional_metrics={
                "sample_size": len(x),
                "correlation_type": "Kendall Tau",
            },
        )