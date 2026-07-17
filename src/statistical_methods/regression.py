"""
regression.py

Regression statistical methods.

Contains:
    • Simple Linear Regression
    • Logistic Regression (placeholder)

Input:
    StatisticalDataset

Output:
    StatisticalResult

No dataframe manipulation occurs here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress

from src.config import settings
from src.models import (
    StatisticalDataset,
    StatisticalResult,
)


class RegressionMethods:
    """
    Collection of regression methods.
    """

    # ==========================================================
    # Shared Validation
    # ==========================================================

    @staticmethod
    def _validate(dataset: StatisticalDataset):

        x = np.asarray(dataset.x, dtype=float)
        y = np.asarray(dataset.y, dtype=float)

        if len(x) != len(y):
            return None, None, "Variables must have equal observations."

        if len(x) < 3:
            return None, None, "At least three observations are required."

        return x, y, None

    # ==========================================================
    # Linear Regression
    # ==========================================================

    @staticmethod
    def linear_regression(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:

        x, y, error = RegressionMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Linear Regression",
                interpretation=error,
            )

        result = linregress(x, y)

        interpretation = (
            "Regression model is statistically significant."
            if result.pvalue < settings.significance_level
            else "Regression model is not statistically significant."
        )

        return StatisticalResult(
            test_name="Linear Regression",
            statistic=float(result.slope),
            p_value=float(result.pvalue),
            interpretation=interpretation,
            additional_metrics={
                "slope": float(result.slope),
                "intercept": float(result.intercept),
                "r_value": float(result.rvalue),
                "r_squared": float(result.rvalue ** 2),
                "standard_error": float(result.stderr),
                "sample_size": len(x),
            },
        )

    # ==========================================================
    # Logistic Regression
    # ==========================================================

    @staticmethod
    def logistic_regression(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Binary Logistic Regression.

        Placeholder.

        Future implementation will use
        statsmodels.Logit for coefficient estimation,
        confidence intervals,
        odds ratios,
        Wald statistics,
        and prediction probabilities.
        """

        x, y, error = RegressionMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Logistic Regression",
                interpretation=error,
            )

        return StatisticalResult(
            test_name="Logistic Regression",
            interpretation=(
                "Logistic Regression will be implemented "
                "using StatsModels Logit in Phase 6."
            ),
            additional_metrics={
                "sample_size": len(x),
            },
        )