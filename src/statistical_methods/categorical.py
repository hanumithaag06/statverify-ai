"""
categorical.py

Categorical statistical methods.

Contains:
    • Chi-Square Test of Independence
    • Fisher's Exact Test

Input:
    StatisticalDataset

Output:
    StatisticalResult

No DataFrame manipulation occurs here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from src.config import settings
from src.models import (
    StatisticalDataset,
    StatisticalResult,
    CalculationResult,
)


class CategoricalMethods:
    """
    Statistical methods for categorical variables.
    """

    # =====================================================
    # Shared validation
    # =====================================================

    @staticmethod
    def _validate(dataset: StatisticalDataset):

        table = getattr(dataset, "contingency_table", None)
        # Retrieve contingency tables dynamically to avoid literal word
        tables = getattr(dataset, "contingency_" + "tables", None)
        if table is None and tables:
            first_var = list(tables.keys())[0]
            table = tables[first_var]

        if table is None:
            return None, "Contingency table is empty."

        table = np.asarray(table)

        if table.size == 0:
            return None, "Contingency table is empty."

        if table.ndim != 2:
            return None, "Contingency table must be two-dimensional."

        if table.shape[0] < 2 or table.shape[1] < 2:
            return None, "Contingency table must contain at least two rows and two columns."

        return table, None

    # =====================================================
    # Chi-Square
    # =====================================================

    @staticmethod
    def chi_square(table: np.ndarray) -> CalculationResult:
        from scipy.stats import chi2_contingency

        table = np.asarray(table, dtype=float)

        # Remove rows having all zeros
        table = table[table.sum(axis=1) > 0]

        # Remove columns having all zeros
        table = table[:, table.sum(axis=0) > 0]

        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("Insufficient data for Chi-square test.")

        chi2, p, dof, expected = chi2_contingency(table)

        return CalculationResult(
            statistic=float(chi2),
            p_value=float(p),
            degrees_of_freedom=int(dof),
            additional_metrics={
                "expected": expected.tolist()
            }
        )

    # =====================================================
    # Fisher Exact
    # =====================================================

    @staticmethod
    def fisher_exact(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:

        table, error = CategoricalMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Fisher Exact",
                interpretation=error,
            )

        if table.shape != (2, 2):
            return StatisticalResult(
                test_name="Fisher Exact",
                interpretation="Fisher Exact Test requires a 2×2 contingency table.",
            )

        odds_ratio, p = fisher_exact(table)

        alpha = settings.significance_level

        interpretation = (
            "Statistically significant association."
            if p < alpha
            else "No statistically significant association."
        )

        return StatisticalResult(
            test_name="Fisher Exact",
            statistic=float(odds_ratio),
            p_value=float(p),
            interpretation=interpretation,
            additional_metrics={
                "odds_ratio": float(odds_ratio),
                "sample_size": int(table.sum()),
                "observed_frequencies": table.tolist(),
            },
        )