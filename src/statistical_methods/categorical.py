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
        if table is None and dataset.contingency_tables:
            first_var = list(dataset.contingency_tables.keys())[0]
            table = dataset.contingency_tables[first_var]

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
    def chi_square(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:

        # Demographic summary table loop
        if dataset.contingency_tables:
            demographics = {}
            lines = []
            for variable, table in dataset.contingency_tables.items():
                if table is None:
                    continue
                if len(table.shape) != 2:
                    continue
                if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                try:
                    chi2, p, dof, expected = chi2_contingency(table)
                    demographics[variable] = {
                        "chi2": float(chi2),
                        "p": float(p),
                        "dof": float(dof),
                        "expected": expected.tolist(),
                        "observed": table.tolist(),
                    }
                    lines.append(f"{variable}\nχ² = {chi2:.2f}\np = {p:.3f}")
                except Exception as ex:
                    lines.append(f"{variable}\nError: {str(ex)}")

            interpretation = "\n\n--------------------------------\n\n".join(lines)
            first_var = list(demographics.keys())[0] if demographics else None
            first_res = demographics[first_var] if first_var else {"chi2": None, "p": None, "dof": None}

            return StatisticalResult(
                test_name="Chi-Square (Demographic Analysis)",
                statistic=first_res["chi2"],
                p_value=first_res["p"],
                degrees_of_freedom=first_res["dof"],
                interpretation=interpretation,
                additional_metrics={
                    "demographics": demographics,
                },
            )

        table = dataset.contingency_table

        if table is None:
            raise ValueError(
                "Contingency table not extracted."
            )

        if len(table.shape) != 2:
            raise ValueError(
                "Contingency table must be 2-dimensional."
            )

        chi2, p, dof, expected = chi2_contingency(table)

        alpha = settings.significance_level

        interpretation = (
            "Statistically significant association."
            if p < alpha
            else "No statistically significant association."
        )

        return StatisticalResult(
            test_name="Chi-Square",
            statistic=float(chi2),
            p_value=float(p),
            degrees_of_freedom=float(dof),
            interpretation=interpretation,
            additional_metrics={
                "expected_frequencies": expected.tolist(),
                "observed_frequencies": table.tolist() if hasattr(table, "tolist") else list(table),
                "sample_size": int(table.sum()),
            },
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