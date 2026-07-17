"""
Parametric statistical methods.

Contains:
    • Independent t-test
    • Paired t-test (future)
    • One-way ANOVA (future)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind

from src.config import settings
from src.models import (
    StatisticalDataset,
    StatisticalResult,
)


class ParametricMethods:
    """
    Parametric statistical methods.
    """

    @staticmethod
    def independent_t_test(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Welch Independent Two-Sample t-test.
        """

        if len(dataset.groups) != 2:
            return StatisticalResult(
                test_name="Independent t-test",
                interpretation="Exactly two groups are required.",
            )

        names = list(dataset.groups.keys())

        g1 = np.asarray(dataset.groups[names[0]], dtype=float)
        g2 = np.asarray(dataset.groups[names[1]], dtype=float)

        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) < 2 or len(g2) < 2:
            return StatisticalResult(
                test_name="Independent t-test",
                interpretation="Insufficient observations.",
            )

        result = ttest_ind(
            g1,
            g2,
            equal_var=False,
        )

        alpha = settings.significance_level

        interpretation = (
            "Statistically significant difference."
            if result.pvalue < alpha
            else "No statistically significant difference."
        )

        return StatisticalResult(
            test_name="Independent t-test",
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            interpretation=interpretation,
            additional_metrics={
                "group_1": names[0],
                "group_2": names[1],
                "group1_mean": float(np.mean(g1)),
                "group2_mean": float(np.mean(g2)),
                "group1_std": float(np.std(g1, ddof=1)),
                "group2_std": float(np.std(g2, ddof=1)),
                "group1_n": int(len(g1)),
                "group2_n": int(len(g2)),
            },
        )

    # --------------------------------------------------

    @staticmethod
    def paired_t_test(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Reserved for future implementation.
        """

        return StatisticalResult(
            test_name="Paired t-test",
            interpretation="Not implemented.",
        )

    # --------------------------------------------------

    @staticmethod
    def one_way_anova(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Reserved for future implementation.
        """

        return StatisticalResult(
            test_name="One-way ANOVA",
            interpretation="Not implemented.",
        )