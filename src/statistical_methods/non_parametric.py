"""
non_parametric.py

Non-parametric statistical methods.

Contains:
    • Mann-Whitney U Test
    • Wilcoxon Signed-Rank Test
    • Kruskal-Wallis Test
    • Friedman Test

Input:
    StatisticalDataset

Output:
    StatisticalResult

No dataframe manipulation occurs here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import (
    mannwhitneyu,
    wilcoxon,
    kruskal,
    friedmanchisquare,
)

from src.config import settings
from src.models import (
    StatisticalDataset,
    StatisticalResult,
)


class NonParametricMethods:
    """
    Collection of non-parametric statistical methods.
    """

    # =====================================================
    # Shared Helpers
    # =====================================================

    @staticmethod
    def _interpret(p: float) -> str:
        return (
            "Statistically significant difference."
            if p < settings.significance_level
            else "No statistically significant difference."
        )

    # =====================================================
    # Mann-Whitney U
    # =====================================================

    @staticmethod
    def mann_whitney(dataset: StatisticalDataset) -> StatisticalResult:

        if len(dataset.groups) != 2:
            return StatisticalResult(
                test_name="Mann-Whitney U",
                interpretation="Exactly two groups are required.",
            )

        groups = list(dataset.groups.values())

        if len(groups[0]) < 2 or len(groups[1]) < 2:
            return StatisticalResult(
                test_name="Mann-Whitney U",
                interpretation="Each group must contain at least two observations.",
            )

        statistic, p = mannwhitneyu(
            groups[0],
            groups[1],
            alternative="two-sided",
        )

        return StatisticalResult(
            test_name="Mann-Whitney U",
            statistic=float(statistic),
            p_value=float(p),
            interpretation=NonParametricMethods._interpret(p),
            additional_metrics={
                "sample_sizes": [len(groups[0]), len(groups[1])]
            },
        )

    # =====================================================
    # Wilcoxon Signed Rank
    # =====================================================

    @staticmethod
    def wilcoxon(dataset: StatisticalDataset) -> StatisticalResult:

        x = np.asarray(dataset.x, dtype=float)
        y = np.asarray(dataset.y, dtype=float)

        if len(x) != len(y):
            return StatisticalResult(
                test_name="Wilcoxon",
                interpretation="Paired variables must have equal length.",
            )

        if len(x) < 2:
            return StatisticalResult(
                test_name="Wilcoxon",
                interpretation="Insufficient paired observations.",
            )

        statistic, p = wilcoxon(x, y)

        return StatisticalResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=float(statistic),
            p_value=float(p),
            interpretation=NonParametricMethods._interpret(p),
            additional_metrics={
                "sample_size": len(x)
            },
        )

    # =====================================================
    # Kruskal-Wallis
    # =====================================================

    @staticmethod
    def kruskal(dataset: StatisticalDataset) -> StatisticalResult:

        groups = list(dataset.groups.values())

        if len(groups) < 2:
            return StatisticalResult(
                test_name="Kruskal-Wallis",
                interpretation="At least two groups are required.",
            )

        statistic, p = kruskal(*groups)

        return StatisticalResult(
            test_name="Kruskal-Wallis",
            statistic=float(statistic),
            p_value=float(p),
            interpretation=NonParametricMethods._interpret(p),
            additional_metrics={
                "groups": len(groups),
                "sample_sizes": [len(g) for g in groups],
            },
        )

    # =====================================================
    # Friedman Test
    # =====================================================

    @staticmethod
    def friedman(dataset: StatisticalDataset) -> StatisticalResult:

        groups = list(dataset.groups.values())

        if len(groups) < 3:
            return StatisticalResult(
                test_name="Friedman",
                interpretation="At least three paired groups are required.",
            )

        lengths = {len(g) for g in groups}

        if len(lengths) != 1:
            return StatisticalResult(
                test_name="Friedman",
                interpretation="All paired groups must have equal sample size.",
            )

        statistic, p = friedmanchisquare(*groups)

        return StatisticalResult(
            test_name="Friedman",
            statistic=float(statistic),
            p_value=float(p),
            interpretation=NonParametricMethods._interpret(p),
            additional_metrics={
                "groups": len(groups),
                "sample_size": len(groups[0]),
            },
        )