"""
Statistical assumption checker.

This module determines whether the recommended
statistical test satisfies all required assumptions.

It NEVER performs statistical calculations.
"""

from __future__ import annotations

from src.models import (
    StatisticalDataset,
    TestRecommendation,
    AssumptionResult,
    AssumptionStatus,
)


class AssumptionChecker:
    """
    Performs generic assumption checks before
    executing a statistical test.
    """

    def check(
        self,
        dataset: StatisticalDataset,
        recommendation: TestRecommendation,
    ) -> list[AssumptionResult]:

        results = []

        # ----------------------------
        # Sample size
        # ----------------------------

        if dataset.sample_size < 3:

            results.append(
                AssumptionResult(
                    name="Sample Size",
                    status=AssumptionStatus.FAIL,
                    message=f"Dataset contains only {dataset.sample_size} observations.",
                )
            )

        else:

            results.append(
                AssumptionResult(
                    name="Sample Size",
                    status=AssumptionStatus.PASS,
                    message=f"{dataset.sample_size} observations available.",
                )
            )

        # ----------------------------
        # Missing values
        # ----------------------------

        missing = int(dataset.dataframe.isna().sum().sum())

        if missing:

            results.append(
                AssumptionResult(
                    name="Missing Data",
                    status=AssumptionStatus.WARNING,
                    message=f"{missing} missing values remain.",
                )
            )

        else:

            results.append(
                AssumptionResult(
                    name="Missing Data",
                    status=AssumptionStatus.PASS,
                    message="No missing values.",
                )
            )

        # ----------------------------
        # Group check
        # ----------------------------

        if dataset.groups:

            if len(dataset.groups) == 2:

                results.append(
                    AssumptionResult(
                        name="Groups",
                        status=AssumptionStatus.PASS,
                        message="Two groups detected.",
                    )
                )

            else:

                results.append(
                    AssumptionResult(
                        name="Groups",
                        status=AssumptionStatus.WARNING,
                        message=f"{len(dataset.groups)} groups detected.",
                    )
                )

        return results