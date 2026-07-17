from __future__ import annotations

from src.comparison import ComparisonEngine
from src.models import (
    StatisticalResult,
    VerificationResult,
)


class VerificationEngine:
    """
    Verification Engine.

    Compares reported statistical results with
    calculated statistical results.
    """

    def __init__(self) -> None:
        self.comparison_engine = ComparisonEngine()

    def verify(
        self,
        reported: StatisticalResult,
        calculated: StatisticalResult,
    ) -> VerificationResult:
        """
        Verify reported statistics against calculated statistics.
        """

        comparison = self.comparison_engine.compare(
            reported,
            calculated,
        )

        if comparison.passed:
            message = (
                "Verification successful. "
                "All reported statistical values match the calculated values."
            )
        else:
            message = (
                "Verification failed. "
                "Differences were detected between the reported and calculated values."
            )

        return VerificationResult(
            verified=comparison.passed,
            comparison=comparison,
            reported_result=reported,
            calculated_result=calculated,
            message=message,
        )