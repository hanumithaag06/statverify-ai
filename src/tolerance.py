"""
tolerance.py

Tolerance policies used by the verification engine.

Different statistical values require different acceptable error
margins because of floating-point precision and rounding in
published research papers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TolerancePolicy:
    """
    Global tolerance configuration.

    Values represent the maximum acceptable absolute
    difference between reported and calculated statistics.
    """

    p_value: float = 0.005
    statistic: float = 0.01
    effect_size: float = 0.02
    confidence_interval: float = 0.05


DEFAULT_TOLERANCE = TolerancePolicy()


class ToleranceChecker:
    """
    Utility methods for numerical tolerance checking.
    """

    @staticmethod
    def within(
        reported: float | None,
        calculated: float | None,
        tolerance: float,
    ) -> bool:
        """
        Returns True if two values are within tolerance.
        """

        if reported is None or calculated is None:
            return False

        return abs(reported - calculated) <= tolerance

    @staticmethod
    def difference(
        reported: float | None,
        calculated: float | None,
    ) -> float | None:
        """
        Returns absolute difference between two values.
        """

        if reported is None or calculated is None:
            return None

        return abs(reported - calculated)

    @staticmethod
    def is_close(
        reported: float,
        calculated: float,
        tolerance: float = 0.01,
        relative_threshold: float = 0.01,
    ) -> bool:
        """
        Determine if two floats are close.

        Checks both absolute and relative tolerance.
        """

        # Absolute tolerance check
        abs_diff = abs(reported - calculated)
        if abs_diff <= tolerance:
            return True

        # Relative tolerance check (avoid division by zero)
        if reported != 0 and calculated != 0:
            relative_diff = abs_diff / max(abs(reported), abs(calculated))
            if relative_diff <= relative_threshold:
                return True

        return False