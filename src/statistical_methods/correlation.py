"""
correlation.py

Correlation statistical methods.

Contains:
    • Pearson Correlation  (with 95% CI via Fisher Z, R²)
    • Spearman Correlation (with 95% CI via Fisher Z approximation)
    • Kendall Tau Correlation

Input:
    StatisticalDataset

Output:
    StatisticalResult

No dataframe manipulation occurs here.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr, norm

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

        # Remove pairs where either is NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

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
        """
        Pearson Product-Moment Correlation.

        Computes:
        - r (correlation coefficient)
        - p-value (two-tailed)
        - R² (coefficient of determination)
        - 95% CI via Fisher Z transformation
        """

        x, y, error = CorrelationMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Pearson Correlation",
                interpretation=error,
            )

        n = len(x)
        r, p = pearsonr(x, y)
        r = float(r)
        p = float(p)

        # Fisher Z 95% CI
        ci_lower, ci_upper = _fisher_z_ci(r, n, settings.significance_level)

        r_squared = r ** 2
        significant = bool(p < settings.significance_level)

        interpretation = (
            f"Statistically significant linear relationship "
            f"(r={r:.3f}, p={p:.4f}, R²={r_squared:.3f})."
            if significant
            else f"No statistically significant linear relationship "
            f"(r={r:.3f}, p={p:.4f}, R²={r_squared:.3f})."
        )

        return StatisticalResult(
            test_name="Pearson Correlation",
            statistic=round(r, 4),
            p_value=round(p, 4),
            degrees_of_freedom=float(n - 2),
            effect_size=round(r_squared, 4),
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            interpretation=interpretation,
            additional_metrics={
                "r": round(r, 4),
                "r_squared": round(r_squared, 4),
                "r_magnitude": _r_label(r),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "sample_size": n,
                "correlation_type": "Pearson",
                "significant": significant,
                "alpha": settings.significance_level,
            },
        )

    # =====================================================
    # Spearman Correlation
    # =====================================================

    @staticmethod
    def spearman(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Spearman Rank Correlation.

        Computes:
        - ρ (rho) and p-value
        - 95% CI via Fisher Z approximation
        - ρ² as effect size
        """

        x, y, error = CorrelationMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Spearman Correlation",
                interpretation=error,
            )

        n = len(x)
        rho, p = spearmanr(x, y)
        rho = float(rho)
        p = float(p)

        # Fisher Z CI (approximate — valid for n ≥ 10)
        ci_lower, ci_upper = _fisher_z_ci(rho, n, settings.significance_level)
        rho_squared = rho ** 2
        significant = bool(p < settings.significance_level)

        interpretation = (
            f"Statistically significant monotonic relationship "
            f"(ρ={rho:.3f}, p={p:.4f})."
            if significant
            else f"No statistically significant monotonic relationship "
            f"(ρ={rho:.3f}, p={p:.4f})."
        )

        return StatisticalResult(
            test_name="Spearman Correlation",
            statistic=round(rho, 4),
            p_value=round(p, 4),
            degrees_of_freedom=float(n - 2),
            effect_size=round(rho_squared, 4),
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            interpretation=interpretation,
            additional_metrics={
                "rho": round(rho, 4),
                "rho_squared": round(rho_squared, 4),
                "rho_magnitude": _r_label(rho),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "sample_size": n,
                "correlation_type": "Spearman",
                "significant": significant,
                "alpha": settings.significance_level,
            },
        )

    # =====================================================
    # Kendall Tau
    # =====================================================

    @staticmethod
    def kendall(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Kendall Tau-b Correlation.

        Computes:
        - τ (tau) and p-value
        - τ² as effect size
        """

        x, y, error = CorrelationMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Kendall Tau",
                interpretation=error,
            )

        n = len(x)
        tau, p = kendalltau(x, y)
        tau = float(tau)
        p = float(p)

        tau_squared = tau ** 2
        significant = bool(p < settings.significance_level)

        interpretation = (
            f"Statistically significant ordinal association "
            f"(τ={tau:.3f}, p={p:.4f})."
            if significant
            else f"No statistically significant ordinal association "
            f"(τ={tau:.3f}, p={p:.4f})."
        )

        return StatisticalResult(
            test_name="Kendall Tau",
            statistic=round(tau, 4),
            p_value=round(p, 4),
            degrees_of_freedom=None,
            effect_size=round(tau_squared, 4),
            interpretation=interpretation,
            additional_metrics={
                "tau": round(tau, 4),
                "tau_squared": round(tau_squared, 4),
                "tau_magnitude": _r_label(tau),
                "sample_size": n,
                "correlation_type": "Kendall Tau",
                "significant": significant,
                "alpha": settings.significance_level,
            },
        )


# =====================================================
# Shared Helpers
# =====================================================

def _fisher_z_ci(r: float, n: int, alpha: float) -> tuple[float, float]:
    """
    Compute 95% CI for a correlation via Fisher Z transformation.

    Parameters
    ----------
    r : correlation coefficient
    n : sample size
    alpha : significance level (e.g. 0.05)
    """
    # Clip r to avoid atanh domain error
    r = max(-0.9999, min(0.9999, r))
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3) if n > 3 else 1.0
    z_crit = float(norm.ppf(1 - alpha / 2))
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    return math.tanh(z_lo), math.tanh(z_hi)


def _r_label(r: float) -> str:
    """Magnitude label for correlation coefficients (Cohen 1988)."""
    r = abs(r)
    if r < 0.10:
        return "negligible"
    if r < 0.30:
        return "small"
    if r < 0.50:
        return "medium"
    return "large"