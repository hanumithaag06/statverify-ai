"""
regression.py

Regression statistical methods.

Contains:
    • Simple Linear Regression  (OLS via StatsModels / SciPy, slope CI, adjusted R², RMSE, F-stat)
    • Binary Logistic Regression (statsmodels Logit, Odds Ratios with 95% CI, McFadden Pseudo R²)

Input:
    StatisticalDataset

Output:
    StatisticalResult

No dataframe manipulation occurs here.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import linregress, t as t_dist

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from src.config import settings
from src.models import (
    StatisticalDataset,
    StatisticalResult,
)


class RegressionMethods:
    """
    Collection of regression methods using SciPy and StatsModels.
    """

    # ==========================================================
    # Shared Validation
    # ==========================================================

    @staticmethod
    def _validate(dataset: StatisticalDataset):

        if dataset.x is None or dataset.y is None:
            return None, None, "Regression requires numeric x and y variables."

        x = np.asarray(dataset.x, dtype=float)
        y = np.asarray(dataset.y, dtype=float)

        # Remove pairs where either is NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) != len(y):
            return None, None, "Variables must have equal observations."

        if len(x) < 3:
            return None, None, "At least three observations are required."

        return x, y, None

    # ==========================================================
    # Simple Linear Regression
    # ==========================================================

    @staticmethod
    def linear_regression(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Ordinary Least Squares Simple Linear Regression.

        Computes:
        - slope, intercept, r, R², adjusted R²
        - F-statistic and model p-value
        - 95% confidence interval on slope
        - Standard error of slope
        - RMSE (root mean squared error)
        """

        x, y, error = RegressionMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Linear Regression",
                interpretation=error,
            )

        n = len(x)
        alpha = settings.significance_level

        if HAS_STATSMODELS:
            try:
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()

                intercept, slope = float(model.params[0]), float(model.params[1])
                se_slope = float(model.bse[1])
                f_stat = float(model.fvalue)
                p = float(model.f_pvalue)
                r_squared = float(model.rsquared)
                adj_r_squared = float(model.rsquared_adj)
                r = math.sqrt(r_squared) * math.copysign(1.0, slope)

                conf_int = model.conf_int(alpha=alpha)
                slope_ci_lower, slope_ci_upper = float(conf_int[1, 0]), float(conf_int[1, 1])
                rmse = float(math.sqrt(model.mse_resid))

            except Exception:
                # Fallback to SciPy linregress
                return RegressionMethods._scipy_linear_regression(x, y, n, alpha)
        else:
            return RegressionMethods._scipy_linear_regression(x, y, n, alpha)

        significant = bool(p < alpha)
        interpretation = (
            f"Regression model is statistically significant "
            f"(F={f_stat:.3f}, p={p:.4f}, R²={r_squared:.3f}, slope={slope:.4f})."
            if significant
            else f"Regression model is not statistically significant "
            f"(F={f_stat:.3f}, p={p:.4f}, R²={r_squared:.3f}, slope={slope:.4f})."
        )

        return StatisticalResult(
            test_name="Linear Regression",
            statistic=round(f_stat, 4),
            p_value=round(p, 4),
            degrees_of_freedom=float(n - 2),
            effect_size=round(r_squared, 4),
            confidence_interval=(round(slope_ci_lower, 4), round(slope_ci_upper, 4)),
            interpretation=interpretation,
            additional_metrics={
                "slope": round(slope, 4),
                "intercept": round(intercept, 4),
                "r": round(r, 4),
                "r_squared": round(r_squared, 4),
                "adjusted_r_squared": round(adj_r_squared, 4),
                "f_statistic": round(f_stat, 4),
                "rmse": round(rmse, 4),
                "standard_error_slope": round(se_slope, 4),
                "slope_ci_95_lower": round(slope_ci_lower, 4),
                "slope_ci_95_upper": round(slope_ci_upper, 4),
                "sample_size": n,
                "significant": significant,
                "alpha": alpha,
                "engine": "StatsModels OLS" if HAS_STATSMODELS else "SciPy",
            },
        )

    @staticmethod
    def _scipy_linear_regression(x: np.ndarray, y: np.ndarray, n: int, alpha: float) -> StatisticalResult:
        result = linregress(x, y)
        slope = float(result.slope)
        intercept = float(result.intercept)
        r = float(result.rvalue)
        p = float(result.pvalue)
        se_slope = float(result.stderr)

        r_squared = r ** 2
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else r_squared
        f_stat = (r_squared / (1 - r_squared)) * (n - 2) if r_squared < 1.0 else float("inf")

        y_pred = slope * x + intercept
        residuals = y - y_pred
        rmse = float(math.sqrt(np.mean(residuals ** 2)))

        t_crit = float(t_dist.ppf(1 - alpha / 2, df=n - 2))
        slope_ci_lower = slope - t_crit * se_slope
        slope_ci_upper = slope + t_crit * se_slope

        significant = bool(p < alpha)
        interpretation = (
            f"Regression model is statistically significant "
            f"(F={f_stat:.3f}, p={p:.4f}, R²={r_squared:.3f}, slope={slope:.4f})."
            if significant
            else f"Regression model is not statistically significant "
            f"(F={f_stat:.3f}, p={p:.4f}, R²={r_squared:.3f}, slope={slope:.4f})."
        )

        return StatisticalResult(
            test_name="Linear Regression",
            statistic=round(f_stat, 4),
            p_value=round(p, 4),
            degrees_of_freedom=float(n - 2),
            effect_size=round(r_squared, 4),
            confidence_interval=(round(slope_ci_lower, 4), round(slope_ci_upper, 4)),
            interpretation=interpretation,
            additional_metrics={
                "slope": round(slope, 4),
                "intercept": round(intercept, 4),
                "r": round(r, 4),
                "r_squared": round(r_squared, 4),
                "adjusted_r_squared": round(adj_r_squared, 4),
                "f_statistic": round(f_stat, 4),
                "rmse": round(rmse, 4),
                "standard_error_slope": round(se_slope, 4),
                "slope_ci_95_lower": round(slope_ci_lower, 4),
                "slope_ci_95_upper": round(slope_ci_upper, 4),
                "sample_size": n,
                "significant": significant,
                "alpha": alpha,
                "engine": "SciPy",
            },
        )

    # ==========================================================
    # Binary Logistic Regression
    # ==========================================================

    @staticmethod
    def logistic_regression(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        r"""
        Binary Logistic Regression using StatsModels Logit.

        Computes:
        - Coefficient ($\beta$) and Odds Ratio ($\exp(\beta)$)
        - 95% Confidence Interval for Odds Ratio
        - McFadden Pseudo $R^2$
        - Wald $z$-statistic and $p$-value
        """

        x, y, error = RegressionMethods._validate(dataset)

        if error:
            return StatisticalResult(
                test_name="Logistic Regression",
                interpretation=error,
            )

        unique_y = np.unique(y)
        if not HAS_STATSMODELS:
            return StatisticalResult(
                test_name="Logistic Regression",
                interpretation="StatsModels is required for Logistic Regression.",
            )

        if len(unique_y) != 2:
            return StatisticalResult(
                test_name="Logistic Regression",
                interpretation=f"Logistic regression requires a binary outcome (y), found {len(unique_y)} unique values.",
            )

        # Standardize binary outcome to 0 and 1
        y_binary = (y == unique_y[1]).astype(float)

        try:
            X = sm.add_constant(x)
            model = sm.Logit(y_binary, X).fit(disp=False)

            beta = float(model.params[1])
            intercept = float(model.params[0])
            z_stat = float(model.tvalues[1])
            p_val = float(model.pvalues[1])
            pseudo_r2 = float(model.prsquared)

            odds_ratio = float(np.exp(beta))
            conf_int = model.conf_int(alpha=settings.significance_level)
            or_ci_lower = float(np.exp(conf_int[1, 0]))
            or_ci_upper = float(np.exp(conf_int[1, 1]))

            significant = bool(p_val < settings.significance_level)
            interpretation = (
                f"Binary logistic regression is statistically significant "
                f"(z={z_stat:.3f}, p={p_val:.4f}, Odds Ratio={odds_ratio:.3f}, McFadden Pseudo R²={pseudo_r2:.3f})."
                if significant
                else f"Binary logistic regression is not statistically significant "
                f"(z={z_stat:.3f}, p={p_val:.4f}, Odds Ratio={odds_ratio:.3f}, McFadden Pseudo R²={pseudo_r2:.3f})."
            )

            return StatisticalResult(
                test_name="Logistic Regression",
                statistic=round(z_stat, 4),
                p_value=round(p_val, 4),
                degrees_of_freedom=1.0,
                effect_size=round(pseudo_r2, 4),
                confidence_interval=(round(or_ci_lower, 4), round(or_ci_upper, 4)),
                interpretation=interpretation,
                additional_metrics={
                    "beta": round(beta, 4),
                    "intercept": round(intercept, 4),
                    "odds_ratio": round(odds_ratio, 4),
                    "or_ci_95_lower": round(or_ci_lower, 4),
                    "or_ci_95_upper": round(or_ci_upper, 4),
                    "mcfadden_pseudo_r2": round(pseudo_r2, 4),
                    "z_statistic": round(z_stat, 4),
                    "sample_size": len(x),
                    "significant": significant,
                    "alpha": settings.significance_level,
                    "engine": "StatsModels Logit",
                },
            )

        except Exception as ex:
            return StatisticalResult(
                test_name="Logistic Regression",
                interpretation=f"Logistic regression model failed to converge: {str(ex)}",
            )