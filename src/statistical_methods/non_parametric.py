"""
non_parametric.py

Non-parametric statistical methods.

Contains:
    • Mann-Whitney U Test       (with rank-biserial correlation effect size)
    • Wilcoxon Signed-Rank Test (with matched-pairs r effect size)
    • Kruskal-Wallis Test       (with η² effect size)
    • Friedman Test             (with Kendall's W effect size)

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
    def _interpret(p: float, test: str = "", stat: float | None = None) -> str:
        significant = p < settings.significance_level
        p_str = f"p={p:.4f}"
        s_str = f", U={stat:.3f}" if stat is not None else ""
        if significant:
            return f"Statistically significant difference ({p_str}{s_str})."
        return f"No statistically significant difference ({p_str}{s_str})."

    # =====================================================
    # Mann-Whitney U
    # =====================================================

    @staticmethod
    def mann_whitney(dataset: StatisticalDataset) -> StatisticalResult:
        """
        Mann-Whitney U Test (non-parametric independent two-sample test).

        Computes:
        - U statistic and p-value
        - Rank-biserial correlation (r) as effect size
          r = 1 - 2U / (n1 * n2)
        """

        if len(dataset.groups) != 2:
            return StatisticalResult(
                test_name="Mann-Whitney U",
                interpretation="Exactly two groups are required.",
            )

        names = list(dataset.groups.keys())
        g1 = np.asarray(dataset.groups[names[0]], dtype=float)
        g2 = np.asarray(dataset.groups[names[1]], dtype=float)
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) < 2 or len(g2) < 2:
            return StatisticalResult(
                test_name="Mann-Whitney U",
                interpretation="Each group must contain at least two observations.",
            )

        statistic, p = mannwhitneyu(g1, g2, alternative="two-sided")
        u = float(statistic)
        p = float(p)

        n1, n2 = len(g1), len(g2)

        # Rank-biserial correlation: r = 1 - 2U / (n1 * n2)
        rank_biserial = 1.0 - (2.0 * u) / (n1 * n2)

        significant = bool(p < settings.significance_level)
        interpretation = (
            f"Statistically significant difference between '{names[0]}' and '{names[1]}' "
            f"(U={u:.3f}, p={p:.4f}, r={rank_biserial:.3f})."
            if significant
            else f"No statistically significant difference between '{names[0]}' and '{names[1]}' "
            f"(U={u:.3f}, p={p:.4f}, r={rank_biserial:.3f})."
        )

        return StatisticalResult(
            test_name="Mann-Whitney U",
            statistic=round(u, 4),
            p_value=round(p, 4),
            effect_size=round(rank_biserial, 4),
            interpretation=interpretation,
            additional_metrics={
                "u_statistic": round(u, 4),
                "rank_biserial_r": round(rank_biserial, 4),
                "effect_magnitude": _r_label(rank_biserial),
                "group_1": names[0],
                "group_2": names[1],
                "group1_n": n1,
                "group2_n": n2,
                "group1_median": round(float(np.median(g1)), 4),
                "group2_median": round(float(np.median(g2)), 4),
                "significant": significant,
                "alpha": settings.significance_level,
            },
        )

    # =====================================================
    # Wilcoxon Signed Rank
    # =====================================================

    @staticmethod
    def wilcoxon(dataset: StatisticalDataset) -> StatisticalResult:
        """
        Wilcoxon Signed-Rank Test (non-parametric paired test).

        Computes:
        - W statistic and p-value
        - Matched-pairs r effect size: r = Z / sqrt(N)
        """

        if dataset.x is None or dataset.y is None:
            return StatisticalResult(
                test_name="Wilcoxon Signed-Rank",
                interpretation="Paired variables x and y are required.",
            )

        x = np.asarray(dataset.x, dtype=float)
        y = np.asarray(dataset.y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) != len(y):
            return StatisticalResult(
                test_name="Wilcoxon Signed-Rank",
                interpretation="Paired variables must have equal length.",
            )

        if len(x) < 2:
            return StatisticalResult(
                test_name="Wilcoxon Signed-Rank",
                interpretation="Insufficient paired observations.",
            )

        statistic, p = wilcoxon(x, y)
        w = float(statistic)
        p = float(p)
        n = len(x)

        # Matched-pairs r = Z / sqrt(N)
        # Approximate Z from Wilcoxon W using normal approximation
        mu_w = n * (n + 1) / 4
        sigma_w = float(np.sqrt(n * (n + 1) * (2 * n + 1) / 24))
        z_approx = (w - mu_w) / sigma_w if sigma_w > 0 else 0.0
        matched_r = z_approx / float(np.sqrt(n))

        significant = bool(p < settings.significance_level)
        interpretation = (
            f"Statistically significant paired difference (W={w:.3f}, p={p:.4f}, r={matched_r:.3f})."
            if significant
            else f"No statistically significant paired difference (W={w:.3f}, p={p:.4f}, r={matched_r:.3f})."
        )

        return StatisticalResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=round(w, 4),
            p_value=round(p, 4),
            effect_size=round(matched_r, 4),
            interpretation=interpretation,
            additional_metrics={
                "w_statistic": round(w, 4),
                "matched_pairs_r": round(matched_r, 4),
                "effect_magnitude": _r_label(matched_r),
                "sample_size": n,
                "median_x": round(float(np.median(x)), 4),
                "median_y": round(float(np.median(y)), 4),
                "significant": significant,
                "alpha": settings.significance_level,
            },
        )

    # =====================================================
    # Kruskal-Wallis
    # =====================================================

    @staticmethod
    def kruskal(dataset: StatisticalDataset) -> StatisticalResult:
        """
        Kruskal-Wallis H Test (non-parametric one-way ANOVA).

        Computes:
        - H statistic and p-value
        - η² (eta-squared) effect size: η² = (H - k + 1) / (N - k)
        """

        groups_dict = dataset.groups
        if len(groups_dict) < 2:
            return StatisticalResult(
                test_name="Kruskal-Wallis",
                interpretation="At least two groups are required.",
            )

        names = list(groups_dict.keys())
        arrays = [
            np.asarray(v, dtype=float)
            for v in groups_dict.values()
        ]
        arrays = [g[~np.isnan(g)] for g in arrays]
        arrays = [g for g in arrays if len(g) >= 2]

        if len(arrays) < 2:
            return StatisticalResult(
                test_name="Kruskal-Wallis",
                interpretation="Each group must have at least 2 valid observations.",
            )

        statistic, p = kruskal(*arrays)
        h = float(statistic)
        p = float(p)

        k = len(arrays)
        n_total = sum(len(g) for g in arrays)

        # η² = (H - k + 1) / (N - k)
        eta_squared = max(0.0, (h - k + 1) / (n_total - k)) if n_total > k else 0.0

        group_stats = {
            names[i]: {
                "n": int(len(arrays[i])),
                "median": round(float(np.median(arrays[i])), 4),
                "mean_rank": None,  # placeholder
            }
            for i in range(len(arrays))
        }

        significant = bool(p < settings.significance_level)
        interpretation = (
            f"Statistically significant difference among {k} groups "
            f"(H={h:.3f}, p={p:.4f}, η²={eta_squared:.3f})."
            if significant
            else f"No statistically significant difference among {k} groups "
            f"(H={h:.3f}, p={p:.4f}, η²={eta_squared:.3f})."
        )

        return StatisticalResult(
            test_name="Kruskal-Wallis",
            statistic=round(h, 4),
            p_value=round(p, 4),
            degrees_of_freedom=float(k - 1),
            effect_size=round(eta_squared, 4),
            interpretation=interpretation,
            additional_metrics={
                "h_statistic": round(h, 4),
                "eta_squared": round(eta_squared, 4),
                "effect_magnitude": _eta_sq_label(eta_squared),
                "n_groups": k,
                "n_total": n_total,
                "group_stats": group_stats,
                "significant": significant,
                "alpha": settings.significance_level,
            },
        )

    # =====================================================
    # Friedman Test
    # =====================================================

    @staticmethod
    def friedman(dataset: StatisticalDataset) -> StatisticalResult:
        """
        Friedman Test (non-parametric repeated measures ANOVA).

        Computes:
        - χ² statistic and p-value
        - Kendall's W coefficient of concordance (effect size)
          W = χ² / (k(n - 1))  where k = groups, n = observations per group
        """

        groups_dict = dataset.groups
        if len(groups_dict) < 3:
            return StatisticalResult(
                test_name="Friedman",
                interpretation="At least three paired groups are required.",
            )

        groups = list(groups_dict.values())
        lengths = {len(g) for g in groups}

        if len(lengths) != 1:
            return StatisticalResult(
                test_name="Friedman",
                interpretation="All paired groups must have equal sample size.",
            )

        statistic, p = friedmanchisquare(*groups)
        chi2 = float(statistic)
        p = float(p)

        k = len(groups)
        n = len(groups[0])

        # Kendall's W = χ² / (k * (n - 1))
        kendalls_w = chi2 / (k * (n - 1)) if (k * (n - 1)) > 0 else 0.0

        significant = bool(p < settings.significance_level)
        interpretation = (
            f"Statistically significant differences across {k} conditions "
            f"(χ²={chi2:.3f}, p={p:.4f}, W={kendalls_w:.3f})."
            if significant
            else f"No statistically significant differences across {k} conditions "
            f"(χ²={chi2:.3f}, p={p:.4f}, W={kendalls_w:.3f})."
        )

        return StatisticalResult(
            test_name="Friedman",
            statistic=round(chi2, 4),
            p_value=round(p, 4),
            degrees_of_freedom=float(k - 1),
            effect_size=round(kendalls_w, 4),
            interpretation=interpretation,
            additional_metrics={
                "chi2_statistic": round(chi2, 4),
                "kendalls_w": round(kendalls_w, 4),
                "effect_magnitude": _w_label(kendalls_w),
                "n_groups": k,
                "n_per_group": n,
                "significant": significant,
                "alpha": settings.significance_level,
            },
        )


# =====================================================
# Label helpers
# =====================================================

def _r_label(r: float) -> str:
    """Magnitude label for correlation-type effect sizes."""
    r = abs(r)
    if r < 0.10:
        return "negligible"
    if r < 0.30:
        return "small"
    if r < 0.50:
        return "medium"
    return "large"


def _eta_sq_label(eta2: float) -> str:
    """η² magnitude label (Cohen, 1988)."""
    if eta2 < 0.01:
        return "negligible"
    if eta2 < 0.06:
        return "small"
    if eta2 < 0.14:
        return "medium"
    return "large"


def _w_label(w: float) -> str:
    """Kendall's W magnitude label."""
    if w < 0.10:
        return "negligible"
    if w < 0.30:
        return "weak"
    if w < 0.70:
        return "moderate"
    return "strong"