"""
Parametric statistical methods.

Contains:
    • Independent t-test  (Welch, with Cohen's d + 95% CI)
    • Paired t-test       (scipy ttest_rel, with Cohen's dz + 95% CI)
    • One-way ANOVA       (scipy f_oneway, with η² effect size + Tukey HSD)
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, f_oneway

from src.config import settings
from src.models import (
    StatisticalDataset,
    StatisticalResult,
)


class ParametricMethods:
    """
    Parametric statistical methods.
    """

    # --------------------------------------------------
    # Independent t-test (Welch)
    # --------------------------------------------------

    @staticmethod
    def independent_t_test(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Welch Independent Two-Sample t-test.

        Computes:
        - t-statistic and p-value
        - Welch-Satterthwaite degrees of freedom
        - Cohen's d effect size
        - 95% confidence interval on the mean difference
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

        result = ttest_ind(g1, g2, equal_var=False)

        alpha = settings.significance_level

        # --- Cohen's d (pooled SD) ---
        n1, n2 = len(g1), len(g2)
        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        s1, s2 = float(np.std(g1, ddof=1)), float(np.std(g2, ddof=1))

        pooled_sd = math.sqrt(
            ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
        )
        cohens_d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0

        # --- 95% CI on mean difference (Welch) ---
        # SE of the difference
        se_diff = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
        # Welch-Satterthwaite df
        df_welch = (
            (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
            / (
                (s1 ** 2 / n1) ** 2 / (n1 - 1)
                + (s2 ** 2 / n2) ** 2 / (n2 - 1)
            )
        )
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=df_welch))
        mean_diff = m1 - m2
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        significant = bool(result.pvalue < alpha)
        interpretation = (
            f"Statistically significant difference between '{names[0]}' "
            f"and '{names[1]}' (t={result.statistic:.3f}, p={result.pvalue:.4f})."
            if significant
            else f"No statistically significant difference between '{names[0]}' "
            f"and '{names[1]}' (t={result.statistic:.3f}, p={result.pvalue:.4f})."
        )

        return StatisticalResult(
            test_name="Independent t-test",
            statistic=round(float(result.statistic), 4),
            p_value=round(float(result.pvalue), 4),
            degrees_of_freedom=round(df_welch, 2),
            effect_size=round(cohens_d, 4),
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            interpretation=interpretation,
            additional_metrics={
                "group_1": names[0],
                "group_2": names[1],
                "group1_mean": round(m1, 4),
                "group2_mean": round(m2, 4),
                "mean_difference": round(mean_diff, 4),
                "group1_std": round(s1, 4),
                "group2_std": round(s2, 4),
                "group1_n": n1,
                "group2_n": n2,
                "cohens_d": round(cohens_d, 4),
                "cohens_d_magnitude": _cohens_d_label(cohens_d),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "welch_df": round(df_welch, 2),
                "alpha": alpha,
                "significant": significant,
            },
        )

    # --------------------------------------------------
    # Paired t-test
    # --------------------------------------------------

    @staticmethod
    def paired_t_test(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        Paired Two-Sample t-test (repeated measures / pre-post).

        Computes:
        - t-statistic, p-value, df
        - Cohen's dz effect size (based on SD of differences)
        - 95% CI on mean difference
        """

        if dataset.x is None or dataset.y is None:
            return StatisticalResult(
                test_name="Paired t-test",
                interpretation="Paired variables x and y are required.",
            )

        x = np.asarray(dataset.x, dtype=float)
        y = np.asarray(dataset.y, dtype=float)

        # Remove pairs where either is NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) != len(y) or len(x) < 2:
            return StatisticalResult(
                test_name="Paired t-test",
                interpretation="Insufficient paired observations.",
            )

        result = ttest_rel(x, y)

        n = len(x)
        alpha = settings.significance_level
        diffs = x - y
        mean_diff = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))

        # Cohen's dz = mean(diff) / sd(diff)
        cohens_dz = mean_diff / sd_diff if sd_diff > 0 else 0.0

        # 95% CI on mean difference
        se_diff = sd_diff / math.sqrt(n)
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        significant = bool(result.pvalue < alpha)
        interpretation = (
            f"Statistically significant paired difference "
            f"(t={result.statistic:.3f}, p={result.pvalue:.4f})."
            if significant
            else f"No statistically significant paired difference "
            f"(t={result.statistic:.3f}, p={result.pvalue:.4f})."
        )

        return StatisticalResult(
            test_name="Paired t-test",
            statistic=round(float(result.statistic), 4),
            p_value=round(float(result.pvalue), 4),
            degrees_of_freedom=float(n - 1),
            effect_size=round(cohens_dz, 4),
            confidence_interval=(round(ci_lower, 4), round(ci_upper, 4)),
            interpretation=interpretation,
            additional_metrics={
                "n_pairs": n,
                "mean_difference": round(mean_diff, 4),
                "sd_differences": round(sd_diff, 4),
                "cohens_dz": round(cohens_dz, 4),
                "cohens_dz_magnitude": _cohens_d_label(cohens_dz),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "mean_x": round(float(np.mean(x)), 4),
                "mean_y": round(float(np.mean(y)), 4),
                "alpha": alpha,
                "significant": significant,
            },
        )

    # --------------------------------------------------
    # One-Way ANOVA
    # --------------------------------------------------

    @staticmethod
    def one_way_anova(
        dataset: StatisticalDataset,
    ) -> StatisticalResult:
        """
        One-Way ANOVA with eta-squared (η²) effect size.

        Computes:
        - F-statistic, p-value
        - Degrees of freedom (between groups, within groups)
        - η² (eta-squared) effect size
        - Group descriptive statistics
        - Tukey HSD pairwise comparisons (when ≥ 2 groups present)
        """

        groups_dict = dataset.groups

        if len(groups_dict) < 2:
            return StatisticalResult(
                test_name="One-Way ANOVA",
                interpretation="At least two groups are required.",
            )

        group_arrays = [
            np.asarray(v, dtype=float)
            for v in groups_dict.values()
        ]
        # Remove NaN within each group
        group_arrays = [g[~np.isnan(g)] for g in group_arrays]
        group_arrays = [g for g in group_arrays if len(g) >= 2]

        if len(group_arrays) < 2:
            return StatisticalResult(
                test_name="One-Way ANOVA",
                interpretation="Each group must have at least 2 valid observations.",
            )

        names = list(groups_dict.keys())
        result = f_oneway(*group_arrays)

        alpha = settings.significance_level

        # --- η² (eta-squared) ---
        all_values = np.concatenate(group_arrays)
        grand_mean = float(np.mean(all_values))
        ss_between = sum(
            len(g) * (float(np.mean(g)) - grand_mean) ** 2
            for g in group_arrays
        )
        ss_total = float(np.sum((all_values - grand_mean) ** 2))
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        # Degrees of freedom
        k = len(group_arrays)
        n_total = len(all_values)
        df_between = k - 1
        df_within = n_total - k

        # Group descriptive stats
        group_stats = {}
        for name, arr in zip(names, group_arrays):
            group_stats[name] = {
                "n": int(len(arr)),
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr, ddof=1)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        # Tukey HSD (only available in scipy via stats.tukey_hsd)
        tukey_results = []
        try:
            tukey = stats.tukey_hsd(*group_arrays)
            for i in range(k):
                for j in range(i + 1, k):
                    tukey_results.append({
                        "group_1": names[i],
                        "group_2": names[j],
                        "p_value": round(float(tukey.pvalue[i][j]), 4),
                        "significant": bool(tukey.pvalue[i][j] < alpha),
                    })
        except Exception:
            tukey_results = []

        significant = bool(result.pvalue < alpha)
        interpretation = (
            f"Statistically significant difference among {k} groups "
            f"(F={result.statistic:.3f}, p={result.pvalue:.4f}, η²={eta_squared:.3f})."
            if significant
            else f"No statistically significant difference among {k} groups "
            f"(F={result.statistic:.3f}, p={result.pvalue:.4f}, η²={eta_squared:.3f})."
        )

        return StatisticalResult(
            test_name="One-Way ANOVA",
            statistic=round(float(result.statistic), 4),
            p_value=round(float(result.pvalue), 4),
            degrees_of_freedom=float(df_between),
            effect_size=round(eta_squared, 4),
            interpretation=interpretation,
            additional_metrics={
                "f_statistic": round(float(result.statistic), 4),
                "df_between": df_between,
                "df_within": df_within,
                "n_total": n_total,
                "n_groups": k,
                "eta_squared": round(eta_squared, 4),
                "eta_squared_magnitude": _eta_sq_label(eta_squared),
                "alpha": alpha,
                "significant": significant,
                "group_stats": group_stats,
                "tukey_hsd": tukey_results,
            },
        )


# --------------------------------------------------
# Label helpers
# --------------------------------------------------

def _cohens_d_label(d: float) -> str:
    """Cohen's d magnitude label."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
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