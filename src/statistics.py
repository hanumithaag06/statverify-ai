"""
Statistical analysis engines for StatVerify AI.

Contains:
- StatisticalDecisionEngine: Recommends the appropriate test.
- AssumptionChecker:         Validates prerequisites before execution.
- StatisticalEngine:         Executes calculations via a runner registry.
"""

from __future__ import annotations

import numpy as np

from src.config import settings
from src.models import (
    ParsedTable,
    StatisticalContext,
    StatisticalResult,
    StatisticalTest,
    TestRecommendation,
    StatisticalDataset,
    VariableType,
    AssumptionResult,
    AssumptionStatus,
)

from src.statistical_methods import (
    ParametricMethods,
    CategoricalMethods,
    CorrelationMethods,
    NonParametricMethods,
    RegressionMethods,
)

from src.utils import logger


# ==============================================================
# Decision Engine
# ==============================================================


class StatisticalDecisionEngine:
    """
    Recommends a statistical test based on column metadata.

    Decision logic (in priority order):
    1. Summary table  → Chi-Square (always)
    2. Raw dataset:
       a. Binary + Continuous   → T-Test (2 groups) or ANOVA (3+ groups)
       b. Two continuous cols   → Pearson correlation
       c. Categorical grouping  → Chi-Square
       d. Fallback              → T-Test
    """

    def recommend(
        self,
        table: ParsedTable,
        context: StatisticalContext | None = None,
    ) -> TestRecommendation:
        """
        Recommend a statistical test.
        """

        logger.info("Recommending statistical test...")

        # --------------------------------------------------------
        # Summary Table (keyword-detected)
        # --------------------------------------------------------

        if getattr(table.metadata, "analysis_type", None) == "summary":
            if table.dataframe.shape[1] == 4:
                return TestRecommendation(
                    test=StatisticalTest.CHI_SQUARE,
                    reason="Summary frequency table detected (4 columns).",
                    confidence=0.98,
                )

        if bool(table.summary_metadata) or getattr(table.metadata, "analysis_type", None) == "summary":

            group_variable = (
                context.group_variable
                if context is not None
                else None
            )

            group_count = (
                len(group_variable)
                if isinstance(group_variable, list)
                else (1 if group_variable else 0)
            )

            if group_count == 2:
                return TestRecommendation(
                    test=StatisticalTest.CHI_SQUARE,
                    reason="Summary table with two independent groups → Chi-Square.",
                    confidence=0.96,
                )

            if group_count > 2:
                return TestRecommendation(
                    test=StatisticalTest.CHI_SQUARE,
                    reason="Summary table with more than two groups → Chi-Square.",
                    confidence=0.94,
                )

            return TestRecommendation(
                test=StatisticalTest.CHI_SQUARE,
                reason="Categorical summary table → Chi-Square.",
                confidence=0.90,
            )

        # --------------------------------------------------------
        # Raw Dataset
        # --------------------------------------------------------

        test = StatisticalTest.T_TEST
        reason = "Default fallback test (independent t-test)."
        confidence = 0.50

        numeric_cols = [col for col in table.metadata if col.is_numeric]
        binary_cols = [col for col in table.metadata if col.is_binary]
        categorical_cols = [
            col
            for col in table.metadata
            if col.variable_type == VariableType.CATEGORICAL
        ]

        # Binary grouping present: T-Test (2 groups) or ANOVA (3+ groups)
        if binary_cols and numeric_cols:
            n_groups = self._count_groups(table, binary_cols[0].name)
            if n_groups >= 3:
                test = StatisticalTest.ANOVA
                reason = (
                    f"Detected {n_groups} distinct groups in '{binary_cols[0].name}' "
                    f"with numeric outcome '{numeric_cols[0].name}'. "
                    f"One-Way ANOVA is suitable."
                )
                confidence = 0.92
            else:
                test = StatisticalTest.T_TEST
                reason = (
                    f"Detected binary grouping variable '{binary_cols[0].name}' "
                    f"and numeric outcome variable '{numeric_cols[0].name}'. "
                    f"Two-sample t-test is suitable."
                )
                confidence = 0.95

        elif len(numeric_cols) >= 2:
            test = StatisticalTest.PEARSON
            reason = (
                f"Detected {len(numeric_cols)} numeric columns "
                f"({', '.join(c.name for c in numeric_cols)}). "
                f"Pearson correlation is suitable to check linear relationship."
            )
            confidence = 0.90

        elif categorical_cols:
            n_groups = self._count_groups(table, categorical_cols[0].name)
            if n_groups >= 3 and numeric_cols:
                test = StatisticalTest.ANOVA
                reason = (
                    f"Detected {n_groups} categories in '{categorical_cols[0].name}' "
                    f"with numeric outcome '{numeric_cols[0].name}'. "
                    f"One-Way ANOVA is suitable."
                )
                confidence = 0.88
            else:
                test = StatisticalTest.CHI_SQUARE
                reason = (
                    f"Detected categorical column '{categorical_cols[0].name}'. "
                    f"Chi-Square test of independence is suitable."
                )
                confidence = 0.85

        return TestRecommendation(
            test=test,
            reason=reason,
            confidence=confidence,
        )

    @staticmethod
    def _count_groups(table: ParsedTable, column: str) -> int:
        """Count unique non-null values in a column."""
        try:
            return int(table.dataframe[column].nunique(dropna=True))
        except Exception:
            return 2


# ==============================================================
# Assumption Checker
# ==============================================================


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
                    message=f"{missing} missing values remain in dataset.",
                )
            )

        else:

            results.append(
                AssumptionResult(
                    name="Missing Data",
                    status=AssumptionStatus.PASS,
                    message="No missing values detected.",
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
                        message="Two groups detected — appropriate for t-test.",
                    )
                )

            elif len(dataset.groups) >= 3:

                results.append(
                    AssumptionResult(
                        name="Groups",
                        status=AssumptionStatus.PASS,
                        message=f"{len(dataset.groups)} groups detected — appropriate for ANOVA.",
                    )
                )

            else:

                results.append(
                    AssumptionResult(
                        name="Groups",
                        status=AssumptionStatus.WARNING,
                        message=f"{len(dataset.groups)} group(s) detected. At least 2 are required.",
                    )
                )

        # ----------------------------
        # Normality hint (for parametric tests)
        # ----------------------------

        test = recommendation.test
        parametric_tests = {
            StatisticalTest.T_TEST,
            StatisticalTest.PAIRED_T_TEST,
            StatisticalTest.ANOVA,
        }

        if test in parametric_tests and dataset.groups:
            all_large = all(len(v) >= 30 for v in dataset.groups.values())
            if all_large:
                results.append(
                    AssumptionResult(
                        name="Normality (CLT)",
                        status=AssumptionStatus.PASS,
                        message="Group sizes ≥ 30; Central Limit Theorem supports normality assumption.",
                    )
                )
            else:
                results.append(
                    AssumptionResult(
                        name="Normality",
                        status=AssumptionStatus.WARNING,
                        message=(
                            "Group sizes < 30. Normality assumption should be verified "
                            "(e.g. Shapiro-Wilk test). Consider Mann-Whitney U if violated."
                        ),
                    )
                )

        return results


# ==============================================================
# Statistical Engine
# ==============================================================


class StatisticalEngine:
    """
    Executes statistical tests via a runner registry.

    New tests are added by registering a runner method —
    no if-else chain grows.
    """

    def __init__(self) -> None:

        self.runners = {
            StatisticalTest.T_TEST:
                ParametricMethods.independent_t_test,

            StatisticalTest.PAIRED_T_TEST:
                ParametricMethods.paired_t_test,

            StatisticalTest.ANOVA:
                ParametricMethods.one_way_anova,

            StatisticalTest.CHI_SQUARE:
                CategoricalMethods.chi_square,

            StatisticalTest.FISHER_EXACT:
                CategoricalMethods.fisher_exact,

            StatisticalTest.PEARSON:
                CorrelationMethods.pearson,

            StatisticalTest.SPEARMAN:
                CorrelationMethods.spearman,

            StatisticalTest.KENDALL:
                CorrelationMethods.kendall,

            StatisticalTest.MANN_WHITNEY:
                NonParametricMethods.mann_whitney,

            StatisticalTest.WILCOXON:
                NonParametricMethods.wilcoxon,

            StatisticalTest.KRUSKAL:
                NonParametricMethods.kruskal,

            StatisticalTest.FRIEDMAN:
                NonParametricMethods.friedman,

            StatisticalTest.LINEAR_REGRESSION:
                RegressionMethods.linear_regression,

            StatisticalTest.LOGISTIC_REGRESSION:
                RegressionMethods.logistic_regression,
        }

        self.assumption_checker = AssumptionChecker()

    def execute(
        self,
        dataset: StatisticalDataset,
        recommendation: TestRecommendation,
    ) -> StatisticalResult:

        logger.info(
            f"Executing statistical test: {recommendation.test.value}..."
        )

        # Summary chi-square handled specially
        if recommendation.test == StatisticalTest.CHI_SQUARE:
            if dataset.is_summary:
                return self._summary_chi_square(dataset)
            return self._chi_square(dataset)

        # T-Test
        if recommendation.test == StatisticalTest.T_TEST:
            return self._t_test(dataset, recommendation)

        # ANOVA
        if recommendation.test == StatisticalTest.ANOVA:
            return self._anova(dataset, recommendation)

        # All other registered tests
        runner = self.runners.get(recommendation.test)

        if runner is None:
            return StatisticalResult(
                test_name=recommendation.test.value,
                interpretation=(
                    f"Test '{recommendation.test.value}' is not yet implemented."
                ),
            )

        result = runner(dataset)

        result.assumptions = self.assumption_checker.check(
            dataset,
            recommendation,
        )

        return result

    # ----------------------------------------------------------
    # Summary Chi-Square (multi-variable demographic table)
    # ----------------------------------------------------------

    def _summary_chi_square(self, dataset: StatisticalDataset) -> StatisticalResult:
        from scipy.stats import chi2_contingency

        results = []
        all_assumptions_ok = True

        group_names = (
            dataset.group_names
            or list(dataset.sample_sizes.keys())
            or ["Experimental", "Control"]
        )

        for variable, raw_table in dataset.contingency_tables.items():
            table = np.asarray(raw_table, dtype=float)
            labels = list(dataset.category_labels.get(variable, []))

            row_mask = table.sum(axis=1) > 0
            if labels:
                labels = [label for label, keep in zip(labels, row_mask) if keep]
            table = table[row_mask]

            col_mask = table.sum(axis=0) > 0
            table_for_test = table[:, col_mask]

            if table_for_test.shape[0] < 2 or table_for_test.shape[1] < 2:
                continue

            try:
                # correction=False: standard Pearson chi-square — matches
                # conventional reporting in research tables.
                chi2, p, dof, expected = chi2_contingency(
                    table_for_test, correction=False
                )

                significant = bool(p < settings.significance_level)
                low_expected = bool((expected < 5).any())

                if low_expected:
                    all_assumptions_ok = False

                if significant:
                    var_interp = (
                        f"Statistically significant association "
                        f"between groups for '{variable}' "
                        f"(χ²={chi2:.3f}, df={dof}, p={p:.4f})."
                    )
                else:
                    var_interp = (
                        f"No statistically significant association "
                        f"for '{variable}' "
                        f"(χ²={chi2:.3f}, df={dof}, p={p:.4f})."
                    )

                # Per-category No./% breakdown using detected group sample sizes
                breakdown = []
                usable_groups = group_names[: table.shape[1]]
                if labels and table.shape[1] >= len(usable_groups):
                    for row_idx, category in enumerate(labels):
                        entry = {"category": category}
                        for col_idx, group in enumerate(usable_groups):
                            count = float(table[row_idx, col_idx])
                            n = dataset.sample_sizes.get(group)
                            pct = round(count / n * 100, 1) if n else None
                            entry[f"{group}_n"] = int(round(count))
                            entry[f"{group}_pct"] = pct
                        breakdown.append(entry)

                results.append(
                    {
                        "variable": variable,
                        "chi_square": round(float(chi2), 3),
                        "p_value": round(float(p), 4),
                        "dof": int(dof),
                        "significant": significant,
                        "low_expected_freq": low_expected,
                        "interpretation": var_interp,
                        "groups": usable_groups,
                        "breakdown": breakdown,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "variable": variable,
                        "error": str(e),
                    }
                )

        if not results:
            return StatisticalResult(
                test_name="Chi-Square",
                interpretation="No valid contingency tables found.",
            )

        sig_count = sum(1 for r in results if r.get("significant") is True)
        non_sig_count = len(results) - sig_count

        summary_interp = (
            f"{len(results)} demographic variables analysed. "
            f"{sig_count} significant, {non_sig_count} non-significant "
            f"at α={settings.significance_level}."
        )

        assumptions = [
            AssumptionResult(
                name="Independent Observations",
                status=AssumptionStatus.PASS,
                message="Assumed independent (groups are separate).",
            ),
            AssumptionResult(
                name="Expected Cell Counts ≥ 5",
                status=(
                    AssumptionStatus.PASS
                    if all_assumptions_ok
                    else AssumptionStatus.WARNING
                ),
                message=(
                    "All expected frequencies ≥ 5."
                    if all_assumptions_ok
                    else "Some expected frequencies < 5 in one or more variables. "
                    "Consider Fisher Exact Test for those cells."
                ),
            ),
        ]

        return StatisticalResult(
            test_name="Chi-Square",
            statistic=None,
            p_value=None,
            degrees_of_freedom=None,
            assumptions=assumptions,
            interpretation=summary_interp,
            additional_results=results,
            additional_metrics={
                "variables_tested": len(results),
                "significant_variables": sig_count,
                "non_significant_variables": non_sig_count,
                "significance_level": settings.significance_level,
            },
        )

    # ----------------------------------------------------------
    # Raw Chi-Square
    # ----------------------------------------------------------

    def _chi_square(
        self,
        dataset: StatisticalDataset,
    ) -> StatisticalResult:

        results = []

        for variable, table in dataset.contingency_tables.items():
            try:
                result = CategoricalMethods.chi_square(table)

                results.append(
                    {
                        "variable": variable,
                        "statistic": result.statistic,
                        "p_value": result.p_value,
                        "degrees_of_freedom": result.degrees_of_freedom,
                    }
                )

            except Exception:
                continue

        if not results:

            return StatisticalResult(
                test_name="Chi-Square",
                interpretation="No valid contingency tables found.",
            )

        best = results[0]

        significant = (
            best["p_value"] < settings.significance_level
            if best["p_value"] is not None
            else False
        )

        interpretation = (
            f"Chi-Square test: χ²={best['statistic']:.3f}, "
            f"df={int(best['degrees_of_freedom'])}, p={best['p_value']:.4f}. "
            + ("Statistically significant association." if significant
               else "No statistically significant association.")
        )

        stat_result = StatisticalResult(
            test_name="Chi-Square",
            statistic=round(best["statistic"], 4) if best["statistic"] is not None else None,
            p_value=round(best["p_value"], 4) if best["p_value"] is not None else None,
            degrees_of_freedom=best["degrees_of_freedom"],
            additional_results=results,
            interpretation=interpretation,
        )

        stat_result.assumptions = self.assumption_checker.check(
            dataset,
            TestRecommendation(
                test=StatisticalTest.CHI_SQUARE,
                reason="Chi-Square",
                confidence=1.0,
            ),
        )

        return stat_result

    # ----------------------------------------------------------
    # T-Test dispatch
    # ----------------------------------------------------------

    def _t_test(
        self,
        dataset: StatisticalDataset,
        recommendation: TestRecommendation,
    ) -> StatisticalResult:
        result = ParametricMethods.independent_t_test(dataset)
        result.assumptions = self.assumption_checker.check(
            dataset,
            TestRecommendation(
                test=StatisticalTest.T_TEST,
                reason="Independent t-test",
                confidence=1.0,
            ),
        )
        return result

    # ----------------------------------------------------------
    # ANOVA dispatch
    # ----------------------------------------------------------

    def _anova(
        self,
        dataset: StatisticalDataset,
        recommendation: TestRecommendation,
    ) -> StatisticalResult:
        result = ParametricMethods.one_way_anova(dataset)
        result.assumptions = self.assumption_checker.check(
            dataset,
            TestRecommendation(
                test=StatisticalTest.ANOVA,
                reason="One-Way ANOVA",
                confidence=1.0,
            ),
        )
        return result

    # ----------------------------------------------------------
    # Fallback
    # ----------------------------------------------------------

    def _unsupported(
        self,
        recommendation: TestRecommendation,
        assumptions: list[AssumptionResult],
    ) -> StatisticalResult:

        logger.warning(
            f"No runner registered for: {recommendation.test.value}"
        )

        return StatisticalResult(
            test_name=recommendation.test.value,
            assumptions=assumptions,
            interpretation=(
                f"Test '{recommendation.test.value}' is not yet implemented."
            ),
        )
