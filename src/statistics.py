"""
Statistical analysis engines for StatVerify AI.

Contains:
- StatisticalDecisionEngine: Recommends the appropriate test.
- AssumptionChecker:         Validates prerequisites before execution.
- StatisticalEngine:         Executes tests via a runner registry.
"""

from __future__ import annotations

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
        # Summary Table
        # --------------------------------------------------------

        if getattr(table.metadata, "analysis_type", None) == "summary":
            if table.dataframe.shape[1] == 4:
                return TestRecommendation(
                    test=StatisticalTest.CHI_SQUARE,
                    reason="Summary frequency table detected.",
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

            # Two numeric columns (Experimental vs Control)
            if group_count == 2:
                return TestRecommendation(
                    test=StatisticalTest.T_TEST,
                    reason="Summary table with two independent groups.",
                    confidence=0.96,
                )

            # More than two groups
            if group_count > 2:
                return TestRecommendation(
                    test=StatisticalTest.ANOVA,
                    reason="Summary table with more than two groups.",
                    confidence=0.94,
                )

            # Categorical summary (no clear numeric groups)
            return TestRecommendation(
                test=StatisticalTest.CHI_SQUARE,
                reason="Categorical summary table.",
                confidence=0.90,
            )

        # --------------------------------------------------------
        # Raw Dataset
        # --------------------------------------------------------

        test = StatisticalTest.T_TEST
        reason = "Default fallback test."
        confidence = 0.50

        numeric_cols = [col for col in table.metadata if col.is_numeric]
        binary_cols = [col for col in table.metadata if col.is_binary]
        categorical_cols = [
            col
            for col in table.metadata
            if col.variable_type == VariableType.CATEGORICAL
        ]

        if len(numeric_cols) >= 2:
            test = StatisticalTest.PEARSON
            reason = (
                f"Detected {len(numeric_cols)} numeric columns "
                f"({', '.join(c.name for c in numeric_cols)}). "
                f"Pearson correlation is suitable to check linear relationship."
            )
            confidence = 0.90

        elif len(binary_cols) >= 1 and len(numeric_cols) >= 1:
            test = StatisticalTest.T_TEST
            reason = (
                f"Detected binary grouping variable '{binary_cols[0].name}' "
                f"and numeric outcome variable '{numeric_cols[0].name}'. "
                f"Two-sample t-test is suitable."
            )
            confidence = 0.95

        elif len(categorical_cols) >= 1:
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
        }

        self.assumption_checker = AssumptionChecker()

    def execute(
        self,
        dataset: StatisticalDataset,
        recommendation: TestRecommendation | None = None,
    ) -> StatisticalResult:
        """
        Execute statistical analysis by dispatching on dataset properties.
        """
        # Handle legacy tests that pass ParsedTable directly
        if not hasattr(dataset, "contingency_table"):
            from src.extractor import StatisticalDataExtractor
            from src.planner import AnalysisPlanner
            planner = AnalysisPlanner()
            extractor = StatisticalDataExtractor()
            context = planner.build_context(dataset)  # type: ignore
            dataset = extractor.extract(dataset, context)  # type: ignore

        if (
            getattr(dataset, "contingency_table", None) is not None
            or getattr(dataset, "contingency_tables", None)
        ):
            return self._chi_square(dataset)
        elif dataset.numeric_columns:
            return self._t_test(dataset)
        elif dataset.groups:
            return self._anova(dataset)

        if recommendation is not None:
            runner = self.runners.get(recommendation.test)
            if runner is not None:
                try:
                    result = runner(dataset)
                    result.assumptions = self.assumption_checker.check(dataset, recommendation)
                    return result
                except Exception as err:
                    logger.warning(f"Runner failed: {err}")
                    return StatisticalResult(
                        test_name=recommendation.test.value,
                        statistic=None,
                        p_value=None,
                        assumptions=self.assumption_checker.check(dataset, recommendation),
                        interpretation=f"Execution failed: {err}",
                    )

        return StatisticalResult(
            test_name="Unknown Test",
            interpretation="Could not determine the statistical test to run.",
        )

    def _chi_square(self, dataset: StatisticalDataset) -> StatisticalResult:
        result = CategoricalMethods.chi_square(dataset)
        result.assumptions = self.assumption_checker.check(
            dataset,
            TestRecommendation(
                test=StatisticalTest.CHI_SQUARE,
                reason="Chi-Square Test",
                confidence=1.0,
            ),
        )
        return result

    def _t_test(self, dataset: StatisticalDataset) -> StatisticalResult:
        result = ParametricMethods.independent_t_test(dataset)
        result.assumptions = self.assumption_checker.check(
            dataset,
            TestRecommendation(
                test=StatisticalTest.T_TEST,
                reason="T-Test",
                confidence=1.0,
            ),
        )
        return result

    def _anova(self, dataset: StatisticalDataset) -> StatisticalResult:
        result = ParametricMethods.one_way_anova(dataset)
        result.assumptions = self.assumption_checker.check(
            dataset,
            TestRecommendation(
                test=StatisticalTest.ANOVA,
                reason="ANOVA Test",
                confidence=1.0,
            ),
        )
        return result

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

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
