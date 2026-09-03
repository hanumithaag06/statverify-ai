"""
ai.py

AI Explanation Layer

Responsibilities
----------------
✓ Explain statistical results in plain language
✓ Never perform calculations
✓ Never verify numbers
✓ Uses Gemini 2.5 Flash
"""

from __future__ import annotations

import os
from typing import Any

import google.generativeai as genai

from src.models import (
    AIExplanation,
    StatisticalResult,
    AssumptionStatus,
)


# ==========================================================
# Gemini Client
# ==========================================================


class GeminiClient:
    """
    Thin wrapper around Gemini.
    """

    MODEL = "gemini-2.5-flash"

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.MODEL)
        else:
            self.model = None

    def generate(self, prompt: str) -> str:
        """
        Generate explanation from Gemini.

        Falls back to deterministic text if
        API key is unavailable.
        """

        if self.model is None:
            return (
                "This statistical result has been computed without using an external AI service. "
                "See the statistics section for full numerical results."
            )

        try:
            response = self.model.generate_content(prompt)

            return response.text

        except Exception:
            return (
                "Unable to contact Gemini. "
                "Returning fallback explanation."
            )


# ==========================================================
# Prompt Builder
# ==========================================================


class PromptBuilder:
    """
    Builds rich prompts for Gemini.

    IMPORTANT:
    AI must NEVER calculate statistics.
    All numbers are pre-computed by SciPy.
    """

    @staticmethod
    def build(result: StatisticalResult) -> str:

        # Format assumption results
        assumption_lines = []
        for a in (result.assumptions or []):
            icon = "✅" if a.status == AssumptionStatus.PASS else ("⚠️" if a.status == AssumptionStatus.WARNING else "❌")
            assumption_lines.append(f"  {icon} {a.name}: {a.message}")
        assumptions_text = "\n".join(assumption_lines) if assumption_lines else "  (no assumptions recorded)"

        # Format additional metrics (group means, effect size labels, etc.)
        metrics = result.additional_metrics or {}
        metric_lines = []

        # Common metrics to surface
        keys_to_show = [
            ("group1_mean", "Group 1 Mean"),
            ("group2_mean", "Group 2 Mean"),
            ("mean_difference", "Mean Difference"),
            ("group1_n", "Group 1 N"),
            ("group2_n", "Group 2 N"),
            ("cohens_d", "Cohen's d"),
            ("cohens_d_magnitude", "Effect Magnitude"),
            ("eta_squared", "η² (Eta-Squared)"),
            ("eta_squared_magnitude", "Effect Magnitude"),
            ("rank_biserial_r", "Rank-Biserial r"),
            ("r_squared", "R²"),
            ("adjusted_r_squared", "Adjusted R²"),
            ("f_statistic", "F Statistic"),
            ("rmse", "RMSE"),
            ("slope", "Regression Slope"),
            ("intercept", "Intercept"),
            ("n_groups", "Number of Groups"),
            ("n_total", "Total N"),
            ("sample_size", "Sample Size"),
            ("variables_tested", "Variables Tested"),
            ("significant_variables", "Significant Variables"),
        ]

        for key, label in keys_to_show:
            val = metrics.get(key)
            if val is not None:
                metric_lines.append(f"  {label}: {val}")

        metrics_text = "\n".join(metric_lines) if metric_lines else "  (no additional metrics)"

        # Confidence interval
        ci_text = "Not available"
        if result.confidence_interval:
            ci_text = f"[{result.confidence_interval[0]}, {result.confidence_interval[1]}]"

        # Effect size
        es_text = str(result.effect_size) if result.effect_size is not None else "Not available"

        # p-value interpretation guidance
        p_val = result.p_value
        if p_val is not None:
            if p_val < 0.001:
                p_guidance = "The p-value is very highly significant (p < 0.001)."
            elif p_val < 0.01:
                p_guidance = "The p-value is highly significant (p < 0.01)."
            elif p_val < 0.05:
                p_guidance = "The p-value is significant (p < 0.05)."
            else:
                p_guidance = "The p-value is not significant (p ≥ 0.05)."
        else:
            p_guidance = "p-value not available."

        return f"""
You are a statistical interpretation assistant for a research verification tool.

IMPORTANT RULES:
- DO NOT perform calculations.
- DO NOT recalculate or verify any number.
- DO NOT change or second-guess any statistic provided.
- Only interpret and explain the pre-computed results below.

══════════════════════════════════════
STATISTICAL TEST RESULT
══════════════════════════════════════

Test: {result.test_name}
Statistic: {result.statistic}
p-value: {result.p_value}
Degrees of Freedom: {result.degrees_of_freedom}
Effect Size: {es_text}
95% Confidence Interval: {ci_text}

p-value context: {p_guidance}

══════════════════════════════════════
ADDITIONAL METRICS
══════════════════════════════════════
{metrics_text}

══════════════════════════════════════
STATISTICAL ASSUMPTIONS
══════════════════════════════════════
{assumptions_text}

══════════════════════════════════════
COMPUTED INTERPRETATION
══════════════════════════════════════
{result.interpretation}

══════════════════════════════════════
YOUR TASK
══════════════════════════════════════
Write a clear, structured explanation for a researcher who may not be a statistician. Include:

1. **Summary** (1–2 sentences): What was the main finding?
2. **What the statistics mean**: Explain the test statistic, p-value, and effect size in plain language.
3. **Confidence interval** (if available): What range of values is consistent with the data?
4. **Effect size**: Is the effect negligible, small, medium, or large? What does this mean practically?
5. **Assumptions**: Were any assumptions violated or concerning? What should the researcher be aware of?
6. **Practical implication**: What does this result mean for the research question?
7. **Recommendation**: Should the researcher consider any alternative analyses or caveats?

Keep the explanation professional but accessible. Use plain English, not jargon.
""".strip()


# ==========================================================
# AI Engine
# ==========================================================


class AIEngine:
    """
    Generates explanations only.
    """

    def __init__(self, client: GeminiClient | None = None):

        if client is None:
            self.client = GeminiClient()
        else:
            self.client = client

    def explain(
        self,
        result: StatisticalResult,
    ) -> AIExplanation:

        prompt = PromptBuilder.build(result)

        explanation = self.client.generate(prompt)

        # Generate a concise summary line from the result
        summary = _build_summary(result)

        return AIExplanation(
            summary=summary,
            explanation=explanation,
            recommendation=_build_recommendation(result),
        )


# ==========================================================
# Summary & Recommendation helpers
# ==========================================================

def _build_summary(result: StatisticalResult) -> str:
    """
    Build a one-line summary from the StatisticalResult
    without performing any calculation.
    """

    test = result.test_name or "Statistical test"
    p = result.p_value

    if p is None:
        return f"{test} completed — see detailed results."

    if p < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p < 0.01:
        sig = f"significant (p = {p:.3f})"
    elif p < 0.05:
        sig = f"significant (p = {p:.3f})"
    else:
        sig = f"not significant (p = {p:.3f})"

    return f"{test}: result is {sig}."


def _build_recommendation(result: StatisticalResult) -> str | None:
    """
    Build a recommendation string based on assumption checks.
    """

    if not result.assumptions:
        return None

    failing = [
        a for a in result.assumptions
        if a.status == AssumptionStatus.FAIL
    ]
    warning = [
        a for a in result.assumptions
        if a.status == AssumptionStatus.WARNING
    ]

    if failing:
        names = ", ".join(a.name for a in failing)
        return (
            f"Critical assumption(s) failed: {names}. "
            f"Interpret results with caution and consider an alternative test."
        )

    if warning:
        names = ", ".join(a.name for a in warning)
        return (
            f"Warning on: {names}. "
            f"Review assumption details and consider robustness checks."
        )

    return "All assumptions passed. Results can be interpreted with confidence."