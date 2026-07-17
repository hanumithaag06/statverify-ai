"""
AI Explanation Layer

Responsibilities
----------------
✓ Explain statistical results
✓ Never perform calculations
✓ Uses Gemini 2.5 Flash
"""

from __future__ import annotations

import os

import google.generativeai as genai

from src.models import (
    AIExplanation,
    StatisticalResult,
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
                "This statistical result has been interpreted "
                "without using an external AI service."
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
    Builds prompts for Gemini.

    IMPORTANT:
    AI must NEVER calculate statistics.
    """

    @staticmethod
    def build(result: StatisticalResult) -> str:

        return f"""
You are a statistical interpretation assistant.

DO NOT perform calculations.
DO NOT recalculate statistics.
DO NOT verify numbers.

Only explain the supplied statistical result.

Test:
{result.test_name}

Statistic:
{result.statistic}

p-value:
{result.p_value}

Interpretation:
{result.interpretation}

Provide:

1. Simple summary
2. Meaning of the result
3. Practical interpretation
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

        return AIExplanation(
            summary="AI Explanation",
            explanation=explanation,
            recommendation=None,
        )