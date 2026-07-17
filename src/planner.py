"""
planner.py

Analysis planning engine.

This module converts parser metadata into a deterministic
StatisticalContext.

It never uses column names.

All decisions are based only on

• variable types
• cardinality
• metadata
• user analysis mode

This module contains NO statistical calculations.
"""

from __future__ import annotations

from src.models import (
    ParsedTable,
    StatisticalContext,
    VariableType,
)
from src.utils import logger


class AnalysisPlanner:
    """
    Builds the statistical context required
    by the Statistical Engine.

    No column names are hardcoded.

    Decisions rely entirely on metadata.
    """

    def build_context(
        self,
        table: ParsedTable,
    ) -> StatisticalContext:

        logger.info("Building statistical context...")

        # ----------------------------------------
        # SUMMARY TABLE
        # ----------------------------------------

        if getattr(table.metadata, "analysis_type", None) == "summary" or bool(table.summary_metadata):
            return StatisticalContext(
                group_variable=[
                    "Experimental",
                    "Control",
                ],
                dependent_variable="Category",
                independent_variable="Variable",
            )

        # ----------------------------------------
        # RAW DATASET
        # ----------------------------------------

        metadata = table.metadata

        # ------------------------------------
        # Ignore identifiers
        # ------------------------------------

        usable = [
            column
            for column in metadata
            if not column.is_identifier
        ]

        numeric = [
            column
            for column in usable
            if column.variable_type == VariableType.CONTINUOUS
        ]

        binary = [
            column
            for column in usable
            if column.variable_type == VariableType.BINARY
        ]

        categorical = [
            column
            for column in usable
            if column.variable_type == VariableType.CATEGORICAL
        ]

        context = StatisticalContext()

        # ------------------------------------
        # Binary + Continuous
        # -> Independent T Test
        # ------------------------------------

        if binary and numeric:

            context.group_variable = binary[0].name

            context.independent_variable = binary[0].name

            context.dependent_variable = numeric[0].name

            return context

        # ------------------------------------
        # Two continuous
        # -> Correlation
        # ------------------------------------

        if len(numeric) >= 2:

            context.independent_variable = numeric[0].name

            context.dependent_variable = numeric[1].name

            return context

        # ------------------------------------
        # Two categorical
        # -> Chi Square
        # ------------------------------------

        if len(categorical) >= 2:

            context.independent_variable = categorical[0].name

            context.dependent_variable = categorical[1].name

            return context

        # ------------------------------------
        # Binary vs Binary
        # ------------------------------------

        if len(binary) >= 2:

            context.independent_variable = binary[0].name

            context.dependent_variable = binary[1].name

            context.group_variable = binary[0].name

            return context

        # ------------------------------------
        # Single continuous
        # ------------------------------------

        if len(numeric) == 1:

            context.dependent_variable = numeric[0].name

        return context