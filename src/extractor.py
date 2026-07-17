"""
extractor.py

Statistical data extraction layer.

This module converts a ParsedTable into clean statistical datasets
required by the statistical engine.

Responsibilities
----------------
✓ Remove missing values
✓ Convert numeric columns
✓ Build grouped datasets
✓ Build contingency tables
✓ Never perform statistical calculations
"""

from __future__ import annotations

import pandas as pd

from src.models import (
    ParsedTable,
    StatisticalContext,
    StatisticalDataset,
)
from src.utils import logger


class StatisticalDataExtractor:
    """
    Extract clean statistical datasets from ParsedTable.

    This class performs ONLY data preparation.

    It never computes statistics.
    """

    # ==========================================================
    # Public API
    # ==========================================================

    def extract(
        self,
        table: ParsedTable,
        context: StatisticalContext,
    ) -> StatisticalDataset:
        """
        Build a StatisticalDataset from the parsed table.
        """

        logger.info("Extracting statistical dataset...")

        dataset = StatisticalDataset(
            dataframe=table.dataframe.copy()
        )

        if getattr(table.metadata, "analysis_type", None) == "summary":
            df = table.dataframe
            tables = {}
            total_sum = 0
            if "Variable" in df.columns:
                for variable in df["Variable"].unique():
                    subset = df[df["Variable"] == variable]
                    cols = [c for c in ["Experimental", "Control"] if c in df.columns]
                    if cols:
                        tables[variable] = subset[cols].astype(float).to_numpy()
                        total_sum += float(tables[variable].sum())
            else:
                cols = [c for c in ["Experimental", "Control"] if c in df.columns]
                if cols:
                    tables["Summary"] = df[cols].astype(float).to_numpy()
                    total_sum += float(tables["Summary"].sum())
            dataset.contingency_tables = tables
            dataset.sample_size = int(total_sum)
            return dataset

        if getattr(table, "is_summary_table", False) or bool(
            table.summary_metadata
        ):
            return self._extract_summary(dataset, context)

        return self._extract_raw(dataset, context)

    # ==========================================================
    # Summary Extraction
    # ==========================================================

    def _extract_summary(
        self,
        dataset: StatisticalDataset,
        context: StatisticalContext,
    ) -> StatisticalDataset:
        """
        Extract summary-table counts.
        """

        df = dataset.dataframe

        group_columns = context.group_variable

        if not group_columns:
            return dataset

        for column in group_columns:

            dataset.numeric_columns[column] = (
                df[column]
                .fillna(0)
                .astype(float)
                .tolist()
            )

        dataset.sample_size = int(
            sum(dataset.numeric_columns[group_columns[0]])
        )

        return dataset

    # ==========================================================
    # Raw Extraction
    # ==========================================================

    def _extract_raw(
        self,
        dataset: StatisticalDataset,
        context: StatisticalContext,
    ) -> StatisticalDataset:
        """
        Extract data from a raw dataset.
        """

        if context.group_variable:
            self._extract_groups(dataset, context)

        elif (
            context.independent_variable
            and context.dependent_variable
        ):
            self._extract_numeric(dataset, context)

        return dataset

    # ==========================================================
    # Numeric Extraction
    # ==========================================================

    def _extract_numeric(
        self,
        dataset: StatisticalDataset,
        context: StatisticalContext,
    ) -> None:
        """
        Extract paired numeric variables.

        Used for:
        - Correlation
        - Regression
        """

        df = self._remove_missing(
            dataset.dataframe,
            [
                context.independent_variable,
                context.dependent_variable,
            ],
        )

        dataset.numeric_columns[
            context.dependent_variable
        ] = (
            pd.to_numeric(
                df[context.dependent_variable],
                errors="coerce",
            )
            .dropna()
            .tolist()
        )

        if context.independent_variable:

            dataset.categorical_columns[
                context.independent_variable
            ] = (
                df[context.independent_variable]
                .dropna()
                .astype(str)
                .tolist()
            )

        dataset.sample_size = len(df)

    # ==========================================================
    # Group Extraction
    # ==========================================================

    def _extract_groups(
        self,
        dataset: StatisticalDataset,
        context: StatisticalContext,
    ) -> None:
        """
        Extract grouped numeric observations.

        Used for:
        - T-Test
        - ANOVA
        - Mann-Whitney
        - Kruskal-Wallis
        """

        df = self._remove_missing(
            dataset.dataframe,
            [
                context.group_variable,
                context.dependent_variable,
            ],
        )

        group_column = context.group_variable
        value_column = context.dependent_variable

        groups = {}

        for group_name, group_df in df.groupby(group_column):

            values = (
                pd.to_numeric(
                    group_df[value_column],
                    errors="coerce",
                )
                .dropna()
                .tolist()
            )

            groups[str(group_name)] = values

        dataset.groups = groups

        dataset.sample_size = sum(
            len(values)
            for values in groups.values()
        )

    # ==========================================================
    # Contingency Table
    # ==========================================================

    def extract_contingency(
        self,
        table: ParsedTable,
        row_variable: str,
        column_variable: str,
    ) -> list[list[int]]:
        """
        Build a contingency table.

        Used by Chi-Square.
        """

        df = self._remove_missing(
            table.dataframe,
            [
                row_variable,
                column_variable,
            ],
        )

        contingency = pd.crosstab(
            df[row_variable],
            df[column_variable],
        )

        return contingency.values.tolist()

    # ==========================================================
    # Shared Utilities
    # ==========================================================

    def _remove_missing(
        self,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Remove rows with missing values
        in the specified columns.
        """

        cleaned = dataframe.dropna(
            subset=columns
        ).copy()

        cleaned.reset_index(
            drop=True,
            inplace=True,
        )

        return cleaned