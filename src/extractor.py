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
✓ Populate sample_size for summary datasets
✓ Propagate group_names from summary_metadata
✓ Never perform statistical calculations
"""

from __future__ import annotations

import pandas as pd
import numpy as np

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

            detected_sizes = table.summary_metadata.get("_sample_sizes", {})

            # Infer group_names from summary_metadata (first variable's groups list)
            inferred_group_names: list[str] = []
            for var_key, var_meta in table.summary_metadata.items():
                if var_key.startswith("_"):
                    continue
                if isinstance(var_meta, dict) and "groups" in var_meta:
                    inferred_group_names = list(var_meta["groups"])
                    break

            dataset = StatisticalDataset(
                dataframe=df,
                metadata=table.metadata,
                summary_metadata=table.summary_metadata,
                sample_sizes=detected_sizes,
                group_names=inferred_group_names,
                is_summary=True,
            )

            convert_percentages = (
                "Experimental" in detected_sizes
                and "Control" in detected_sizes
            )

            tables = {}
            category_labels = {}

            for variable in df["Variable"].dropna().unique():
                sub = df[df["Variable"] == variable]
                contingency = (
                    sub[["Experimental", "Control"]]
                    .fillna(0)
                    .astype(float)
                )
                contingency.index = sub["Sub Variable"]

                if contingency.values.sum() == 0:
                    continue

                table_arr = contingency.to_numpy()

                if convert_percentages:
                    exp_n = detected_sizes["Experimental"]
                    ctrl_n = detected_sizes["Control"]

                    counts = []
                    for row in table_arr:
                        counts.append([
                            round(row[0] * exp_n / 100),
                            round(row[1] * ctrl_n / 100),
                        ])

                    table_arr = np.asarray(counts, dtype=float)

                tables[variable] = table_arr
                category_labels[variable] = contingency.index.astype(str).tolist()

            dataset.contingency_tables = tables
            dataset.category_labels = category_labels

            # Populate sample_size as total subjects across all groups
            if detected_sizes:
                dataset.sample_size = int(sum(detected_sizes.values()))
            elif tables:
                # Fall back: sum all counts in first contingency table
                first_table = next(iter(tables.values()))
                dataset.sample_size = int(np.sum(first_table))

            logger.info(
                f"Summary dataset extracted: {len(tables)} variables, "
                f"N={dataset.sample_size}, groups={inferred_group_names}"
            )
            return dataset

        if getattr(table, "is_summary_table", False) or bool(
            table.summary_metadata
        ):
            return self._extract_summary(dataset, context)

        return self._extract_raw(dataset, context)

    # ==========================================================
    # Role-Mapping Extraction
    # ==========================================================

    def extract_with_role_mapping(
        self,
        table: ParsedTable,
        mapping: "ColumnRoleMapping",
    ) -> StatisticalDataset:
        """
        Build a StatisticalDataset from a user-confirmed column role
        mapping. Used for summary tables that weren't recognized via
        header keywords — grouping/sub-category/value column names,
        and count-vs-percentage semantics, come entirely from the
        mapping. Nothing here is hardcoded or dataset-specific.
        """

        from src.models import ValueType

        df = table.dataframe.copy()

        if mapping.grouping_column:
            df[mapping.grouping_column] = (
                df[mapping.grouping_column].replace("", None).ffill()
            )
            group_iter = df.groupby(mapping.grouping_column)
        else:
            # Single-variable table: treat the whole frame as one group.
            group_iter = [("Value", df)]

        contingency_tables: dict[str, np.ndarray] = {}
        category_labels: dict[str, list[str]] = {}

        convert_percentages = mapping.value_type == ValueType.PERCENTAGE

        for variable_name, sub_df in group_iter:

            values = (
                sub_df[mapping.group_value_columns]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
            )

            if values.values.sum() == 0:
                continue

            table_arr = values.to_numpy(dtype=float)

            if convert_percentages:
                counts = []
                for row in table_arr:
                    counts.append(
                        [
                            round(
                                row[i]
                                * mapping.group_sample_sizes.get(col, 0)
                                / 100
                            )
                            for i, col in enumerate(mapping.group_value_columns)
                        ]
                    )
                table_arr = np.asarray(counts, dtype=float)

            contingency_tables[str(variable_name)] = table_arr
            category_labels[str(variable_name)] = (
                sub_df[mapping.sub_category_column].astype(str).tolist()
            )

        # Compute total sample size across all groups
        total_n = int(sum(mapping.group_sample_sizes.values())) if mapping.group_sample_sizes else 0
        if total_n == 0 and contingency_tables:
            first = next(iter(contingency_tables.values()))
            total_n = int(np.sum(first))

        dataset = StatisticalDataset(
            dataframe=df,
            metadata=table.metadata,
            summary_metadata=table.summary_metadata,
            sample_sizes=mapping.group_sample_sizes,
            group_names=list(mapping.group_value_columns),
            sample_size=total_n,
            is_summary=True,
        )
        dataset.contingency_tables = contingency_tables
        dataset.category_labels = category_labels

        logger.info(
            f"Role-mapped dataset extracted: {len(contingency_tables)} variables, "
            f"N={total_n}, groups={list(mapping.group_value_columns)}"
        )

        return dataset

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

        dependent = pd.to_numeric(
            df[context.dependent_variable],
            errors="coerce",
        )

        independent_numeric = pd.to_numeric(
            df[context.independent_variable],
            errors="coerce",
        )

        is_independent_numeric = (
            independent_numeric.notna().mean() > 0.95
        )

        if is_independent_numeric:

            paired = pd.DataFrame(
                {
                    "x": independent_numeric,
                    "y": dependent,
                }
            ).dropna()

            dataset.x = paired["x"].tolist()
            dataset.y = paired["y"].tolist()

            dataset.numeric_columns[context.independent_variable] = (
                paired["x"].tolist()
            )
            dataset.numeric_columns[context.dependent_variable] = (
                paired["y"].tolist()
            )

        else:

            dataset.numeric_columns[
                context.dependent_variable
            ] = dependent.dropna().tolist()

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

        # Filter to only columns that actually exist in the dataframe
        valid_cols = [c for c in columns if c is not None and c in dataframe.columns]

        if not valid_cols:
            return dataframe.copy()

        cleaned = dataframe.dropna(
            subset=valid_cols
        ).copy()

        cleaned.reset_index(
            drop=True,
            inplace=True,
        )

        return cleaned