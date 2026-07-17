"""
Universal Research Table Parser.

The parser converts multiple research table formats into a
normalized pandas DataFrame.

Current Supported Inputs
------------------------
✔ Clipboard Text
✔ CSV
✔ Excel (.xlsx, .xls)

Future Support
--------------
□ PDF
□ Images
□ OCR
□ HTML Tables

Workflow
--------
Input
    ↓
Detect Input Type
    ↓
Load Data
    ↓
Return DataFrame

Cleaning, schema inference and validation are implemented
in later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.models import (
    ColumnMetadata,
    DatasetMetadata,
    InputType,
    ParsedTable,
    VariableType,
)
from src.utils import logger, log_end, log_error, log_start


class ParserError(Exception):
    """Raised when parser fails to load the input."""


class UnsupportedInputError(ParserError):
    """Raised for unsupported file formats."""


class UniversalParser:
    """
    Universal parser for research tables.

    Notes
    -----
    This class is responsible ONLY for loading data.

    It does not:

    - clean data
    - validate data
    - infer schema
    - perform statistics

    Those responsibilities belong to later phases.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv": InputType.CSV,
        ".xlsx": InputType.EXCEL,
        ".xls": InputType.EXCEL,
    }

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def parse(
        self,
        source: str | Path | BinaryIO,
        input_type: InputType | None = None,
    ) -> ParsedTable:
        """
        Parse any supported input.

        Parameters
        ----------
        source
            File path, clipboard string or uploaded file.

        input_type
            Optional explicit input type.

        Returns
        -------
        pandas.DataFrame
        """

        log_start("Universal Parser")

        try:

            detected = input_type or self._detect_input_type(source)

            logger.info(f"Detected input type: {detected}")

            if detected == InputType.CSV:
                dataframe = self._load_csv(source)

            elif detected == InputType.EXCEL:
                dataframe = self._load_excel(source)

            elif detected == InputType.CLIPBOARD:
                dataframe = self._load_clipboard(source)

            else:
                raise UnsupportedInputError(
                    f"{detected} is currently unsupported."
                )

            dataframe = self._clean_dataframe(dataframe)

            table_type = self._detect_table_type(dataframe)

            if table_type == "SUMMARY":

                logger.info("Summary table detected.")

                dataframe = self._normalize_summary_table(dataframe)

                summary_metadata = self._infer_summary_metadata(dataframe)

            else:

                summary_metadata = {}

            log_end("Universal Parser")

            metadata = self._infer_schema(dataframe)

            dataset = DatasetMetadata(
                rows=len(dataframe),
                columns=len(dataframe.columns),
                duplicate_rows=int(dataframe.duplicated().sum()),
                duplicate_columns=int(dataframe.columns.duplicated().sum()),
                total_missing_cells=int(dataframe.isna().sum().sum()),
                memory_usage_bytes=int(
                    dataframe.memory_usage(deep=True).sum()
                ),
            )

            return ParsedTable(
                dataframe=dataframe,
                metadata=metadata,
                summary_metadata=summary_metadata,
            )

        except Exception as error:

            log_error("Universal Parser", error)

            raise ParserError(str(error)) from error

    # ---------------------------------------------------------
    # Input Detection
    # ---------------------------------------------------------

    def _detect_input_type(
        self,
        source: str | Path | BinaryIO,
    ) -> InputType:
        """
        Automatically detect input type.
        """

        # Uploaded Streamlit file
        if hasattr(source, "name"):

            suffix = Path(source.name).suffix.lower()

            if suffix in self.SUPPORTED_EXTENSIONS:
                return self.SUPPORTED_EXTENSIONS[suffix]

            raise UnsupportedInputError(
                f"Unsupported extension: {suffix}"
            )

        # Local file
        if isinstance(source, (str, Path)):

            path = Path(source)

            if path.exists():

                suffix = path.suffix.lower()

                if suffix in self.SUPPORTED_EXTENSIONS:
                    return self.SUPPORTED_EXTENSIONS[suffix]

                raise UnsupportedInputError(
                    f"Unsupported extension: {suffix}"
                )

            # Otherwise assume clipboard text
            return InputType.CLIPBOARD

        raise UnsupportedInputError(
            "Unable to determine input type."
        )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    def _load_csv(
        self,
        source: str | Path | BinaryIO,
    ) -> pd.DataFrame:
        """
        Load CSV.
        """

        logger.info("Loading CSV...")

        return pd.read_csv(source)

    # ---------------------------------------------------------
    # Excel
    # ---------------------------------------------------------

    def _load_excel(
        self,
        source: str | Path | BinaryIO,
    ) -> pd.DataFrame:
        """
        Load Excel.
        """

        logger.info("Loading Excel...")

        return pd.read_excel(source)

    # ---------------------------------------------------------
    # Clipboard
    # ---------------------------------------------------------

    def _load_clipboard(
        self,
        source: str,
    ) -> pd.DataFrame:
        """
        Parse clipboard text.

        Supports tab-separated values copied directly
        from Excel.
        """

        logger.info("Parsing clipboard text...")

        from io import StringIO

        return pd.read_csv(
            StringIO(source),
            sep=r"\t|,",
            engine="python",
        )

    # ---------------------------------------------------------
    # Cleaning Pipeline
    # ---------------------------------------------------------

    def _clean_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the complete cleaning pipeline.

        This method intentionally performs only generic
        cleaning operations. It never applies research-
        specific assumptions.
        """

        logger.info("Cleaning dataframe...")

        dataframe = dataframe.copy()

        dataframe = self._remove_empty_rows(dataframe)

        dataframe = self._remove_empty_columns(dataframe)

        dataframe = self._normalize_column_names(dataframe)

        dataframe = self._normalize_string_values(dataframe)

        dataframe = self._replace_missing_values(dataframe)

        logger.info("Cleaning completed.")

        return dataframe

    # ---------------------------------------------------------
    # Empty Rows
    # ---------------------------------------------------------

    def _remove_empty_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove rows where every value is missing.
        """

        before = len(dataframe)

        dataframe = dataframe.dropna(how="all")

        removed = before - len(dataframe)

        logger.info(f"Removed {removed} empty rows.")

        return dataframe.reset_index(drop=True)

    # ---------------------------------------------------------
    # Empty Columns
    # ---------------------------------------------------------

    def _remove_empty_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove columns where every value is missing.
        """

        before = dataframe.shape[1]

        dataframe = dataframe.dropna(axis=1, how="all")

        removed = before - dataframe.shape[1]

        logger.info(f"Removed {removed} empty columns.")

        return dataframe

    # ---------------------------------------------------------
    # Header Cleaning
    # ---------------------------------------------------------

    def _normalize_column_names(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize column names.

        Operations:
        - Trim spaces
        - Replace multiple spaces
        - Convert to string
        - Ensure uniqueness
        """

        columns = []

        seen = {}

        for column in dataframe.columns:

            name = " ".join(str(column).split())

            if name == "":
                name = "Unnamed"

            if name not in seen:

                seen[name] = 1
                columns.append(name)

            else:

                seen[name] += 1

                columns.append(f"{name}_{seen[name]}")

        dataframe.columns = columns

        return dataframe

    # ---------------------------------------------------------
    # Cell Cleaning
    # ---------------------------------------------------------

    def _normalize_string_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Trim whitespace from every string cell.
        """

        dataframe = dataframe.map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

        return dataframe

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def _replace_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Standardize common missing-value representations.

        These are converted into pandas.NA.
        """

        missing_tokens = {

            "",

            "NA",

            "N/A",

            "NULL",

            "None",

            "-",

            "--",

            "nan",

            "NaN",

        }

        dataframe = dataframe.replace(
            list(missing_tokens),
            pd.NA,
        )

        return dataframe

    # ==========================================================
    # Summary Table Detection
    # ==========================================================

    def _detect_table_type(self, dataframe: pd.DataFrame) -> str:
        """
        Detect whether the dataframe is a RAW dataset or a SUMMARY table.

        A keyword-scoring approach is used so that tables from research
        papers are recognised even when column names vary slightly.

        Returns
        -------
        "RAW"
        "SUMMARY"
        """

        columns = [
            str(c).strip().lower()
            for c in dataframe.columns
        ]

        summary_keywords = [
            "sub variables",
            "sub-variable",
            "subcategory",
            "experimental",
            "control",
            "percentage",
            "%",
            "frequency",
            "count",
        ]

        score = 0

        for keyword in summary_keywords:
            if any(keyword in col for col in columns):
                score += 1

        if score >= 2:
            return "SUMMARY"

        return "RAW"

    # ==========================================================
    # Summary Table Normalization
    # ==========================================================

    def _normalize_summary_table(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize summary tables copied from research papers.

        - Strips whitespace from column names.
        - Renames common columns to canonical names
          (``Variable``, ``Sub Variable``, ``Experimental``, ``Control``).
        - Forward-fills the ``Variable`` column to fill merged cells.
        - Drops fully-empty rows.

        Example
        -------
        Before::

            Age   | 1–2 Years
            (blank)| 2–3 Years

        After::

            Age   | 1–2 Years
            Age   | 2–3 Years
        """

        df = dataframe.copy()

        # Remove whitespace from column names
        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        # Rename common columns to canonical names
        renamed = {}

        for column in df.columns:

            lower = column.lower()

            if "experimental" in lower:
                renamed[column] = "Experimental"

            elif "control" in lower:
                renamed[column] = "Control"

            elif "sub" in lower:
                renamed[column] = "Sub Variable"

            elif "variable" in lower:
                renamed[column] = "Variable"

        df.rename(columns=renamed, inplace=True)

        # Forward-fill merged Variable cells
        if "Variable" in df.columns:
            df["Variable"] = (
                df["Variable"]
                .replace("", None)
                .ffill()
            )

        # Remove fully-empty rows
        df = df.dropna(how="all")

        return df.reset_index(drop=True)

    # ==========================================================
    # Summary Metadata
    # ==========================================================

    def _infer_summary_metadata(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Generate variable-level metadata for summary tables.

        Returns
        -------
        dict
            Keys are variable names; values are dicts with
            ``categories``, ``groups``, and ``rows``.
        """

        metadata: dict = {}

        if "Variable" not in dataframe.columns:
            return metadata

        variables = dataframe["Variable"].unique()

        for variable in variables:

            subset = dataframe[
                dataframe["Variable"] == variable
            ]

            categories: list[str] = []

            if "Sub Variable" in subset.columns:
                categories = (
                    subset["Sub Variable"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

            groups: list[str] = []

            if "Experimental" in subset.columns:
                groups.append("Experimental")

            if "Control" in subset.columns:
                groups.append("Control")

            metadata[variable] = {
                "categories": categories,
                "groups": groups,
                "rows": len(subset),
            }

        return metadata

    # ---------------------------------------------------------
    # Feature Inference
    # ---------------------------------------------------------

    def _infer_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> list[ColumnMetadata]:
        """
        Infer metadata for every dataframe column.
        """

        logger.info("Inferring dataframe schema...")

        metadata = []

        total_rows = len(dataframe)

        for column in dataframe.columns:

            series = dataframe[column]

            missing = int(series.isna().sum())

            unique = int(series.nunique(dropna=True))

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            non_missing = series.notna().sum()

            numeric_ratio = (
                numeric.notna().sum() / non_missing
                if non_missing
                else 0
            )

            text_ratio = (
                1 - numeric_ratio
                if non_missing
                else 0
            )

            inferred = self._infer_variable_type(column, series)

            is_constant = (unique == 1 and int(non_missing) > 0)

            is_identifier = inferred == VariableType.IDENTIFIER

            is_binary = inferred == VariableType.BINARY


            metadata.append(
                ColumnMetadata(
                    name=column,
                    variable_type=inferred,
                    pandas_dtype=str(series.dtype),
                    total_values=total_rows,
                    non_missing_values=int(non_missing),
                    missing_values=int(missing),
                    unique_values=int(unique),
                    unique_ratio=(
                        unique / non_missing
                        if non_missing
                        else 0
                    ),
                    numeric_ratio=numeric_ratio,
                    text_ratio=text_ratio,
                    is_numeric=numeric_ratio > 0.95,
                    is_text=text_ratio > 0.95,
                    is_binary=is_binary,
                    is_constant=is_constant,
                    is_identifier=is_identifier,
                    inferred_type=inferred.value,
                )
            )

        return metadata

    def _infer_variable_type(
        self,
        column_name: str,
        series: pd.Series,
    ) -> VariableType:
        """
        Infer the statistical variable type.
        """

        clean = series.dropna()

        if clean.empty:
            return VariableType.UNKNOWN

        unique = clean.nunique()

        numeric = pd.to_numeric(
            clean,
            errors="coerce",
        )

        numeric_ratio = numeric.notna().mean()

        text_ratio = 1 - numeric_ratio

        # -------------------------
        # Pre-computed flags
        # -------------------------

        column_lower = column_name.lower()

        identifier_keywords = {
            "id",
            "patient_id",
            "subject_id",
            "participant_id",
            "sample_id",
            "record_id",
            "uuid",
            "uid",
            "roll",
            "employee_id",
            "student_id",
        }

        name_match = any(
            keyword in column_lower
            for keyword in identifier_keywords
        )

        is_identifier = (
            name_match
            and unique / len(clean) > 0.95
        )

        is_binary_text = {
            str(v).strip().lower()
            for v in clean.unique()
        } in [
            {"yes", "no"},
            {"true", "false"},
            {"male", "female"},
            {"m", "f"},
            {"0", "1"},
            {"control", "experimental"},
            {"case", "control"},
        ]

        is_binary = unique == 2 and (numeric_ratio > 0.95 or is_binary_text)

        # -------------------------
        # Classification
        # -------------------------

        if is_identifier:
            return VariableType.IDENTIFIER

        if is_binary:
            return VariableType.BINARY

        if numeric_ratio > 0.95:
            return VariableType.CONTINUOUS

        if text_ratio > 0.95:
            return VariableType.CATEGORICAL

        return VariableType.UNKNOWN