"""
Universal Research Table Parser.

The parser converts multiple research table formats into a
normalized pandas DataFrame.

Supported Inputs
----------------
✔ CSV          (.csv)
✔ Excel        (.xlsx, .xls)
✔ PDF Tables   (.pdf via pdfplumber / pypdf)
✔ Image Tables (.png, .jpg, .jpeg, .tiff via PIL / pytesseract / Gemini Vision)
✔ Clipboard    (Tab-separated or comma-separated raw text)
✔ Multi-File   (Batch processing of multiple heterogeneous files)

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

import re
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
from src.config import settings
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
        # Stubs — recognized but not yet parseable
        ".pdf": InputType.PDF,
        ".png": InputType.IMAGE,
        ".jpg": InputType.IMAGE,
        ".jpeg": InputType.IMAGE,
        ".tiff": InputType.IMAGE,
        ".tif": InputType.IMAGE,
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

            elif detected == InputType.PDF:
                dataframe = self._load_pdf(source)

            elif detected == InputType.IMAGE:
                dataframe = self._load_image(source)

            else:
                raise UnsupportedInputError(
                    f"{detected} is currently unsupported."
                )

            dataframe = self._clean_dataframe(dataframe)

            table_type = self._detect_table_type(dataframe)

            is_summary_shaped = False
            requires_role_confirmation = False
            candidate_grouping = None
            candidate_subcategory = None
            candidate_values: list[str] = []

            if table_type == "SUMMARY":

                logger.info("Summary table detected via header keywords.")

                dataframe, group_sample_sizes = self._normalize_summary_table(dataframe)

                summary_metadata = self._infer_summary_metadata(dataframe)

                is_summary_shaped = True

                looks_like_percentages = self._values_look_like_percentages(dataframe)

                if group_sample_sizes:
                    # N was found in the headers — fully deterministic,
                    # no confirmation needed regardless of count/percentage.
                    summary_metadata["_sample_sizes"] = group_sample_sizes

                elif looks_like_percentages:
                    # Values look like percentages but no N was found in
                    # headers — group sizes cannot be assumed, so ask
                    # the user instead of guessing.
                    logger.info(
                        "Percentage-shaped summary table with no N in "
                        "headers; role confirmation required."
                    )
                    requires_role_confirmation = True

                    value_cols = [
                        c for c in ("Experimental", "Control")
                        if c in dataframe.columns
                    ]

                    candidate_grouping = (
                        "Variable" if "Variable" in dataframe.columns else None
                    )
                    candidate_subcategory = (
                        "Sub Variable"
                        if "Sub Variable" in dataframe.columns
                        else None
                    )
                    candidate_values = value_cols

                # else: raw counts with no N — nothing to convert,
                # existing deterministic path handles it as-is.

            else:

                summary_metadata = {}

                # Header keywords didn't match at all — fall back to
                # structural detection so arbitrarily-named
                # group-comparison tables are still recognized.
                (
                    is_summary_shaped,
                    candidate_grouping,
                    candidate_subcategory,
                    candidate_values,
                ) = self._detect_summary_shape(dataframe)

                if is_summary_shaped:
                    logger.info(
                        "Summary-shaped table detected structurally; "
                        "role confirmation required."
                    )
                    requires_role_confirmation = True

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
                is_summary_shaped=is_summary_shaped,
                requires_role_confirmation=requires_role_confirmation,
                candidate_grouping_column=candidate_grouping,
                candidate_subcategory_column=candidate_subcategory,
                candidate_value_columns=candidate_values,
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
    # PDF Loader
    # ---------------------------------------------------------

    def _load_pdf(
        self,
        source: str | Path | BinaryIO,
    ) -> pd.DataFrame:
        """
        Extract tabular data from PDF documents using pdfplumber or pypdf.
        """

        logger.info("Extracting tables from PDF...")

        # 1. Try pdfplumber for structured PDF tables
        try:
            import pdfplumber

            tables = []
            with pdfplumber.open(source) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_tables()
                    for table in extracted:
                        if table and len(table) > 1:
                            # Use first row as header if valid
                            df_table = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df_table)

            if tables:
                logger.info(f"pdfplumber extracted {len(tables)} table(s) from PDF.")
                return pd.concat(tables, ignore_index=True)

        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}. Trying pypdf...")

        # 2. Try pypdf text extraction fallback
        try:
            from pypdf import PdfReader
            from io import StringIO

            reader = PdfReader(source)
            text_lines = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_lines.extend(text.splitlines())

            if text_lines:
                raw_text = "\n".join(text_lines)
                logger.info("pypdf extracted text lines from PDF.")
                return pd.read_csv(
                    StringIO(raw_text),
                    sep=r"\t|,|\s{2,}",
                    engine="python",
                )

        except Exception as e2:
            logger.error(f"pypdf extraction failed: {e2}")

        raise ParserError(
            "Could not extract tabular data from the PDF file. "
            "Ensure the PDF contains selectable text or table elements."
        )

    # ---------------------------------------------------------
    # Image Loader
    # ---------------------------------------------------------

    def _load_image(
        self,
        source: str | Path | BinaryIO,
    ) -> pd.DataFrame:
        """
        Extract tabular data from images using PIL, pytesseract, or Gemini Vision.
        """

        logger.info("Extracting tables from Image...")

        # 1. Try pytesseract if installed
        try:
            import pytesseract
            from PIL import Image
            from io import StringIO

            img = Image.open(source)
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                logger.info("pytesseract OCR extracted text from image.")
                return pd.read_csv(
                    StringIO(text.strip()),
                    sep=r"\t|,|\s{2,}",
                    engine="python",
                )

        except Exception as e:
            logger.warning(f"pytesseract OCR not available or failed: {e}")

        # 2. Try Gemini Vision fallback if GEMINI_API_KEY is available
        import os
        api_key = getattr(settings, "gemini_api_key", "") or os.getenv("GEMINI_API_KEY")
        if api_key:
            import google.generativeai as genai
            from PIL import Image
            from io import StringIO

            genai.configure(api_key=api_key)
            img = Image.open(source)
            prompt = (
                "Extract the table from this image into clean CSV format. "
                "Return ONLY valid raw CSV text with headers. Do not wrap in markdown or add commentary."
            )

            for model_name in ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-1.5-flash", "gemini-flash-latest"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content([prompt, img])
                    csv_text = response.text.strip().replace("```csv", "").replace("```", "").strip()
                    if csv_text:
                        logger.info(f"Gemini Vision ({model_name}) extracted CSV table from image.")
                        return pd.read_csv(StringIO(csv_text))
                except Exception as e_m:
                    logger.warning(f"Model {model_name} extraction failed: {e_m}")

        raise ParserError(
            "Image table OCR requires pytesseract or a valid GEMINI_API_KEY for vision model extraction."
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
    # Structural Summary Shape Detection
    # ==========================================================

    def _detect_summary_shape(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[bool, str | None, str | None, list[str]]:
        """
        Structurally detect a group-comparison summary table without
        relying on header keywords.

        A table qualifies if it has:
          - at least 2 columns that are ≥90% numeric (candidate group
            value columns), and
          - at least 1 remaining non-numeric column (candidate
            sub-category / grouping column).

        Returns
        -------
        (is_summary_shaped, candidate_grouping_col, candidate_subcategory_col, candidate_value_cols)
        """

        numeric_cols = []
        text_cols = []

        for column in dataframe.columns:

            series = dataframe[column]
            non_missing = series.notna().sum()

            if non_missing == 0:
                continue

            numeric_ratio = (
                pd.to_numeric(series, errors="coerce").notna().sum()
                / non_missing
            )

            if numeric_ratio >= 0.9:
                numeric_cols.append(column)
            else:
                text_cols.append(column)

        if len(numeric_cols) < 2 or len(text_cols) < 1:
            return False, None, None, []

        if len(text_cols) == 1:
            # Single categorical axis — e.g. just "Sub Variable"
            return True, None, text_cols[0], numeric_cols

        # Two or more text columns: the one with fewer unique
        # non-null values relative to row count is more likely the
        # grouping column (merged/repeated cells in the source table);
        # the other is the sub-category column.
        text_cols_sorted = sorted(
            text_cols,
            key=lambda c: dataframe[c].nunique(dropna=True),
        )

        grouping_col = text_cols_sorted[0]
        subcategory_col = text_cols_sorted[1]

        return True, grouping_col, subcategory_col, numeric_cols

    # ==========================================================
    # Summary Table Normalization
    # ==========================================================

    def _normalize_summary_table(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Normalize summary tables copied from research papers.

        Returns the normalized dataframe alongside any group sample
        sizes detected from the original headers (e.g. "N=30"), so
        percentages can later be converted to counts without any
        hardcoded assumption about group size.
        """

        df = dataframe.copy()

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        renamed = {}
        sample_sizes: dict[str, int] = {}

        for column in df.columns:

            lower = column.lower()

            n_match = re.search(r"n\s*=\s*(\d+)", lower)
            n_value = int(n_match.group(1)) if n_match else None

            if "experimental" in lower:
                renamed[column] = "Experimental"
                if n_value is not None:
                    sample_sizes["Experimental"] = n_value

            elif "control" in lower:
                renamed[column] = "Control"
                if n_value is not None:
                    sample_sizes["Control"] = n_value

            elif "sub" in lower or "category" in lower:
                renamed[column] = "Sub Variable"

            elif "variable" in lower:
                renamed[column] = "Variable"

        df.rename(columns=renamed, inplace=True)

        if "Variable" in df.columns:
            df["Variable"] = (
                df["Variable"]
                .replace("", None)
                .ffill()
            )

        df = df.dropna(how="all")

        return df.reset_index(drop=True), sample_sizes

    def parse_multiple(
        self,
        sources: list[str | Path | BinaryIO],
    ) -> list[ParsedTable]:
        """
        Parse a list of uploaded files (batch multi-input).

        Parameters
        ----------
        sources
            List of file paths or uploaded file objects.

        Returns
        -------
        list[ParsedTable]
        """

        logger.info(f"Parsing batch of {len(sources)} input sources...")
        tables = []
        for src in sources:
            try:
                table = self.parse(src)
                tables.append(table)
            except Exception as ex:
                logger.error(f"Failed to parse source '{src}': {ex}")

        return tables

    # ==========================================================
    # Summary Metadata
    # ==========================================================

    def _values_look_like_percentages(
        self,
        dataframe: pd.DataFrame,
    ) -> bool:
        """
        Heuristically detect whether Experimental/Control values are
        percentages rather than raw counts, by checking whether values
        sum to ~100 per variable group. This is a generic signal that
        works for any dataset — it never assumes a specific N.
        """

        if "Variable" not in dataframe.columns:
            return False

        group_cols = [
            c for c in ("Experimental", "Control")
            if c in dataframe.columns
        ]

        if not group_cols:
            return False

        near_100_count = 0
        total_checked = 0

        for _, group_df in dataframe.groupby("Variable"):
            for col in group_cols:
                total = pd.to_numeric(
                    group_df[col], errors="coerce"
                ).fillna(0).sum()

                total_checked += 1
                if 98 <= total <= 102:
                    near_100_count += 1

        if total_checked == 0:
            return False

        return (near_100_count / total_checked) > 0.7

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