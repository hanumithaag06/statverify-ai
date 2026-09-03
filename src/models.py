"""
Pydantic models used across the application.

These models provide a strongly typed interface between modules.

Workflow

UI
 ↓
AnalysisRequest
 ↓
Parser
 ↓
ParsedTable
 ↓
Statistics Engine
 ↓
StatisticsResult
 ↓
Report Generator
 ↓
Export

No dictionaries should be exchanged between modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ==========================================================
# Enums
# ==========================================================


class AnalysisMode(str, Enum):
    """Supported analysis modes."""

    CALCULATION = "calculation"
    VERIFICATION = "verification"


class InputType(str, Enum):
    """Supported input formats."""

    CLIPBOARD = "clipboard"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    IMAGE = "image"


class ValueType(str, Enum):
    COUNT = "count"
    PERCENTAGE = "percentage"


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


class StatisticalTest(str, Enum):

    # Parametric
    T_TEST = "t_test"
    PAIRED_T_TEST = "paired_t_test"
    ANOVA = "anova"

    # Non Parametric
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL = "kruskal"
    FRIEDMAN = "friedman"

    # Categorical
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"

    # Correlation
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"

    # Regression
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    
    # Fallback
    UNKNOWN = "unknown"


# ==========================================================
# User Request
# ==========================================================


class ColumnRoleMapping(BaseModel):
    """
    User-confirmed mapping of raw columns to statistical roles for a
    group-comparison summary table (e.g. Experimental vs Control).

    This is built interactively via the UI — it is never inferred
    silently, since column names and count-vs-percentage semantics
    cannot be reliably guessed for arbitrary research tables.
    """

    grouping_column: str | None = None
    """Column identifying the demographic/clinical variable
    (e.g. 'Demographic Variables'). None for single-variable tables."""

    sub_category_column: str

    group_value_columns: list[str]
    """The 2+ columns holding values per group, in display order."""

    value_type: ValueType

    group_sample_sizes: dict[str, int] = Field(default_factory=dict)
    """Required only when value_type == PERCENTAGE. Keys match
    group_value_columns."""


class AnalysisRequest(BaseModel):
    """
    Represents one user analysis request.
    """

    mode: AnalysisMode

    input_type: InputType

    source_name: str | None = None

    user_notes: str | None = None

    role_mapping: ColumnRoleMapping | None = None


# ==========================================================
# Column Metadata
# ==========================================================


class ColumnMetadata(BaseModel):
    """
    Rich statistical metadata for one dataframe column.
    """

    name: str

    variable_type: VariableType

    pandas_dtype: str

    total_values: int

    non_missing_values: int

    missing_values: int

    unique_values: int

    unique_ratio: float

    numeric_ratio: float

    text_ratio: float

    is_numeric: bool

    is_text: bool

    is_binary: bool

    is_constant: bool

    is_identifier: bool

    inferred_type: str


# ==========================================================
# Dataset Metadata
# ==========================================================


class DatasetMetadata(BaseModel):
    """
    Metadata describing the entire dataset.
    """

    rows: int

    columns: int

    duplicate_rows: int

    duplicate_columns: int

    total_missing_cells: int

    memory_usage_bytes: int


# ==========================================================
# Validation Report
# ==========================================================


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationMessage(BaseModel):
    """
    Represents a single validation finding.
    """

    severity: ValidationSeverity

    message: str

    column: str | None = None


class ValidationReport(BaseModel):
    """
    Complete validation report for a parsed table.
    """

    messages: list[ValidationMessage] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(
            msg.severity == ValidationSeverity.ERROR
            for msg in self.messages
        )

    @property
    def has_warnings(self) -> bool:
        return any(
            msg.severity == ValidationSeverity.WARNING
            for msg in self.messages
        )


# ==========================================================
# Parsed Table
# ==========================================================


class ColumnMetadataList(list):
    analysis_type: str = "raw"


class ParsedTable(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    dataframe: pd.DataFrame = Field(exclude=True)

    metadata: list[ColumnMetadata]

    validation: ValidationReport = Field(
        default_factory=ValidationReport
    )

    summary_metadata: dict[str, Any] = Field(default_factory=dict)
    """Variable-level metadata inferred from summary tables.

    Structure (per variable key):
        {
            "categories": list[str],
            "groups":     list[str],
            "rows":       int,
        }

    Empty dict when the input is a raw dataset.
    """

    requires_sample_size_input: bool = False
    """True when the parser detected a summary/percentage table but
    could not determine group sample sizes from the headers. The UI
    must prompt the user for sizes before percent-to-count conversion
    can occur — sizes are never assumed or hardcoded."""

    is_summary_shaped: bool = False
    """True when structural detection found a group-comparison shape
    (grouping/subcategory column + 2+ value columns), regardless of
    header wording."""

    requires_role_confirmation: bool = False
    """True when the table looks like a summary table but couldn't be
    parsed via known header keywords — the UI must collect a
    ColumnRoleMapping before statistics can run."""

    candidate_grouping_column: str | None = None
    candidate_subcategory_column: str | None = None
    candidate_value_columns: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        meta_list = ColumnMetadataList(self.metadata)
        if self.summary_metadata:
            meta_list.analysis_type = "summary"
        else:
            meta_list.analysis_type = "raw"
        object.__setattr__(self, "metadata", meta_list)

    @property
    def rows(self) -> int:
        return len(self.dataframe)

    @property
    def columns(self) -> int:
        return len(self.dataframe.columns)


# ==========================================================
# Statistical Context and Recommendation
# ==========================================================


class StatisticalContext(BaseModel):
    """
    Context for executing statistical tests.
    """

    dependent_variable: str | None = None

    independent_variable: str | None = None

    group_variable: list[str] | str | None = None


class TestRecommendation(BaseModel):
    """
    Recommended statistical test.
    """

    test: StatisticalTest

    reason: str

    confidence: float


class CalculationResult(BaseModel):
    """
    Result of a statistical calculation.
    """

    statistic: float

    p_value: float

    degrees_of_freedom: float | None = None

    additional_metrics: dict[str, Any] = Field(default_factory=dict)

# ==========================================================
# Statistical Assumptions
# ==========================================================

class AssumptionStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class AssumptionResult(BaseModel):
    """
    Result of one statistical assumption.
    """

    name: str

    status: AssumptionStatus

    message: str


class AssumptionReport(BaseModel):
    """
    Complete assumption checking report.
    """

    assumptions: list[AssumptionResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(
            item.status != AssumptionStatus.FAIL
            for item in self.assumptions
        )



# ==========================================================
# Statistical Test Result
# ==========================================================


class StatisticalResult(BaseModel):
    """
    Generic statistical result.

    Works for every statistical test.
    """

    test_name: str

    statistic: float | None = None

    p_value: float | None = None

    degrees_of_freedom: float | None = None

    confidence_interval: tuple[float, float] | None = None

    effect_size: float | None = None

    assumptions: list[AssumptionResult] = Field(default_factory=list)

    interpretation: str | None = None

    additional_metrics: dict[str, Any] = Field(default_factory=dict)

    additional_results: list[dict[str, Any]] = Field(default_factory=list)


# ==========================================================
# Comparison Models
# ==========================================================


class ComparisonItem(BaseModel):
    """
    Comparison for one statistical metric.
    """

    metric: str

    reported: float | None = None

    calculated: float | None = None

    matched: bool

    difference: float | None = None


class ComparisonResult(BaseModel):
    """
    Collection of all comparisons.
    """

    passed: bool

    items: list[ComparisonItem]

    summary: str


# ==========================================================
# Verification Result
# ==========================================================


class VerificationResult(BaseModel):
    """
    Final verification output.
    """

    verified: bool

    comparison: ComparisonResult

    reported_result: StatisticalResult

    calculated_result: StatisticalResult

    message: str


# ==========================================================
# AI Explanation
# ==========================================================


class AIExplanation(BaseModel):
    """
    Response generated by the AI layer.

    AI never performs calculations.
    """

    summary: str

    explanation: str

    recommendation: str | None = None


# ==========================================================
# Final Report
# ==========================================================


class ReportData(BaseModel):
    """
    Complete report passed to exporters.
    """

    request: AnalysisRequest

    statistics: StatisticalResult

    verification: VerificationResult | None = None

    ai: AIExplanation | None = None


# ==========================================================
# Statistical Dataset
# ==========================================================


class StatisticalDataset(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame = Field(exclude=True)

    metadata: list[ColumnMetadata] = Field(default_factory=list)

    summary_metadata: dict[str, Any] = Field(default_factory=dict)

    sample_size: int = 0

    groups: dict[str, list[float]] = Field(default_factory=dict)

    numeric_columns: dict[str, list[float]] = Field(default_factory=dict)

    categorical_columns: dict[str, list[str]] = Field(default_factory=dict)

    contingency_tables: dict[str, np.ndarray] = Field(default_factory=dict)

    category_labels: dict[str, list[str]] = Field(default_factory=dict)
    """Sub-variable category names aligned by row-index to each
    entry in `contingency_tables`. Used to render per-category
    No./% breakdowns without ever hardcoding category names."""

    sample_sizes: dict[str, int] = Field(default_factory=dict)

    group_names: list[str] = Field(default_factory=list)
    """Explicit, ordered group names (column labels the user confirmed
    or that were detected via header keywords). Used instead of relying
    on dict key order, which isn't guaranteed to reflect table order."""

    x: list[float] | None = None
    y: list[float] | None = None

    is_summary: bool = False
