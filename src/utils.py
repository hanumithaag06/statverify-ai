"""
Shared utility functions.

This module contains only generic reusable helpers.

DO NOT place business logic here.

Examples of what belongs here:
    - Logging
    - Time helpers
    - DataFrame checks
    - Safe numeric conversion
    - Text normalization

Examples of what DOES NOT belong here:
    - Statistical calculations
    - AI prompts
    - Parsing CSV
    - Report generation
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.config import settings

# ==========================================================
# Logger Configuration
# ==========================================================

LOG_FILE = Path(settings.log_dir) / "statverify.log"

logger.remove()

logger.add(
    sink=LOG_FILE,
    level=settings.log_level,
    rotation="5 MB",
    retention="10 days",
    enqueue=True,
)

logger.add(
    sink=lambda msg: print(msg, end=""),
    level=settings.log_level,
)

# ==========================================================
# Time Utilities
# ==========================================================


def current_timestamp() -> str:
    """
    Return current timestamp.

    Returns
    -------
    str
        ISO formatted timestamp.
    """

    return datetime.now().isoformat(timespec="seconds")


# ==========================================================
# DataFrame Utilities
# ==========================================================


def dataframe_is_empty(df: pd.DataFrame) -> bool:
    """
    Check whether DataFrame is empty.

    Parameters
    ----------
    df
        Input DataFrame.

    Returns
    -------
    bool
    """

    return df.empty


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names.

    - remove extra spaces
    - replace multiple spaces
    - convert to string

    Parameters
    ----------
    df

    Returns
    -------
    DataFrame
    """

    dataframe = df.copy()

    dataframe.columns = (
        dataframe.columns.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )

    return dataframe


# ==========================================================
# Type Utilities
# ==========================================================


def safe_float(value: Any) -> float | None:
    """
    Safely convert any object to float.

    Returns None if conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    """
    Safely convert object to integer.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ==========================================================
# Text Utilities
# ==========================================================


def normalize_text(text: str) -> str:
    """
    Normalize whitespace.

    Parameters
    ----------
    text

    Returns
    -------
    str
    """

    return " ".join(text.split())


# ==========================================================
# Logging Helpers
# ==========================================================


def log_start(task: str) -> None:
    """
    Log start of an operation.
    """

    logger.info(f"Started : {task}")


def log_end(task: str) -> None:
    """
    Log completion of an operation.
    """

    logger.success(f"Finished: {task}")


def log_error(task: str, error: Exception) -> None:
    """
    Log error details.
    """

    logger.exception(f"{task} failed -> {error}")
