"""
Centralized application configuration.

This module is the single source of truth for all application settings.
No other module should directly access environment variables.

Usage:
    from src.config import settings

Example:
    print(settings.app_name)
    print(settings.model_name)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------

load_dotenv()

# -------------------------------------------------------------------
# Project Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EXPORT_DIR = PROJECT_ROOT / "exports"
LOG_DIR = PROJECT_ROOT / "logs"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Automatically create required directories
for directory in (DATA_DIR, EXPORT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Application Settings
# -------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Application configuration loaded from .env.

    Environment variables automatically override defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===============================================================
    # Application
    # ===============================================================

    app_name: str = Field(default="StatVerify AI")
    app_version: str = Field(default="1.0.0")

    app_env: Literal["development", "testing", "production"] = "development"

    # ===============================================================
    # AI
    # ===============================================================

    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://openrouter.ai/api/v1")
    model_name: str = Field(default="qwen/qwen3-8b")

    # ===============================================================
    # Logging
    # ===============================================================

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # ===============================================================
    # Database
    # ===============================================================

    database_url: str = Field(default="sqlite:///statverify.db")

    # ===============================================================
    # Export
    # ===============================================================

    default_export_format: Literal[
        "pdf",
        "excel",
        "csv",
        "docx",
    ] = "pdf"

    # ===============================================================
    # Statistics
    # ===============================================================

    significance_level: float = Field(default=0.05, ge=0.0, le=1.0)

    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)

    # ===============================================================
    # Statistical Thresholds
    # ===============================================================

    default_alpha: float = Field(default=0.05)

    normality_alpha: float = Field(default=0.05)

    variance_alpha: float = Field(default=0.05)

    max_missing_ratio: float = Field(default=0.30)

    minimum_expected_frequency: int = Field(default=5)

    minimum_sample_size: int = Field(default=3)

    correlation_method: Literal[
        "pearson",
        "spearman",
    ] = "pearson"

    # ===============================================================
    # Directories
    # ===============================================================

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def export_dir(self) -> Path:
        return EXPORT_DIR

    @property
    def log_dir(self) -> Path:
        return LOG_DIR

    @property
    def assets_dir(self) -> Path:
        return ASSETS_DIR


# -------------------------------------------------------------------
# Singleton Settings Object
# -------------------------------------------------------------------

settings = Settings()
